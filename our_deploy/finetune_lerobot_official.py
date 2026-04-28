"""
finetune_lerobot_official.py

Fine-tuning script for UniVLA on LeRobot data, following the official UniVLA training style.

Usage:
    python finetune_lerobot_official.py \
        --vla_path /Data/lzl/huggingface/univla-7b \
        --data_dir ./converted_data/block_hz_4 \
        --run_dir ./runs/lerobot_block
"""

import os
import json
import argparse
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
import h5py
import numpy as np
from PIL import Image

from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.policy.transformer_utils import MAPBlock

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ActionDecoder(nn.Module):
    """Action decoder for predicting continuous actions."""
    def __init__(self, window_size=10, hidden_dim=512, action_dim=7):
        super().__init__()
        self.latent_action_pool = MAPBlock(
            n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64
        )
        self.visual_pool = MAPBlock(
            n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, action_dim * window_size),
            nn.Tanh(),
        )

    def forward(self, latent_action_tokens, visual_embed):
        visual_embed = self.visual_pool(visual_embed)
        latent_action_tokens = latent_action_tokens[:, -4:]
        action_token = self.latent_action_pool(latent_action_tokens, init_embed=visual_embed)
        action = self.proj(action_token)
        return action


class WrappedModel(nn.Module):
    """Wrapper for VLA with action decoder."""
    def __init__(self, vla, freeze_vla=False, window_size=10, action_dim=7):
        super().__init__()
        self.vla = vla
        self.window_size = window_size
        self.action_dim = action_dim
        self.action_decoder = ActionDecoder(window_size=window_size, action_dim=action_dim)

        if freeze_vla:
            self.vla.requires_grad_(False)

    def forward(self, input_ids, attention_mask, pixel_values, actions=None):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vla_output = self.vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
            )

        # Extract visual embeddings and latent tokens
        visual_embed = vla_output.hidden_states[-1][:, :256].float()  # 256 patches
        latent_tokens = vla_output.hidden_states[-1][:, 256:].float()

        # For now, use all latent tokens (simplified compared to official)
        pred_action = self.action_decoder(latent_tokens, visual_embed).reshape(-1, self.window_size, self.action_dim)

        if actions is not None:
            loss = nn.functional.l1_loss(pred_action, actions, reduction='none')
            loss_one_step = loss[:, 0].mean()
            loss_total = loss.mean()
            return vla_output, loss_total, loss_one_step, pred_action

        return vla_output, None, None, pred_action


class LeRobotHDF5Dataset(Dataset):
    """Dataset for LeRobot HDF5 format data."""
    def __init__(
        self,
        data_dir: str,
        image_transform=None,
        window_size: int = 10,
        action_dim: int = 7,
        prompt_builder_fn=None,
    ):
        self.data_dir = Path(data_dir)
        self.image_transform = image_transform
        self.window_size = window_size
        self.action_dim = action_dim
        self.prompt_builder_fn = prompt_builder_fn or PurePromptBuilder

        # Load normalization stats
        norm_stats_path = self.data_dir / "norm_stats.json"
        if norm_stats_path.exists():
            with open(norm_stats_path) as f:
                self.norm_stats = json.load(f)
            self.action_mean = np.array(self.norm_stats['action_mean'])
            self.action_std = np.array(self.norm_stats['action_std'])
        else:
            self.action_mean = np.zeros(action_dim)
            self.action_std = np.ones(action_dim)
            self.norm_stats = None

        # Load episodes
        self.episodes = sorted(self.data_dir.glob("episode_*.hdf5"))
        print(f"Found {len(self.episodes)} episodes")

        # Build sample index
        self.samples = []
        for ep_idx, ep_path in enumerate(self.episodes):
            with h5py.File(ep_path, 'r') as f:
                num_frames = f['action'].shape[0]
                for i in range(num_frames - window_size + 1):
                    self.samples.append({
                        'ep_idx': ep_idx,
                        'ep_path': str(ep_path),
                        'start_idx': i,
                    })
        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        with h5py.File(sample['ep_path'], 'r') as f:
            # Load image
            img = f['top_rgb'][sample['start_idx']]
            img = Image.fromarray(img)
            if self.image_transform:
                pixel_values = self.image_transform(img)
            else:
                pixel_values = transforms.ToTensor()(img)

            # Load actions
            actions = f['action'][sample['start_idx']:sample['start_idx'] + self.window_size]
            actions = (actions - self.action_mean) / self.action_std
            actions = torch.tensor(actions, dtype=torch.float32)

        # Build prompt (simple task description)
        prompt = "What action should the robot take to pick up the block?"

        return {
            'pixel_values': pixel_values,
            'actions': actions,
            'prompt': prompt,
        }


def collate_fn(batch, tokenizer, max_length=512):
    """Collate batch with tokenization."""
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    actions = torch.stack([b['actions'] for b in batch])
    prompts = [b['prompt'] for b in batch]

    # Tokenize
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'pixel_values': pixel_values,
        'actions': actions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vla_path', type=str, required=True, help='Path to UniVLA model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to converted data')
    parser.add_argument('--run_dir', type=str, default='./runs/lerobot')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=3.5e-4)
    parser.add_argument('--grad_accumulation_steps', type=int, default=2)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--action_dim', type=int, default=7)
    parser.add_argument('--use_lora', type=bool, default=True)
    parser.add_argument('--lora_rank', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--freeze_vla', type=bool, default=False)
    parser.add_argument('--image_aug', type=bool, default=False)
    args = parser.parse_args()

    # Set device
    assert torch.cuda.is_available(), "Fine-tuning requires GPU!"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Create run directory
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Register OpenVLA model classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load processor and model
    print(f"Loading model from {args.vla_path}")
    processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    vla = vla.to(device)

    # Apply LoRA
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=min(args.lora_rank, 16),
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Create wrapped model
    model = WrappedModel(
        vla=vla,
        freeze_vla=args.freeze_vla,
        window_size=args.window_size,
        action_dim=args.action_dim,
    ).to(device)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_params:,}")

    # Create optimizer and scheduler
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(args.max_steps * 0.8), gamma=0.1
    )

    # Create dataset and dataloader
    image_transform = processor.image_processor.apply_transform

    dataset = LeRobotHDF5Dataset(
        data_dir=args.data_dir,
        image_transform=image_transform,
        window_size=args.window_size,
        action_dim=args.action_dim,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda b: collate_fn(b, processor.tokenizer),
    )

    # Save normalization stats
    if dataset.norm_stats:
        with open(run_dir / 'dataset_statistics.json', 'w') as f:
            json.dump(dataset.norm_stats, f)

    # Training loop
    print("Starting training...")
    recent_losses = deque(maxlen=args.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=args.grad_accumulation_steps)

    model.train()
    optimizer.zero_grad()

    step = 0
    epoch = 0

    with tqdm.tqdm(total=args.max_steps, desc="Training") as progress:
        while step < args.max_steps:
            epoch += 1
            for batch in dataloader:
                if step >= args.max_steps:
                    break

                # Move batch to device
                batch['input_ids'] = batch['input_ids'].to(device)
                batch['attention_mask'] = batch['attention_mask'].to(device)
                batch['pixel_values'] = batch['pixel_values'].to(torch.bfloat16).to(device)
                batch['actions'] = batch['actions'].to(device)

                # Forward pass
                vla_output, loss, loss_one_step, pred_action = model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['pixel_values'],
                    batch['actions'],
                )

                # Compute total loss (only use action decoder loss for now)
                total_loss = loss

                # Normalize for gradient accumulation
                normalized_loss = total_loss / args.grad_accumulation_steps

                # Backward pass
                normalized_loss.backward()

                # Store metrics
                recent_losses.append(total_loss.item())
                recent_l1_losses.append(loss.item())

                # Optimizer step
                if (step + 1) % args.grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    progress.update(1)

                # Logging
                if step % 10 == 0:
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    avg_l1 = sum(recent_l1_losses) / len(recent_l1_losses)
                    progress.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'l1': f'{avg_l1:.4f}',
                        'step': step,
                    })

                # Save checkpoint
                if step > 0 and step % args.save_steps == 0:
                    print(f"\nSaving checkpoint at step {step}")
                    torch.save(
                        model.action_decoder.state_dict(),
                        run_dir / f'action_decoder-{step}.pt',
                    )
                    if args.use_lora:
                        model.vla.save_pretrained(run_dir / 'adapter')
                    else:
                        model.vla.save_pretrained(run_dir)

                step += 1

    # Final save
    print("Training complete!")
    torch.save(model.action_decoder.state_dict(), run_dir / 'action_decoder-final.pt')
    if args.use_lora:
        model.vla.save_pretrained(run_dir / 'adapter')
    else:
        model.vla.save_pretrained(run_dir)
    print(f"Model saved to {run_dir}")


if __name__ == "__main__":
    main()
