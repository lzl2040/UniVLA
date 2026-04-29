"""
finetune_lerobot_full.py

Full fine-tuning script for UniVLA on LeRobot data, following the official UniVLA training style.
Includes LAM (Latent Action Model) support for generating latent action tokens.

Usage:
    python finetune_lerobot_full.py \
        --vla_path /Data/lzl/huggingface/univla-7b \
        --lam_path /Data/lzl/huggingface/univla-latent-action-model/lam-stage-2.ckpt \
        --data_dir /Data/rlds_raw/lerobot_converted_data/block \
        --run_dir ./runs/lerobot_full
"""

import os
import json
import argparse
import logging
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Optional, List, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
import h5py
import numpy as np
from PIL import Image
from einops import rearrange

from accelerate import PartialState, Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.policy.transformer_utils import MAPBlock
from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_logging(log_dir: str, log_name: str = "train.log") -> logging.Logger:
    """Setup logging to file and console.
    
    Args:
        log_dir: Directory to save log files
        log_name: Name of the log file
    
    Returns:
        Logger instance
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / log_name
    
    # Create logger
    logger = logging.getLogger("UniVLA")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


class ActionDecoder(nn.Module):
    """Action decoder for predicting continuous actions with proprio support."""
    def __init__(self, window_size=10, hidden_dim=512, action_dim=7, proprio_dim=8):
        super().__init__()
        self.attn_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, action_dim * window_size),
        )

    def forward(self, latent_action_tokens, visual_embed, proprio):
        proprio = self.proprio_proj(proprio)
        visual_embed = self.visual_pool(visual_embed)
        action = self.proj(torch.cat([self.attn_pool(latent_action_tokens, init_embed=visual_embed), proprio], dim=-1))
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

    def forward(self, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vla_output = self.vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                output_hidden_states=True,
            )
        loss, loss_one_step, latent_action_tokens = self.action_decoder_forward(batch, vla_output)
        return vla_output, loss, loss_one_step, latent_action_tokens

    def action_decoder_forward(self, batch, vla_output):
        # Extract visual embeddings and latent tokens
        visual_embed = vla_output.hidden_states[-1][:, :256].to(torch.float)  # 256 patches
        latent_tokens = vla_output.hidden_states[-1][:, 256:].to(torch.float)
        
        # Get action tokens from labels
        action_gt = batch["labels"].to(latent_tokens.device)
        mask = action_gt > 32000  # Action tokens have IDs > 32000

        # Extract latent action tokens
        latent_action_tokens = []
        for idx, per_sample_latent_tokens in enumerate(latent_tokens):
            per_sample_latent_action_tokens = per_sample_latent_tokens[mask[idx], :]
            latent_action_tokens.append(per_sample_latent_action_tokens)
        
        if len(latent_action_tokens[0]) > 0:
            latent_action_tokens = torch.stack(latent_action_tokens).to(torch.float)
            pred_action = self.action_decoder(latent_action_tokens, visual_embed, batch['proprio']).reshape(-1, self.window_size, self.action_dim)
        else:
            # Fallback if no action tokens found
            pred_action = torch.zeros(batch['actions'].shape, device=batch['actions'].device)
        
        loss = nn.functional.l1_loss(pred_action, batch['actions'], reduction='none')
        loss_one_step = loss[:, 0].mean()
        loss = loss.mean()

        return loss, loss_one_step, latent_action_tokens


class LeRobotHDF5Dataset(Dataset):
    """Dataset for LeRobot HDF5 format data with LAM support."""
    def __init__(
        self,
        data_dir: str,
        norm_stats: dict,
        image_transform=None,
        window_size: int = 10,
        action_dim: int = 7,
        proprio_dim: int = 7,
    ):
        self.data_dir = Path(data_dir)
        self.image_transform = image_transform
        self.window_size = window_size
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.norm_stats = norm_stats
        self.resize_img = transforms.Resize((224, 224))
        self.image_transform_lam = transforms.ToTensor()
        self.color_aug = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

        # Load episodes
        self.episodes = sorted(self.data_dir.glob("episode_*.hdf5"))
        print(f"Found {len(self.episodes)} episodes")

        # Load all data into memory
        self.images = []
        self.actions = []
        self.proprios = []
        self.instructions = []
        
        for ep_path in self.episodes:
            with h5py.File(ep_path, 'r') as f:
                num_frames = f['action'].shape[0]
                
                # Load images
                imgs = f['right_rgb'][:]  # (T, H, W, 3)
                self.images.append(imgs)
                
                # Load actions
                actions = f['action'][:]
                self.actions.append(actions)
                
                # Load proprio (use action as proprio if not available)
                if 'observations' in f and 'qpos' in f['observations']:
                    proprio = f['observations/qpos'][:]
                else:
                    # Use joint positions from raw_action if available, else zeros
                    if 'raw_action' in f:
                        proprio = f['raw_action'][:, :proprio_dim]
                    else:
                        proprio = np.zeros((num_frames, proprio_dim))
                self.proprios.append(proprio)
                
                # Use dataset_info.json for instructions
                # if "block" in data_dir:
                #     instruction = "Put the building block into the corresponding slot."  # Default
                # elif "cup" in data_dir:
                #     instruction = ""
                # print(f['task'])
                task = f.attrs['task']
                # print(task)
                self.instructions.append(task)

        print(f"Total episodes: {len(self.episodes)}")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        images = self.images[idx]
        actions = self.actions[idx]
        proprio = self.proprios[idx]
        instruction = self.instructions[idx]
        
        num_frames = len(actions)
        extra_frame_num = np.random.randint(0, 1)
        window_size = self.window_size + extra_frame_num
        
        # Random start index
        image_index = np.random.choice(max(1, num_frames - window_size))
        
        # Get image chunking
        image_chunking = images[image_index:image_index + window_size]
        
        # VLA image (current frame with color augmentation)
        image_vla = Image.fromarray(image_chunking[extra_frame_num])
        image_vla = self.color_aug(image_vla)
        pixel_values = self.image_transform(image_vla)
        
        # LAM images (initial and target)
        goal_image = Image.fromarray(image_chunking[-1])
        initial_pixel_values = self.image_transform_lam(self.resize_img(image_vla))
        target_pixel_values = self.image_transform_lam(self.resize_img(goal_image))
        
        # History frames (for optional history conditioning)
        initial_pixel_values_hist, target_pixel_values_hist = None, None
        if extra_frame_num > 0 and len(image_chunking) > self.window_size:
            hist_frame_prev = Image.fromarray(image_chunking[0])
            hist_frame_goal = Image.fromarray(image_chunking[self.window_size])
            initial_pixel_values_hist = self.image_transform_lam(self.resize_img(hist_frame_prev))
            target_pixel_values_hist = self.image_transform_lam(self.resize_img(hist_frame_goal))
        
        # Get actions
        actions_chunking = np.zeros((self.window_size, self.action_dim))
        actual_len = min(num_frames - image_index, self.window_size)
        actions_chunking[:actual_len] = actions[image_index:image_index + actual_len]
        
        # Normalize actions
        action_tensor = torch.tensor(actions_chunking, dtype=torch.float32)
        action_tensor = (action_tensor - torch.tensor(self.norm_stats['action_mean'])) / torch.tensor(self.norm_stats['action_std'])
        
        # Get proprio (current position)
        proprio_tensor = torch.tensor(proprio[image_index], dtype=torch.float32)
        if 'qpos_mean' in self.norm_stats:
            proprio_tensor = (proprio_tensor - torch.tensor(self.norm_stats['qpos_mean'])) / torch.tensor(self.norm_stats['qpos_std'])
        
        return {
            'pixel_values': pixel_values,
            'initial_pixel_values': initial_pixel_values,
            'target_pixel_values': target_pixel_values,
            'initial_pixel_values_hist': initial_pixel_values_hist,
            'target_pixel_values_hist': target_pixel_values_hist,
            'actions': action_tensor,
            'proprio': proprio_tensor,
            'instruction': instruction,
            'with_hist': extra_frame_num > 0,
        }


class PaddedCollatorForActionPrediction:
    """Collator for action prediction with LAM support."""
    def __init__(self, model_max_length: int, pad_token_id: int, padding_side: str = "right"):
        self.model_max_length = model_max_length
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Stack images
        pixel_values = torch.stack([inst["pixel_values"] for inst in instances])
        initial_pixel_values = torch.stack([inst["initial_pixel_values"] for inst in instances])
        target_pixel_values = torch.stack([inst["target_pixel_values"] for inst in instances])
        
        # Handle history images
        initial_pixel_values_hist, target_pixel_values_hist, with_hist = [], [], []
        for inst in instances:
            if inst["initial_pixel_values_hist"] is not None:
                initial_pixel_values_hist.append(inst["initial_pixel_values_hist"])
                target_pixel_values_hist.append(inst["target_pixel_values_hist"])
                with_hist.append(torch.tensor(True))
            else:
                with_hist.append(torch.tensor(False))
        
        # Stack actions and proprio
        actions = torch.stack([inst["actions"] for inst in instances])
        proprio = torch.stack([inst["proprio"] for inst in instances])
        instructions = [inst["instruction"] for inst in instances]
        
        with_hist = torch.stack(with_hist)
        initial_pixel_values_hist = torch.stack(initial_pixel_values_hist) if len(initial_pixel_values_hist) > 0 else []
        target_pixel_values_hist = torch.stack(target_pixel_values_hist) if len(target_pixel_values_hist) > 0 else []

        return {
            'pixel_values': pixel_values,
            'initial_pixel_values': initial_pixel_values,
            'target_pixel_values': target_pixel_values,
            'initial_pixel_values_hist': initial_pixel_values_hist,
            'target_pixel_values_hist': target_pixel_values_hist,
            'actions': actions,
            'proprio': proprio,
            'instructions': instructions,
            'with_hist': with_hist,
        }


def get_norm_stats(data_dir: str, action_dim: int = 7):
    """Compute normalization statistics from all episodes."""
    data_path = Path(data_dir)
    episodes = sorted(data_path.glob("episode_*.hdf5"))
    
    all_actions = []
    all_proprios = []
    
    for ep_path in episodes:
        with h5py.File(ep_path, 'r') as f:
            actions = f['action'][:]
            all_actions.append(torch.from_numpy(actions))
            
            if 'observations' in f and 'qpos' in f['observations']:
                proprio = f['observations/qpos'][:]
            elif 'raw_action' in f:
                proprio = f['raw_action'][:, :action_dim]
            else:
                proprio = np.zeros((actions.shape[0], action_dim))
            all_proprios.append(torch.from_numpy(proprio))
    
    all_actions = torch.cat(all_actions, dim=0).float()
    all_proprios = torch.cat(all_proprios, dim=0).float()
    
    action_mean = all_actions.mean(dim=0).numpy()
    action_std = all_actions.std(dim=0).clamp(min=1e-2).numpy()
    
    qpos_mean = all_proprios.mean(dim=0).numpy()
    qpos_std = all_proprios.std(dim=0).clamp(min=1e-2).numpy()
    
    return {
        'action_mean': action_mean,
        'action_std': action_std,
        'qpos_mean': qpos_mean,
        'qpos_std': qpos_std,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vla_path', type=str, default="/mnt/wangxiaofa/pt_weights/univla-7b", help='Path to UniVLA model')
    parser.add_argument('--lam_path', type=str, default="/mnt/wangxiaofa/pt_weights/univla-latent-action-model/lam-stage-2.ckpt", help='Path to LAM checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to converted data')
    parser.add_argument('--run_dir', type=str, default='./runs/lerobot_full')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--max_steps', type=int, default=40000)
    parser.add_argument('--save_steps', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=3.5e-4)
    parser.add_argument('--grad_accumulation_steps', type=int, default=2)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--action_dim', type=int, default=7)
    parser.add_argument('--proprio_dim', type=int, default=7)
    parser.add_argument('--use_lora', type=bool, default=True)
    parser.add_argument('--lora_rank', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--freeze_vla', type=bool, default=False)
    parser.add_argument('--image_aug', type=bool, default=False)
    parser.add_argument('--log_dir', type=str, default=None, help='Directory to save logs (default: run_dir/logs)')
    args = parser.parse_args()

    # Create run directory
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_dir = Path(args.log_dir) if args.log_dir else run_dir / 'logs'
    logger = setup_logging(log_dir, log_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger.info("=" * 50)
    logger.info("UniVLA Fine-tuning Configuration")
    logger.info("=" * 50)
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("=" * 50)

    # Set device
    # assert torch.cuda.is_available(), "Fine-tuning requires GPU!"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    logger.info(f"Using device: {device}")

    # Register OpenVLA model classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load processor and model
    logger.info(f"Loading model from {args.vla_path}")
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
    logger.info(f"Total trainable parameters: {trainable_params:,}")

    # Load LAM model if provided
    lam = None
    if args.lam_path and os.path.exists(args.lam_path):
        logger.info(f"Loading LAM from {args.lam_path}")
        lam = ControllableDINOLatentActionModel(
            in_dim=3,
            model_dim=768,
            latent_dim=128,
            num_latents=16,
            patch_size=14,
            enc_blocks=12,
            dec_blocks=12,
            num_heads=12,
        )
        lam_ckpt = torch.load(args.lam_path)['state_dict']
        new_ckpt = {}
        for key in lam_ckpt.keys():
            new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]
        lam.load_state_dict(new_ckpt, strict=True)
        lam = lam.to(device).eval()
        logger.info("LAM loaded successfully!")

    # Create optimizer and scheduler
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(args.max_steps * 0.8), gamma=0.1
    )

    # Get normalization stats
    norm_stats = get_norm_stats(args.data_dir, args.action_dim)
    
    # Save normalization stats
    with open(run_dir / 'norm_stats.json', 'w') as f:
        json.dump({k: v.tolist() if hasattr(v, 'tolist') else v for k, v in norm_stats.items()}, f)

    # Create dataset and dataloader
    image_transform = processor.image_processor.apply_transform

    dataset = LeRobotHDF5Dataset(
        data_dir=args.data_dir,
        norm_stats=norm_stats,
        image_transform=image_transform,
        window_size=args.window_size,
        action_dim=args.action_dim,
        proprio_dim=args.proprio_dim,
    )

    collator = PaddedCollatorForActionPrediction(
        model_max_length=processor.tokenizer.model_max_length,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collator,
    )

    # Training loop
    logger.info("Starting training...")
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
                batch['pixel_values'] = batch['pixel_values'].to(torch.bfloat16).to(device)
                batch['initial_pixel_values'] = batch['initial_pixel_values'].to(device)
                batch['target_pixel_values'] = batch['target_pixel_values'].to(device)
                batch['actions'] = batch['actions'].to(device)
                batch['proprio'] = batch['proprio'].to(device)

                # Generate latent action tokens using LAM
                if lam is not None:
                    with torch.no_grad():
                        video = torch.stack([batch["initial_pixel_values"], batch["target_pixel_values"]], dim=1)
                        lam_output = lam.vq_encode(video)
                        latent_action_idx = lam_output['indices'].squeeze()
                    
                    # Build input_ids and labels from latent action tokens
                    input_ids_list = []
                    labels_list = []
                    
                    for idx in range(len(batch['instructions'])):
                        # Create action tokens like <ACT_0><ACT_1>...
                        action_vocab = [f'<ACT_{i.item()}>' for i in latent_action_idx[idx]]
                        action_tokens = ''.join(action_vocab)
                        
                        # Build prompt
                        input_prompt = f"What action should the robot take to {batch['instructions'][idx].lower()}?"
                        
                        prompt_builder = PurePromptBuilder("openvla")
                        conversation = [
                            {"from": "human", "value": input_prompt},
                            {"from": "gpt", "value": action_tokens},
                        ]
                        for turn in conversation:
                            prompt_builder.add_turn(turn["from"], turn["value"])
                        
                        # Tokenize
                        input_ids = processor.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
                        labels = list(input_ids)
                        
                        input_ids = torch.tensor(input_ids)
                        labels = torch.tensor(labels)
                        
                        # Mask non-action tokens in labels
                        labels[:-(len(action_vocab) + 1)] = -100
                        
                        input_ids_list.append(input_ids)
                        labels_list.append(labels)
                    
                    # Pad sequences
                    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
                    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
                    
                    # Truncate if needed
                    input_ids = input_ids[:, :processor.tokenizer.model_max_length]
                    labels = labels[:, :processor.tokenizer.model_max_length]
                    
                    attention_mask = input_ids.ne(processor.tokenizer.pad_token_id)
                    
                    batch['input_ids'] = input_ids.to(device)
                    batch['attention_mask'] = attention_mask.to(device)
                    batch['labels'] = labels.to(device)
                else:
                    # Without LAM, use simple prompt (no latent action conditioning)
                    prompts = [f"What action should the robot take to {inst.lower()}?" for inst in batch['instructions']]
                    tokenized = processor.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                    batch['input_ids'] = tokenized['input_ids'].to(device)
                    batch['attention_mask'] = tokenized['attention_mask'].to(device)
                    batch['labels'] = torch.full_like(tokenized['input_ids'], -100).to(device)

                # Forward pass
                vla_output, loss, loss_one_step, latent_action_tokens = model(batch)

                # Compute total loss
                total_loss = loss if args.freeze_vla else loss + vla_output.loss

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
                    logger.info(f"Step {step}: loss={avg_loss:.4f}, l1={avg_l1:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

                # Save checkpoint
                if step > 0 and step % args.save_steps == 0:
                    logger.info(f"Saving checkpoint at step {step}")
                    torch.save(
                        model.action_decoder.state_dict(),
                        run_dir / f'action_decoder-{step}.pt',
                    )
                    if args.use_lora:
                        model.vla.save_pretrained(run_dir / f'adapter-{step}')
                    else:
                        model.vla.save_pretrained(run_dir / f'model-{step}')

                step += 1

    # Final save
    logger.info("Training complete!")
    torch.save(model.action_decoder.state_dict(), run_dir / 'action_decoder-final.pt')
    if args.use_lora:
        model.vla.save_pretrained(run_dir / 'adapter')
    else:
        model.vla.save_pretrained(run_dir)
    logger.info(f"Model saved to {run_dir}")
    logger.info(f"Logs saved to {log_dir}")


if __name__ == "__main__":
    main()
