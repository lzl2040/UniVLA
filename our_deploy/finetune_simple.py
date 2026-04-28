"""
finetune_simple.py

Simplified fine-tuning script for UniVLA on LeRobot data.
Does not depend on complex prismatic imports - uses HF transformers directly.

Usage:
    python finetune_simple.py \
        --vla_path /Data/lzl/huggingface/univla-7b \
        --data_dir ./converted_data/block_hz_4 \
        --batch_size 2 \
        --max_steps 10
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.utils.data import DataLoader
import tqdm
from transformers import AutoProcessor

# Import local copies of OpenVLA model classes (no external dependencies)
from configuration_prismatic import OpenVLAConfig
from modeling_prismatic import OpenVLAForActionPrediction
from processing_prismatic import PrismaticProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MAPBlock(nn.Module):
    """Multi-Action Prediction block for pooling latent tokens."""
    def __init__(self, n_latents=1, vis_dim=4096, embed_dim=512, n_heads=8):
        super().__init__()
        self.n_latents = n_latents
        self.embed_dim = embed_dim
        self.latents = nn.Parameter(torch.randn(1, n_latents, embed_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.proj_in = nn.Linear(vis_dim, embed_dim)
        
    def forward(self, x, init_embed=None):
        B = x.shape[0]
        latents = self.latents.expand(B, -1, -1)
        x_proj = self.proj_in(x)
        
        if init_embed is not None:
            init_proj = self.proj_in(init_embed)
            latents = latents + init_proj
        
        latents_norm = self.norm1(latents)
        x_norm = self.norm1(x_proj)
        attn_out, _ = self.cross_attn(latents_norm, x_norm, x_norm)
        latents = latents + attn_out
        
        latents_norm = self.norm2(latents)
        attn_out, _ = self.self_attn(latents_norm, latents_norm, latents_norm)
        latents = latents + attn_out
        
        return latents.squeeze(1)


class ActionDecoder(nn.Module):
    """Action decoder for predicting continuous actions."""
    def __init__(self, window_size=10, hidden_dim=512, action_dim=7, vis_dim=4096):
        super().__init__()
        self.window_size = window_size
        self.action_dim = action_dim
        self.latent_action_pool = MAPBlock(n_latents=1, vis_dim=vis_dim, embed_dim=hidden_dim)
        self.visual_pool = MAPBlock(n_latents=1, vis_dim=vis_dim, embed_dim=hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, action_dim * window_size),
            nn.Tanh(),
        )
    
    def forward(self, latent_action_tokens, visual_embed):
        visual_embed = self.visual_pool(visual_embed)
        latent_action_tokens = latent_action_tokens[:, -4:]
        action_token = self.latent_action_pool(latent_action_tokens, init_embed=visual_embed)
        action = self.proj(action_token)
        return action.reshape(-1, self.window_size, self.action_dim)


class SimpleVLAModel(nn.Module):
    """Wrapper for VLA with action decoder."""
    def __init__(self, vla_path, window_size=10, action_dim=7, use_lora=True, lora_rank=32):
        super().__init__()
        self.window_size = window_size
        self.action_dim = action_dim
        self.vis_dim = 4096
        
        print(f"Loading VLA from {vla_path}")
        self.vla = OpenVLAForActionPrediction.from_pretrained(
            vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        self.action_decoder = ActionDecoder(
            window_size=window_size,
            action_dim=action_dim,
            vis_dim=self.vis_dim,
        )
        
        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=min(lora_rank, 16),
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            self.vla = get_peft_model(self.vla, lora_config)
            print("Applied LoRA")
            self.vla.print_trainable_parameters()
    
    def forward(self, input_ids, attention_mask, pixel_values, actions=None):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
            )
        
        hidden_states = outputs.hidden_states[-1]
        num_patches = 256
        visual_embed = hidden_states[:, :num_patches].float()
        latent_tokens = hidden_states[:, num_patches:].float()
        
        pred_action = self.action_decoder(latent_tokens, visual_embed)
        
        if actions is not None:
            loss = nn.functional.l1_loss(pred_action, actions, reduction='none')
            loss_one_step = loss[:, 0].mean()
            loss_total = loss.mean()
            return loss_total, loss_one_step, pred_action
        
        return pred_action


def load_dataset(dataset_dir, window_size=10, action_dim=7):
    """Load HDF5 dataset."""
    import h5py
    import numpy as np
    
    dataset_dir = Path(dataset_dir)
    episodes = sorted(dataset_dir.glob("episode_*.h5"))
    
    print(f"Found {len(episodes)} episodes")
    
    norm_stats_path = dataset_dir / "norm_stats.json"
    if norm_stats_path.exists():
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)
        action_mean = np.array(norm_stats['action_mean'])
        action_std = np.array(norm_stats['action_std'])
    else:
        action_mean = np.zeros(action_dim)
        action_std = np.ones(action_dim)
    
    samples = []
    for ep_path in episodes:
        with h5py.File(ep_path, 'r') as f:
            num_frames = f['actions'].shape[0]
            for i in range(num_frames - window_size + 1):
                samples.append({
                    'ep_path': str(ep_path),
                    'start_idx': i,
                })
    
    print(f"Total samples: {len(samples)}")
    return samples, action_mean, action_std


class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for HDF5 files."""
    def __init__(self, samples, action_mean, action_std, window_size=10, action_dim=7):
        self.samples = samples
        self.action_mean = action_mean
        self.action_std = action_std
        self.window_size = window_size
        self.action_dim = action_dim
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        import h5py
        import numpy as np
        from PIL import Image
        
        sample = self.samples[idx]
        
        with h5py.File(sample['ep_path'], 'r') as f:
            img = f['top_rgb'][sample['start_idx']]
            img = Image.fromarray(img)
            pixel_values = self.image_transform(img)
            
            actions = f['actions'][sample['start_idx']:sample['start_idx']+self.window_size]
            actions = (actions - self.action_mean) / self.action_std
            actions = torch.tensor(actions, dtype=torch.float32)
        
        return {
            'pixel_values': pixel_values,
            'actions': actions,
            'instruction': "pick up the block",
        }


def collate_fn(batch, tokenizer, max_length=512):
    """Collate batch with tokenization."""
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    actions = torch.stack([b['actions'] for b in batch])
    
    instructions = [b['instruction'] for b in batch]
    prompts = [f"What action should the robot take to {inst}?" for inst in instructions]
    
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
    parser.add_argument('--vla_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--run_dir', type=str, default='./runs/simple')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=50)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--action_dim', type=int, default=7)
    parser.add_argument('--learning_rate', type=float, default=3.5e-4)
    parser.add_argument('--use_lora', type=bool, default=True)
    parser.add_argument('--lora_rank', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.run_dir, exist_ok=True)
    
    model = SimpleVLAModel(
        vla_path=args.vla_path,
        window_size=args.window_size,
        action_dim=args.action_dim,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    
    samples, action_mean, action_std = load_dataset(
        args.data_dir,
        window_size=args.window_size,
        action_dim=args.action_dim,
    )
    
    dataset = SimpleDataset(
        samples,
        action_mean,
        action_std,
        window_size=args.window_size,
        action_dim=args.action_dim,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-3)
    
    with open(os.path.join(args.run_dir, 'norm_stats.json'), 'w') as f:
        json.dump({
            'action_mean': action_mean.tolist(),
            'action_std': action_std.tolist(),
        }, f)
    
    print("Starting training...")
    model.train()
    optimizer.zero_grad()
    
    step = 0
    epoch = 0
    
    while step < args.max_steps:
        epoch += 1
        print(f"Epoch {epoch}")
        
        for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch}"):
            if step >= args.max_steps:
                break
            
            batch['pixel_values'] = batch['pixel_values'].to(device)
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            batch['actions'] = batch['actions'].to(device)
            
            loss, loss_one_step, pred_action = model(
                batch['input_ids'],
                batch['attention_mask'],
                batch['pixel_values'],
                batch['actions'],
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            step += 1
            
            if step % 5 == 0:
                print(f"Step {step}: loss={loss.item():.4f}, loss_1step={loss_one_step.item():.4f}")
            
            if step % args.save_steps == 0:
                print(f"Saving checkpoint at step {step}")
                torch.save(
                    model.action_decoder.state_dict(),
                    os.path.join(args.run_dir, f'action_decoder-{step}.pt'),
                )
    
    print("Training complete!")
    torch.save(
        model.action_decoder.state_dict(),
        os.path.join(args.run_dir, 'action_decoder-final.pt'),
    )


if __name__ == "__main__":
    main()