"""
finetune_lerobot.py

Fine-tune UniVLA on LeRobot v2.1 format dataset.

Usage:
    # First, convert your LeRobot data to HDF5 format:
    python lerobot_to_univla.py --lerobot_dir /path/to/lerobot/data --output_dir /path/to/hdf5/output
    
    # Then, fine-tune UniVLA:
    torchrun --standalone --nnodes 1 --nproc-per-node 8 finetune_lerobot.py \
        --vla_path /path/to/univla-7b \
        --lam_path /path/to/lam.ckpt \
        --data_dir /path/to/hdf5/output \
        --batch_size 4 \
        --max_steps 10000 \
        --window_size 10
"""

import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import draccus
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor

import wandb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.policy.transformer_utils import MAPBlock

# Register OpenVLA model classes BEFORE importing AutoModelForVision2Seq
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)

# Now import AutoModelForVision2Seq after registration
from transformers import AutoModelForVision2Seq
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

from lerobot_dataset import LeRobotHDF5Dataset, PaddedCollatorForActionPrediction, get_norm_stats

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ActionDecoder(torch.nn.Module):
    """Action decoder for predicting continuous actions from latent action tokens."""
    
    def __init__(self, window_size=12, hidden_dim=512, action_dim=7):
        super().__init__()
        self.latent_action_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
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


class WrappedModel(torch.nn.Module):
    """Wrapper model combining VLA and action decoder."""
    
    def __init__(self, vla, freeze_vla=False, window_size=12, action_dim=7):
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
        visual_embed = vla_output.hidden_states[-1][:, :self.vla.vision_backbone.featurizer.patch_embed.num_patches].to(torch.float)
        latent_tokens = vla_output.hidden_states[-1][:, self.vla.vision_backbone.featurizer.patch_embed.num_patches:]
        action_gt = batch["labels"].to(latent_tokens.device)
        mask = action_gt > 32000
        
        latent_action_tokens = []
        for idx, per_sample_latent_tokens in enumerate(latent_tokens):
            per_sample_latent_action_tokens = per_sample_latent_tokens[mask[idx], :]
            latent_action_tokens.append(per_sample_latent_action_tokens)
        latent_action_tokens = torch.stack(latent_action_tokens).to(torch.float)
        
        pred_action = self.action_decoder(latent_action_tokens, visual_embed).reshape(-1, self.window_size, self.action_dim)
        loss = torch.nn.functional.l1_loss(pred_action, batch['actions'], reduction='none')
        loss_one_step = loss[:, 0].mean()
        loss = loss.mean()
        
        return loss, loss_one_step, latent_action_tokens


@dataclass
class FinetuneConfig:
    # Model Paths
    vla_path: str = "/path/to/your/pretrained-univla-7b"
    lam_path: str = "latent_action_model/logs/task_centric_lam_stage2/epoch=0-step=200000.ckpt"
    
    # Data Paths
    data_dir: Path = Path("/path/to/converted/hdf5/data")
    run_root_dir: Path = Path("runs")
    adapter_tmp_dir: Path = Path("adapter-tmp")
    
    # Data Settings
    camera_names: Optional[List[str]] = None  # e.g., ["top_rgb", "wrist_rgb"]
    action_dim: int = 7
    qpos_dim: int = 7
    
    # Training Parameters
    batch_size: int = 4
    max_steps: int = 10000
    save_steps: int = 2500
    learning_rate: float = 3.5e-4
    grad_accumulation_steps: int = 2
    image_aug: bool = True
    
    # Window size for action chunking
    window_size: int = 10
    
    # LAM settings
    codebook_size: int = 16
    lam_model_dim: int = 768
    lam_latent_dim: int = 128
    lam_patch_size: int = 14
    lam_enc_blocks: int = 12
    lam_dec_blocks: int = 12
    lam_num_heads: int = 12
    
    # LoRA Settings
    freeze_vla: bool = False
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False
    
    # Tracking
    wandb_project: str = "finetune-lerobot"
    wandb_entity: str = "your-entity"
    run_id_note: Optional[str] = None


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning UniVLA Model `{cfg.vla_path}` on LeRobot data from `{cfg.data_dir}`")
    
    # Validate GPU availability
    assert torch.cuda.is_available(), "Fine-tuning requires GPU!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()
    
    # Configure experiment ID
    exp_id = (
        f"univla+lerobot"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}"
    if cfg.run_id_note:
        exp_id += f"--{cfg.run_id_note}"
    exp_id += f"-ws-{cfg.window_size}"
    
    # Create directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)
    
    # Quantization config
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
    
    # Register model classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    # Device placement
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)
    
    # LoRA wrapping
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()
    
    # Create wrapped model
    wrapped_model = WrappedModel(
        vla=vla,
        freeze_vla=cfg.freeze_vla,
        window_size=cfg.window_size,
        action_dim=cfg.action_dim,
    ).to(device_id)
    
    trainable_total_params = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
    print(f'Total Trainable Params: {trainable_total_params}')
    
    # DDP wrapper
    wrapped_model = DDP(wrapped_model, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
    
    # Optimizer
    trainable_params = [param for param in wrapped_model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.max_steps * 0.8), gamma=0.1)
    
    # Load normalization stats
    norm_stats = get_norm_stats(str(cfg.data_dir))
    print("Normalization stats loaded")
    
    # Create dataset and dataloader
    train_dataset = LeRobotHDF5Dataset(
        dataset_dir=str(cfg.data_dir),
        camera_names=cfg.camera_names,
        norm_stats=norm_stats,
        window_size=cfg.window_size,
        action_dim=cfg.action_dim,
        qpos_dim=cfg.qpos_dim,
        image_transform=processor.image_processor.apply_transform,
    )
    
    collator = PaddedCollatorForActionPrediction(
        model_max_length=processor.tokenizer.model_max_length,
        pad_token_id=processor.tokenizer.pad_token_id,
        padding_side="right"
    )
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=8,
        pin_memory=False,
    )
    
    # Initialize W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")
    
    # Training metrics
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_losses = deque(maxlen=cfg.grad_accumulation_steps)
    
    # Training loop
    print("Starting training...")
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        wrapped_model.train()
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            # Prepare batch for VLA
            # Note: We need to create input_ids, attention_mask, labels from instructions
            instructions = batch["instructions"]
            
            # Tokenize instructions
            input_ids = []
            attention_masks = []
            labels = []
            
            for instruction in instructions:
                # Create a simple prompt
                prompt = f"What action should the robot take to {instruction}?"
                tokens = processor.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                input_ids.append(tokens["input_ids"])
                attention_masks.append(tokens["attention_mask"])
                # Create dummy labels (will be overwritten by action loss)
                labels.append(tokens["input_ids"].clone())
            
            batch["input_ids"] = torch.stack(input_ids).squeeze(1).to(device_id)
            batch["attention_mask"] = torch.stack(attention_masks).squeeze(1).to(device_id)
            batch["labels"] = torch.stack(labels).squeeze(1).to(device_id)
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16).to(device_id)
            batch["actions"] = batch["actions"].to(device_id)
            
            # Forward pass
            output, act_loss, loss_one_step, latent_action_proj = wrapped_model(batch)
            loss = act_loss if cfg.freeze_vla else act_loss + output.loss
            
            # Backward pass
            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), max_norm=1.)
            
            # Store metrics
            recent_losses.append(loss.item())
            recent_action_losses.append(act_loss.item())
            
            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            
            # Smoothened metrics
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_loss = sum(recent_action_losses) / len(recent_action_losses)
            
            # Log to W&B
            if distributed_state.is_main_process and gradient_step_idx % 5 == 0:
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "action_loss": act_loss.item(),
                        "action_loss_1step": loss_one_step.item(),
                        "lr": optimizer.state_dict()['param_groups'][0]['lr'],
                    },
                    step=gradient_step_idx,
                )
            
            # Optimizer step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                progress.update()
            
            # Save checkpoint
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")
                    
                    save_dir = adapter_dir if cfg.use_lora else run_dir
                    
                    # Save processor and weights
                    if not cfg.freeze_vla:
                        processor.save_pretrained(run_dir)
                        wrapped_model.module.vla.save_pretrained(save_dir)
                    
                    # Save action decoder
                    torch.save(
                        wrapped_model.module.action_decoder.state_dict(),
                        str(run_dir) + f'/action_decoder-{gradient_step_idx}.pt'
                    )
                    
                    # Save norm stats
                    import json
                    with open(run_dir / 'norm_stats.json', 'w') as f:
                        json.dump({
                            'action_mean': norm_stats['action_mean'].tolist() if hasattr(norm_stats['action_mean'], 'tolist') else norm_stats['action_mean'],
                            'action_std': norm_stats['action_std'].tolist() if hasattr(norm_stats['action_std'], 'tolist') else norm_stats['action_std'],
                            'qpos_mean': norm_stats['qpos_mean'].tolist() if hasattr(norm_stats['qpos_mean'], 'tolist') else norm_stats['qpos_mean'],
                            'qpos_std': norm_stats['qpos_std'].tolist() if hasattr(norm_stats['qpos_std'], 'tolist') else norm_stats['qpos_std'],
                        }, f, indent=2)
                
                dist.barrier()
                
                # Merge LoRA weights
                if cfg.use_lora:
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    )
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        merged_vla.save_pretrained(run_dir)
                        print(f"Saved merged model to {run_dir}")
                
                dist.barrier()
            
            # Stop at max steps
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break
    
    # Final save
    if distributed_state.is_main_process:
        print("Training complete!")
        wandb.finish()


if __name__ == "__main__":
    finetune()
