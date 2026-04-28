"""
inference.py

Inference script for deploying fine-tuned UniVLA on real robot.

Usage:
    python inference.py \
        --vla_path ./runs/univla+lerobot-xxx \
        --decoder_path ./runs/univla+lerobot-xxx/action_decoder-10000.pt \
        --norm_stats_path ./runs/univla+lerobot-xxx/norm_stats.json
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torchvision
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
from PIL import Image

import draccus
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.policy.transformer_utils import MAPBlock


class ActionDecoder(torch.nn.Module):
    """Action decoder for predicting continuous actions."""
    
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


class UniVLAInference:
    """
    Inference class for UniVLA model.
    
    Usage:
        # Initialize
        policy = UniVLAInference(
            vla_path="/path/to/finetuned/model",
            decoder_path="/path/to/action_decoder.pt",
            norm_stats_path="/path/to/norm_stats.json",
            window_size=10,
            action_dim=7
        )
        
        # Get action
        action = policy.step(
            curr_image=camera_image,  # PIL Image or numpy array [H, W, 3]
            task_instruction="pick up the cup",
            proprio=current_joint_positions  # [action_dim] or [qpos_dim]
        )
    """
    
    def __init__(
        self,
        vla_path: str,
        decoder_path: str,
        norm_stats_path: str = None,
        window_size: int = 10,
        action_dim: int = 7,
        device: str = "cuda",
        pred_action_horizon: int = 10,
    ):
        self.vla_path = vla_path
        self.decoder_path = decoder_path
        self.window_size = window_size
        self.action_dim = action_dim
        self.device = device
        self.pred_action_horizon = pred_action_horizon
        
        # Load normalization stats
        self.norm_stats = None
        if norm_stats_path and os.path.exists(norm_stats_path):
            with open(norm_stats_path, 'r') as f:
                stats = json.load(f)
            self.norm_stats = {
                'action_mean': np.array(stats['action_mean']),
                'action_std': np.array(stats['action_std']),
                'qpos_mean': np.array(stats['qpos_mean']),
                'qpos_std': np.array(stats['qpos_std']),
            }
        
        # Register model classes
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        
        # Load model
        self.processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.vla.eval()
        
        # Load action decoder
        self.action_decoder = ActionDecoder(
            window_size=window_size,
            action_dim=action_dim
        ).to(device)
        
        if os.path.exists(decoder_path):
            state_dict = torch.load(decoder_path, map_location=device)
            self.action_decoder.load_state_dict(state_dict)
            print(f"Loaded action decoder from {decoder_path}")
        else:
            print(f"Warning: Decoder path {decoder_path} not found, using random weights")
        
        self.action_decoder.eval()
        
        # Image transforms
        self.resize = torchvision.transforms.Resize((224, 224))
        self.to_tensor = torchvision.transforms.ToTensor()
        
        # Action queue for temporal ensembling
        self.action_queue = []
    
    def preprocess_image(self, image):
        """Preprocess image for the model."""
        if isinstance(image, np.ndarray):
            # Assume RGB format
            image = Image.fromarray(image)
        
        # Resize to 224x224
        image = self.resize(image)
        
        # Convert to tensor
        pixel_values = self.to_tensor(image)
        
        return pixel_values
    
    def step(
        self,
        curr_image,
        task_instruction: str,
        proprio=None,
        use_action_queue: bool = False,
    ):
        """
        Get action from the model.
        
        Args:
            curr_image: Current camera image (PIL Image or numpy array [H, W, 3])
            task_instruction: Task description string
            proprio: Current proprioceptive state (joint positions) [action_dim]
            use_action_queue: Whether to use action queue for temporal ensembling
        
        Returns:
            action: Predicted action [action_dim] numpy array
        """
        # Use action queue if available
        if use_action_queue and len(self.action_queue) > 0:
            return self.action_queue.pop(0)
        
        # Preprocess image
        pixel_values = self.preprocess_image(curr_image).unsqueeze(0).to(self.device)
        
        # Prepare prompt
        prompt = f"What action should the robot take to {task_instruction}?"
        
        # Tokenize
        inputs = self.processor.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Create dummy labels
        labels = input_ids.clone()
        
        # Forward pass
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vla_output = self.vla(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values.to(torch.bfloat16),
                    labels=labels,
                    output_hidden_states=True,
                )
        
        # Extract latent action tokens
        visual_embed = vla_output.hidden_states[-1][:, :self.vla.vision_backbone.featurizer.patch_embed.num_patches].to(torch.float)
        latent_tokens = vla_output.hidden_states[-1][:, self.vla.vision_backbone.featurizer.patch_embed.num_patches:]
        
        # Get latent action tokens (last 4 tokens)
        latent_action_tokens = latent_tokens[:, -4:, :].to(torch.float)
        
        # Predict action
        with torch.no_grad():
            pred_action = self.action_decoder(latent_action_tokens, visual_embed)
            pred_action = pred_action.reshape(-1, self.window_size, self.action_dim)
        
        # Denormalize action
        action = pred_action[0, 0].cpu().numpy()  # Get first step action
        
        if self.norm_stats is not None:
            action = action * self.norm_stats['action_std'] + self.norm_stats['action_mean']
        
        # Fill action queue for temporal ensembling
        if use_action_queue:
            all_actions = pred_action[0].cpu().numpy()  # [window_size, action_dim]
            if self.norm_stats is not None:
                all_actions = all_actions * self.norm_stats['action_std'] + self.norm_stats['action_mean']
            
            # Add actions to queue (limit to pred_action_horizon)
            for i in range(min(self.pred_action_horizon, self.window_size)):
                self.action_queue.append(all_actions[i])
        
        return action
    
    def reset(self):
        """Reset the action queue."""
        self.action_queue = []
    
    def get_all_actions(self, curr_image, task_instruction: str, proprio=None):
        """
        Get all actions in the window.
        
        Returns:
            actions: [window_size, action_dim] numpy array
        """
        # Preprocess image
        pixel_values = self.preprocess_image(curr_image).unsqueeze(0).to(self.device)
        
        # Prepare prompt
        prompt = f"What action should the robot take to {task_instruction}?"
        
        # Tokenize
        inputs = self.processor.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        labels = input_ids.clone()
        
        # Forward pass
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vla_output = self.vla(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values.to(torch.bfloat16),
                    labels=labels,
                    output_hidden_states=True,
                )
        
        # Extract latent action tokens
        visual_embed = vla_output.hidden_states[-1][:, :self.vla.vision_backbone.featurizer.patch_embed.num_patches].to(torch.float)
        latent_tokens = vla_output.hidden_states[-1][:, self.vla.vision_backbone.featurizer.patch_embed.num_patches:]
        latent_action_tokens = latent_tokens[:, -4:, :].to(torch.float)
        
        # Predict all actions
        with torch.no_grad():
            pred_actions = self.action_decoder(latent_action_tokens, visual_embed)
            pred_actions = pred_actions.reshape(-1, self.window_size, self.action_dim)
        
        # Denormalize
        actions = pred_actions[0].cpu().numpy()
        
        if self.norm_stats is not None:
            actions = actions * self.norm_stats['action_std'] + self.norm_stats['action_mean']
        
        return actions


@dataclass
class InferenceConfig:
    vla_path: str = "./runs/univla+lerobot-xxx"
    decoder_path: str = "./runs/univla+lerobot-xxx/action_decoder-10000.pt"
    norm_stats_path: str = "./runs/univla+lerobot-xxx/norm_stats.json"
    window_size: int = 10
    action_dim: int = 7
    task_instruction: str = "pick up the cup"


def demo_inference(cfg: InferenceConfig):
    """Demo inference with a test image."""
    
    # Initialize inference
    policy = UniVLAInference(
        vla_path=cfg.vla_path,
        decoder_path=cfg.decoder_path,
        norm_stats_path=cfg.norm_stats_path,
        window_size=cfg.window_size,
        action_dim=cfg.action_dim,
    )
    
    # Create a dummy test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Get action
    action = policy.step(
        curr_image=test_image,
        task_instruction=cfg.task_instruction,
        proprio=np.zeros(cfg.action_dim),
    )
    
    print(f"Predicted action shape: {action.shape}")
    print(f"Predicted action: {action}")
    
    # Get all actions
    all_actions = policy.get_all_actions(
        curr_image=test_image,
        task_instruction=cfg.task_instruction,
    )
    
    print(f"All actions shape: {all_actions.shape}")


if __name__ == "__main__":
    demo_inference()