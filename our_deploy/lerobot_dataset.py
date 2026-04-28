"""
lerobot_dataset.py

Dataset class for loading converted LeRobot data in UniVLA training pipeline.
"""

import numpy as np
import torch
import os
import glob
import h5py
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import random
import logging
from PIL import Image
from einops import rearrange, repeat
from dataclasses import dataclass
from typing import Dict, Sequence, Optional

logger = logging.getLogger(__name__)


class LeRobotHDF5Dataset(torch.utils.data.Dataset):
    """
    Dataset class for loading LeRobot data converted to HDF5 format.
    
    Expected HDF5 structure:
    - /action: [episode_len, action_dim]
    - /observations/qpos: [episode_len, qpos_dim]
    - /observations/images/{cam_name}: [episode_len, H, W, 3]
    """
    
    def __init__(
        self,
        dataset_dir: str,
        camera_names: list = None,
        norm_stats: dict = None,
        window_size: int = 16,
        min_window_size: int = 16,
        max_window_size: int = 16,
        image_transform=None,
        action_dim: int = 7,
        qpos_dim: int = 7,
    ) -> None:
        super(LeRobotHDF5Dataset).__init__()
        
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names if camera_names else ['top_rgb']  # Default camera
        self.norm_stats = norm_stats
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.action_dim = action_dim
        self.qpos_dim = qpos_dim
        
        self.resize_img = torchvision.transforms.Resize((224, 224))
        self.image_transform_lam = torchvision.transforms.ToTensor()
        self.image_transform = image_transform
        self.color_aug = torchvision.transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )
        
        # Load all episodes
        self.image_dict, self.qpos, self.action, self.tasks_embedding, self.episode_lens = \
            self.load_all_episodes(dataset_dir)
    
    def __len__(self):
        return len(self.action)
    
    def load_all_episodes(self, dataset_dir):
        """Load all episodes from HDF5 files."""
        image_dict = {cam_name: [] for cam_name in self.camera_names}
        qpos = []
        actions = []
        instructions = []
        episode_lens = []
        
        # Find all HDF5 files
        hdf5_files = sorted(glob.glob(os.path.join(dataset_dir, 'episode_*.hdf5')))
        
        if not hdf5_files:
            raise ValueError(f"No HDF5 files found in {dataset_dir}")
        
        print(f"Found {len(hdf5_files)} HDF5 files in {dataset_dir}")
        
        for hdf5_path in hdf5_files:
            print(f"Loading {hdf5_path}")
            
            with h5py.File(hdf5_path, 'r') as root:
                # Load action
                action_data = np.array(root['/action'])
                
                # Handle action dimension mismatch
                if action_data.shape[1] > self.action_dim:
                    action_data = action_data[:, :self.action_dim]
                
                # Load qpos
                qpos_data = np.array(root['/observations/qpos'])
                
                # Handle qpos dimension mismatch
                if qpos_data.shape[1] > self.qpos_dim:
                    qpos_data = qpos_data[:, :self.qpos_dim]
                
                episode_len = len(action_data)
                episode_lens.append(episode_len)
                
                # Load images for each camera
                for cam_name in self.camera_names:
                    if f'/observations/images/{cam_name}' in root:
                        images = root[f'/observations/images/{cam_name}'][()]
                    elif f'observations/images/{cam_name}' in root:
                        images = root[f'observations/images/{cam_name}'][()]
                    else:
                        print(f"Warning: Camera {cam_name} not found in {hdf5_path}")
                        continue
                    
                    # Process images: resize and convert
                    processed_images = []
                    for i in range(images.shape[0]):
                        img = torch.from_numpy(images[i]).float()
                        # Resize to 224x224
                        img = F.interpolate(
                            img.permute(2, 0, 1).unsqueeze(0),
                            size=(224, 224),
                            mode='bilinear',
                            align_corners=False
                        )[0]
                        processed_images.append(img)
                    
                    image_dict[cam_name].append(torch.stack(processed_images, dim=0))
                
                qpos.append(torch.from_numpy(qpos_data).float())
                actions.append(torch.from_numpy(action_data).float())
                
                # Get task instruction
                task_instruction = root.attrs.get('task', 'unknown task')
                instructions.append(task_instruction)
        
        # Stack all data
        for cam_name in self.camera_names:
            if image_dict[cam_name]:
                image_dict[cam_name] = torch.stack(image_dict[cam_name], dim=0)
        
        qpos = torch.stack(qpos, dim=0) if qpos else torch.tensor([])
        actions = torch.stack(actions, dim=0) if actions else torch.tensor([])
        
        print(f"Loaded {len(actions)} episodes, total frames: {sum(episode_lens)}")
        
        return image_dict, qpos, actions, instructions, episode_lens
    
    def __getitem__(self, clip_index):
        """Get a single sample."""
        extra_frame_num = random.randint(0, 1)
        window_size = self.window_size + extra_frame_num
        
        episode_len = self.episode_lens[clip_index]
        image_index = np.random.choice(max(1, episode_len - window_size))
        
        # Initialize action chunking
        actions_chunking = torch.zeros((self.window_size, self.action_dim))
        is_not_padding = torch.zeros((self.window_size,))
        
        # Get actions
        action_len = min(episode_len - image_index, self.window_size)
        actions_chunking[:action_len] = self.action[clip_index, image_index:image_index + action_len]
        is_not_padding[:action_len] = 1
        
        # Get qpos
        qpos_chunking = self.qpos[clip_index][image_index]
        
        # Get images (use first camera by default)
        cam_name = self.camera_names[0]
        if cam_name in self.image_dict and len(self.image_dict[cam_name]) > clip_index:
            image_chunking = self.image_dict[cam_name][clip_index][image_index:image_index + window_size]
        else:
            # Fallback: create dummy images
            image_chunking = torch.zeros((window_size, 3, 224, 224))
        
        # Prepare VLA image
        if image_chunking.shape[0] > extra_frame_num:
            image_vla = Image.fromarray(
                np.transpose(image_chunking[extra_frame_num].cpu().numpy().astype(np.uint8), (1, 2, 0))
            )
            image_vla = self.color_aug(image_vla)
            goal_image = Image.fromarray(
                np.transpose(image_chunking[-1].cpu().numpy().astype(np.uint8), (1, 2, 0))
            )
        else:
            image_vla = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            goal_image = image_vla
        
        pixel_values = self.image_transform(image_vla) if self.image_transform else self.image_transform_lam(image_vla)
        
        initial_pixel_values = self.image_transform_lam(self.resize_img(image_vla))
        target_pixel_values = self.image_transform_lam(self.resize_img(goal_image))
        
        initial_pixel_values_hist, target_pixel_values_hist = None, None
        if extra_frame_num > 0 and image_chunking.shape[0] > self.min_window_size:
            hist_frame_prev = Image.fromarray(
                np.transpose(image_chunking[0].cpu().numpy().astype(np.uint8), (1, 2, 0))
            )
            hist_frame_goal = Image.fromarray(
                np.transpose(image_chunking[self.min_window_size].cpu().numpy().astype(np.uint8), (1, 2, 0))
            )
            initial_pixel_values_hist = self.image_transform_lam(self.resize_img(hist_frame_prev))
            target_pixel_values_hist = self.image_transform_lam(self.resize_img(hist_frame_goal))
        
        # Normalize actions and qpos
        qpos_tensor = qpos_chunking.float()
        action_tensor = actions_chunking.float()
        
        if self.norm_stats is not None:
            action_tensor = (action_tensor - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
            qpos_tensor = (qpos_tensor - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        
        task_embed = self.tasks_embedding[clip_index]
        dataset_name = 'lerobot'
        
        return dict(
            pixel_values=pixel_values,
            initial_pixel_values=initial_pixel_values,
            target_pixel_values=target_pixel_values,
            initial_pixel_values_hist=initial_pixel_values_hist,
            target_pixel_values_hist=target_pixel_values_hist,
            dataset_name=dataset_name,
            actions=action_tensor,
            lang=task_embed,
            proprio=qpos_tensor,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        initial_pixel_values = [instance["initial_pixel_values"] for instance in instances]
        target_pixel_values = [instance["target_pixel_values"] for instance in instances]

        initial_pixel_values_hist, target_pixel_values_hist = [], []
        with_hist = []
        for instance in instances:
            if instance["initial_pixel_values_hist"] is not None:
                initial_pixel_values_hist.append(instance["initial_pixel_values_hist"])
                target_pixel_values_hist.append(instance["target_pixel_values_hist"])
                with_hist.append(torch.tensor(True))
            else:
                with_hist.append(torch.tensor(False))

        pixel_values = [instance["pixel_values"] for instance in instances]
        dataset_names = [instance.get("dataset_name", "unknown") for instance in instances]

        # For low-level policy training
        actions = [instance["actions"] for instance in instances]
        actions = torch.stack(actions, dim=0)

        proprio = [instance["proprio"] for instance in instances]
        proprio = torch.stack(proprio, dim=0)

        instructions = [instance["lang"] for instance in instances]

        # Stack pixel values
        pixel_values = torch.stack(pixel_values)
        initial_pixel_values = torch.stack(initial_pixel_values)
        target_pixel_values = torch.stack(target_pixel_values)
        initial_pixel_values_hist = torch.stack(initial_pixel_values_hist) if len(initial_pixel_values_hist) > 0 else []
        target_pixel_values_hist = torch.stack(target_pixel_values_hist) if len(target_pixel_values_hist) > 0 else []
        with_hist = torch.stack(with_hist)

        output = dict(
            pixel_values=pixel_values,
            initial_pixel_values=initial_pixel_values,
            target_pixel_values=target_pixel_values,
            initial_pixel_values_hist=initial_pixel_values_hist,
            target_pixel_values_hist=target_pixel_values_hist,
            instructions=instructions,
            with_hist=with_hist,
            actions=actions,
            proprio=proprio
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output


def get_norm_stats(dataset_dir):
    """Load or compute normalization statistics."""
    norm_stats_path = os.path.join(dataset_dir, 'norm_stats.npz')
    
    if os.path.exists(norm_stats_path):
        stats = np.load(norm_stats_path)
        return {
            "action_mean": stats['action_mean'],
            "action_std": stats['action_std'],
            "action_min": stats['action_min'],
            "action_max": stats['action_max'],
            "qpos_mean": stats['qpos_mean'],
            "qpos_std": stats['qpos_std'],
            "qpos_min": stats['qpos_min'],
            "qpos_max": stats['qpos_max'],
        }
    
    # Compute from scratch
    all_qpos = []
    all_actions = []
    
    hdf5_files = sorted(glob.glob(os.path.join(dataset_dir, 'episode_*.hdf5')))
    
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            action = root['/action'][()]
        all_qpos.append(torch.from_numpy(qpos))
        all_actions.append(torch.from_numpy(action))
    
    all_qpos = torch.cat(all_qpos, dim=0)
    all_actions = torch.cat(all_actions, dim=0)
    
    action_mean = all_qpos.mean(dim=0, keepdim=True).numpy().squeeze()
    action_std = all_actions.std(dim=0, keepdim=True).numpy().squeeze()
    action_std = np.clip(action_std, 1e-2, np.inf)
    
    qpos_mean = all_qpos.mean(dim=0, keepdim=True).numpy().squeeze()
    qpos_std = all_qpos.std(dim=0, keepdim=True).numpy().squeeze()
    qpos_std = np.clip(qpos_std, 1e-2, np.inf)
    
    stats = {
        "action_mean": all_actions.mean(dim=0).numpy(),
        "action_std": all_actions.std(dim=0).numpy(),
        "action_min": all_actions.min(dim=0)[0].numpy(),
        "action_max": all_actions.max(dim=0)[0].numpy(),
        "qpos_mean": qpos_mean,
        "qpos_std": qpos_std,
    }
    
    return stats


def load_data_lerobot(
    dataset_dir,
    camera_names=None,
    batch_size_train=4,
    window_size=16,
    min_window_size=16,
    max_window_size=16,
    image_transform=None,
    action_dim=7,
    qpos_dim=7,
    num_workers=8,
):
    """Load LeRobot HDF5 dataset for UniVLA training."""
    
    # Get normalization stats
    norm_stats = get_norm_stats(dataset_dir)
    print("Normalization stats:", norm_stats)
    
    # Create dataset
    train_dataset = LeRobotHDF5Dataset(
        dataset_dir=dataset_dir,
        camera_names=camera_names,
        norm_stats=norm_stats,
        window_size=window_size,
        min_window_size=min_window_size,
        max_window_size=max_window_size,
        image_transform=image_transform,
        action_dim=action_dim,
        qpos_dim=qpos_dim,
    )
    
    # Create collator
    collator = PaddedCollatorForActionPrediction(
        model_max_length=512,  # Default value
        pad_token_id=0,  # Default value
        padding_side="right"
    )
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collator,
    )
    
    return train_dataloader, norm_stats
