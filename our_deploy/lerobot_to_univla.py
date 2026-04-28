"""
lerobot_to_univla.py

Convert LeRobot v2.1 format dataset to UniVLA-compatible format.

LeRobot v2.1 format:
- data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
- videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4
- meta/info.json, meta/tasks.jsonl

UniVLA expects:s

- HDF5 files with:
  - /action: [episode_len, action_dim]
  - /observations/qpos: [episode_len, state_dim]
  - /observations/images/{cam_name}: [episode_len, H, W, 3]
"""

import os
import json
import h5py
import cv2
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
import argparse
import av  # PyAV for video decoding


def read_tasks_jsonl(tasks_path):
    """Read tasks.jsonl and return a dict mapping task_index to task description."""
    tasks = {}
    with open(tasks_path, 'r') as f:
        for line in f:
            task = json.loads(line.strip())
            tasks[task['task_index']] = task['task']
    return tasks


def decode_video_frames(video_path):
    """Decode all frames from a video file using PyAV."""
    frames = []
    container = av.open(video_path)
    for frame in container.decode(video=0):
        # Convert to numpy array (RGB)
        img = frame.to_ndarray(format='rgb24')
        frames.append(img)
    container.close()
    return np.stack(frames, axis=0) if frames else None


def decode_video_frames_opencv(video_path):
    """Decode all frames from a video file using OpenCV (fallback)."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.close()
    return np.stack(frames, axis=0) if frames else None


def extract_state_for_qpos(states, state_dim=17, mode='xyz_quat_gripper'):
    """
    Extract relevant dimensions from full state for qpos.
    
    State layout (17 dim): xyz(0-2) + rpy(3-5) + quat(6-9) + ort6d(10-15) + gripper(16)
    
    Args:
        states: Full state array [episode_len, 17]
        state_dim: Full state dimension
        mode: Extraction mode
            - 'xyz_quat_gripper': xyz(3) + quat(4) + gripper(1) = 8 dim
            - 'xyz_rpy_gripper': xyz(3) + rpy(3) + gripper(1) = 7 dim  
            - 'full': use all dimensions
            - 'first_7': use first 7 dimensions (for cup dataset with 8 dim state)
    
    Returns:
        qpos: Extracted state [episode_len, qpos_dim]
    """
    if mode == 'xyz_quat_gripper':
        # Extract xyz(0:3) + quat(6:10) + gripper(16)
        if states.shape[1] >= 17:
            qpos = np.concatenate([
                states[:, 0:3],   # xyz
                states[:, 6:10],  # quat (rx, ry, rz, rw)
                states[:, 16:17]  # gripper
            ], axis=1)
        else:
            # Fallback for smaller state dim
            qpos = states[:, :min(8, states.shape[1])]
    elif mode == 'xyz_rpy_gripper':
        # Extract xyz(0:3) + rpy(3:6) + gripper(16)
        if states.shape[1] >= 17:
            qpos = np.concatenate([
                states[:, 0:3],   # xyz
                states[:, 3:6],   # rpy
                states[:, 16:17]  # gripper
            ], axis=1)
        else:
            qpos = states[:, :min(7, states.shape[1])]
    elif mode == 'first_7':
        qpos = states[:, :7]
    elif mode == 'full':
        qpos = states
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return qpos.astype(np.float32)


def extract_action(actions, mode='xyz_rpy_gripper'):
    """
    Extract relevant dimensions from full action.
    
    Action layout (17 dim): xyz(0-2) + rpy(3-5) + quat(6-9) + ort6d(10-15) + gripper(16)
    Same layout as state.
    
    Args:
        actions: Full action array [episode_len, action_dim]
        mode: Extraction mode
            - 'xyz_rpy_gripper': xyz(3) + rpy(3) + gripper(1) = 7 dim (default)
            - 'xyz_quat_gripper': xyz(3) + quat(4) + gripper(1) = 8 dim
            - 'full': use all dimensions
            - 'first_7': use first 7 dimensions
    
    Returns:
        extracted_action: Extracted action [episode_len, action_dim]
    """
    if mode == 'xyz_rpy_gripper':
        # Extract xyz(0:3) + rpy(3:6) + gripper(16) = 7 dim
        if actions.shape[1] >= 17:
            extracted = np.concatenate([
                actions[:, 0:3],   # xyz
                actions[:, 3:6],   # rpy
                actions[:, 16:17]  # gripper
            ], axis=1)
        else:
            extracted = actions[:, :min(7, actions.shape[1])]
    elif mode == 'xyz_quat_gripper':
        # Extract xyz(0:3) + quat(6:10) + gripper(16) = 8 dim
        if actions.shape[1] >= 17:
            extracted = np.concatenate([
                actions[:, 0:3],   # xyz
                actions[:, 6:10],  # quat
                actions[:, 16:17]  # gripper
            ], axis=1)
        else:
            extracted = actions[:, :min(8, actions.shape[1])]
    elif mode == 'first_7':
        extracted = actions[:, :7]
    elif mode == 'full':
        extracted = actions
    else:
        raise ValueError(f"Unknown action extraction mode: {mode}")
    
    return extracted.astype(np.float32)


def convert_lerobot_to_hdf5(
    lerobot_dir,
    output_dir,
    camera_names=None,
    use_video=True,
    compress=False,
    task_instruction=None,
    state_extract_mode='xyz_quat_gripper',
    action_extract_mode='xyz_rpy_gripper',
):
    """
    Convert LeRobot v2.1 format to HDF5 format for UniVLA.
    
    Args:
        lerobot_dir: Path to LeRobot dataset directory
        output_dir: Path to output directory for HDF5 files
        camera_names: List of camera names to use. If None, use all available.
        use_video: Whether to include video frames in HDF5
        compress: Whether to compress images in HDF5
        task_instruction: Override task instruction for all episodes
        state_extract_mode: How to extract qpos from state:
            - 'xyz_quat_gripper': xyz(3) + quat(4) + gripper(1) = 8 dim
            - 'xyz_rpy_gripper': xyz(3) + rpy(3) + gripper(1) = 7 dim
            - 'first_7': first 7 dimensions
            - 'full': use full state
        action_extract_mode: How to extract action:
            - 'xyz_rpy_gripper': xyz(3) + rpy(3) + gripper(1) = 7 dim (default)
            - 'xyz_quat_gripper': xyz(3) + quat(4) + gripper(1) = 8 dim
            - 'first_7': first 7 dimensions
            - 'full': use full action
    """
    lerobot_dir = Path(lerobot_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read meta info
    with open(lerobot_dir / 'meta' / 'info.json', 'r') as f:
        info = json.load(f)
    
    # Read tasks
    tasks = read_tasks_jsonl(lerobot_dir / 'meta' / 'tasks.jsonl')
    
    # Get available cameras from info
    video_keys = [k for k in info['features'].keys() if info['features'][k].get('dtype') == 'video']
    
    if camera_names is None:
        # Use all available cameras
        camera_names = video_keys
    else:
        # Map simple camera names to full keys
        camera_name_map = {}
        for key in video_keys:
            # Extract camera name from key like "observation.images.cam_high"
            parts = key.split('.')
            if len(parts) >= 3:
                simple_name = parts[2]  # e.g., "cam_high"
                camera_name_map[simple_name] = key
            camera_name_map[key] = key  # Also allow full key
        
        camera_names = [camera_name_map.get(cn, cn) for cn in camera_names]
    
    print(f"Using cameras: {camera_names}")
    
    # Get episode indices
    total_episodes = info['total_episodes']
    episode_chunk = 0
    
    print(f"Converting {total_episodes} episodes...")
    
    for episode_idx in tqdm(range(total_episodes)):
        # Read parquet file
        parquet_path = lerobot_dir / 'data' / f'chunk-{episode_chunk:03d}' / f'episode_{episode_idx:06d}.parquet'
        
        if not parquet_path.exists():
            print(f"Warning: {parquet_path} not found, skipping...")
            continue
        
        df = pq.read_table(parquet_path).to_pandas()
        
        # Get episode length
        episode_len = len(df)
        
        # Extract action and state
        raw_actions = np.stack(df['action'].apply(lambda x: np.array(x)).values).astype(np.float32)
        raw_states = np.stack(df['observation.state'].apply(lambda x: np.array(x)).values).astype(np.float32)
        
        # Extract qpos and action based on modes
        qpos = extract_state_for_qpos(raw_states, mode=state_extract_mode)
        actions = extract_action(raw_actions, mode=action_extract_mode)
        
        # Get task instruction
        task_idx = df['task_index'].iloc[0]
        task_desc = task_instruction if task_instruction else tasks.get(task_idx, f"task_{task_idx}")
        
        # Create HDF5 file
        hdf5_path = output_dir / f'episode_{episode_idx:06d}.hdf5'
        
        with h5py.File(hdf5_path, 'w') as hf:
            # Store action (extracted)
            hf.create_dataset('/action', data=actions, compression='gzip' if compress else None)
            
            # Store raw action for reference
            hf.create_dataset('/raw_action', data=raw_actions, compression='gzip' if compress else None)
            
            # Create observations group
            obs_grp = hf.create_group('/observations')
            
            # Print info only for first episode
            if episode_idx == 0:
                print(f"Raw state shape: {raw_states.shape}, Qpos shape: {qpos.shape} (mode={state_extract_mode})")
                print(f"Raw action shape: {raw_actions.shape}, Action shape: {actions.shape} (mode={action_extract_mode})")
            obs_grp.create_dataset('qpos', data=qpos, compression='gzip' if compress else None)
            
            # Store full state as well for reference
            obs_grp.create_dataset('state', data=raw_states, compression='gzip' if compress else None)
            
            # Store images
            if use_video:
                images_grp = obs_grp.create_group('/images')
                
                for cam_key in camera_names:
                    # Get video path
                    video_path = lerobot_dir / 'videos' / f'chunk-{episode_chunk:03d}' / cam_key / f'episode_{episode_idx:06d}.mp4'
                    
                    if not video_path.exists():
                        print(f"Warning: {video_path} not found, skipping camera...")
                        continue
                    
                    # Decode video frames
                    try:
                        frames = decode_video_frames(str(video_path))
                    except Exception as e:
                        print(f"PyAV failed for {video_path}, trying OpenCV: {e}")
                        frames = decode_video_frames_opencv(str(video_path))
                    
                    if frames is None:
                        print(f"Warning: Could not decode {video_path}")
                        continue
                    
                    # Use simple camera name for HDF5 key
                    simple_cam_name = cam_key.split('.')[-1] if '.' in cam_key else cam_key
                    images_grp.create_dataset(f'/{simple_cam_name}', data=frames, compression='gzip' if compress else None)
            
            # Store metadata
            hf.attrs['task'] = task_desc
            hf.attrs['episode_index'] = episode_idx
            hf.attrs['compress'] = compress
    
    # Save normalization stats
    print("Computing normalization statistics...")
    all_actions = []
    all_qpos = []
    
    for episode_idx in range(total_episodes):
        hdf5_path = output_dir / f'episode_{episode_idx:06d}.hdf5'
        if hdf5_path.exists():
            with h5py.File(hdf5_path, 'r') as hf:
                all_actions.append(hf['/action'][()])
                all_qpos.append(hf['/observations/qpos'][()])
    
    all_actions = np.concatenate(all_actions, axis=0)
    all_qpos = np.concatenate(all_qpos, axis=0)
    
    norm_stats = {
        'action_mean': all_actions.mean(axis=0),
        'action_std': all_actions.std(axis=0),
        'action_min': all_actions.min(axis=0),
        'action_max': all_actions.max(axis=0),
        'qpos_mean': all_qpos.mean(axis=0),
        'qpos_std': all_qpos.std(axis=0),
        'qpos_min': all_qpos.min(axis=0),
        'qpos_max': all_qpos.max(axis=0),
    }
    
    np.savez(output_dir / 'norm_stats.npz', **norm_stats)
    
    # Save dataset info
    dataset_info = {
        'total_episodes': total_episodes,
        'action_dim': actions.shape[1],
        'qpos_dim': qpos.shape[1],
        'camera_names': [cn.split('.')[-1] if '.' in cn else cn for cn in camera_names],
        'tasks': tasks,
    }
    
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Conversion complete! Output saved to {output_dir}")
    print(f"Total episodes: {total_episodes}")
    print(f"Action dim: {actions.shape[1]}")
    print(f"Qpos dim: {qpos.shape[1]}")
    
    return norm_stats


def main():
    parser = argparse.ArgumentParser(description='Convert LeRobot v2.1 format to UniVLA HDF5 format')
    parser.add_argument('--lerobot_dir', type=str, required=True,
                        help='Path to LeRobot dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory for HDF5 files')
    parser.add_argument('--cameras', type=str, nargs='+', default=None,
                        help='Camera names to use (default: all available)')
    parser.add_argument('--no_video', action='store_true',
                        help='Do not include video frames in HDF5')
    parser.add_argument('--compress', action='store_true',
                        help='Compress HDF5 datasets')
    parser.add_argument('--task_instruction', type=str, default=None,
                        help='Override task instruction for all episodes')
    parser.add_argument('--state_extract_mode', type=str, default='xyz_quat_gripper',
                        choices=['xyz_quat_gripper', 'xyz_rpy_gripper', 'first_7', 'full'],
                        help='How to extract qpos from state (default: xyz_quat_gripper)')
    parser.add_argument('--action_extract_mode', type=str, default='xyz_rpy_gripper',
                        choices=['xyz_rpy_gripper', 'xyz_quat_gripper', 'first_7', 'full'],
                        help='How to extract action (default: xyz_rpy_gripper)')
    
    args = parser.parse_args()
    
    convert_lerobot_to_hdf5(
        args.lerobot_dir,
        args.output_dir,
        camera_names=args.cameras,
        use_video=not args.no_video,
        compress=args.compress,
        task_instruction=args.task_instruction,
        state_extract_mode=args.state_extract_mode,
        action_extract_mode=args.action_extract_mode,
    )


if __name__ == '__main__':
    main()