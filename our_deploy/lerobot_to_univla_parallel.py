"""
lerobot_to_univla_parallel.py

Multi-threaded version of LeRobot v2.1 to UniVLA HDF5 format converter.

Uses multiprocessing for parallel video decoding and HDF5 file writing.
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
import av
from multiprocessing import Pool, cpu_count
from functools import partial
import time


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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.close()
    return np.stack(frames, axis=0) if frames else None


def extract_state_for_qpos(states, mode='xyz_quat_gripper'):
    """
    Extract relevant dimensions from full state for qpos.
    
    State layout (17 dim): xyz(0-2) + rpy(3-5) + quat(6-9) + ort6d(10-15) + gripper(16)
    """
    if mode == 'xyz_quat_gripper':
        if states.shape[1] >= 17:
            qpos = np.concatenate([
                states[:, 0:3],   # xyz
                states[:, 6:10],  # quat
                states[:, 16:17]  # gripper
            ], axis=1)
        else:
            qpos = states[:, :min(8, states.shape[1])]
    elif mode == 'xyz_rpy_gripper':
        if states.shape[1] >= 17:
            qpos = np.concatenate([
                states[:, 0:3],
                states[:, 3:6],
                states[:, 16:17]
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
    """
    if mode == 'xyz_rpy_gripper':
        if actions.shape[1] >= 17:
            extracted = np.concatenate([
                actions[:, 0:3],   # xyz
                actions[:, 3:6],   # rpy
                actions[:, 16:17]  # gripper
            ], axis=1)
        else:
            extracted = actions[:, :min(7, actions.shape[1])]
    elif mode == 'xyz_quat_gripper':
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


def process_single_episode(args):
    """
    Process a single episode - designed for parallel execution.
    
    Args:
        args: Tuple of (episode_idx, lerobot_dir, output_dir, camera_names, 
                       use_video, compress, task_instruction, state_extract_mode,
                       action_extract_mode, episode_chunk, tasks)
    
    Returns:
        Tuple of (episode_idx, action_shape, qpos_shape, success)
    """
    (episode_idx, lerobot_dir, output_dir, camera_names, use_video, compress,
     task_instruction, state_extract_mode, action_extract_mode, episode_chunk, tasks) = args
    
    try:
        lerobot_dir = Path(lerobot_dir)
        output_dir = Path(output_dir)
        
        # Read parquet file
        parquet_path = lerobot_dir / 'data' / f'chunk-{episode_chunk:03d}' / f'episode_{episode_idx:06d}.parquet'
        
        if not parquet_path.exists():
            return (episode_idx, None, None, False)
        
        df = pq.read_table(parquet_path).to_pandas()
        episode_len = len(df)
        
        # Extract raw action and state
        raw_actions = np.stack(df['action'].apply(lambda x: np.array(x)).values).astype(np.float32)
        raw_states = np.stack(df['observation.state'].apply(lambda x: np.array(x)).values).astype(np.float32)
        
        # Get task instruction
        task_idx = df['task_index'].iloc[0]
        task_desc = task_instruction if task_instruction else tasks.get(task_idx, f"task_{task_idx}")
        
        # Extract qpos and action based on modes
        qpos = extract_state_for_qpos(raw_states, mode=state_extract_mode)
        actions = extract_action(raw_actions, mode=action_extract_mode)
        
        # Create HDF5 file
        hdf5_path = output_dir / f'episode_{episode_idx:06d}.hdf5'
        
        with h5py.File(hdf5_path, 'w') as hf:
            # Store action (extracted)
            hf.create_dataset('/action', data=actions, compression='gzip' if compress else None)
            
            # Store raw action for reference
            hf.create_dataset('/raw_action', data=raw_actions, compression='gzip' if compress else None)
            
            # Create observations group
            obs_grp = hf.create_group('/observations')
            
            # Store qpos and state
            obs_grp.create_dataset('qpos', data=qpos, compression='gzip' if compress else None)
            obs_grp.create_dataset('state', data=raw_states, compression='gzip' if compress else None)
            
            # Store images
            if use_video and camera_names:
                images_grp = obs_grp.create_group('/images')
                
                for cam_key in camera_names:
                    video_path = lerobot_dir / 'videos' / f'chunk-{episode_chunk:03d}' / cam_key / f'episode_{episode_idx:06d}.mp4'
                    
                    if not video_path.exists():
                        continue
                    
                    try:
                        frames = decode_video_frames(str(video_path))
                    except Exception:
                        frames = decode_video_frames_opencv(str(video_path))
                    
                    if frames is None:
                        continue
                    
                    simple_cam_name = cam_key.split('.')[-1] if '.' in cam_key else cam_key
                    images_grp.create_dataset(f'/{simple_cam_name}', data=frames, compression='gzip' if compress else None)
            
            # Store metadata
            hf.attrs['task'] = task_desc
            hf.attrs['episode_index'] = episode_idx
        
        return (episode_idx, actions.shape, qpos.shape, True)
    
    except Exception as e:
        print(f"Error processing episode {episode_idx}: {e}")
        return (episode_idx, None, None, False)


def convert_lerobot_to_hdf5_parallel(
    lerobot_dir,
    output_dir,
    camera_names=None,
    use_video=True,
    compress=False,
    task_instruction=None,
    state_extract_mode='xyz_quat_gripper',
    action_extract_mode='xyz_rpy_gripper',
    num_workers=None
):
    """
    Convert LeRobot v2.1 format to HDF5 format using parallel processing.
    
    Args:
        lerobot_dir: Path to LeRobot dataset directory
        output_dir: Path to output directory for HDF5 files
        camera_names: List of camera names to use
        use_video: Whether to include video frames
        compress: Whether to compress HDF5 datasets
        task_instruction: Override task instruction
        state_extract_mode: How to extract qpos from state
        action_extract_mode: How to extract action
        num_workers: Number of parallel workers (default: CPU count)
    """
    lerobot_dir = Path(lerobot_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read meta info
    with open(lerobot_dir / 'meta' / 'info.json', 'r') as f:
        info = json.load(f)
    
    tasks = read_tasks_jsonl(lerobot_dir / 'meta' / 'tasks.jsonl')
    
    # Get available cameras
    video_keys = [k for k in info['features'].keys() if info['features'][k].get('dtype') == 'video']
    
    if camera_names is None:
        camera_names = video_keys
    else:
        camera_name_map = {}
        for key in video_keys:
            parts = key.split('.')
            if len(parts) >= 3:
                simple_name = parts[2]
                camera_name_map[simple_name] = key
            camera_name_map[key] = key
        
        camera_names = [camera_name_map.get(cn, cn) for cn in camera_names]
    
    print(f"Using cameras: {camera_names}")
    
    total_episodes = info['total_episodes']
    episode_chunk = 0
    
    if num_workers is None:
        num_workers = min(cpu_count(), 16)  # Limit to 16 workers max
    
    print(f"Converting {total_episodes} episodes using {num_workers} workers...")
    
    # Prepare arguments for parallel processing
    process_args = [
        (episode_idx, str(lerobot_dir), str(output_dir), camera_names, use_video,
         compress, task_instruction, state_extract_mode, action_extract_mode, episode_chunk, tasks)
        for episode_idx in range(total_episodes)
    ]
    
    # Process episodes in parallel
    start_time = time.time()
    results = []
    
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap(process_single_episode, process_args), 
                          total=total_episodes, desc="Converting"):
            results.append(result)
    
    elapsed_time = time.time() - start_time
    
    # Process results
    success_count = sum(1 for r in results if r[3])
    print(f"\nSuccessfully converted {success_count}/{total_episodes} episodes in {elapsed_time:.2f}s")
    
    # Get shapes from first successful result
    action_shape = None
    qpos_shape = None
    for r in results:
        if r[3]:
            action_shape = r[1]
            qpos_shape = r[2]
            break
    
    # Compute normalization stats
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
        'action_dim': action_shape[1] if action_shape else 0,
        'qpos_dim': qpos_shape[1] if qpos_shape else 0,
        'camera_names': [cn.split('.')[-1] if '.' in cn else cn for cn in camera_names],
        'tasks': tasks,
    }
    
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Conversion complete! Output saved to {output_dir}")
    print(f"Action dim: {action_shape[1] if action_shape else 0}")
    print(f"Qpos dim: {qpos_shape[1] if qpos_shape else 0}")
    
    return norm_stats


def main():
    parser = argparse.ArgumentParser(description='Convert LeRobot v2.1 to UniVLA HDF5 (Parallel)')
    parser.add_argument('--lerobot_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--cameras', type=str, nargs='+', default=None)
    parser.add_argument('--no_video', action='store_true')
    parser.add_argument('--compress', action='store_true')
    parser.add_argument('--task_instruction', type=str, default=None)
    parser.add_argument('--state_extract_mode', type=str, default='xyz_quat_gripper',
                        choices=['xyz_quat_gripper', 'xyz_rpy_gripper', 'first_7', 'full'])
    parser.add_argument('--action_extract_mode', type=str, default='xyz_rpy_gripper',
                        choices=['xyz_rpy_gripper', 'xyz_quat_gripper', 'first_7', 'full'],
                        help='How to extract action (default: xyz_rpy_gripper)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    
    args = parser.parse_args()
    
    convert_lerobot_to_hdf5_parallel(
        args.lerobot_dir,
        args.output_dir,
        camera_names=args.cameras,
        use_video=not args.no_video,
        compress=args.compress,
        task_instruction=args.task_instruction,
        state_extract_mode=args.state_extract_mode,
        action_extract_mode=args.action_extract_mode,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    main()