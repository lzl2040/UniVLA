# UniVLA Fine-tuning on LeRobot v2.1 Data

本教程介绍如何将 LeRobot v2.1 格式的真机数据转换为 UniVLA 可用的格式，并进行微调和部署。

## 目录

1. [环境准备](#环境准备)
2. [数据格式说明](#数据格式说明)
3. [数据转换](#数据转换)
4. [模型微调](#模型微调)
5. [模型推理](#模型推理)
6. [常见问题](#常见问题)

---

## 环境准备

### 1. 安装依赖

```bash
cd UniVLA
pip install -e .
pip install av h5py pyarrow tqdm
```

### 2. 准备预训练模型

下载 UniVLA 预训练模型和 LAM (Latent Action Model) 权重：

```bash
# 设置模型路径
export VLA_PATH="/path/to/univla-7b"
export LAM_PATH="/path/to/lam.ckpt"
```

---

## 数据格式说明

### LeRobot v2.1 格式

你的数据目录结构应该如下：

```
/Data/lerobot_data/real_world/cup_hz_2.5_plus/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.top_rgb/
│       │   ├── episode_000000.mp4
│       │   └── ...
│       ├── observation.images.wrist_rgb/
│       └── ...
├── meta/
│   ├── info.json
│   ├── tasks.jsonl
│   └── episodes.jsonl
└── merged.parquet
```

**关键字段说明：**

| 字段 | 说明 | 形状 |
|------|------|------|
| `observation.state` | 机器人状态（关节角度等） | `[episode_len, state_dim]` |
| `action` | 机器人动作 | `[episode_len, 7]` |
| `observation.images.*` | 相机图像视频 | `[episode_len, H, W, 3]` |
| `timestamp` | 时间戳 | `[episode_len]` |
| `task_index` | 任务索引 | 标量 |

### UniVLA 期望的 HDF5 格式

```
episode_000000.hdf5
├── /action                      # [episode_len, 7]
├── /observations/
│   ├── /qpos                    # [episode_len, 7] - 关节位置
│   ├── /state                   # [episode_len, state_dim] - 完整状态
│   └── /images/
│       ├── /top_rgb             # [episode_len, H, W, 3]
│       └── /wrist_rgb           # [episode_len, H, W, 3]
└── attrs: task, episode_index
```

---

## 数据转换

### 单个数据集转换

```bash
python lerobot_to_univla.py \
    --lerobot_dir /Data/lerobot_data/real_world/cup_hz_2.5_plus \
    --output_dir /Data/rlds_raw/lerobot_converted_data/cup_hz_2.5_plus \
    --cameras observation.images.top_rgb observation.images.wrist_rgb \
    --compress

python lerobot_to_univla_parallel.py \
    --lerobot_dir /Data/lerobot_data/real_world/block_hz_4  \
    --output_dir /Data/rlds_raw/lerobot_converted_data/block \
    --cameras observation.images.top_rgb observation.images.wrist_rgb \
    --state_extract_mode xyz_quat_gripper \
    --action_extract_mode xyz_rpy_gripper \
    --num_workers 8
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--lerobot_dir` | LeRobot 数据目录路径 |
| `--output_dir` | 输出 HDF5 文件目录 |
| `--cameras` | 要使用的相机名称（可选多个） |
| `--compress` | 是否压缩 HDF5 文件 |
| `--no_video` | 不包含视频帧（仅转换动作和状态） |
| `--task_instruction` | 覆盖所有 episode 的任务描述 |

### 批量转换

```bash
# 转换多个数据集
for dataset in cup_hz_2.5_plus block_hz_4; do
    python lerobot_to_univla.py \
        --lerobot_dir /Data/lerobot_data/real_world/$dataset \
        --output_dir ./converted_data/$dataset \
        --cameras observation.images.top_rgb \
        --compress
done
```

### 转换输出

转换完成后，输出目录包含：

```
./converted_data/cup_hz_2.5_plus/
├── episode_000000.hdf5
├── episode_000001.hdf5
├── ...
├── norm_stats.npz        # 归一化统计信息
└── dataset_info.json     # 数据集元信息
```

---

## 模型微调

### 单 GPU 训练

```bash
python finetune_lerobot.py \
    --vla_path /Data/lzl/huggingface/univla-7b \
    --lam_path /Data/lzl/huggingface/univla-latent-action-model/lam-stage-2.ckpt \
    --data_dir /Data/rlds_raw/lerobot_converted_data/block \
    --batch_size 4 \
    --max_steps 10000
    --save_steps 2500 \
    --window_size 10 \
    --learning_rate 3.5e-4 \
    --use_lora True \
    --lora_rank 32
```

### 多 GPU 分布式训练

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 \
    finetune_lerobot.py \
    --vla_path /path/to/univla-7b \
    --lam_path /path/to/lam.ckpt \
    --data_dir ./converted_data/cup_hz_2.5_plus \
    --run_root_dir ./runs \
    --batch_size 4 \
    --grad_accumulation_steps 2 \
    --max_steps 10000 \
    --window_size 10 \
    --use_lora True
```

### 关键训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vla_path` | 必填 | UniVLA 预训练模型路径 |
| `--lam_path` | 必填 | LAM 检查点路径 |
| `--data_dir` | 必填 | 转换后的 HDF5 数据目录 |
| `--batch_size` | 4 | 批大小 |
| `--max_steps` | 10000 | 最大训练步数 |
| `--save_steps` | 2500 | 保存检查点间隔 |
| `--window_size` | 10 | 动作窗口大小（action chunking） |
| `--learning_rate` | 3.5e-4 | 学习率 |
| `--use_lora` | True | 是否使用 LoRA 微调 |
| `--lora_rank` | 32 | LoRA 秩 |
| `--freeze_vla` | False | 是否冻结 VLA 主干 |
| `--action_dim` | 7 | 动作维度 |
| `--camera_names` | None | 相机名称列表 |

### 训练输出

训练过程中会保存以下文件：

```
./runs/univla+lerobot+b8+lr-3.5e-4+lora-r32-ws-10/
├── config.json                    # 模型配置
├── model.safetensors              # 模型权重
├── action_decoder-2500.pt         # 动作解码器
├── action_decoder-5000.pt
├── action_decoder-10000.pt
├── norm_stats.json                # 归一化统计
└── processor_config.json          # 处理器配置
```

---

## 模型推理

### Python API 使用

```python
from inference import UniVLAInference
import numpy as np

# 初始化模型
policy = UniVLAInference(
    vla_path="./runs/univla+lerobot-xxx",
    decoder_path="./runs/univla+lerobot-xxx/action_decoder-10000.pt",
    norm_stats_path="./runs/univla+lerobot-xxx/norm_stats.json",
    window_size=10,
    action_dim=7,
)

# 获取单个动作
action = policy.step(
    curr_image=camera_image,  # PIL Image 或 numpy array [H, W, 3]
    task_instruction="pick up the cup with blue line",
    proprio=current_joint_positions,
)

# 获取所有窗口内的动作（action chunking）
all_actions = policy.get_all_actions(
    curr_image=camera_image,
    task_instruction="pick up the cup with blue line",
)

# 使用动作队列进行时序集成
action = policy.step(
    curr_image=camera_image,
    task_instruction="pick up the cup",
    use_action_queue=True,  # 启用动作队列
)
```

### 真机部署示例

```python
import cv2
from inference import UniVLAInference

# 初始化
policy = UniVLAInference(
    vla_path="./runs/univla+lerobot-xxx",
    decoder_path="./runs/univla+lerobot-xxx/action_decoder-10000.pt",
    norm_stats_path="./runs/univla+lerobot-xxx/norm_stats.json",
    window_size=10,
    pred_action_horizon=8,  # 预测动作数量
)

# 控制循环
task = "pick up the cup with blue line"
policy.reset()  # 重置动作队列

while not done:
    # 获取相机图像
    image = camera.capture()  # 你的相机接口
    
    # 获取动作
    action = policy.step(
        curr_image=image,
        task_instruction=task,
        use_action_queue=True,
    )
    
    # 执行动作
    robot.execute_action(action)  # 你的机器人接口
```

### 命令行测试

```bash
python inference.py \
    --vla_path ./runs/univla+lerobot-xxx \
    --decoder_path ./runs/univla+lerobot-xxx/action_decoder-10000.pt \
    --norm_stats_path ./runs/univla+lerobot-xxx/norm_stats.json \
    --window_size 10 \
    --task_instruction "pick up the cup with blue line"
```

---

## 一键运行

使用提供的脚本一键完成转换、训练和测试：

```bash
# 编辑 run.sh 中的配置
vim run.sh

# 修改以下变量
# - VLA_PATH: UniVLA 模型路径
# - LAM_PATH: LAM 检查点路径
# - DATASETS: 要处理的数据集

# 运行
chmod +x run.sh
./run.sh
```

---

## 常见问题

### 1. 内存不足 (OOM)

**问题：** 训练时 GPU 内存不足

**解决方案：**
- 减小 `--batch_size`
- 增加 `--grad_accumulation_steps`
- 使用 `--use_quantization True` 进行 4-bit 量化
- 减少 `--window_size`

### 2. 视频解码失败

**问题：** `PyAV failed to decode video`

**解决方案：**
```bash
# 安装 AV1 编解码器
sudo apt-get install libdav1d-dev

# 或使用 OpenCV 后备方案（已内置）
```

### 3. 数据维度不匹配

**问题：** `action_dim` 或 `qpos_dim` 不匹配

**解决方案：**
- 检查 `meta/info.json` 中的 `features` 字段
- 设置正确的 `--action_dim` 和 `--qpos_dim`
- 脚本会自动截取前 N 维

### 4. 找不到相机图像

**问题：** `Camera xxx not found`

**解决方案：**
- 检查 `info.json` 中的视频键名
- 使用完整的键名，如 `observation.images.top_rgb`
- 或使用简写名称 `top_rgb`

### 5. LoRA 合并失败

**问题：** 合并 LoRA 权重时出错

**解决方案：**
```python
# 手动合并
from peft import PeftModel
base_model = AutoModelForVision2Seq.from_pretrained(base_path)
peft_model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(output_path)
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `lerobot_to_univla.py` | 数据格式转换脚本 |
| `lerobot_dataset.py` | PyTorch Dataset 类 |
| `finetune_lerobot.py` | 微调训练脚本 |
| `inference.py` | 推理部署脚本 |
| `run.sh` | 一键运行脚本 |
| `README.md` | 本教程文档 |

---

## 参考

- [UniVLA 官方仓库](https://github.com/OpenDriveLab/UniVLA)
- [LeRobot 数据格式](https://github.com/huggingface/lerobot)
- [真机部署教程](../docs/real-world-deployment.md)

---

## 联系方式

如有问题，请提交 Issue 或联系作者。