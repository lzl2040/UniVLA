#!/bin/bash

# ============================================
# UniVLA Fine-tuning Pipeline for LeRobot Data
# ============================================

# Configuration
LEROBOT_DATA_DIR="/Data/lerobot_data/real_world"
OUTPUT_BASE_DIR="./converted_data"
RUN_BASE_DIR="./runs"

# Datasets to convert and train
DATASETS=("cup_hz_2.5_plus" "block_hz_4")

# Model paths (downloaded to /Data/lzl/huggingface)
VLA_PATH="/Data/lzl/huggingface/univla-7b"
LAM_PATH="/Data/lzl/huggingface/univla-latent-action-model/lam-stage-2.ckpt"

# Training parameters
BATCH_SIZE=4
MAX_STEPS=10000
SAVE_STEPS=2500
WINDOW_SIZE=10
LEARNING_RATE=3.5e-4

# GPU settings
NUM_GPUS=8

# ============================================
# Step 1: Convert LeRobot data to HDF5 format
# ============================================
echo "=========================================="
echo "Step 1: Converting LeRobot data to HDF5..."
echo "=========================================="

for DATASET in "${DATASETS[@]}"; do
    echo "Converting dataset: $DATASET"
    
    INPUT_DIR="$LEROBOT_DATA_DIR/$DATASET"
    OUTPUT_DIR="$OUTPUT_BASE_DIR/$DATASET"
    
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Warning: $INPUT_DIR not found, skipping..."
        continue
    fi
    
    python lerobot_to_univla.py \
        --lerobot_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --cameras observation.images.top_rgb \
        --compress
    
    echo "Conversion complete for $DATASET"
done

# ============================================
# Step 2: Fine-tune UniVLA on converted data
# ============================================
echo "=========================================="
echo "Step 2: Fine-tuning UniVLA..."
echo "=========================================="

# You can train on a single dataset or merge multiple datasets
# Here we use the first dataset as an example
TRAIN_DATA_DIR="$OUTPUT_BASE_DIR/${DATASETS[0]}"

# Check if converted data exists
if [ ! -d "$TRAIN_DATA_DIR" ]; then
    echo "Error: Converted data not found at $TRAIN_DATA_DIR"
    exit 1
fi

# Run distributed training
torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPUS \
    finetune_lerobot.py \
    --vla_path "$VLA_PATH" \
    --lam_path "$LAM_PATH" \
    --data_dir "$TRAIN_DATA_DIR" \
    --run_root_dir "$RUN_BASE_DIR" \
    --batch_size $BATCH_SIZE \
    --max_steps $MAX_STEPS \
    --save_steps $SAVE_STEPS \
    --window_size $WINDOW_SIZE \
    --learning_rate $LEARNING_RATE \
    --use_lora True \
    --lora_rank 32 \
    --image_aug True \
    --wandb_project "finetune-lerobot" \
    --wandb_entity "your-entity"

echo "=========================================="
echo "Training complete!"
echo "=========================================="

# ============================================
# Step 3 (Optional): Test inference
# ============================================
echo "=========================================="
echo "Step 3: Testing inference..."
echo "=========================================="

# Find the latest checkpoint
LATEST_RUN=$(ls -td "$RUN_BASE_DIR"/univla+lerobot* 2>/dev/null | head -1)

if [ -n "$LATEST_RUN" ]; then
    echo "Found checkpoint: $LATEST_RUN"
    
    # Find the latest action decoder
    LATEST_DECODER=$(ls -t "$LATEST_RUN"/action_decoder-*.pt 2>/dev/null | head -1)
    
    if [ -n "$LATEST_DECODER" ]; then
        echo "Testing with decoder: $LATEST_DECODER"
        
        python inference.py \
            --vla_path "$LATEST_RUN" \
            --decoder_path "$LATEST_DECODER" \
            --norm_stats_path "$LATEST_RUN/norm_stats.json" \
            --window_size $WINDOW_SIZE \
            --task_instruction "pick up the cup"
    else
        echo "No action decoder found in $LATEST_RUN"
    fi
else
    echo "No checkpoint found in $RUN_BASE_DIR"
fi

echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="