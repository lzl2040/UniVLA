#!/bin/bash
# ============================================================
# UniVLA Environment Installation Script
# ============================================================
# This script sets up the conda environment for UniVLA 
# fine-tuning on LeRobot data.
#
# Usage:
#   chmod +x install_env.sh
#   ./install_env.sh
# ============================================================

set -e  # Exit on error

# Configuration
ENV_NAME="univla"
PYTHON_VERSION="3.10"
UNIVLA_PATH="/Data/lzl/huggingface/univla-7b"  # Change this to your model path

echo "============================================"
echo "UniVLA Environment Installation"
echo "============================================"
echo "Environment name: ${ENV_NAME}"
echo "Python version: ${PYTHON_VERSION}"
echo "============================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Initialize conda for current shell
echo "Initializing conda..."
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Using existing environment."
    fi
fi

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

# Activate environment
echo "Activating environment..."
conda activate ${ENV_NAME}

# Verify Python version
echo "Python version: $(python --version)"

# ============================================================
# Install PyTorch and CUDA dependencies
# ============================================================
echo ""
echo "============================================"
echo "Installing PyTorch with CUDA support..."
echo "============================================"

# Detect CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | head -1)
if [ -z "$CUDA_VERSION" ]; then
    CUDA_VERSION="12.1"  # Default fallback
fi
echo "Detected CUDA version: ${CUDA_VERSION}"

# Install PyTorch (adjust CUDA version as needed)
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ============================================================
# Install core dependencies
# ============================================================
echo ""
echo "============================================"
echo "Installing core dependencies..."
echo "============================================"

pip install -q \
    transformers>=4.40.0 \
    accelerate>=0.30.0 \
    peft>=0.10.0 \
    bitsandbytes>=0.43.0 \
    sentencepiece \
    protobuf \
    h5py \
    einops \
    wandb \
    draccus \
    ema-pytorch \
    tqdm \
    pillow \
    opencv-python

# ============================================================
# Install UniVLA package
# ============================================================
echo ""
echo "============================================"
echo "Installing UniVLA package..."
echo "============================================"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
UNIVLA_ROOT="$(dirname "$SCRIPT_DIR")"

echo "UniVLA root: ${UNIVLA_ROOT}"

# Install UniVLA from the parent directory
if [ -f "${UNIVLA_ROOT}/setup.py" ]; then
    pip install -e "${UNIVLA_ROOT}"
elif [ -f "${UNIVLA_ROOT}/pyproject.toml" ]; then
    pip install -e "${UNIVLA_ROOT}"
else
    echo "Warning: UniVLA setup.py/pyproject.toml not found"
    echo "Installing prismatic package directly..."
    pip install -e "${UNIVLA_ROOT}/prismatic"
fi

# ============================================================
# Install additional dependencies for LAM
# ============================================================
echo ""
echo "============================================"
echo "Installing LAM dependencies..."
echo "============================================"

pip install -q \
    lightning \
    piq \
    timm

# Download DINOv2 model (cached by torch.hub)
echo "Pre-caching DINOv2 model..."
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')" 2>/dev/null || echo "DINOv2 will be downloaded on first use"

# ============================================================
# Install flash-attn (optional, for faster inference)
# ============================================================
echo ""
echo "============================================"
echo "Installing flash-attn (optional)..."
echo "============================================"

# flash-attn requires compilation, may take a while
# Skip if installation fails
pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn installation skipped (optional)"

# ============================================================
# Verify installation
# ============================================================
echo ""
echo "============================================"
echo "Verifying installation..."
echo "============================================"

python -c "
import torch
import transformers
import accelerate
import peft

print('✓ PyTorch:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ CUDA version:', torch.version.cuda)
    print('✓ GPU:', torch.cuda.get_device_name(0))
print('✓ Transformers:', transformers.__version__)
print('✓ Accelerate:', accelerate.__version__)
print('✓ PEFT:', peft.__version__)

# Test UniVLA imports
try:
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    print('✓ UniVLA imports: OK')
except Exception as e:
    print('✗ UniVLA imports failed:', e)

# Test LAM imports
try:
    from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel
    print('✓ LAM imports: OK')
except Exception as e:
    print('✗ LAM imports failed:', e)
"

# ============================================================
# Create necessary directories
# ============================================================
echo ""
echo "============================================"
echo "Creating directories..."
echo "============================================"

mkdir -p "${SCRIPT_DIR}/runs"
mkdir -p "${SCRIPT_DIR}/converted_data"

# ============================================================
# Done!
# ============================================================
echo ""
echo "============================================"
echo "Installation complete!"
echo "============================================"
echo ""
echo "To activate the environment, run:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "To test training, run:"
echo "    python finetune_lerobot_full.py --vla_path ${UNIVLA_PATH} --data_dir ./converted_data/block_hz_4 --run_dir ./runs/test"
echo ""
echo "============================================"