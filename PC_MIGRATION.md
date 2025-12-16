# PC Migration Guide

This document contains all the steps needed to set up this project on a new PC.

## Overview

| Component | Size | Transfer Method |
|-----------|------|-----------------|
| Git repo (code + data) | ~31GB | GitHub |
| SeeDoo_Datasets | ~352MB | Included in `external_data/` |
| Docker image | ~33GB | Export/Import OR rebuild |

## Prerequisites on New PC

1. **Windows with WSL2** (or native Linux)
2. **Docker Desktop** with WSL2 backend
3. **Git**
4. **NVIDIA GPU drivers** (for CUDA support)
5. **Tailscale** (for server access)

---

## Step 1: Clone the Repository

```bash
# In WSL or Linux terminal
cd ~/Desktop  # or your preferred location
git clone https://github.com/Shaharak88/LLaMA-Factory-treadmill-movment.git
cd LLaMA-Factory-treadmill-movment
```

All code, data, experiment results, and SeeDoo_Datasets are now on your new PC.

---

## Step 2: Set Up Docker

### Option A: Rebuild Docker Image (Recommended - No file transfer needed)

```bash
# Build the image from scratch (~30 min)
cd docker
docker build -t llamafactory:with-opencv -f Dockerfile ..
```

### Option B: Transfer Docker Image from Old PC

On **old PC**:
```bash
# Export image to file (will be ~15-20GB compressed)
docker save llamafactory:with-opencv | gzip > llamafactory-image.tar.gz
```

Transfer `llamafactory-image.tar.gz` to new PC via USB/network drive.

On **new PC**:
```bash
# Import the image
docker load < llamafactory-image.tar.gz
```

### Option C: Pull from Hetzner Server

If the server is accessible:
```bash
# On server: save and compress
ssh seedoo@hetzner-gpu.tail9e6e7.ts.net
docker save llamafactory:with-opencv-fixed | gzip > /tmp/llamafactory-image.tar.gz

# On new PC: download and load
scp seedoo@hetzner-gpu.tail9e6e7.ts.net:/tmp/llamafactory-image.tar.gz .
docker load < llamafactory-image.tar.gz
docker tag llamafactory:with-opencv-fixed llamafactory:with-opencv
```

---

## Step 3: Set Up Environment Variables

Create `.env` file in the project root:
```bash
cp .env.local .env
# Edit .env with your specific paths and API keys
```

Key variables to configure:
- `HF_TOKEN` - Hugging Face token (for model downloads)
- Any API keys or credentials

---

## Step 4: Verify Tailscale Connection (for Server Access)

```bash
# Install Tailscale if not already
# https://tailscale.com/download

# Login
tailscale up

# Test connection to server
ssh seedoo@hetzner-gpu.tail9e6e7.ts.net "echo 'Connection OK'"
```

---

## Step 5: Test the Setup

```bash
# Test Docker
docker run --rm --gpus all llamafactory:with-opencv nvidia-smi

# Test Python environment inside container
docker run --rm -v $(pwd):/workspace llamafactory:with-opencv python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Project Structure

```
LLaMA-Factory/
├── data/                    # Experiment data (3.1GB)
│   ├── _exp_*/              # Individual experiment folders
│   └── experiments_log.csv  # Master experiment tracker
├── external_data/           # External datasets
│   └── SeeDoo_Datasets/     # SeeDoo backups and datasets (352MB)
├── saves/                   # Trained LoRA adapters
├── analytics/               # Reports and visualizations
├── scripts/                 # Utility scripts
├── src/                     # LLaMA-Factory source code
├── run_experiment.sh        # Main experiment runner
├── run_full_pipeline.py     # Full pipeline script
└── building_dataset.py      # Dataset builder
```

---

## Key Commands

### Run an experiment
```bash
./run_experiment.sh --experiment-name "my_exp" --train-videos 10 --test-videos 5
```

### Run on remote server
```bash
ssh seedoo@hetzner-gpu.tail9e6e7.ts.net
cd /workspace/LLaMA-Factory
./run_experiment.sh ...
```

### Generate report
```bash
python generate_experiment_report.py --experiment-id 123
```

---

## Hetzner Server Info

- **Host**: `hetzner-gpu.tail9e6e7.ts.net` (via Tailscale)
- **User**: `seedoo`
- **Workspace**: `/workspace/LLaMA-Factory`
- **Docker image**: `llamafactory:with-opencv-fixed`

To sync code to server:
```bash
rsync -avz --exclude '__pycache__' --exclude '.git' --exclude 'hf_cache' \
  ./ seedoo@hetzner-gpu.tail9e6e7.ts.net:/workspace/LLaMA-Factory/
```

---

## Troubleshooting

### Docker GPU not working
```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker can see GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Git clone fails (repo too large)
```bash
# Shallow clone first
git clone --depth 1 https://github.com/Shaharak88/LLaMA-Factory-treadmill-movment.git

# Then fetch full history if needed
git fetch --unshallow
```

### WSL runs out of disk space
```bash
# Check WSL disk usage
wsl --list --verbose
df -h

# Clean Docker
docker system prune -a
```

---

## Files NOT in Git (regenerate or download)

These large files are NOT in the repo - regenerate as needed:
- `hf_cache/` - Hugging Face model cache (auto-downloads)
- `ms_cache/` - Microsoft model cache
- Docker images (rebuild or transfer separately)

---

## Last Updated
2024-12-16 - Initial migration guide created
