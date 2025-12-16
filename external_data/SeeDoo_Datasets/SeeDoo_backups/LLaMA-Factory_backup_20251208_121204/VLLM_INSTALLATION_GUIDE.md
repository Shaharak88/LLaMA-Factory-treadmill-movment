# vLLM Installation Guide for LlamaFactory Docker Container

**Date**: November 26, 2025
**Purpose**: Enable high-performance vLLM inference for the treadmill detection pipeline

---

## Overview

This document describes the installation of vLLM into the LlamaFactory Docker container to enable the `run_full_pipeline.py` script to function correctly.

### Background

The LlamaFactory project provides two evaluation approaches:
1. **Simple Evaluation** (`evaluate_treadmill_lora.py`) - Uses standard HuggingFace Transformers
2. **Pipeline Evaluation** (`run_full_pipeline.py`) - Uses vLLM for high-performance batch inference

The pipeline script (`run_full_pipeline.py`) requires vLLM to be installed, but it was not included in the base Docker image. This guide documents the installation process and how to save the updated image.

---

## Prerequisites

- Docker installed and running
- LlamaFactory Docker container running (container name: `llamafactory`)
- Sufficient disk space (vLLM installation requires ~3-4 GB)
- Internet connection for package downloads

---

## Installation Steps

### 1. Verify Container is Running

```bash
docker ps --filter "name=llamafactory"
```

Expected output should show the container is "Up".

### 2. Install vLLM in the Container

```bash
docker exec llamafactory pip install vllm
```

**Installation Time**: Approximately 5-10 minutes (depending on internet speed)

**What Gets Installed**:
- vLLM 0.11.2
- PyTorch 2.9.0 (if not already present or needs upgrade)
- CUDA libraries (cuD NN, cuBLAS, etc.)
- Ray framework for distributed computing
- Various inference optimization libraries

### 3. Verify Installation

After installation completes, verify vLLM is installed:

```bash
docker exec llamafactory python3 -c "import vllm; print('vLLM version:', vllm.__version__)"
```

Expected output:
```
vLLM version: 0.11.2
```

---

## Saving the Updated Docker Image

Once vLLM is successfully installed, save the container as a new Docker image to avoid reinstalling in the future.

###  Stop the Container (Optional but Recommended)

```bash
docker stop llamafactory
```

### Commit the Container as a New Image

```bash
docker commit llamafactory llamafactory:with-vllm
```

This creates a new image named `llamafactory:with-vllm` that includes vLLM.

### Tag the Image with a Date (Optional)

```bash
docker tag llamafactory:with-vllm llamafactory:vllm-20251126
```

### Verify the New Image

```bash
docker images | grep llamafactory
```

Expected output should show:
- `llamafactory:with-vllm`
- `llamafactory:vllm-20251126` (if tagged)
- Original `llamafactory` image

---

## Using the New Image

### Option 1: Update docker-compose.yml

If using docker-compose, update the image reference:

```yaml
services:
  llamafactory:
    image: llamafactory:with-vllm  # Changed from llamafactory:latest
    # ... rest of configuration
```

### Option 2: Run Container Directly

```bash
docker run --gpus all \
  -v /path/to/LLaMA-Factory:/app \
  -p 7860:7860 \
  --name llamafactory \
  llamafactory:with-vllm
```

---

## Testing the Installation

### Test 1: Verify vLLM Import

```bash
docker exec llamafactory python3 -c "from vllm import LLM, SamplingParams; print('✓ vLLM imports successfully')"
```

### Test 2: Run the Pipeline Script

```bash
docker exec llamafactory python3 /app/run_full_pipeline.py \
    --dataset_name check_dataset \
    --texture_type stripes \
    --direction left \
    --view_angle 0.0,30.0 \
    --speed_range 0,12.0 \
    --fps 4 \
    --duration 12.0 \
    --eval_video_fps 4 \
    --eval_video_maxlen 12
```

The pipeline should now run without the `NameError: name 'LLM' is not defined` error.

---

## Disk Space Requirements

| Component | Size |
|-----------|------|
| vLLM package | ~370 MB |
| PyTorch 2.9.0 | ~900 MB |
| CUDA libraries | ~2 GB |
| Dependencies | ~500 MB |
| **Total** | **~3.7 GB** |

---

## Troubleshooting

### Issue: Container Out of Memory During Installation

**Solution**: Increase Docker memory limit in Docker Desktop settings (recommend 8GB minimum).

### Issue: CUDA Version Mismatch

**Error**: `CUDA version mismatch`

**Solution**: vLLM 0.11.2 requires CUDA 12.x. Verify your NVIDIA drivers support CUDA 12:

```bash
nvidia-smi
```

### Issue: Installation Hangs

**Solution**:
1. Check internet connection
2. Clear pip cache: `docker exec llamafactory pip cache purge`
3. Retry installation

---

##Alternative: Using Standard Transformers Inference

If vLLM installation fails or is not needed, use the simpler evaluation script:

```bash
python3 evaluate_treadmill_lora.py
```

This script uses standard HuggingFace Transformers inference and doesn't require vLLM.

---

## Related Files

- `scripts/vllm_infer.py` - vLLM inference script (import bug fixed)
- `run_full_pipeline.py` - Full pipeline script (requires vLLM)
- `evaluate_pipeline.py` - Evaluation wrapper (calls vllm_infer.py)
- `evaluate_treadmill_lora.py` - Alternative evaluation (no vLLM needed)
- `BUGFIX_CHANGELOG_20251126.md` - Bug fix documentation for vllm_infer.py

---

## Version Information

- **vLLM Version**: 0.11.2
- **PyTorch Version**: 2.9.0
- **CUDA Version**: 12.8
- **Python Version**: 3.11
- **Container Base**: LlamaFactory (latest)

---

## Notes

- vLLM provides significant performance improvements for batch inference
- The installation is large but necessary for production-scale inference
- The Docker image with vLLM can be shared or deployed to other systems
- For development/testing, the standard Transformers approach may be sufficient

---

## Support

For issues related to:
- **vLLM**: https://github.com/vllm-project/vllm
- **LlamaFactory**: https://github.com/hiyouga/LLaMA-Factory
- **This Project**: See BUGFIX_CHANGELOG_20251126.md for recent changes
