# Docker Setup Guide for LLaMA Factory

This guide will help you run the entire LLaMA Factory project in Docker without missing anything.

## Prerequisites

1. **Docker** (version 20.10+)
   ```bash
   docker --version
   ```

2. **Docker Compose** (version 2.0+)
   ```bash
   docker compose version
   ```

3. **NVIDIA Docker Runtime** (for GPU support)
   ```bash
   # Install nvidia-docker2
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker

   # Test GPU access
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

## Quick Start

### 1. Build the Docker Image

From the project root directory:

```bash
docker compose build
```

This will:
- Use the CUDA-enabled base image with PyTorch
- Install all Python dependencies from requirements.txt
- Install LLaMA Factory with metrics support
- Fix package versions (transformers 4.57.1, qwen-vl-utils 0.0.14) for proper LoRA support
- Set up the environment

**Build arguments** (optional, modify in docker-compose.yml):
- `PIP_INDEX`: PyPI mirror (default: https://pypi.org/simple)
- `EXTRAS`: Extra dependencies (default: metrics)
- `INSTALL_FLASHATTN`: Install flash-attention (default: false)

### 2. Start the Container

```bash
docker compose up -d
```

This starts the container in detached mode with:
- GPU access (all available GPUs)
- Port 7860 exposed for Gradio UI
- Port 8000 exposed for API service
- All volumes mounted (data, saves, cache, etc.)

### 3. Access the Container

Enter the running container:

```bash
docker compose exec llamafactory bash
```

Or start a new interactive session:

```bash
docker compose run --rm llamafactory bash
```

## What's Included

### Volume Mounts

All important directories are mounted to persist data and allow development:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./src` | `/app/src` | Source code (for live development) |
| `./examples` | `/app/examples` | Training configs and examples |
| `./scripts` | `/app/scripts` | Utility scripts |
| `./data` | `/app/data` | Training data (including treadmill_videos/) |
| `./saves` | `/app/saves` | Model checkpoints and outputs |
| `./hf_cache` | `/root/.cache/huggingface` | HuggingFace model cache |
| `./ms_cache` | `/root/.cache/modelscope` | ModelScope cache |
| `./output` | `/app/output` | Training outputs |

### Exposed Ports

- **7860**: Gradio Web UI (LLaMA Board)
- **8000**: API Service (FastAPI)

### Environment Variables

Configure via `.env` file in the project root. Key variables:

- `CUDA_VISIBLE_DEVICES`: GPU selection
- `GRADIO_SERVER_PORT`: Web UI port
- `API_PORT`: API service port
- `WANDB_*`: Weights & Biases integration
- `HF_HOME`: HuggingFace cache location

## Usage Examples

### Launch Web UI (Gradio)

```bash
# Inside container
llamafactory-cli webui

# Or from host
docker compose exec llamafactory llamafactory-cli webui
```

Access at: http://localhost:7860

### Train a Model (LoRA Fine-tuning)

```bash
# Inside container
llamafactory-cli train examples/train_qlora/qwen25vl_lora_sft.yaml

# Or from host
docker compose exec llamafactory llamafactory-cli train examples/train_qlora/qwen25vl_lora_sft.yaml
```

### Run Inference

```bash
# Inside container
python scripts/vllm_infer.py

# Or from host
docker compose exec llamafactory python scripts/vllm_infer.py
```

### API Service

```bash
# Start API service inside container
llamafactory-cli api examples/inference/qwen2_vl.yaml

# Or from host
docker compose exec llamafactory llamafactory-cli api examples/inference/qwen2_vl.yaml
```

Access API at: http://localhost:8000

### Interactive Chat

```bash
docker compose exec llamafactory llamafactory-cli chat examples/inference/qwen2_vl.yaml
```

## Treadmill Detection Project

Your treadmill detection setup is fully included:

1. **Data**: `./data/treadmill_videos/` is mounted at `/app/data/treadmill_videos/`
2. **Config**: `examples/train_qlora/qwen25vl_lora_sft.yaml` is accessible
3. **Inference Script**: `scripts/vllm_infer.py` is mounted

To train:
```bash
docker compose exec llamafactory llamafactory-cli train examples/train_qlora/qwen25vl_lora_sft.yaml
```

## Managing the Container

### View Logs

```bash
# Follow logs
docker compose logs -f

# View specific service logs
docker compose logs llamafactory
```

### Stop the Container

```bash
docker compose down
```

### Restart the Container

```bash
docker compose restart
```

### Remove Everything (including volumes)

```bash
docker compose down -v
```

### Rebuild After Code Changes

```bash
docker compose build --no-cache
docker compose up -d
```

## Development Workflow

Since source code directories are mounted as volumes, you can:

1. **Edit code on your host** machine using your preferred editor
2. **Changes are immediately reflected** inside the container
3. **No need to rebuild** the image for code changes
4. **Restart the container** if you need to reload Python modules

Example workflow:
```bash
# Edit code on host
vim src/llamafactory/train/tuner.py

# Execute inside container without rebuild
docker compose exec llamafactory python -m llamafactory.train.tuner
```

## Troubleshooting

### GPU Not Detected

```bash
# Check GPU availability inside container
docker compose exec llamafactory nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Permission Issues

If you encounter permission issues with volumes:
```bash
# Fix ownership (run on host)
sudo chown -R $USER:$USER ./hf_cache ./ms_cache ./output ./saves
```

### Port Already in Use

If ports 7860 or 8000 are already in use, modify `docker-compose.yml`:
```yaml
ports:
  - "7861:7860"  # Change host port
  - "8001:8000"  # Change host port
```

### Out of Memory

Increase shared memory or use host IPC (already configured):
```yaml
ipc: host  # Already set in docker-compose.yml
```

Or set shared memory size:
```yaml
shm_size: "16gb"
```

### Container Exits Immediately

Check logs:
```bash
docker compose logs llamafactory
```

The container is configured with `command: bash` and `stdin_open: true`, so it stays running. Access it with:
```bash
docker compose exec llamafactory bash
```

## Advanced Configuration

### Use Multiple GPUs

Modify `.env`:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
```

Or in docker-compose.yml:
```yaml
deploy:
  resources:
    reservations:
      devices:
      - driver: nvidia
        device_ids: ['0', '1', '2', '3']
        capabilities: [gpu]
```

### Use Different Base Image

Modify `docker-compose.yml`:
```yaml
build:
  args:
    BASE_IMAGE: hiyouga/pytorch:th2.6.0-cu124-flashattn2.7.4-cxx11abi0-devel
```

Available base images: https://hub.docker.com/r/hiyouga/pytorch/tags

### Install Flash Attention

Modify `docker-compose.yml`:
```yaml
build:
  args:
    INSTALL_FLASHATTN: "true"
```

Then rebuild:
```bash
docker compose build
```

### Connect to Remote Registry

If using private models or custom registries:
```bash
# Login to HuggingFace
docker compose exec llamafactory huggingface-cli login

# Login to ModelScope
docker compose exec llamafactory modelscope login
```

Tokens are persisted in `./hf_cache`.

## Production Deployment

For production, consider:

1. **Remove development volumes** (source code mounts)
2. **Use specific image tags** instead of `latest`
3. **Set resource limits**:
   ```yaml
   deploy:
     resources:
       limits:
         memory: 32G
   ```
4. **Use secrets** for API keys instead of environment variables
5. **Enable HTTPS** with a reverse proxy (nginx, traefik)
6. **Set up monitoring** (Prometheus, Grafana)

## Additional Resources

- [LLaMA Factory Documentation](https://github.com/hiyouga/LLaMA-Factory)
- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

## Summary Checklist

- [ ] Docker and Docker Compose installed
- [ ] NVIDIA Docker runtime installed (for GPU)
- [ ] `.env` file configured
- [ ] `docker-compose.yml` reviewed
- [ ] Image built: `docker compose build`
- [ ] Container started: `docker compose up -d`
- [ ] GPU accessible: `docker compose exec llamafactory nvidia-smi`
- [ ] Data volumes mounted correctly
- [ ] Ports accessible (7860, 8000)
- [ ] Can access container: `docker compose exec llamafactory bash`

Now you're ready to run LLaMA Factory in Docker!
