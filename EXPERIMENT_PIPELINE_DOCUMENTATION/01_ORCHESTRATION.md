# Orchestration: run_experiment.sh

## Purpose
`run_experiment.sh` is the main entry point for the entire experiment pipeline. It orchestrates code synchronization to a remote GPU server, executes the training pipeline in Docker, retrieves results, and generates reports locally.

## Location
`/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/run_experiment.sh`

## High-Level Flow

```
[Local PC]                      [GPU Server: hetzner-gpu.tail9e6e7.ts.net]
    │                                          │
    ├─1. rsync code ──────────────────────────>│
    │                                          │
    │                                          ├─2. Docker: building_dataset.py
    │                                          │    (Generate videos + datasets)
    │                                          │
    │                                          ├─3. Docker: run_full_pipeline.py
    │                                          │    (Train model + Evaluate)
    │                                          │
    ├─4. rsync results <──────────────────────┤
    │    (Download datasets, models, eval)    │
    │                                          │
    ├─5. generate_experiment_report.py        │
    │    (Create HTML report locally)         │
    │                                          │
    └──────────────────────────────────────────┘
```

## Script Structure

### 1. Configuration Variables

```bash
PROJECT_ROOT="/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory"
REMOTE_USER="seedoo"
REMOTE_HOST="hetzner-gpu.tail9e6e7.ts.net"
REMOTE_PROJECT_ROOT="/workspace"
DOCKER_IMAGE="llamafactory:latest"
```

**Why these values?**
- `PROJECT_ROOT`: Local Windows path (WSL2 mount)
- `REMOTE_USER` + `REMOTE_HOST`: SSH connection to GPU server via Tailscale VPN
- `REMOTE_PROJECT_ROOT`: Docker workspace directory (mounted volume)
- `DOCKER_IMAGE`: Pre-built Docker image with all dependencies

### 2. Stage 1: Sync Code to Server

```bash
echo "=== Step 1: Syncing code to server ==="
rsync -avz --progress \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude 'data/treadmill_videos/' \
  --exclude 'data/videos/' \
  --exclude 'saves/' \
  --exclude '.venv' \
  --exclude 'venv' \
  --exclude 'node_modules' \
  "$PROJECT_ROOT/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_ROOT}/"
```

**Parameters explained:**
- `-a`: Archive mode (preserves permissions, timestamps, symlinks)
- `-v`: Verbose output
- `-z`: Compress during transfer
- `--progress`: Show transfer progress
- `--exclude`: Skip unnecessary files/folders to speed up sync

**Why exclude these directories?**
- `__pycache__/`, `*.pyc`: Python bytecode (regenerated automatically)
- `.git`: Git history (unnecessary, saves bandwidth)
- `data/treadmill_videos/`, `data/videos/`: Large video files (generated on server)
- `saves/`: Model checkpoints (downloaded separately after training)
- `.venv`, `venv`, `node_modules`: Virtual environments (not needed)

**What gets synced?**
- All Python scripts (.py)
- Configuration files (.yaml, .json)
- Shell scripts (.sh)
- Documentation files (.md)

### 3. Stage 2: Build Dataset (Docker)

```bash
echo "=== Step 2: Building dataset on server (Docker) ==="
ssh "${REMOTE_USER}@${REMOTE_HOST}" << 'EOF'
docker run --rm \
  --gpus all \
  -v /workspace:/workspace \
  -w /workspace \
  llamafactory:latest \
  python building_dataset.py \
    --dataset_name "_exp_$(date +%Y%m%d_%H%M%S)" \
    --num_videos 100 \
    --train_split 0.8 \
    --train_texture_type "subtle_gray_stripes" \
    --test_texture_type "subtle_gray_stripes" \
    --train_direction "right" \
    --test_direction "right" \
    --train_view_angle "0" \
    --test_view_angle "52" \
    --train_speed_range "3.0,3.0" \
    --test_speed_range "3.0,3.0" \
    --train_edge_width 15 \
    --test_edge_width 15 \
    --train_distance 3.0 \
    --test_distance 3.0
EOF
```

**Docker parameters explained:**
- `--rm`: Remove container after execution (cleanup)
- `--gpus all`: Expose all GPUs to container (for Blender rendering)
- `-v /workspace:/workspace`: Mount host directory into container
- `-w /workspace`: Set working directory inside container
- `llamafactory:latest`: Docker image name

**Dataset parameters explained:**
- `--dataset_name`: Unique identifier with timestamp (e.g., "_exp_20251207_143033")
- `--num_videos`: Total videos to generate (100)
- `--train_split`: 80% training, 20% testing (0.8)

**Train/Test Configuration:**
- Both use same texture (`subtle_gray_stripes`)
- Both move right
- **Key difference**: Train angle=0°, Test angle=52° (tests generalization)
- Both use same speed (3.0 units), edge width (15px), distance (3.0)

**Why test with different angle?**
- Validates model generalization to new viewpoints
- Angle=0° is frontal view, Angle=52° is elevated/angled view

### 4. Stage 3: Train Model (Docker)

```bash
echo "=== Step 3: Running full pipeline on server (Docker) ==="
DATASET_NAME=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" \
  "ls -t ${REMOTE_PROJECT_ROOT}/data/ | grep '^_exp_' | head -n 1 | sed 's/_train$//' | sed 's/_test$//' | sort -u | head -n 1")
echo "Using dataset: $DATASET_NAME"

ssh "${REMOTE_USER}@${REMOTE_HOST}" << EOF
docker run --rm \
  --gpus all \
  --shm-size=16g \
  -v /workspace:/workspace \
  -w /workspace \
  llamafactory:latest \
  python run_full_pipeline.py \
    --dataset_name "$DATASET_NAME" \
    --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --save_steps 100 \
    --use_dora \
    --eval_method "yesno"
EOF
```

**Dataset name detection:**
```bash
DATASET_NAME=$(ssh ... "ls -t ... | grep '^_exp_' | head -n 1 ...")
```
- `ls -t`: List files sorted by modification time (newest first)
- `grep '^_exp_'`: Filter for experiment datasets
- `head -n 1`: Get most recent dataset
- `sed` commands: Clean up suffixes (_train, _test)

**Docker parameters:**
- `--shm-size=16g`: Increase shared memory for DataLoader (prevents crashes with multiple workers)

**Training parameters:**
- `--model_name_or_path`: Base model from Hugging Face
- `--num_train_epochs`: Training epochs (10)
- `--learning_rate`: LoRA learning rate (5e-5)
- `--per_device_train_batch_size`: Batch size per GPU (1, due to video memory)
- `--gradient_accumulation_steps`: Effective batch size = 1 × 8 = 8
- `--lora_rank`: LoRA rank (32, controls adapter capacity)
- `--lora_alpha`: LoRA alpha (64, scales LoRA weights)
- `--save_steps`: Save checkpoint every 100 steps
- `--use_dora`: Use DoRA (improved LoRA variant)
- `--eval_method`: Evaluation format ("yesno" = yes/no answers)

### 5. Stage 4: Retrieve Results

```bash
echo "=== Step 4: Retrieving results from server ==="

# Create local directories
mkdir -p "$PROJECT_ROOT/data"
mkdir -p "$PROJECT_ROOT/saves"
mkdir -p "$PROJECT_ROOT/evaluation_results"

# Download datasets (videos + JSON files)
echo "Downloading datasets..."
rsync -avz --progress \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_ROOT}/data/${DATASET_NAME}_train/" \
  "$PROJECT_ROOT/data/${DATASET_NAME}_train/"

rsync -avz --progress \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_ROOT}/data/${DATASET_NAME}_test/" \
  "$PROJECT_ROOT/data/${DATASET_NAME}_test/"

rsync -avz --progress \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_ROOT}/data/${DATASET_NAME}_*.json" \
  "$PROJECT_ROOT/data/"

rsync -avz --progress \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_ROOT}/data/dataset_info.json" \
  "$PROJECT_ROOT/data/"

# Download model checkpoints
echo "Downloading model..."
MODEL_NAME=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" \
  "ls -t ${REMOTE_PROJECT_ROOT}/saves/ | grep ${DATASET_NAME} | head -n 1")
echo "Model: $MODEL_NAME"

rsync -avz --progress \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_ROOT}/saves/${MODEL_NAME}/" \
  "$PROJECT_ROOT/saves/${MODEL_NAME}/"

# Download evaluation results
echo "Downloading evaluation results..."
rsync -avz --progress \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_ROOT}/evaluation_results/" \
  "$PROJECT_ROOT/evaluation_results/"

# Download experiment log
echo "Downloading experiment log..."
rsync -avz --progress \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_ROOT}/experiments_log.csv" \
  "$PROJECT_ROOT/"
```

**Why download in stages?**
1. **Datasets first**: Essential for report generation
2. **Model next**: Large files, can be reused
3. **Evaluation results**: Metrics and predictions
4. **Experiment log**: Tracking across all experiments

**What gets downloaded?**
- `data/{dataset}_train/`: All training videos and metadata
- `data/{dataset}_test/`: All test videos and evaluation results
- `data/{dataset}_train.json`, `data/{dataset}_test.json`: Dataset JSON files
- `data/dataset_info.json`: Dataset registry
- `saves/{model_name}/`: Trained model checkpoint
- `evaluation_results/`: Evaluation reports and CSVs
- `experiments_log.csv`: The specific experiment row (appended to local CSV)

**CSV Sync Behavior (Updated 2025-12-08):**

The experiment CSV sync uses `experiment_id` to ensure the correct row is retrieved:

1. After pipeline execution, the script queries the server for the newest `experiment_id`
2. Uses base64-encoded Python scripts to avoid shell escaping issues when passing code through SSH
3. Retrieves the specific row by `experiment_id` (unique identifier) using Python's CSV module
4. If the experiment already exists locally, updates the row; otherwise appends it

**Technical Details:**
The row retrieval uses a heredoc with single-quoted delimiter (`'PYSCRIPT'`) to prevent shell expansion, then base64-encodes the Python script before sending via SSH. This avoids complex escaping of quotes, commas, and newlines that would otherwise corrupt the Python code when interpreted by nested shells (local bash → SSH → remote bash → docker exec → Python).

This approach fixes issues when:
- Running multiple evaluations on the same dataset with different models
- Running with different eval_method (yesno vs moving_stopped) on same dataset
- The server has accumulated experiments that weren't synced locally
- Re-evaluating the same experiment_id (updates local row instead of duplicating)

The `experiment_id` is then passed to the HTML report generator to ensure the correct experiment is reported.

### 6. Stage 5: Generate Report (Local)

```bash
echo "=== Step 5: Generating HTML report locally ==="
cd "$PROJECT_ROOT"
python generate_experiment_report.py \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "analytics/reports" \
  --skip_video_download
```

**Why generate report locally?**
- Faster (no SSH latency)
- Browser access to HTML files
- Easier debugging

**Parameters:**
- `--experiment-id`: Specific experiment ID to report (preferred, unique identifier)
- `--dataset-name`: Which dataset to analyze (fallback, returns most recent experiment with that name)
- `--output_dir`: Where to save HTML report
- `--skip_video_download`: Videos already downloaded in Stage 4

**Note (Updated 2025-12-08):** When running in REMOTE mode, the script passes `--experiment-id` to ensure the correct experiment is reported. In LOCAL mode, it falls back to `--dataset-name` which now returns the most recent experiment with that name (last match, not first).

### 7. Completion

```bash
echo ""
echo "==========================================="
echo "       EXPERIMENT PIPELINE COMPLETE!       "
echo "==========================================="
echo ""
echo "Results:"
echo "  - Dataset: $PROJECT_ROOT/data/${DATASET_NAME}_*"
echo "  - Model: $PROJECT_ROOT/saves/${MODEL_NAME}"
echo "  - Evaluation: $PROJECT_ROOT/evaluation_results/"
echo "  - Experiment Log: $PROJECT_ROOT/experiments_log.csv"
echo "  - Report: $PROJECT_ROOT/analytics/reports/"
echo ""
```

## Error Handling

The script uses `set -e` to exit on any error. If any stage fails:
1. Check SSH connection: `ssh seedoo@hetzner-gpu.tail9e6e7.ts.net`
2. Check Docker: `ssh ... docker ps -a`
3. Check logs on server: `ssh ... tail -f /workspace/run_full_pipeline.log`
4. Check disk space: `ssh ... df -h`

## Customization

### Change training parameters
Edit the docker run command in Stage 3:
```bash
--num_train_epochs 20    # More epochs
--learning_rate 1e-4     # Different LR
```

### Change dataset parameters
Edit the docker run command in Stage 2:
```bash
--num_videos 200         # More videos
--train_view_angle "0,30,52"  # Multiple angles
```

### Skip stages
Comment out stages you don't need:
```bash
# Skip dataset generation (reuse existing)
# echo "=== Step 2: Building dataset..."

# Skip model download (save bandwidth)
# rsync -avz ... saves/...
```

## Dependencies

### Local Machine
- rsync (file synchronization)
- ssh (remote execution)
- Python 3.10+ (for report generation)

### GPU Server
- Docker with NVIDIA runtime
- llamafactory:latest image
- Tailscale (for VPN connectivity)

## Security Notes

- SSH keys must be configured for passwordless login
- Tailscale provides encrypted VPN connection
- Docker container runs with workspace volume only (no host access)

## Performance Tips

1. **Initial sync is slow**: First rsync transfers all files (~10-20 min)
2. **Subsequent syncs are fast**: Only changed files (~1-2 min)
3. **Video downloads**: Large files, use `--compress` for rsync
4. **Model downloads**: Checkpoints are ~5-10 GB

## Next Steps
- Read `02_DATA_GENERATION.md` for dataset building details
- Read `03_TRAINING.md` for training pipeline details
