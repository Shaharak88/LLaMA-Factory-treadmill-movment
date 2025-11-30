# Complete Workflow: Running Experiments with Tracking

This guide shows the **complete end-to-end workflow** for running training experiments with automatic CSV tracking, including local development, server deployment, and result retrieval.

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Workflow Steps](#workflow-steps)
4. [Example: Training with Custom Textures](#example-training-with-custom-textures)
5. [CSV Structure](#csv-structure)
6. [Troubleshooting](#troubleshooting)

---

## Overview

**The Big Picture:**
```
Local PC (Windows/WSL)          Remote Server (Docker)            Results
─────────────────────           ──────────────────────           ─────────
1. Make code changes    ───►    2. SSH + Rsync           ───►    5. Rsync back
   - Modify scripts                - Sync all files                  - experiments_log.csv
   - Test locally                  - Files go to /app                - evaluation reports
                                                                       - trained models
                                 3. Build datasets
                                    - building_dataset.py
                                    - Run twice (train/test)

                                 4. Run training pipeline
                                    - run_full_pipeline.py
                                    - Auto-tracks to CSV
                                    - Training + evaluation
```

**Key Points:**
- ✅ Experiment tracking happens automatically on the server
- ✅ CSV file is created/updated in the container
- ✅ You rsync the CSV back to your local PC
- ✅ All parameters, metrics, and breakdowns are captured

---

## Prerequisites

### On Local PC:
- WSL or native Linux environment
- SSH access to remote server
- Git repository with experiment tracking feature
- rsync installed

### On Remote Server:
- Docker container running (usually named `llamafactory`)
- Port 8002 available (or another free port)
- SSH access configured
- Container has /app directory mounted

### Files You Need:
```
LLaMA-Factory/
├── experiment_tracker.py          # NEW - Tracking module
├── run_full_pipeline.py           # MODIFIED - Integrated tracking
├── building_dataset.py            # For dataset generation
├── evaluate_pipeline_simple.py    # For evaluation
└── experiments_log.csv            # AUTO-CREATED on first run
```

---

## Workflow Steps

### Step 1: Make Local Changes (Optional)

If you're modifying code, test locally first:

```bash
# On local PC/WSL
cd /mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory

# Test script syntax
python3 -m py_compile experiment_tracker.py
python3 -m py_compile run_full_pipeline.py

# Test imports
python3 -c "from experiment_tracker import ExperimentTracker; print('OK')"
```

### Step 2: Sync Code to Server

**Read DEPLOYMENT.md for server details**, then rsync:

```bash
# Example rsync command (adjust hostname and paths)
rsync -avz --progress \
  --exclude='data/' \
  --exclude='saves/' \
  --exclude='*.mp4' \
  --exclude='__pycache__/' \
  ./ user@server:/path/to/LLaMA-Factory/

# Or if using SSH alias from DEPLOYMENT.md:
rsync -avz --progress \
  --exclude='data/' \
  --exclude='saves/' \
  ./ seedoo@hetzner-gpu:/app/
```

**Important:** Make sure to sync:
- ✅ experiment_tracker.py
- ✅ run_full_pipeline.py
- ✅ building_dataset.py
- ✅ Any other modified scripts

### Step 3: SSH into Server

```bash
# SSH to server
ssh user@server

# Or using alias:
ssh seedoo@hetzner-gpu

# Enter Docker container
docker exec -it llamafactory bash

# Verify you're in /app
pwd  # Should show /app
ls   # Should see your Python scripts
```

### Step 4: Build Datasets

**IMPORTANT:** Always build train and test datasets separately!

```bash
# Inside Docker container (/app)

# Step 4a: Generate TRAINING dataset
# Use timestamp for unique naming
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATASET_NAME="subtle_gray_exp_${TIMESTAMP}"

python3 /app/building_dataset.py \
  --dataset_name ${DATASET_NAME}_train \
  --texture_type subtle_gray_stripes \
  --view_angle 0.0,15.0,30.0,45.0 \
  --background_gray 15,16,17 \
  --stripe_gray 20,21,22 \
  --speed_range 0.0,14.0 \
  --fps 4 \
  --duration 12.0

# Step 4b: Generate TEST dataset
python3 /app/building_dataset.py \
  --dataset_name ${DATASET_NAME}_test \
  --texture_type subtle_gray_stripes \
  --view_angle 0.0,15.0,30.0,45.0 \
  --background_gray 10,11,12 \
  --stripe_gray 15,16,17 \
  --speed_range 0.0,14.0 \
  --fps 4 \
  --duration 12.0

# Verify datasets were created
ls data/${DATASET_NAME}_train/
ls data/${DATASET_NAME}_test/
```

**What this does:**
- Creates all combinations: 4 angles × 3 bg_gray × 3 stripe_gray × 2 speeds = 72 videos each
- Train set: 72 videos with bg_gray 15-17, stripe_gray 20-22
- Test set: 72 videos with bg_gray 10-12, stripe_gray 15-17 (different values!)

### Step 5: Run Training + Evaluation with Tracking

```bash
# Inside Docker container (/app)

# Run full pipeline with experiment tracking
python3 /app/run_full_pipeline.py \
  --dataset_name ${DATASET_NAME} \
  --skip_dataset \
  --num_train_epochs 5 \
  --eval_video_fps 4 \
  --eval_video_maxlen 128 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --save_steps 100

# Monitor progress
# The script will:
# 1. Create/update experiments_log.csv
# 2. Log experiment with unique ID
# 3. Train the model
# 4. Evaluate both base and fine-tuned models
# 5. Update CSV with all results
```

**What happens automatically:**

| Stage | CSV Update | What Gets Logged |
|-------|------------|------------------|
| Start | Experiment row created | All parameters, configs, seeds |
| Dataset | Status → in_progress | (skipped in this case) |
| Training | Status → in_progress → completed | All LoRA hyperparameters |
| Evaluation | Status → in_progress → completed | All metrics + breakdowns |

**CSV Columns Filled:**
- experiment_id: Sequential (1, 2, 3, ...)
- All dataset params: texture_type, view_angle, background_gray, etc.
- All training params: lora_rank, learning_rate, epochs, etc.
- All results: accuracy, F1, precision, recall for both models
- Per-texture breakdown: JSON with each texture's metrics
- Per-angle breakdown: JSON with each angle's metrics

### Step 6: Retrieve Results

```bash
# Exit Docker container
exit

# From server, rsync results back to local PC
rsync -avz --progress \
  /path/to/LLaMA-Factory/experiments_log.csv \
  /path/to/LLaMA-Factory/evaluation_results_*/ \
  /path/to/LLaMA-Factory/saves/ \
  user@local-pc:/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/

# Or pull from local PC:
rsync -avz --progress \
  seedoo@hetzner-gpu:/app/experiments_log.csv \
  seedoo@hetzner-gpu:/app/evaluation_results_*/ \
  ./
```

**Files to retrieve:**
- ✅ `experiments_log.csv` - Complete experiment history
- ✅ `evaluation_results_<timestamp>/` - Detailed reports
- ✅ `saves/<model_name>/` - Trained LoRA adapters (optional)

---

## Example: Training with Custom Textures

Here's a complete example following your workflow pattern:

```bash
#############################
# 1. ON LOCAL PC - Prepare
#############################
cd /mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory
git checkout feature/experiment-tracking

# Verify experiment tracking is present
ls experiment_tracker.py  # Should exist

#############################
# 2. SYNC TO SERVER
#############################
rsync -avz --progress \
  --exclude='data/' \
  --exclude='saves/' \
  --exclude='*.mp4' \
  ./ seedoo@hetzner-gpu:/app/

#############################
# 3. SSH TO SERVER
#############################
ssh seedoo@hetzner-gpu
docker exec -it llamafactory bash

#############################
# 4. BUILD DATASETS
#############################
# Create unique dataset name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATASET="factory_textures_${TIMESTAMP}"

echo "Creating dataset: $DATASET"

# Build training dataset
python3 /app/building_dataset.py \
  --dataset_name ${DATASET}_train \
  --texture_type factory_dark,factory_dark_stripes,subtle_gray_stripes \
  --direction left,right,up,down \
  --view_angle 0.0,15.0,30.0,45.0 \
  --speed_range 0.0,12.0 \
  --fps 4 \
  --duration 12.0 \
  --background_gray 15,16,17 \
  --stripe_gray 20,21,22

# Build test dataset (different parameters!)
python3 /app/building_dataset.py \
  --dataset_name ${DATASET}_test \
  --texture_type factory_dark,factory_dark_stripes,subtle_gray_stripes \
  --direction left,right,up,down \
  --view_angle 0.0,22.5,52.0 \
  --speed_range 0.0,14.0 \
  --fps 4 \
  --duration 12.0 \
  --background_gray 10,11,12 \
  --stripe_gray 15,16,17

#############################
# 5. RUN TRAINING WITH TRACKING
#############################
python3 /app/run_full_pipeline.py \
  --dataset_name ${DATASET} \
  --skip_dataset \
  --num_train_epochs 5 \
  --eval_video_fps 4 \
  --eval_video_maxlen 128 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --save_steps 50 \
  --logging_steps 10

# Wait for completion...
# Check CSV was updated
tail -1 /app/experiments_log.csv

#############################
# 6. RETRIEVE RESULTS TO LOCAL PC
#############################
exit  # Exit container
exit  # Exit server

# On local PC:
rsync -avz --progress \
  seedoo@hetzner-gpu:/app/experiments_log.csv \
  seedoo@hetzner-gpu:/app/evaluation_results_*/ \
  ./

# Open CSV in Excel/LibreOffice
excel experiments_log.csv
# Or
libreoffice experiments_log.csv
```

---

## CSV Structure

**Total Columns: 86**

### Experiment Metadata (3 columns)
- experiment_id, timestamp, run_timestamp

### Dataset Configuration (28 columns)
- dataset_name, train/test names, num_videos, train_split, seed
- All video parameters: texture_type, direction, speed_range, resolution, fps, duration
- Camera/lighting: view_angle, brightness, contrast, lighting settings
- Edge/stripe parameters: edge_width, stripe_width, stripe_spacing, etc.

### Model & Training (30 columns)
- Model: model_name_or_path, template
- LoRA: output_dir, rank, alpha, dropout, cutoff_len
- Training: batch_size, grad_accumulation, learning_rate, epochs, scheduler, etc.
- Evaluation: eval settings, GPU memory, etc.
- Pipeline: skip flags, use_docker

### Status (3 columns)
- dataset_status, training_status, evaluation_status

### Results - Base Model (12 columns)
- Accuracy, F1, precision, recall
- Moving/stopped: correct & total counts
- Per-texture results (JSON)
- Per-angle results (JSON)

### Results - Fine-tuned Model (12 columns)
- Same as base model metrics

### Manual Notes (2 columns)
- notes_1, notes_2 (for your annotations)

**Example JSON in per_texture_results:**
```json
{
  "factory_dark": {"accuracy": 95.5, "f1_score": 94.2, "correct": 43, "total": 45},
  "factory_dark_stripes": {"accuracy": 88.9, "f1_score": 87.1, "correct": 40, "total": 45},
  "subtle_gray_stripes": {"accuracy": 82.2, "f1_score": 80.5, "correct": 37, "total": 45}
}
```

---

## Troubleshooting

### Issue: experiments_log.csv not created

**Cause:** Script failed before experiment tracking initialized

**Solution:**
```bash
# Check if experiment_tracker.py exists
ls /app/experiment_tracker.py

# Test import
docker exec llamafactory python3 -c "from experiment_tracker import ExperimentTracker; print('OK')"

# Run with verbose logging
docker exec llamafactory python3 /app/run_full_pipeline.py --help
```

### Issue: CSV exists but not updated

**Cause:** Pipeline failed during execution

**Solution:**
```bash
# Check the CSV - experiment should be marked as "failed"
docker exec llamafactory tail -1 /app/experiments_log.csv

# Check logs for errors
docker logs llamafactory --tail 100
```

### Issue: Can't sync CSV back to local PC

**Cause:** File permissions or rsync configuration

**Solution:**
```bash
# Check if file exists on server
ssh seedoo@hetzner-gpu ls -la /app/experiments_log.csv

# Fix permissions if needed
ssh seedoo@hetzner-gpu chmod 644 /app/experiments_log.csv

# Try rsync with verbose mode
rsync -avzP seedoo@hetzner-gpu:/app/experiments_log.csv ./
```

### Issue: Port 8002 busy

**Cause:** Another container or process using the port

**Solution:**
```bash
# Check what's using the port
docker ps | grep 8002
lsof -i :8002

# Use a different port or stop the conflicting container
# NEVER force-kill busy containers without checking first!
```

### Issue: Dataset not found

**Cause:** Dataset names don't match

**Solution:**
```bash
# List all datasets
ls /app/data/*_train.json
ls /app/data/*_test.json

# Verify dataset_info.json has your datasets
docker exec llamafactory python3 -c "
import json
with open('/app/data/dataset_info.json') as f:
    datasets = json.load(f)
    for name in datasets:
        if 'train' in name or 'test' in name:
            print(name)
"
```

---

## Summary Checklist

**Before each training run:**
- [ ] Code synced to server
- [ ] SSH into container
- [ ] Choose unique dataset name (use timestamp!)
- [ ] Build train dataset
- [ ] Build test dataset
- [ ] Verify datasets created
- [ ] Check port 8002 is free
- [ ] Run training with --skip_dataset

**After training completes:**
- [ ] Verify experiments_log.csv updated
- [ ] Check evaluation_results_<timestamp>/ exists
- [ ] Rsync CSV back to local PC
- [ ] Rsync evaluation results
- [ ] Open CSV in spreadsheet software
- [ ] Add notes in notes_1, notes_2 columns

---

## Quick Reference Commands

```bash
# Sync to server
rsync -avz --exclude='data/' --exclude='saves/' ./ seedoo@hetzner-gpu:/app/

# Enter container
docker exec -it llamafactory bash

# Build datasets (inside container)
python3 /app/building_dataset.py --dataset_name NAME_train [args]
python3 /app/building_dataset.py --dataset_name NAME_test [args]

# Run training (inside container)
python3 /app/run_full_pipeline.py --dataset_name NAME --skip_dataset [args]

# Retrieve results (from local PC)
rsync -avz seedoo@hetzner-gpu:/app/experiments_log.csv ./
rsync -avz seedoo@hetzner-gpu:/app/evaluation_results_*/ ./
```

---

**End of Workflow Guide**
