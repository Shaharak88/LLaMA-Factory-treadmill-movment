# Treadmill Motion Detection Pipeline - Usage Guide

Complete guide for training and evaluating vision models to detect treadmill belt motion.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Pipeline Overview](#pipeline-overview)
3. [Scripts Reference](#scripts-reference)
4. [Parameter Modes](#parameter-modes)
5. [Common Workflows](#common-workflows)
6. [Video Generation Parameters](#video-generation-parameters)
7. [Training Parameters](#training-parameters)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Enter Docker Container
```bash
docker exec -it llamafactory bash
```

### Run Complete Pipeline (Simple)
```bash
python3 /app/run_full_pipeline.py \
  --dataset_name my_experiment \
  --num_videos 100 \
  --num_train_epochs 1
```

This creates 100 videos (90% train, 10% test), trains a LoRA adapter, and evaluates both base and fine-tuned models.

---

## Pipeline Overview

### Three Main Scripts

**1. `run_full_pipeline.py`** - Master orchestrator (recommended)
- Runs all three steps: dataset generation → training → evaluation
- Handles data splitting (90% train, 10% test)
- All-in-one solution

**2. `building_dataset.py`** - Dataset generation only
- Creates synthetic treadmill videos
- Generates dataset JSON for LLaMA-Factory
- Use when you only need videos

**3. `evaluate_pipeline.py`** - Evaluation only
- Tests base and fine-tuned models
- Compares accuracy on held-out test set
- Use after training completes

---

## Scripts Reference

### run_full_pipeline.py

**Purpose**: End-to-end pipeline orchestration

**Basic Usage**:
```bash
python3 /app/run_full_pipeline.py \
  --dataset_name <name> \
  [dataset options] \
  [training options] \
  [evaluation options]
```

**Key Arguments**:
- `--dataset_name` (required): Base name for datasets (creates `<name>_train` and `<name>_test`)
- `--skip_dataset`: Skip video generation (use existing dataset)
- `--skip_training`: Skip training (only generate videos)
- `--skip_evaluation`: Skip evaluation (only generate and train)
- `--train_split`: Train/test split ratio (default: 0.9 = 90% train, 10% test)

**Output**:
- Datasets: `data/<name>_train/` and `data/<name>_test/`
- Dataset JSONs: `data/<name>_train.json` and `data/<name>_test.json`
- Model: `saves/qwen2vl-treadmill-lora-pipeline/`
- Evaluation: `evaluation_results_<timestamp>/`

---

### building_dataset.py

**Purpose**: Generate synthetic treadmill videos and dataset JSON

**Basic Usage**:
```bash
python3 /app/building_dataset.py \
  --dataset_name <name> \
  --num_videos <count> \
  [video parameters]
```

**Key Arguments**:
- `--dataset_name` (required): Dataset name
- `--num_videos`: Total videos to generate (default: 10)
- `--moving_ratio`: Ratio of moving videos (default: 0.5 = 50/50 split)
- `--vary_parameters`: Randomize all parameters for diversity
- `--seed`: Random seed for reproducibility (default: 42)

**Output**:
- Videos: `data/<name>/`
- Dataset JSON: `data/<name>.json`
- Updates: `data/dataset_info.json`

---

### evaluate_pipeline.py

**Purpose**: Evaluate models on test dataset

**Basic Usage**:
```bash
python3 /app/evaluate_pipeline.py \
  --test_dataset <dataset_name> \
  --adapter_name_or_path <lora_dir> \
  [inference options]
```

**Key Arguments**:
- `--test_dataset` (required): Test dataset name (must exist in `dataset_info.json`)
- `--adapter_name_or_path`: Path to LoRA adapter (optional, compares base vs fine-tuned)
- `--model_name_or_path`: Base model (default: Qwen/Qwen2.5-VL-3B-Instruct)
- `--video_fps`: FPS for video processing (default: 2.0)
- `--video_maxlen`: Max frames per video (default: 128)

**Output**:
- Results: `evaluation_results/`
- Metrics: Accuracy, predictions, confusion analysis

---

## Parameter Modes

### Mode 1: Simple Mode (Default)

Uses `--num_videos` to generate fixed count with **varied speeds only**.

**Example**:
```bash
--num_videos 10 --texture_type stripes --direction right
```

**Result**: 10 videos (5 moving, 5 stopped)
- All have same texture (stripes) and direction (right)
- Moving videos have random speeds from `--speed_range`
- Stopped videos have speed 0

---

### Mode 2: Combination Mode

Use **comma-separated values** to generate **all combinations**.

**Example 1**: 2 textures × 2 angles
```bash
--texture_type stripes,noise --view_angle 0.0,30.0 --speed_range 0,5.0
```

**Result**: 8 videos (2×2×2)
- `--num_videos` is **ignored** in this mode
- Creates every combination:
  - stripes, 0°, speed 0
  - stripes, 0°, speed 5.0
  - stripes, 30°, speed 0
  - stripes, 30°, speed 5.0
  - noise, 0°, speed 0
  - noise, 0°, speed 5.0
  - noise, 30°, speed 0
  - noise, 30°, speed 5.0

**Example 2**: Complex combinations
```bash
--texture_type stripes,noise,rubber --direction left,right --view_angle 0.0,15.0 --speed_range 0,6.0
```

**Result**: 3×2×2×2 = **24 videos**

**Rule**: Total videos = multiply all option counts

---

### Mode 3: Varied Mode

Use `--vary_parameters` to randomize everything.

**Example**:
```bash
--num_videos 100 --vary_parameters --seed 42
```

**Result**: 100 videos (50 moving, 50 stopped)
- Randomizes: texture, direction, angle, brightness, contrast, lighting, noise
- Speed varies within `--speed_range`
- Reproducible with same seed

---

## Common Workflows

### 1. Quick Test (Small Dataset)
```bash
python3 /app/run_full_pipeline.py \
  --dataset_name quick_test \
  --num_videos 20 \
  --num_train_epochs 1 \
  --skip_evaluation
```

**Creates**: 20 videos → trains for 1 epoch → skips evaluation

---

### 2. Diverse Training Dataset
```bash
python3 /app/run_full_pipeline.py \
  --dataset_name diverse_treadmill \
  --num_videos 200 \
  --vary_parameters \
  --seed 42 \
  --num_train_epochs 3
```

**Creates**: 200 varied videos → trains for 3 epochs → evaluates

---

### 3. Specific Combinations (High Quality)
```bash
python3 /app/run_full_pipeline.py \
  --dataset_name high_quality_combo \
  --texture_type stripes,noise,factory_dark_stripes,factory_dark \
  --direction left,right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 0,6.0 \
  --fps 4 \
  --duration 12.0 \
  --num_train_epochs 1 \
  --eval_video_fps 4 \
  --eval_video_maxlen 12
```

**Creates**: 4×2×3×2 = 48 videos (12s each at 4 FPS) → trains → evaluates

---

### 4. Only Generate Videos (No Training)
```bash
python3 /app/building_dataset.py \
  --dataset_name video_only \
  --texture_type stripes,noise,rubber,grid \
  --speed_range 2.0,5.0,8.0 \
  --fps 30 \
  --duration 5.0
```

**Creates**: 4×3 = 12 videos only

---

### 5. Resume Training with Existing Dataset
```bash
python3 /app/run_full_pipeline.py \
  --dataset_name existing_exp \
  --skip_dataset \
  --num_train_epochs 5
```

**Uses**: Existing `data/existing_exp_train/` and `data/existing_exp_test/` → trains for 5 epochs

---

### 6. Evaluate Only
```bash
python3 /app/evaluate_pipeline.py \
  --test_dataset my_test \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline \
  --video_fps 2.0 \
  --video_maxlen 128
```

**Evaluates**: Both base and fine-tuned models on `data/my_test/`

---

## Video Generation Parameters

### Texture Types
Available: `stripes`, `noise`, `rubber`, `grid`, `diamond_plate`, `factory_dark`, `factory_dark_stripes`

**Example**: `--texture_type stripes,noise,rubber`

---

### Motion Direction
Available: `left`, `right`, `up`, `down`

**Example**: `--direction left,right`

---

### Speed Range
- Format: `min,max` for range OR `speed1,speed2,speed3` for specific values
- Use `0` for stopped videos

**Examples**:
- `--speed_range 1.0,8.0` (range: random between 1-8)
- `--speed_range 0,3.0,6.0` (exact: 0, 3.0, 6.0)

---

### Video Settings
- `--fps`: Frames per second (default: 30)
- `--duration`: Video length in seconds (default: 5.0)
- `--resolution`: Video size (default: 640x480)

**Example**: `--fps 4 --duration 12.0` (12 seconds at 4 FPS = 48 frames)

---

### Camera & Lighting
- `--view_angle`: Camera angle in degrees (default: 0.0, range: -30 to 30)
- `--brightness`: Brightness adjustment (default: 0.0, range: -0.2 to 0.2)
- `--contrast`: Contrast adjustment (default: 1.0, range: 0.7 to 1.3)
- `--lighting_variation`: Lighting type (`none`, `vignette`, `gradient_lr`, `gradient_tb`, `spotlight`)
- `--lighting_intensity`: Lighting strength (default: 0.5, range: 0.3 to 0.8)

---

### Advanced Effects
- `--motion_blur`: Motion blur amount (default: 0, range: 0-3)
- `--camera_noise`: Noise level (default: 0.0, range: 0.0 to 0.3)
- `--edge_width`: Belt edge width percentage (default: 0.1, range: 0.05 to 0.15)

---

## Training Parameters

### LoRA Configuration
- `--lora_rank`: LoRA rank (default: 8, higher = more parameters)
- `--lora_alpha`: LoRA alpha (default: 16, scaling factor)
- `--lora_dropout`: Dropout rate (default: 0.05)

---

### Training Hyperparameters
- `--num_train_epochs`: Training epochs (default: 1)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--per_device_train_batch_size`: Batch size per device (default: 1)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 8)
- `--lr_scheduler_type`: Scheduler type (default: cosine)
- `--warmup_ratio`: Warmup ratio (default: 0.1)

---

### Precision & Quantization
- `--fp16`: Use FP16 precision (default: enabled)
- `--bf16`: Use BF16 precision (default: disabled)
- `--quantization_bit`: Quantization bits (default: 4)

---

### Evaluation Settings
- `--eval_video_fps`: FPS for evaluation (default: 2.0)
- `--eval_video_maxlen`: Max frames for evaluation (default: 128)
- `--eval_max_new_tokens`: Max tokens to generate (default: 128)
- `--eval_batch_size`: Batch size for inference (default: 1024)

---

## Troubleshooting

### Issue: "No videos were successfully generated"
**Cause**: Missing dependencies (opencv-python)

**Solution**:
```bash
docker exec llamafactory pip install "opencv-python-headless<4.10" "numpy<2.0.0"
```

---

### Issue: "Dataset not found in dataset_info.json"
**Cause**: Dataset JSON not registered

**Solution**: Run `building_dataset.py` first or check `data/dataset_info.json`

---

### Issue: Videos all look the same
**Cause**: Using simple mode without variation

**Solution**: Use combination mode or `--vary_parameters`:
```bash
--texture_type stripes,noise,rubber --speed_range 0,3.0,6.0
# OR
--vary_parameters --num_videos 100
```

---

### Issue: Out of memory during training
**Cause**: Batch size too large

**Solution**: Reduce batch size or increase gradient accumulation:
```bash
--per_device_train_batch_size 1 --gradient_accumulation_steps 16
```

---

### Issue: Training too slow
**Cause**: Too many videos or epochs

**Solution**: Reduce dataset size or epochs:
```bash
--num_videos 50 --num_train_epochs 1
```

---

## Exit Container
```bash
exit
```

---

**For more details, run any script with `--help`**
