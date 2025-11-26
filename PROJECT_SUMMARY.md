# Treadmill Motion Detection Project - Complete Summary

## Project Overview
Vision model training for detecting moving vs. stationary treadmill surfaces using Qwen2.5-VL with LoRA fine-tuning.

---

## Phase 1: Initial Setup & Environment

### Docker Environment Setup
- **Container**: `llamafactory:fixed-v1`
- **Base**: LLaMA-Factory framework with CUDA 12.4
- **Ports**: 7860 (WebUI), 8000 (API)
- **Key Fix**: Resolved dependency conflicts (numpy, torch versions)
- **Status**: ✅ Running and tested

### Repository Structure
```
LLaMA-Factory/
├── data/
│   ├── treadmill_videos/          # Training video data
│   └── synthetic_treadmill/        # Synthetic data generator
├── examples/train_qlora/
│   └── qwen25vl_lora_sft.yaml     # LoRA training config
├── saves/                          # Model checkpoints
├── output/                         # Training outputs
├── scripts/
│   └── vllm_infer.py              # Inference script
└── synthetic_treadmill/            # Data generation tool
```

---

## Phase 2: LoRA Training for Treadmill Detection

### Training Configuration
**File**: `examples/train_qlora/qwen25vl_lora_sft.yaml`

#### Model Setup
- **Base Model**: Qwen2.5-VL (Vision-Language Model)
- **Method**: LoRA (Low-Rank Adaptation)
- **LoRA Rank**: 8
- **LoRA Alpha**: 16
- **Target Modules**: All attention layers

#### Training Parameters
- **Batch Size**: 1 (per device)
- **Gradient Accumulation**: 8 steps
- **Learning Rate**: 1e-4
- **Epochs**: 3
- **Max Samples**: 500
- **Val Split**: 0.1
- **Max Length**: 8192 tokens
- **Video Frames**: 8 frames per video

#### Hardware & Optimization
- **GPU**: CUDA-enabled
- **Precision**: bf16 (mixed precision)
- **Flash Attention**: Enabled (FA2)
- **Gradient Checkpointing**: Enabled
- **Upcast Attention**: Enabled for stability

### Dataset Format
**File**: `data/dataset_info.json`
```json
{
  "treadmill_motion": {
    "file_name": "treadmill_motion.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content"
    }
  }
}
```

### Training Task
- **Input**: Video clips of treadmill surface
- **Output**: Binary classification
  - "moving" - treadmill is in motion
  - "stationary" - treadmill is stopped
- **Goal**: Fine-tune vision model to detect treadmill motion

### Evaluation Scripts
1. **evaluate_treadmill_lora.py** - Standard evaluation
2. **evaluate_bootstrap.py** - Bootstrap confidence intervals

---

## Phase 3: Synthetic Data Generation System

### Problem Statement
Need diverse training data with full control over visual parameters for robust treadmill motion detection.

### Solution: Synthetic Video Generator
**Location**: `synthetic_treadmill/`

#### Core Features

**1. Texture Types (5 options)**
- Stripes: Conveyor belt segments
- Noise: Rough rubber surface
- Rubber: Textured with bumps
- Grid: Tiled pattern
- Diamond Plate: Metal tread plate

**2. Motion Control**
- Directions: left, right, up, down
- Variable speed: 0.5 to 20+ px/frame
- Seamless infinite scrolling

**3. Camera Effects**
- View angle: -45° to +45° (perspective transform)
- Brightness: -1.0 to 1.0
- Contrast: 0.5 to 2.0
- Lighting variations: vignette, gradients, spotlight
- Motion blur: 0-10 pixels
- Camera noise: 0.0 to 1.0

**4. Control Parameters**
- Total: 16 degrees of freedom
- Reproducible: seed-based generation
- Output: MP4 videos with parameter-encoded filenames

#### Critical Design Fix - Stripe Orientation
**Problem**: Stripes parallel to motion are invisible
**Solution**: Auto-orient stripes perpendicular to motion
- Left/Right motion → Vertical stripes
- Up/Down motion → Horizontal stripes
- Ensures optical flow is always detectable

#### Technical Architecture
```python
TreadmillTextureGenerator    # 5 texture types
    ↓
TreadmillMotionSimulator     # Seamless scrolling
    ↓
CameraEffectsProcessor       # Lighting, perspective, blur
    ↓
SyntheticVideoGenerator      # MP4 output
```

#### Key Files
- `synthetic_data_generation.py` - Main script (1,630 lines)
- `README.md` - Complete documentation
- `QUICK_START.md` - Quick reference
- `CHANGELOG.md` - Version history
- `requirements.txt` - Dependencies

#### Usage Example
```bash
# Generate diverse dataset
python synthetic_data_generation.py \
  --num_videos 100 \
  --vary_parameters \
  --seed 42 \
  --output_dir ./dataset
```

#### Docker Integration
```bash
# Install dependencies
docker exec llamafactory pip install opencv-python-headless "numpy<2.0.0"

# Generate videos
docker exec llamafactory python3 /app/data/synthetic_treadmill/synthetic_data_generation.py \
  --num_videos 20 \
  --vary_parameters \
  --output_dir /app/data/synthetic_treadmill/output
```

---

## Phase 4: Complete Workflow

### Training Pipeline
1. **Prepare Data**
   - Collect real treadmill videos OR
   - Generate synthetic videos with varied parameters

2. **Format Dataset**
   - Create `treadmill_motion.json` in ShareGPT format
   - Each entry: video path + "moving"/"stationary" label

3. **Train LoRA Adapter**
   ```bash
   llamafactory-cli train examples/train_qlora/qwen25vl_lora_sft.yaml
   ```

4. **Evaluate Model**
   ```bash
   python evaluate_treadmill_lora.py
   python evaluate_bootstrap.py  # With confidence intervals
   ```

5. **Deploy**
   - Use `scripts/vllm_infer.py` for inference
   - Serve on port 8000

### Data Augmentation Strategy
**Synthetic Data Advantages:**
- ✅ Unlimited diverse examples
- ✅ Perfect ground truth labels
- ✅ Control over difficulty (speed, lighting, angle)
- ✅ No manual annotation needed
- ✅ Reproducible experiments

**Recommended Mix:**
- 70% synthetic (diverse conditions)
- 30% real (domain adaptation)

---

## Git History

### Commits
1. **494115f** - "Add Docker setup with fixed dependencies for LoRA training"
2. **45c46a0** - "Add LoRA training setup for treadmill motion detection"
3. **92efa8c** - "Add synthetic treadmill video dataset generator"
4. **bbe09f0** - "Fix stripe orientation for visible motion detection"

### Branch
- **Current**: `treadmill-detection`
- **Main**: `main`

---

## Key Technical Decisions

### 1. Why LoRA?
- Memory efficient (train 1% of parameters)
- Fast training (minutes vs hours)
- Easy to swap adapters
- Preserves base model knowledge

### 2. Why Qwen2.5-VL?
- State-of-art vision-language model
- Native video understanding
- Supports multi-frame input
- Strong instruction following

### 3. Why Synthetic Data?
- Real treadmill videos are limited
- Hard to get diverse lighting/angles
- Synthetic allows systematic testing
- Controllable difficulty progression

### 4. Why Stripe Orientation Fix?
- Parallel stripes create no optical flow
- CV models cannot learn from invisible motion
- Perpendicular orientation ensures detectable features

---

## Dependencies

### Core Requirements
```
transformers>=4.40.0
torch>=2.1.0
opencv-python-headless>=4.8.0
numpy>=1.24.0,<2.0.0
llamafactory
vllm
```

### Installation
```bash
pip install -r requirements.txt
pip install -r synthetic_treadmill/requirements.txt
```

---

## Performance Metrics

### Synthetic Generation Speed
- 640x480 @ 30fps, 5s: ~0.5-1.0 seconds
- 1280x720 @ 30fps, 5s: ~2-3 seconds
- 1920x1080 @ 60fps, 10s: ~10-15 seconds

### Training Time (Estimated)
- 100 videos, 3 epochs: ~30-60 minutes
- Depends on: GPU, video length, batch size

---

## Future Enhancements

### Synthetic Data
1. 3D rendering for true perspective
2. Object placement on belt
3. Defect simulation (scratches, stains)
4. Shadow casting
5. Real texture photos as base

### Model Training
1. Multi-speed classification (slow/medium/fast)
2. Speed regression (exact px/frame)
3. Direction detection (left/right/up/down)
4. Anomaly detection (unusual patterns)

### Deployment
1. Real-time inference pipeline
2. Edge deployment (TensorRT)
3. Multi-camera support
4. Confidence thresholds

---

## Project Status: ✅ Complete

### Deliverables
- ✅ Docker environment configured
- ✅ LoRA training pipeline setup
- ✅ Synthetic data generator (16 DOF)
- ✅ Stripe orientation fix implemented
- ✅ Complete documentation
- ✅ Git version control
- ✅ Tested in Docker container

### Next Steps
1. Generate large synthetic dataset (100-1000 videos)
2. Train LoRA adapter on synthetic data
3. Evaluate on real treadmill videos
4. Fine-tune with real data if needed
5. Deploy inference pipeline

---

**Project Duration**: Session of 2025-11-26
**Total Commits**: 4
**Total Files Created**: 10+
**Lines of Code**: 2,000+

🎯 **Ready for Training!**
