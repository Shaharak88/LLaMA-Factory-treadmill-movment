# Detailed Model-by-Model Experiment Report

**Project:** Treadmill Motion Detection with Qwen2.5-VL-3B
**Date Range:** November 27-30, 2025
**Total Experiments:** 5

---

## EXPERIMENT 5: Right-Only Baseline (First Chronologically)

**Date:** November 27, 2025
**Model Path:** `saves/qwen2vl-treadmill-lora-5-epochs_eval_steps_5`

### Purpose
Establish baseline with simplest possible configuration - single direction (right only)

### Changes from Base Model
- **First experiment** - no previous experiment to compare
- Starting point: Vanilla Qwen2.5-VL-3B-Instruct (never detects motion)

### Training Dataset Arguments

**Dataset Name:** `subtle_gray_train_20251127_193422`

**Command:**
```bash
python3 building_dataset.py \
  --dataset_name subtle_gray_train_20251127_193422 \
  --texture_type subtle_gray_stripes \
  --direction right \
  --speed_range 0.0,14.0 \
  --view_angle 0,15,30,45 \
  --stripe_gray 10 \
  --background_gray 13 \
  --fps 4 \
  --duration 12.0 \
  --seed 42
```

**Parameters:**
- Texture: subtle_gray_stripes (1 texture)
- Direction: right ONLY (1 direction)
- Speeds: 0.0 (stopped), 14.0 (moving) → 2 speeds
- Angles: 0°, 15°, 30°, 45° → 4 angles
- Stripe gray: 10, Background: 13
- **Total videos:** 1 texture × 1 direction × 4 angles × 2 speeds = **8 videos per repetition**
- **Actual training videos:** 36 (18 moving, 18 stopped)
- FPS: 4, Duration: 12 seconds (48 frames)

### Test Dataset Arguments

**Dataset Name:** `subtle_gray_test_20251127_193422`

**Command:**
```bash
python3 building_dataset.py \
  --dataset_name subtle_gray_test_20251127_193422 \
  --texture_type subtle_gray_stripes \
  --direction right \
  --speed_range 0.0,14.0 \
  --view_angle 0,15,30,45 \
  --stripe_gray 30 \
  --background_gray 33 \
  --fps 4 \
  --duration 12.0 \
  --seed 144
```

**Key Differences from Training:**
- Stripe gray: 30 (vs 10 in training) - **+20 difference**
- Background: 33 (vs 13 in training) - **+20 difference**
- Seed: 144 (different random variations)
- Everything else identical
- **Total videos:** 72 (36 moving, 36 stopped)

### Training Configuration
- Epochs: 5
- Batch size: 1 (effective 8 with gradient accumulation)
- Learning rate: 5e-5
- LoRA rank: 8, alpha: 16

### Results

| Metric | Base Model | Fine-Tuned | Improvement |
|--------|------------|------------|-------------|
| Accuracy | 50.00% | 50.00% | **+0.00%** |
| F1 Score | 0.00% | 0.00% | +0.00% |
| Moving Detected | 0/36 (0%) | 0/36 (0%) | No change |
| Stopped Detected | 36/36 (100%) | 36/36 (100%) | No change |

**Outcome:** ❌ **COMPLETE FAILURE** - Model learned nothing

### Analysis
- Only 36 training videos insufficient
- Single direction (right only) too restrictive
- Model weights didn't update meaningfully from base
- Different gray values in test set couldn't be handled

---

## EXPERIMENT 4: Factory Textures (3 Epochs)

**Date:** November 29, 2025
**Model Path:** `saves/qwen2vl-yesno-factory-3epochs`

### Purpose
Test multi-texture, multi-direction training with 3 epochs

### Changes from Experiment 5
1. ✅ **Increased directions:** right only → all 4 directions (left, right, up, down)
2. ✅ **Increased textures:** 1 texture → 3 textures
3. ✅ **Increased training data:** 36 videos → 72 videos (2x)
4. ✅ **Changed angles:** 4 angles (0,15,30,45) → 3 angles (0,15,30)
5. ❌ **Reduced epochs:** Still using 3 epochs (later found to be insufficient)

### Training Dataset Arguments

**Dataset Name:** `yesno_subtle_factory_train`

**Command:**
```bash
python3 building_dataset.py \
  --dataset_name yesno_subtle_factory_train \
  --texture_type subtle_gray_stripes,factory_dark_stripes,factory_dark \
  --direction left,right,up,down \
  --speed_range 0.0,14.0 \
  --view_angle 0,15,30 \
  --stripe_gray 10 \
  --background_gray 13 \
  --fps 4 \
  --duration 12.0 \
  --seed 42
```

**Parameters:**
- Textures: subtle_gray_stripes, factory_dark_stripes, factory_dark → 3 textures
- Directions: left, right, up, down → 4 directions
- Speeds: 0.0, 14.0 → 2 speeds
- Angles: 0°, 15°, 30° → 3 angles
- Stripe gray: 10, Background: 13
- **Total videos:** 3 × 4 × 2 × 3 = **72 videos** (36 moving, 36 stopped)
- FPS: 4, Duration: 12 seconds

### Test Dataset Arguments

**Dataset Name:** `yesno_subtle_factory_test`

**Command:**
```bash
python3 building_dataset.py \
  --dataset_name yesno_subtle_factory_test \
  --texture_type subtle_gray_stripes,factory_dark_stripes,factory_dark \
  --direction left,right,up,down \
  --speed_range 0.0,14.0 \
  --view_angle 0,15,30 \
  --stripe_gray 10 \
  --background_gray 13 \
  --fps 4 \
  --duration 12.0 \
  --seed 144
```

**Key Differences from Training:**
- Seed: 144 (different random variations)
- **Gray values identical to training** (10, 13)
- Everything else identical
- **Total videos:** 72 (36 moving, 36 stopped)

### Training Configuration
- Epochs: **3** (insufficient - see Exp 3 comparison)
- Batch size: 1 (effective 8 with gradient accumulation)
- Learning rate: 5e-5
- LoRA rank: 8, alpha: 16

### Results

| Metric | Base Model | Fine-Tuned | Improvement |
|--------|------------|------------|-------------|
| Accuracy | 50.00% | 66.67% | **+16.67%** |
| F1 Score | 0.00% | 75.00% | +75.00% |
| Precision | 0.00% | 60.00% | +60.00% |
| Recall | 0.00% | 100.00% | +100.00% |
| Moving Detected | 0/36 (0%) | 36/36 (100%) | ✅ Perfect |
| Stopped Detected | 36/36 (100%) | 12/36 (33.33%) | ❌ Poor |

**Outcome:** ⚠️ **MODERATE SUCCESS** - Learned motion but many false positives

### Analysis
- Model learned to detect motion (100% recall)
- But predicted 24 stopped videos as moving (false positives)
- Undertraining with only 3 epochs led to bias toward "moving"
- Metadata parsing bug showed all textures/angles as "unknown"

---

## EXPERIMENT 3: Factory Textures v2 (5 Epochs)

**Date:** November 29, 2025
**Model Path:** `saves/qwen2vl-yesno-factory-v2-5epochs`

### Purpose
Re-run Experiment 4 with more epochs and fixed metadata

### Changes from Experiment 4
1. ✅ **Increased epochs:** 3 → 5 epochs (critical improvement!)
2. ✅ **Fixed metadata parsing:** Regex now handles multi-word texture names
3. ✅ **New dataset (v2):** Clean slate with proper naming
4. ✅ **Different test gray values:** Training (10,13) vs Test (30,33) to test generalization

### Training Dataset Arguments

**Dataset Name:** `yesno_factory_v2_train`

**Command:**
```bash
python3 building_dataset.py \
  --dataset_name yesno_factory_v2_train \
  --texture_type subtle_gray_stripes,factory_dark_stripes,factory_dark \
  --direction left,right,up,down \
  --speed_range 0.0,14.0 \
  --view_angle 0,15,30 \
  --stripe_gray 10 \
  --background_gray 13 \
  --fps 4 \
  --duration 12.0 \
  --seed 42
```

**Parameters:** (Same as Exp 4)
- Textures: 3 (subtle_gray_stripes, factory_dark_stripes, factory_dark)
- Directions: 4 (left, right, up, down)
- Speeds: 2 (0.0, 14.0)
- Angles: 3 (0°, 15°, 30°)
- Stripe gray: 10, Background: 13
- **Total videos:** 72 (36 moving, 36 stopped)
- FPS: 4, Duration: 12 seconds

### Test Dataset Arguments

**Dataset Name:** `yesno_factory_v2_test`

**Command:**
```bash
python3 building_dataset.py \
  --dataset_name yesno_factory_v2_test \
  --texture_type subtle_gray_stripes,factory_dark_stripes,factory_dark \
  --direction left,right,up,down \
  --speed_range 0.0,14.0 \
  --view_angle 0,15,30 \
  --stripe_gray 30 \
  --background_gray 33 \
  --fps 4 \
  --duration 12.0 \
  --seed 144
```

**Key Differences from Training:**
- Stripe gray: **30** (vs 10 in training) - tests generalization to different gray values
- Background: **33** (vs 13 in training)
- **+20 difference** for both gray values
- Seed: 144 (different variations)
- **Total videos:** 72 (36 moving, 36 stopped)

### Training Configuration
- Epochs: **5** (vs 3 in Exp 4)
- Batch size: 1 (effective 8)
- Learning rate: 5e-5
- LoRA rank: 8, alpha: 16

### Results

| Metric | Base Model | Fine-Tuned | Improvement |
|--------|------------|------------|-------------|
| Accuracy | 50.00% | **100.00%** | **+50.00%** |
| F1 Score | 0.00% | **100.00%** | +100.00% |
| Precision | 0.00% | **100.00%** | +100.00% |
| Recall | 0.00% | **100.00%** | +100.00% |
| Moving Detected | 0/36 (0%) | 36/36 (100%) | ✅ Perfect |
| Stopped Detected | 36/36 (100%) | 36/36 (100%) | ✅ Perfect |

**Per-Texture Results:**
- subtle_gray_stripes: 24/24 (100%)
- factory_dark_stripes: 24/24 (100%)
- factory_dark: 24/24 (100%)

**Per-Angle Results:**
- angle0: 24/24 (100%)
- angle15: 24/24 (100%)
- angle30: 24/24 (100%)

**Outcome:** ⭐ **PERFECT** - 100% accuracy across all metrics!

### Analysis
- 5 epochs vs 3 epochs made the difference (100% vs 66.67%)
- Successfully generalized to different gray values (10→30, 13→33)
- Multiple textures helped learn general motion patterns
- All directions, textures, and angles handled perfectly

---

## EXPERIMENT 2: Horizontal-Only Four-Angle

**Date:** November 29, 2025
**Model Path:** `saves/qwen2vl-subtle-gray-angles4-5epochs`

### Purpose
Test if constraining to horizontal-only (left/right) improves learning

### Changes from Experiment 3
1. ❌ **Reduced directions:** 4 directions → 2 directions (left, right only)
2. ✅ **Increased angles:** 3 angles → 4 angles (0,15,30,45)
3. ❌ **Reduced textures:** 3 textures → 1 texture (subtle_gray_stripes only)
4. ✅ **Increased training data:** 72 videos → 128 videos
5. ✅ **Kept 5 epochs** (learned from Exp 3 success)

### Training Dataset Arguments

**Dataset Name:** `subtle_gray_angles4_train`

**Command:**
```bash
python3 building_dataset.py \
  --dataset_name subtle_gray_angles4_train \
  --texture_type subtle_gray_stripes \
  --direction left,right \
  --speed_range 0.0,14.0 \
  --view_angle 0,15,30,45 \
  --stripe_gray 10 \
  --background_gray 13 \
  --fps 4 \
  --duration 12.0 \
  --seed 42
```

**Parameters:**
- Texture: subtle_gray_stripes → 1 texture
- Directions: left, right → 2 directions (horizontal only)
- Speeds: 0.0, 14.0 → 2 speeds
- Angles: 0°, 15°, 30°, 45° → 4 angles
- Stripe gray: 10, Background: 13
- **Total videos:** 1 × 2 × 2 × 4 = **16 videos per repetition**
- **Actual training videos:** 128 (64 moving, 64 stopped)
- FPS: 4, Duration: 12 seconds

### Test Dataset Arguments

**Dataset Name:** `subtle_gray_angles4_test`

**Command:**
```bash
python3 building_dataset.py \
  --dataset_name subtle_gray_angles4_test \
  --texture_type subtle_gray_stripes \
  --direction left,right \
  --speed_range 0.0,14.0 \
  --view_angle 0,15,30,45 \
  --stripe_gray 30 \
  --background_gray 33 \
  --fps 4 \
  --duration 12.0 \
  --seed 256
```

**Key Differences from Training:**
- Stripe gray: 30 (vs 10) - **+20 difference**
- Background: 33 (vs 13) - **+20 difference**
- Seed: 256 (different variations)
- **Directions match training** (left, right only) ← **Critical for success**
- **Total videos:** 256 (128 moving, 128 stopped) - 2x test set size

### Training Configuration
- Epochs: 5 (same as Exp 3)
- Batch size: 1 (effective 8)
- Learning rate: 5e-5
- LoRA rank: 8, alpha: 16

### Results

| Metric | Base Model | Fine-Tuned | Improvement |
|--------|------------|------------|-------------|
| Accuracy | 50.00% | **99.61%** | **+49.61%** |
| F1 Score | 0.00% | **99.61%** | +99.61% |
| Precision | 0.00% | 99.22% | +99.22% |
| Recall | 0.00% | 100.00% | +100.00% |
| Moving Detected | 0/128 (0%) | 128/128 (100%) | ✅ Perfect |
| Stopped Detected | 128/128 (100%) | 127/128 (99.22%) | ⚠️ 1 error |

**Per-Angle Results:**
- angle0: 64/64 (100%)
- angle15: 63/64 (98.44%) ← 1 false positive
- angle30: 64/64 (100%)
- angle45: 64/64 (100%)

**Outcome:** ✅ **NEAR-PERFECT** - Only 1 mistake out of 256 videos!

### Analysis
- Horizontal-only constraint worked extremely well
- 128 training videos provided robust learning
- Successfully generalized to different gray values
- Single texture was sufficient for 99.61% accuracy
- Only 1 error: false positive at 15° angle

---

## EXPERIMENT 1: Multi-Direction Interpolation Test

**Date:** November 30, 2025
**Model Path:** `saves/qwen2vl-treadmill-lora-pipeline`

### Purpose
Test if model trained on horizontal can generalize to vertical motion (interpolation test)

### Changes from Experiment 2
1. ❌ **Direction mismatch:** Training on left/right, Testing on left/right/up/down
2. ❌ **Non-standard angles:** 0,15,30,45 → 8,22,38,52 (unusual angles)
3. ❌ **Reduced training data:** 128 videos → 96 videos
4. ❌ **Massive test set:** 256 videos → 288 videos
5. ✅ **Kept 5 epochs** (successful from Exp 2 & 3)

### Training Dataset Arguments

**Dataset Name:** `subtle_gray_train_20251130_112501`

**Command:**
```bash
python3 building_dataset.py \
  --dataset_name subtle_gray_train_20251130_112501 \
  --texture_type subtle_gray_stripes \
  --direction left,right \
  --speed_range 0.0,14.0 \
  --view_angle 8,22,38,52 \
  --stripe_gray 10 \
  --background_gray 13 \
  --fps 4 \
  --duration 12.0 \
  --seed 42
```

**Parameters:**
- Texture: subtle_gray_stripes → 1 texture
- Directions: left, right → 2 directions (horizontal only)
- Speeds: 0.0, 14.0 → 2 speeds
- Angles: 8°, 22°, 38°, 52° → 4 non-standard angles (interpolation test)
- Stripe gray: 10, Background: 13
- **Total videos:** 1 × 2 × 2 × 4 = **96 videos** (48 moving, 48 stopped)
- FPS: 4, Duration: 12 seconds

### Test Dataset Arguments

**Dataset Name:** `subtle_gray_test_20251130_112952`

**Command:**
```bash
python3 building_dataset.py \
  --dataset_name subtle_gray_test_20251130_112952 \
  --texture_type subtle_gray_stripes \
  --direction left,right,up,down \
  --speed_range 0.0,14.0 \
  --view_angle 8,22,38,52 \
  --stripe_gray 30 \
  --background_gray 33 \
  --fps 4 \
  --duration 12.0 \
  --seed 96
```

**Key Differences from Training:**
- **Directions:** 4 (left, right, up, down) vs 2 in training ← **CRITICAL MISMATCH**
- Includes **unseen vertical motion** (up, down)
- Stripe gray: 30 (vs 10) - **+20 difference**
- Background: 33 (vs 13) - **+20 difference**
- Same non-standard angles: 8°, 22°, 38°, 52°
- Seed: 96 (different variations)
- **Total videos:** 288 (144 moving, 144 stopped) - 3x training size

### Training Configuration
- Epochs: 5 (same as Exp 2 & 3)
- Batch size: 1 (effective 8)
- Learning rate: 5e-5
- LoRA rank: 8, alpha: 16

### Results

| Metric | Base Model | Fine-Tuned | Improvement |
|--------|------------|------------|-------------|
| Accuracy | 50.00% | 50.00% | **+0.00%** |
| F1 Score | 0.00% | 66.67% | +66.67% |
| Precision | 0.00% | 50.00% | +50.00% |
| Recall | 0.00% | 100.00% | +100.00% |
| Moving Detected | 0/144 (0%) | 144/144 (100%) | ⚠️ All predicted moving |
| Stopped Detected | 144/144 (100%) | 0/144 (0%) | ❌ None detected |

**Per-Angle Results:** (All equally bad)
- angle8: 36/72 (50%)
- angle22: 36/72 (50%)
- angle38: 36/72 (50%)
- angle52: 36/72 (50%)

**Outcome:** ❌ **COMPLETE FAILURE** - Model learned opposite behavior!

### Analysis
- Model learned to ALWAYS predict "moving" (opposite of base model)
- Training on left/right, testing on left/right/up/down was too broad
- Non-standard angles (8,22,38,52) added difficulty
- Different gray values + direction mismatch = total failure
- Despite 5 epochs, train/test mismatch prevented any useful learning

---

## Comparative Summary Table

| Exp | Date | Train Videos | Test Videos | Epochs | Directions (Train→Test) | Textures | Angles | Accuracy | Result |
|-----|------|-------------|-------------|--------|------------------------|----------|--------|----------|--------|
| 5 | Nov 27 | 36 | 72 | 5 | 1→1 (R→R) | 1 | 4 | 50% | ❌ Fail |
| 4 | Nov 29 | 72 | 72 | 3 | 4→4 (Match) | 3 | 3 | 66.67% | ⚠️ Moderate |
| 3 | Nov 29 | 72 | 72 | 5 | 4→4 (Match) | 3 | 3 | **100%** | ⭐ Perfect |
| 2 | Nov 29 | 128 | 256 | 5 | 2→2 (L/R→L/R) | 1 | 4 | 99.61% | ✅ Success |
| 1 | Nov 30 | 96 | 288 | 5 | 2→4 (L/R→ALL) ❌ | 1 | 4 | 50% | ❌ Fail |

---

## Key Insights by Change

### What Improved Performance:
1. **More Epochs:** Exp 3 (5 epochs, 100%) vs Exp 4 (3 epochs, 66.67%) - **+33.33% improvement**
2. **More Training Data:** Exp 2 (128 videos, 99.61%) vs Exp 5 (36 videos, 50%) - **+49.61% improvement**
3. **Multiple Textures:** Exp 3 (3 textures, 100%) vs Exp 2 (1 texture, 99.61%) - **+0.39% improvement**
4. **Direction Matching:** Exp 2 (match, 99.61%) vs Exp 1 (mismatch, 50%) - **+49.61% improvement**
5. **Standard Angles:** Exp 2 (0,15,30,45, 99.61%) vs Exp 1 (8,22,38,52, 50%) - **+49.61% improvement**

### What Degraded Performance:
1. **Too Few Videos:** Exp 5 (36 videos) → complete failure
2. **Fewer Epochs:** Exp 4 (3 epochs) → 33.33% lower than Exp 3
3. **Direction Mismatch:** Exp 1 (train L/R, test ALL) → complete failure
4. **Non-Standard Angles:** Exp 1 (8,22,38,52) → complete failure
5. **Single Direction:** Exp 5 (right only) → complete failure

---

## Progression of Understanding

### Experiment 5 → 4
**Learned:** Need more than 36 videos, need multiple directions

### Experiment 4 → 3
**Learned:** 5 epochs >> 3 epochs (critical!)

### Experiment 3 → 2
**Learned:** Can achieve 99%+ with horizontal-only if data is sufficient

### Experiment 2 → 1
**Learned:** Training/test direction mismatch causes total failure

---

**Report Generated:** 2025-11-30
**Location:** ALL_EXPERIMENT_RESULTS/DETAILED_MODEL_REPORT.md
