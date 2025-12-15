# Adapter Capability Analysis - Experimental Design

## Executive Summary

**Core Problem:** Qwen2.5-VL-3B base model cannot detect treadmill/conveyor belt motion when the belt is empty (no objects). Base model achieves ~50% accuracy.

**Research Goal:** Systematically understand where LoRA adapters CAN and CANNOT improve the model without hurting existing capabilities.

**Approach:** 26 focused experiments with proper train/test separation:
- **Training set:** 50 videos per experiment
- **Test set:** 100 videos per experiment (2x larger, completely separate)
- Clear evidence of specific factors that make adapters better/worse

---

## Research Questions

1. **Where do adapters help?** What conditions allow adapters to successfully teach the model to detect empty belt motion without hurting any other ability?
2. **Where do adapters fail?** What conditions cause complete failure (0% improvement)?
3. **Speed generalization:** Can adapters trained on slow motion generalize to fast motion?
4. **Lighting generalization:** Can adapters trained in good lighting generalize to poor lighting?
5. **Blur generalization:** Can adapters trained on clear videos generalize to blurry videos?
6. **Batch size impact:** Does training batch size affect adapter quality?
7. **Capability preservation:** Do adapters hurt the model's ability to detect motion when objects are present?

---

## Phase 1: Baseline & Failure Mode Analysis

### Experiment 1.1: Optimal Batch Size Discovery
**Goal:** Find the optimal batch size for adapter training

**Method:** Train with identical config but vary effective batch size by adjusting per_device_train_batch_size (primary) and gradient_accumulation_steps (secondary)

**Setup:**
```bash
# Create separate train and test datasets (shared across all batch size experiments)
python3 building_dataset.py \
  --dataset_name exp1_1_train \
  --num_videos 50 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 0,5.0 \
  --seed 100

python3 building_dataset.py \
  --dataset_name exp1_1_test \
  --num_videos 100 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 0,5.0 \
  --seed 200
```

**Training Commands (vary batch size):**
```bash
# Batch size 1 (per_device=1, grad_accum=1)
python3 run_full_pipeline.py \
  --dataset_name exp1_1_bs1 \
  --skip_dataset \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --skip_evaluation

# Batch size 2 (per_device=2, grad_accum=1)
python3 run_full_pipeline.py \
  --dataset_name exp1_1_bs2 \
  --skip_dataset \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --skip_evaluation

# Batch size 4 (per_device=4, grad_accum=1)
python3 run_full_pipeline.py \
  --dataset_name exp1_1_bs4 \
  --skip_dataset \
  --num_train_epochs 5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --skip_evaluation

# Batch size 8 (per_device=8, grad_accum=1) - if GPU memory allows
python3 run_full_pipeline.py \
  --dataset_name exp1_1_bs8 \
  --skip_dataset \
  --num_train_epochs 5 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --skip_evaluation

# Batch size 8 via accumulation (per_device=2, grad_accum=4) - if GPU limited
python3 run_full_pipeline.py \
  --dataset_name exp1_1_bs8_accum \
  --skip_dataset \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --skip_evaluation

# Batch size 16 (per_device=4, grad_accum=4)
python3 run_full_pipeline.py \
  --dataset_name exp1_1_bs16 \
  --skip_dataset \
  --num_train_epochs 5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --skip_evaluation
```

**Evaluation (same test set for all):**
```bash
# Evaluate each trained adapter on the same test set
for adapter in exp1_1_bs1 exp1_1_bs2 exp1_1_bs4 exp1_1_bs8 exp1_1_bs8_accum exp1_1_bs16; do
  python3 evaluate_pipeline_simple.py \
    --test_dataset exp1_1_test \
    --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-${adapter}
done
```

**Expected:** Moderate batch size (4-8) achieves best accuracy (>90%), BS=1 may be noisy

---

### Experiment 1.2: Minimum Data Requirements
**Goal:** Find minimum training videos needed for adapter success

**Method:** Train with identical config but vary number of training videos (10, 20, 30, 40, 50)

**Setup:**
```bash
# Create separate train datasets with different sizes
python3 building_dataset.py \
  --dataset_name exp1_2_train_n10 \
  --num_videos 10 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 0,5.0 \
  --seed 110

python3 building_dataset.py \
  --dataset_name exp1_2_train_n20 \
  --num_videos 20 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 0,5.0 \
  --seed 120

python3 building_dataset.py \
  --dataset_name exp1_2_train_n30 \
  --num_videos 30 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 0,5.0 \
  --seed 130

python3 building_dataset.py \
  --dataset_name exp1_2_train_n40 \
  --num_videos 40 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 0,5.0 \
  --seed 140

python3 building_dataset.py \
  --dataset_name exp1_2_train_n50 \
  --num_videos 50 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 0,5.0 \
  --seed 150

# Create shared test dataset (100 videos)
python3 building_dataset.py \
  --dataset_name exp1_2_test \
  --num_videos 100 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 0,5.0 \
  --seed 210
```

**Training Commands:**
```bash
# Train on 10 videos
python3 run_full_pipeline.py \
  --dataset_name exp1_2_n10 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

# Train on 20 videos
python3 run_full_pipeline.py \
  --dataset_name exp1_2_n20 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

# Train on 30 videos
python3 run_full_pipeline.py \
  --dataset_name exp1_2_n30 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

# Train on 40 videos
python3 run_full_pipeline.py \
  --dataset_name exp1_2_n40 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

# Train on 50 videos
python3 run_full_pipeline.py \
  --dataset_name exp1_2_n50 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation
```

**Evaluation (same test set for all):**
```bash
for adapter in exp1_2_n10 exp1_2_n20 exp1_2_n30 exp1_2_n40 exp1_2_n50; do
  python3 evaluate_pipeline_simple.py \
    --test_dataset exp1_2_test \
    --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-${adapter}
done
```

**Expected:** Performance increases with data, find minimum N where accuracy >90%

---

### Experiment 1.3: Epoch Sensitivity
**Goal:** Find optimal number of training epochs

**Method:** Train with identical config but vary number of epochs (1, 3, 5, 7, 10)

**Setup:**
```bash
# Create train dataset (50 videos)
python3 building_dataset.py \
  --dataset_name exp1_3_train \
  --num_videos 50 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 0,5.0 \
  --seed 160

# Create test dataset (100 videos)
python3 building_dataset.py \
  --dataset_name exp1_3_test \
  --num_videos 100 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 0,5.0 \
  --seed 260
```

**Training Commands:**
```bash
# 1 epoch
python3 run_full_pipeline.py \
  --dataset_name exp1_3_e1 \
  --skip_dataset \
  --num_train_epochs 1 \
  --skip_evaluation

# 3 epochs
python3 run_full_pipeline.py \
  --dataset_name exp1_3_e3 \
  --skip_dataset \
  --num_train_epochs 3 \
  --skip_evaluation

# 5 epochs
python3 run_full_pipeline.py \
  --dataset_name exp1_3_e5 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

# 7 epochs
python3 run_full_pipeline.py \
  --dataset_name exp1_3_e7 \
  --skip_dataset \
  --num_train_epochs 7 \
  --skip_evaluation

# 10 epochs
python3 run_full_pipeline.py \
  --dataset_name exp1_3_e10 \
  --skip_dataset \
  --num_train_epochs 10 \
  --skip_evaluation
```

**Evaluation (same test set for all):**
```bash
for adapter in exp1_3_e1 exp1_3_e3 exp1_3_e5 exp1_3_e7 exp1_3_e10; do
  python3 evaluate_pipeline_simple.py \
    --test_dataset exp1_3_test \
    --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-${adapter}
done
```

**Expected:** 5-7 epochs optimal, 10 may overfit

---

## Phase 2: Speed Generalization

### Experiment 2.1: Train Slow → Test Slow (Control)
**Goal:** Establish baseline for slow motion detection

**Method:** Train and test on same slow speed range

**Setup:**
```bash
# Train on slow speeds (1-3 px/frame)
python3 building_dataset.py \
  --dataset_name exp2_1_train \
  --num_videos 50 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,3.0 \
  --seed 170

# Test on slow speeds (1-3 px/frame)
python3 building_dataset.py \
  --dataset_name exp2_1_test \
  --num_videos 100 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,3.0 \
  --seed 270
```

**Training & Evaluation:**
```bash
python3 run_full_pipeline.py \
  --dataset_name exp2_1 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

python3 evaluate_pipeline_simple.py \
  --test_dataset exp2_1_test \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-exp2_1
```

**Expected:** >95% accuracy (control baseline)

---

### Experiment 2.2: Train Slow → Test Fast (Generalization Test)
**Goal:** Test if adapters can generalize from slow to fast motion

**Method:** Train on slow speeds (1-3), test on fast speeds (6-8)

**Setup:**
```bash
# Train on slow speeds (1-3 px/frame)
python3 building_dataset.py \
  --dataset_name exp2_2_train \
  --num_videos 50 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,3.0 \
  --seed 180

# Test on FAST speeds (6-8 px/frame)
python3 building_dataset.py \
  --dataset_name exp2_2_test \
  --num_videos 100 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 6.0,8.0 \
  --seed 280
```

**Training & Evaluation:**
```bash
python3 run_full_pipeline.py \
  --dataset_name exp2_2 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

python3 evaluate_pipeline_simple.py \
  --test_dataset exp2_2_test \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-exp2_2
```

**Expected:** <70% accuracy (failed generalization)

---

### Experiment 2.3: Train Fast → Test Slow (Reverse Test)
**Goal:** Test if adapters can generalize from fast to slow motion

**Method:** Train on fast speeds (6-8), test on slow speeds (1-3)

**Setup:**
```bash
# Train on FAST speeds (6-8 px/frame)
python3 building_dataset.py \
  --dataset_name exp2_3_train \
  --num_videos 50 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 6.0,8.0 \
  --seed 190

# Test on slow speeds (1-3 px/frame)
python3 building_dataset.py \
  --dataset_name exp2_3_test \
  --num_videos 100 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,3.0 \
  --seed 290
```

**Training & Evaluation:**
```bash
python3 run_full_pipeline.py \
  --dataset_name exp2_3 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

python3 evaluate_pipeline_simple.py \
  --test_dataset exp2_3_test \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-exp2_3
```

**Expected:** <70% accuracy (failed generalization)

---

### Experiment 2.4: Train Mixed → Test All Speeds
**Goal:** Test if mixed speed training enables full generalization

**Method:** Train on full speed range (1-8), test on full range

**Setup:**
```bash
# Train on FULL speed range (1-8 px/frame)
python3 building_dataset.py \
  --dataset_name exp2_4_train \
  --num_videos 50 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --seed 191

# Test on FULL speed range (1-8 px/frame)
python3 building_dataset.py \
  --dataset_name exp2_4_test \
  --num_videos 100 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --seed 291
```

**Training & Evaluation:**
```bash
python3 run_full_pipeline.py \
  --dataset_name exp2_4 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

python3 evaluate_pipeline_simple.py \
  --test_dataset exp2_4_test \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-exp2_4
```

**Expected:** >90% accuracy across all speeds

---

## Phase 3: Lighting Generalization

### Experiment 3.1: Train Normal → Test Normal (Control)
**Goal:** Establish baseline for normal lighting

**Method:** Train and test with normal lighting

**Setup:**
```bash
# Train on normal lighting
python3 building_dataset.py \
  --dataset_name exp3_1_train \
  --num_videos 50 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --brightness 0.0 \
  --contrast 1.0 \
  --seed 300

# Test on normal lighting
python3 building_dataset.py \
  --dataset_name exp3_1_test \
  --num_videos 100 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --brightness 0.0 \
  --contrast 1.0 \
  --seed 400
```

**Training & Evaluation:**
```bash
python3 run_full_pipeline.py \
  --dataset_name exp3_1 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

python3 evaluate_pipeline_simple.py \
  --test_dataset exp3_1_test \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-exp3_1
```

**Expected:** >95% accuracy (control baseline)

---

### Experiment 3.2: Train Normal → Test Dark (Generalization Test)
**Goal:** Test if adapters can generalize to dark conditions

**Method:** Train on normal lighting, test on dark lighting

**Setup:**
```bash
# Train on normal lighting
python3 building_dataset.py \
  --dataset_name exp3_2_train \
  --num_videos 50 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --brightness 0.0 \
  --contrast 1.0 \
  --seed 310

# Test on DARK lighting
python3 building_dataset.py \
  --dataset_name exp3_2_test \
  --num_videos 100 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --brightness -0.2 \
  --contrast 0.7 \
  --seed 410
```

**Training & Evaluation:**
```bash
python3 run_full_pipeline.py \
  --dataset_name exp3_2 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

python3 evaluate_pipeline_simple.py \
  --test_dataset exp3_2_test \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-exp3_2
```

**Expected:** <70% accuracy (failed generalization)

---

### Experiment 3.3: Train Dark → Test Normal (Reverse Test)
**Goal:** Test if adapters can generalize from dark to normal

**Method:** Train on dark lighting, test on normal lighting

**Setup:**
```bash
# Train on DARK lighting
python3 building_dataset.py \
  --dataset_name exp3_3_train \
  --num_videos 50 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --brightness -0.2 \
  --contrast 0.7 \
  --seed 320

# Test on normal lighting
python3 building_dataset.py \
  --dataset_name exp3_3_test \
  --num_videos 100 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --brightness 0.0 \
  --contrast 1.0 \
  --seed 420
```

**Training & Evaluation:**
```bash
python3 run_full_pipeline.py \
  --dataset_name exp3_3 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

python3 evaluate_pipeline_simple.py \
  --test_dataset exp3_3_test \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-exp3_3
```

**Expected:** >80% accuracy (easier generalization direction)

---

### Experiment 3.4: Train Varied → Test All Lighting
**Goal:** Test if varied lighting training enables full generalization

**Method:** Train on varied lighting, test on full range

**Setup:**
```bash
# Train with varied lighting
python3 building_dataset.py \
  --dataset_name exp3_4_train \
  --num_videos 50 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --vary_parameters \
  --seed 330

# Test on FULL lighting range (mix of normal and dark)
python3 building_dataset.py \
  --dataset_name exp3_4_test \
  --num_videos 100 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --vary_parameters \
  --seed 430
```

**Training & Evaluation:**
```bash
python3 run_full_pipeline.py \
  --dataset_name exp3_4 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

python3 evaluate_pipeline_simple.py \
  --test_dataset exp3_4_test \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-exp3_4
```

**Expected:** >85% across all lighting conditions

---

## Phase 4: Blur Generalization

### Experiment 4.1: Train Clear → Test Clear (Control)
**Goal:** Establish baseline for clear videos

**Method:** Train and test with no blur

**Setup:**
```bash
# Train on clear videos
python3 building_dataset.py \
  --dataset_name exp4_1_train \
  --num_videos 50 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --motion_blur 0 \
  --camera_noise 0.0 \
  --seed 340

# Test on clear videos
python3 building_dataset.py \
  --dataset_name exp4_1_test \
  --num_videos 100 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --motion_blur 0 \
  --camera_noise 0.0 \
  --seed 440
```

**Training & Evaluation:**
```bash
python3 run_full_pipeline.py \
  --dataset_name exp4_1 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

python3 evaluate_pipeline_simple.py \
  --test_dataset exp4_1_test \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-exp4_1
```

**Expected:** >95% accuracy (control baseline)

---

### Experiment 4.2: Train Clear → Test Blurry (Generalization Test)
**Goal:** Test if adapters can generalize to blurry conditions

**Method:** Train on clear videos, test on blurry videos

**Setup:**
```bash
# Train on clear videos
python3 building_dataset.py \
  --dataset_name exp4_2_train \
  --num_videos 50 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --motion_blur 0 \
  --camera_noise 0.0 \
  --seed 350

# Test on BLURRY videos
python3 building_dataset.py \
  --dataset_name exp4_2_test \
  --num_videos 100 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --motion_blur 2 \
  --camera_noise 0.2 \
  --seed 450
```

**Training & Evaluation:**
```bash
python3 run_full_pipeline.py \
  --dataset_name exp4_2 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

python3 evaluate_pipeline_simple.py \
  --test_dataset exp4_2_test \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-exp4_2
```

**Expected:** <70% accuracy (failed generalization)

---

### Experiment 4.3: Train Blurry → Test Clear (Reverse Test)
**Goal:** Test if adapters can generalize from blurry to clear

**Method:** Train on blurry videos, test on clear videos

**Setup:**
```bash
# Train on BLURRY videos
python3 building_dataset.py \
  --dataset_name exp4_3_train \
  --num_videos 50 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --motion_blur 2 \
  --camera_noise 0.2 \
  --seed 360

# Test on clear videos
python3 building_dataset.py \
  --dataset_name exp4_3_test \
  --num_videos 100 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --motion_blur 0 \
  --camera_noise 0.0 \
  --seed 460
```

**Training & Evaluation:**
```bash
python3 run_full_pipeline.py \
  --dataset_name exp4_3 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

python3 evaluate_pipeline_simple.py \
  --test_dataset exp4_3_test \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-exp4_3
```

**Expected:** >80% accuracy (easier generalization direction)

---

### Experiment 4.4: Train Varied → Test All Blur Levels
**Goal:** Test if varied blur training enables full generalization

**Method:** Train with varied blur, test on full range

**Setup:**
```bash
# Train with varied blur
python3 building_dataset.py \
  --dataset_name exp4_4_train \
  --num_videos 50 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --vary_parameters \
  --seed 370

# Test on FULL blur range
python3 building_dataset.py \
  --dataset_name exp4_4_test \
  --num_videos 100 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --vary_parameters \
  --seed 470
```

**Training & Evaluation:**
```bash
python3 run_full_pipeline.py \
  --dataset_name exp4_4 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

python3 evaluate_pipeline_simple.py \
  --test_dataset exp4_4_test \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-exp4_4
```

**Expected:** >85% across all blur levels

---

## Phase 5: Object Presence - Capability Preservation

### Experiment 5.1: Base Model on Objects (Baseline)
**Goal:** Measure base model's ability to detect motion with objects present

**Method:** Test base model (no adapter) on videos with objects

**Setup:**
```bash
# Generate test set with objects (200 videos for robust baseline)
python3 building_dataset.py \
  --dataset_name exp5_1_test_objects \
  --num_videos 200 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --add_objects true \
  --seed 500
```

**Evaluation (base model only, no adapter):**
```bash
python3 evaluate_pipeline_simple.py \
  --test_dataset exp5_1_test_objects
```

**Expected:** Base model can already detect motion with objects (>70%)

---

### Experiment 5.2: Adapted Model on Objects (Preservation Test)
**Goal:** Ensure adapter trained on empty belts doesn't hurt object detection

**Method:** Train on empty belts, test on videos with objects

**Setup:**
```bash
# Train on EMPTY belts (50 videos)
python3 building_dataset.py \
  --dataset_name exp5_2_train_empty \
  --num_videos 50 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --seed 510

# Test set already created in 5.1 (with objects)
```

**Training & Evaluation:**
```bash
python3 run_full_pipeline.py \
  --dataset_name exp5_2 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

# Evaluate on test set WITH objects
python3 evaluate_pipeline_simple.py \
  --test_dataset exp5_1_test_objects \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-exp5_2
```

**Expected:** Accuracy ≥ Exp 5.1 (no capability regression)

---

### Experiment 5.3: Mixed Training (Empty + Objects)
**Goal:** Test if mixed training maintains both capabilities

**Method:** Train on mix of empty and object videos, test both separately

**Setup:**
```bash
# Train set: 25 empty + 25 with objects = 50 total
python3 building_dataset.py \
  --dataset_name exp5_3_train_empty \
  --num_videos 25 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --seed 520

python3 building_dataset.py \
  --dataset_name exp5_3_train_objects \
  --num_videos 25 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --add_objects true \
  --seed 521

# Test set: 100 empty belts
python3 building_dataset.py \
  --dataset_name exp5_3_test_empty \
  --num_videos 100 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --seed 620

# Test set: 100 with objects (or reuse from 5.1)
```

**Manual Dataset Merge (Required):**
```bash
# Merge the two training JSONs into one
python3 -c "
import json
with open('data/exp5_3_train_empty.json', 'r') as f:
    empty = json.load(f)
with open('data/exp5_3_train_objects.json', 'r') as f:
    objects = json.load(f)
merged = empty + objects
with open('data/exp5_3_train_mixed.json', 'w') as f:
    json.dump(merged, f, indent=2)
"

# Register in dataset_info.json
# (Manual step or modify building_dataset.py)
```

**Training & Evaluation:**
```bash
# Train on merged dataset
python3 run_full_pipeline.py \
  --dataset_name exp5_3 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

# Evaluate on EMPTY belts
python3 evaluate_pipeline_simple.py \
  --test_dataset exp5_3_test_empty \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-exp5_3

# Evaluate on belts WITH objects
python3 evaluate_pipeline_simple.py \
  --test_dataset exp5_1_test_objects \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-exp5_3
```

**Expected:** >90% on both empty and object test sets

---

## Phase 6: Combined Stressors - Robustness Testing

### Experiment 6.1: Train Normal → Test Mild Stressors
**Goal:** Test robustness under multiple mild challenges

**Method:** Train on normal conditions, test with multiple mild stressors

**Setup:**
```bash
# Train on normal conditions (50 videos)
python3 building_dataset.py \
  --dataset_name exp6_1_train \
  --num_videos 50 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --seed 530

# Test with MILD stressors: moderate speed + slight dark + slight blur (100 videos)
python3 building_dataset.py \
  --dataset_name exp6_1_test_mild \
  --num_videos 100 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 4.0,5.0 \
  --brightness -0.1 \
  --motion_blur 1 \
  --seed 630
```

**Training & Evaluation:**
```bash
python3 run_full_pipeline.py \
  --dataset_name exp6_1 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

python3 evaluate_pipeline_simple.py \
  --test_dataset exp6_1_test_mild \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-exp6_1
```

**Expected:** 70-85% accuracy (moderate degradation)

---

### Experiment 6.2: Train Normal → Test Severe Stressors
**Goal:** Test if multiple severe stressors cause catastrophic failure

**Method:** Train on normal conditions, test with multiple severe stressors

**Setup:**
```bash
# Train on normal conditions (reuse from 6.1)

# Test with SEVERE stressors: fast speed + dark + heavy blur (100 videos)
python3 building_dataset.py \
  --dataset_name exp6_2_test_severe \
  --num_videos 100 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 7.0,8.0 \
  --brightness -0.2 \
  --contrast 0.7 \
  --motion_blur 2 \
  --camera_noise 0.2 \
  --seed 640
```

**Evaluation:**
```bash
python3 evaluate_pipeline_simple.py \
  --test_dataset exp6_2_test_severe \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-exp6_1
```

**Expected:** <50% accuracy (catastrophic failure)

---

### Experiment 6.3: Train Robust → Test Severe Stressors
**Goal:** Test if robust training handles combined stressors

**Method:** Train with full variation, test on severe conditions

**Setup:**
```bash
# Train with FULL variation (speed, lighting, blur all varied) (50 videos)
python3 building_dataset.py \
  --dataset_name exp6_3_train_robust \
  --num_videos 50 \
  --texture_type stripes \
  --direction right \
  --view_angle 0.0,15.0,30.0 \
  --speed_range 1.0,8.0 \
  --vary_parameters \
  --seed 550

# Test set already created in 6.2 (severe conditions)
```

**Training & Evaluation:**
```bash
python3 run_full_pipeline.py \
  --dataset_name exp6_3 \
  --skip_dataset \
  --num_train_epochs 5 \
  --skip_evaluation

python3 evaluate_pipeline_simple.py \
  --test_dataset exp6_2_test_severe \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-exp6_3
```

**Expected:** >75% accuracy (robust training helps significantly)

---

## Summary of Expected Results

| Phase | Experiments | Key Finding |
|-------|-------------|-------------|
| **Phase 1** | 1.1-1.3 | Batch size 4-8 optimal, need 30+ videos, 5-7 epochs for success |
| **Phase 2** | 2.1-2.4 | Adapters CANNOT generalize across speeds without mixed training |
| **Phase 3** | 3.1-3.4 | Adapters CANNOT generalize across lighting without varied training |
| **Phase 4** | 4.1-4.4 | Adapters CANNOT generalize across blur without varied training |
| **Phase 5** | 5.1-5.3 | Adapters DON'T hurt object detection capability (preservation verified) |
| **Phase 6** | 6.1-6.3 | Combined stressors cause catastrophic failure without robust training |

---

## Critical ML Engineering Practices Applied

### 1. Proper Train/Test Separation
- ✅ **Always separate datasets:** Train and test are created independently
- ✅ **Different seeds:** Train uses 100-500 range, test uses 200-600 range
- ✅ **2:1 test ratio:** Test set is 2x larger (100 vs 50 videos) for robust evaluation
- ✅ **No data leakage:** Never use `--num_videos` with auto-split

### 2. Batch Size Experiments Done Right
- ✅ **Vary per_device_train_batch_size first:** More efficient than gradient accumulation
- ✅ **Test both approaches:** Direct batch size vs accumulated batch size
- ✅ **GPU-aware:** Provides fallback options if memory limited

### 3. Controlled Experiments
- ✅ **Single variable:** Each experiment changes only ONE factor
- ✅ **Proper controls:** Every generalization test has a matching control experiment
- ✅ **Reproducible:** Fixed seeds for all datasets

### 4. Statistical Rigor
- ✅ **Large test sets:** 100-200 videos for confident metrics
- ✅ **Multiple comparisons:** Same test set used across related experiments
- ✅ **Clear thresholds:** >95% excellent, >90% good, <70% failed

---

## Execution Checklist

### Prerequisites
```bash
# Enter Docker container
docker exec -it llamafactory bash
cd /app

# Verify scripts exist
ls run_full_pipeline.py building_dataset.py evaluate_pipeline_simple.py
```

### Phase Execution Order
1. **Phase 1** - Establish optimal hyperparameters (batch size, data size, epochs)
2. **Phase 2** - Test speed generalization limits
3. **Phase 3** - Test lighting generalization limits
4. **Phase 4** - Test blur generalization limits
5. **Phase 5** - Verify no capability regression on objects
6. **Phase 6** - Test robustness under combined stress

### After Each Experiment
- [ ] Verify datasets created: `ls data/exp*_train/ data/exp*_test/`
- [ ] Check experiments_log.csv: `tail -5 data/experiments_log.csv`
- [ ] Review evaluation output: `cat evaluation_results_*/evaluation_report.txt`
- [ ] Document: base accuracy, finetuned accuracy, improvement %

---

## Timeline Estimate

- **Dataset generation:** ~3 min per dataset (50-100 videos)
- **Training:** ~20 min per experiment (50 videos, 5 epochs)
- **Evaluation:** ~5 min per test set (100 videos)
- **Per experiment:** ~30 minutes average

**Total:**
- 26 experiments × 30 min = **~13 hours** (sequential)
- With 4 GPUs in parallel = **~3-4 hours**

---

## Automation Script (Optional)

```bash
#!/bin/bash
# run_phase1.sh - Example automation for Phase 1

# Experiment 1.1: Batch Size
echo "Running Experiment 1.1: Batch Size Discovery"
python3 building_dataset.py --dataset_name exp1_1_train --num_videos 50 --texture_type stripes --direction right --view_angle 0.0,15.0,30.0 --speed_range 0,5.0 --seed 100
python3 building_dataset.py --dataset_name exp1_1_test --num_videos 100 --texture_type stripes --direction right --view_angle 0.0,15.0,30.0 --speed_range 0,5.0 --seed 200

for bs in 1 2 4 8; do
  echo "Training with batch_size=$bs"
  python3 run_full_pipeline.py --dataset_name exp1_1_bs${bs} --skip_dataset --num_train_epochs 5 --per_device_train_batch_size $bs --gradient_accumulation_steps 1 --skip_evaluation

  echo "Evaluating batch_size=$bs"
  python3 evaluate_pipeline_simple.py --test_dataset exp1_1_test --adapter_name_or_path saves/qwen2vl-treadmill-lora-pipeline-exp1_1_bs${bs}
done

echo "Phase 1.1 Complete!"
```

---

**Document Version:** 3.0 (ML Engineering Compliant)
**Created:** 2025-12-01
**Status:** Ready for Execution
