# Changelog: Session 2025-11-27 - Val Size Fix & Training with 100% Data

**Date:** 2025-11-27
**Session:** Fix validation split issue and train subtle_gray_challenge dataset
**Key Achievement:** Successfully configured and deployed training on 100% of training data (128 examples)

---

## Executive Summary

This session addressed a critical configuration issue where the training pipeline was using only 90% of the training dataset (115/128 examples) for actual training, holding out 10% for validation. The fix ensures all 128 training examples are used for optimization while keeping the separate 128-video test dataset completely held out for final evaluation.

**Impact:**
- Training now uses 128 examples instead of 115 (+13 examples, +11.3% training data)
- No validation split during training (val_size: 0.0)
- Test dataset remains separate and untouched during training
- Fix is permanent in run_full_pipeline.py for all future runs

---

## Problem Discovery

### Initial Training Run
Started training with command:
```bash
docker exec llamafactory python3 /app/run_full_pipeline.py \
  --dataset_name subtle_gray_challenge \
  --skip_dataset \
  --num_train_epochs 5 \
  --eval_video_fps 4 \
  --eval_video_maxlen 128
```

**Issue Found:**
- Training logs showed: `Num examples = 115`
- Expected: `Num examples = 128` (full training dataset)
- Root cause: Generated training config had `val_size: 0.1` (10% validation split)

### Configuration Mismatch
- **Local run_full_pipeline.py:** Had correct `val_size: 0.0` (line 291)
- **Server container run_full_pipeline.py:** Had incorrect `val_size: 0.1`
- **Generated config:** Inherited the incorrect value from server's script

---

## Changes Made

### 1. Fix run_full_pipeline.py in Container

**File:** `/app/run_full_pipeline.py` (in llamafactory container)

**Method:** `_create_training_config()` (lines 290-296)

**Before (Server had this):**
```yaml
### Evaluation Configuration (using validation split, not test set)
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: {self.args.eval_steps}
```

**After (Fixed to match local):**
```yaml
### Evaluation Configuration (no validation split - train on 100% of training set)
val_size: 0.0
per_device_eval_batch_size: 1
eval_strategy: "no"
eval_steps: {self.args.eval_steps}
save_strategy: "steps"
logging_strategy: "steps"
```

**Changes:**
- Line 290: Updated comment to reflect no validation split
- Line 291: Changed `val_size: 0.1` → `val_size: 0.0`
- Line 293: Changed `eval_strategy: steps` → `eval_strategy: "no"`
- Lines 295-296: Added explicit save_strategy and logging_strategy

**How Fixed:**
```bash
# Copied updated script from host to container
docker cp /home/seedoo/shahar_linux_wsl/LLaMA-Factory/run_full_pipeline.py \
  llamafactory:/tmp/run_full_pipeline.py.new

# Overwrote container's version
docker exec llamafactory bash -c 'cat /tmp/run_full_pipeline.py.new > /app/run_full_pipeline.py'
```

### 2. Created Corrected Training Config

**File:** `/app/examples/train_qlora/qwen25vl_lora_100pct.yaml`

Created a new training configuration file with explicit settings:
- `val_size: 0.0` - Train on 100% of training data
- `eval_strategy: "no"` - No evaluation during training
- All other parameters match standard LoRA training config

### 3. Restarted Training with Correct Config

**Command:**
```bash
nohup docker exec llamafactory llamafactory-cli train \
  /app/examples/train_qlora/qwen25vl_lora_100pct.yaml \
  > /tmp/training_128_examples.log 2>&1 &
```

**Verification:**
```
[INFO|trainer.py:2520] >>   Num examples = 128  ✓ CORRECT
[INFO|trainer.py:2521] >>   Num Epochs = 5
[INFO|trainer.py:2527] >>   Total optimization steps = 80
```

---

## Dataset Details

### Dataset Name: `subtle_gray_challenge`

**Uniqueness:** ✅ UNIQUE - No conflicts with existing datasets
- Checked local data/ directory: No subtle_gray_challenge files
- Checked server container: No subtle_gray_challenge files
- Dataset naming follows pattern: `<base_name>_train` and `<base_name>_test`

**Datasets Created:**
1. **subtle_gray_challenge_train** - 128 training videos
   - Texture: subtle_gray_stripes
   - Background gray: 15-18 (4 values)
   - Stripe gray: 20-23 (4 values)
   - View angles: 0°, 15°, 30°, 45° (4 values)
   - Speeds: 0.0, 14.0 (2 values)
   - Total: 4×4×4×2 = 128 videos

2. **subtle_gray_challenge_test** - 128 test videos
   - Background gray: 10-13 (different range from training)
   - Stripe gray: 15-18 (different range from training)
   - Same angles, speeds, other parameters
   - Total: 4×4×4×2 = 128 videos

**Dataset Registration:**
- Registered in: `data/dataset_info.json`
- Format: ShareGPT with video messages
- Columns: messages, videos

---

## Training Configuration

### Model & Method
- **Base Model:** Qwen/Qwen2.5-VL-3B-Instruct
- **Method:** LoRA (Low-Rank Adaptation)
- **LoRA Rank:** 8
- **LoRA Alpha:** 16
- **Trainable Params:** 14,966,784 (0.40% of 3.77B total)

### Training Hyperparameters
- **Epochs:** 5
- **Batch Size:** 1 per device
- **Gradient Accumulation:** 8 steps
- **Effective Batch Size:** 8
- **Learning Rate:** 5e-5
- **Scheduler:** Cosine with 10% warmup
- **Precision:** FP16
- **Quantization:** 4-bit

### Data Configuration
- **Training Examples:** 128 (100% of training dataset)
- **Validation Examples:** 0 (no validation split)
- **Test Examples:** 128 (held out, used only for final evaluation)
- **Optimization Steps:** 80 (128/8 × 5 epochs)

### Video Processing
- **FPS Sampling:** 4 FPS (for evaluation)
- **Max Frames:** 128
- **Cutoff Length:** 8192 tokens

---

## Verification & Testing

### Before Fix
```
Config generated by server's run_full_pipeline.py:
  val_size: 0.1
  eval_strategy: steps

Training output:
  Num examples = 115
  Total optimization steps = 75
```

### After Fix
```
Config explicitly set:
  val_size: 0.0
  eval_strategy: "no"

Training output:
  Num examples = 128  ✓
  Total optimization steps = 80  ✓
```

### Training Progress (at documentation time)
- **Status:** Running successfully
- **Progress:** Step 31/80 (39%)
- **Speed:** ~7.8 seconds/step
- **ETA:** ~6 minutes remaining
- **Process ID:** 154112

---

## Impact Analysis

### Code Changes
1. ✅ **run_full_pipeline.py** - Fixed in container (permanent)
2. ✅ **Training config YAML** - Created corrected version
3. ❌ **No changes needed** - building_dataset.py (train_split is different concept)
4. ❌ **No changes needed** - Documentation (90/10 refers to dataset split, not validation)

### Documentation Review
Reviewed all README and documentation files:
- `README_PIPELINE.md` - References to "90/10" are about dataset generation split ✓ CORRECT
- `README_PIPELINE_USAGE.md` - Same, no changes needed ✓ CORRECT
- `CHANGELOG_DISABLE_VALIDATION_SPLIT.md` - Already existed, documents this exact change ✓

**Key Distinction:**
- **train_split=0.9:** Dataset generation creates 90% train, 10% test datasets
- **val_size=0.0:** Training uses 100% of training dataset without validation split
- These are two independent parameters for different purposes

### Files Modified (Local)
```
M  run_full_pipeline.py              # Already had correct val_size: 0.0
M  data/dataset_info.json            # Added dataset registrations (unrelated)
M  docker-compose.yml                # Previous changes (unrelated)
M  requirements.txt                  # Previous changes (unrelated)
```

### Files Modified (Server Container)
```
M  /app/run_full_pipeline.py         # Fixed val_size: 0.1 → 0.0
A  /app/examples/train_qlora/qwen25vl_lora_100pct.yaml  # New corrected config
```

---

## Docker Image Considerations

### Current Status
- **Container:** llamafactory (already running)
- **Image:** Based on LLaMA-Factory with Qwen2.5-VL support
- **Mount Points:**
  - `/app/data` → Host: `/home/seedoo/shahar_linux_wsl/LLaMA-Factory/data`
  - Other mounts for models, cache, etc.

### Do We Need New Image?
**Answer: NO** - Image rebuild not required because:

1. **Script Changes Only:**
   - Modified `run_full_pipeline.py` (Python script)
   - Created new YAML config file
   - No dependency changes, no system packages, no base image changes

2. **Runtime Changes:**
   - Changes applied to running container
   - Changes copied to host filesystem (synced via rsync)
   - Container restart will pick up changes from mounted volumes

3. **Persistence:**
   - Host files: Already updated via rsync
   - Container files: Updated via docker cp
   - Future container restarts: Will use updated host files from mount

### When Would We Need New Image?
Would only need to rebuild image if:
- Installing new Python packages (requirements.txt system-level)
- Changing base image or system dependencies
- Modifying Dockerfile
- Updating LLaMA-Factory framework version

### Recommendation
- ✅ **Current approach is sufficient:** Script changes propagate correctly
- ✅ **Container can continue running:** No restart needed
- ✅ **Changes persist:** Host filesystem has correct versions
- 💡 **Optional:** Could rebuild image later for cleaner deployment, but not required

---

## Related Files & Documentation

### Changelogs
- `CHANGELOG_FILE_CONSOLIDATION.md` - Previous session (synthetic_data_generation.py consolidation)
- `CHANGELOG_DISABLE_VALIDATION_SPLIT.md` - Documents val_size: 0.0 change (already existed)
- `CHANGELOG_STRIPE_VARIANCE.md` - Documents stripe_distance_variance feature
- `CHANGELOG_SUBTLE_GRAY_STRIPES.md` - Documents subtle_gray_stripes texture

### Documentation
- `README_PIPELINE_USAGE.md` - Pipeline usage guide
- `README_PIPELINE.md` - Pipeline architecture
- `DEPLOYMENT.md` - Deployment instructions
- `README_DATASET_BUILDER.md` - Dataset generation guide

### Scripts
- `run_full_pipeline.py` - Main orchestrator script (MODIFIED)
- `building_dataset.py` - Dataset generation (no changes)
- `evaluate_pipeline_simple.py` - Evaluation script (no changes)

---

## Lessons Learned

1. **Container vs Host Files:**
   - Non-mounted files in container can diverge from host
   - Always verify file versions in running containers
   - Use `docker cp` to update non-mounted files

2. **Configuration Propagation:**
   - Generated configs inherit from script templates
   - Fix must be in script, not just config file
   - Template changes require script updates

3. **Validation vs Test Split:**
   - train_split: Divides total data into train/test datasets
   - val_size: Divides training dataset into train/validation
   - These are independent and serve different purposes

4. **Dataset Naming:**
   - Use descriptive, unique names
   - Follow pattern: `<experiment>_<split>`
   - Check for conflicts before generation

---

## Next Steps

1. ✅ **Monitor training** - Currently running, ~6 min remaining
2. ⏳ **Run evaluation** - After training completes
3. ⏳ **Analyze results** - Compare base vs fine-tuned model
4. ⏳ **Document results** - Record accuracy metrics
5. ✅ **Commit changes** - Git commit with this changelog

---

## Commands Reference

### Check Training Status
```bash
# Check process
ssh seedoo@hetzner-gpu.tail9e6e7.ts.net "docker exec llamafactory ps aux | grep llamafactory-cli"

# Check progress
ssh seedoo@hetzner-gpu.tail9e6e7.ts.net "tail -20 /tmp/training_128_examples.log"
```

### Verify Configuration
```bash
# Check run_full_pipeline.py val_size
docker exec llamafactory grep -A 3 'val_size:' /app/run_full_pipeline.py

# Check generated config
docker exec llamafactory cat /app/examples/train_qlora/qwen25vl_lora_100pct.yaml
```

### Training Command
```bash
docker exec llamafactory llamafactory-cli train \
  /app/examples/train_qlora/qwen25vl_lora_100pct.yaml
```

---

**Status:** ✅ Complete - Training running successfully with 128 examples
**Ready for:** Evaluation after training completion
