# Changelog: Disable Validation Split in Training

**Date:** 2025-11-27
**Feature:** Train on 100% of training dataset without validation split

## Overview
Modified `run_full_pipeline.py` to train on the entire training dataset without holding out a validation split. This change sets `val_size: 0.0` instead of the previous `val_size: 0.1`, allowing the model to learn from all available training data.

## Rationale
- User requested ability to train on 100% of training data intentionally
- Previous configuration automatically held out 10% for validation during training
- With `val_size: 0.0`, all training data is used for optimization
- Test set remains completely separate for final evaluation

## Changes Made

### run_full_pipeline.py

#### Modified `_create_training_config()` Method (Lines 287-288)

**Before:**
```yaml
### Evaluation Configuration (using validation split, not test set)
val_size: 0.1
```

**After:**
```yaml
### Evaluation Configuration (no validation split - train on 100% of training set)
val_size: 0.0
```

**Changes:**
- Line 287: Updated comment to reflect no validation split
- Line 288: Changed `val_size: 0.1` to `val_size: 0.0`

## Verification

### LLaMA-Factory Compatibility
The LLaMA-Factory codebase properly handles `val_size: 0.0`:

From `src/llamafactory/data/data_utils.py:97`:
```python
if data_args.val_size > 1e-6:
    # Create validation split
```

When `val_size: 0.0`, the condition is false and no validation split is created, which is the intended behavior.

### No Other Changes Required
- `eval_strategy: steps` and `eval_steps` settings remain unchanged
- These settings are ignored when no validation set exists
- Test dataset remains separate and is only used in Step 3 (evaluation)
- No changes needed in other files (building_dataset.py, evaluation scripts, etc.)

## Impact

### Training Dataset Split (Before)
- Total → 90% train + 10% test (controlled by `train_split=0.9`)
- Training set → 90% actual training + 10% validation (controlled by `val_size=0.1`)
- Effective training data: 81% of total (0.9 × 0.9)

### Training Dataset Split (After)
- Total → 90% train + 10% test (controlled by `train_split=0.9`)
- Training set → 100% training + 0% validation (controlled by `val_size=0.0`)
- Effective training data: 90% of total (0.9 × 1.0)

### Behavior Changes
1. **During Training:**
   - No validation metrics computed during training
   - No periodic evaluation on validation set
   - Training logs won't show validation loss/accuracy
   - All training examples used for gradient updates

2. **Final Evaluation:**
   - Test set evaluation (Step 3) remains unchanged
   - Test set still completely held out from training
   - Final accuracy metrics computed on test set as before

## Usage

The change is automatic and requires no command-line argument changes:

```bash
# Same command as before - now trains on 100% of training data
python3 run_full_pipeline.py \
  --dataset_name my_experiment \
  --num_videos 100 \
  --texture_type subtle_gray_stripes \
  --seed 42
```

## Files Modified
1. `run_full_pipeline.py` - 2 lines modified (comment + val_size value)

## Testing Recommendations
1. Run training and verify no validation metrics appear in logs
2. Check that training completes without errors
3. Verify test set evaluation still works correctly
4. Compare training curves with/without validation split

## Reverting This Change
To restore validation split behavior, change line 288 back to:
```yaml
val_size: 0.1
```

## Notes
- This change only affects the training phase (Step 2)
- Dataset generation (Step 1) and evaluation (Step 3) are unaffected
- Test set remains independent and is never used during training
- Setting `val_size: 0.0` is the standard way to disable validation in LLaMA-Factory
