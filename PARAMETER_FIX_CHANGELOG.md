# Server Sync Issue Fix - Change Log

**Date:** 2025-12-01
**Issue:** run_full_pipeline.py on server receiving unrecognized --train_* and --test_* parameters

## Problem Description

When running experiments on the remote server, the training step failed with:
```
run_full_pipeline.py: error: unrecognized arguments: --train_texture_type ... --train_direction ... --train_view_angle ... --test_texture_type ...
```

This error occurred even though:
- The **local PC** ran experiments successfully with the same code
- The `run_experiment.sh` script correctly passes metadata parameters to `run_full_pipeline.py`
- The parameters are needed for CSV experiment tracking

## Root Cause

**File Sync Issue:** The server's version of `run_full_pipeline.py` and `experiment_tracker.py` were outdated and did not include the metadata parameter support added in recent commits.

The regular rsync in `run_experiment.sh` (lines 490-505) only checks file modification times. If the local files had older timestamps than the server's old files, rsync wouldn't update them.

## Solution

### Permanent Fix: Modified rsync command

**File Modified:** `run_experiment.sh`

**Lines Changed:** 490-507 (rsync command)

**What Changed:**
- Added `--checksum` flag to rsync command
- rsync now compares file contents using checksums instead of just modification times
- This ensures Python files are ALWAYS synced when they differ, regardless of timestamps

**Why this fixes the issue:**
- The original rsync only checked file modification times
- If local files had older timestamps than server files, rsync wouldn't update them
- The `--checksum` flag forces rsync to compare actual file content
- Files are now synced based on content differences, not timestamp differences

**Code added:**
```bash
# Use --checksum to ensure files are synced based on content, not just timestamps
# This prevents sync issues where local files have older timestamps than server files
run_cmd "rsync -avz --checksum --progress \
```

### Note: Parameter passing is CORRECT

The `--train_*` and `--test_*` parameters passed to `run_full_pipeline.py` are **necessary and correct**:
- They are NOT used for dataset generation (which is skipped via `--skip_dataset`)
- They ARE used for CSV experiment tracking via `experiment_tracker.py`
- The ExperimentTracker reads these metadata parameters using `getattr(args, 'train_texture_type', '')`
- These parameters allow us to track what datasets were used in each experiment

## Impact

**Before Fix:**
- Dataset generation succeeded (4 train videos, 12 test videos)
- Training failed with "unrecognized arguments" error

**After Fix:**
- Dataset generation succeeds (4 train videos, 12 test videos)
- Training should now proceed successfully with only necessary parameters

## Testing

The fix ensures that when `building_dataset.py` is used to pre-generate datasets:
1. Dataset metadata is embedded in the dataset files themselves
2. `run_full_pipeline.py` only needs `--dataset_name` to locate and load these datasets
3. Only training and evaluation hyperparameters are passed to `run_full_pipeline.py`

## Code Comment Added

Added clarifying comment at lines 735-736:
```bash
# When datasets are pre-built, only pass training hyperparameters and evaluation settings
# Dataset generation parameters (--train_* and --test_*) are NOT needed since --skip_dataset is set
```

## Verification Steps

To verify the fix works:
1. Run experiment with different train/test angles:
   ```bash
   ./run_experiment.sh --yes \
     --train-angles "0.0,15.0" \
     --test-angles "7.0,22.0,35.0,30.0,45.0,52.0" \
     --texture subtle_gray_stripes \
     --direction left \
     --speed-range "0.0,14.0" \
     --epochs 2
   ```
2. Verify datasets are generated successfully
3. Verify training proceeds without parameter errors
4. Verify results are logged to experiments_log.csv
