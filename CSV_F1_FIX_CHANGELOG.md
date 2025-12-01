# CSV Sync and F1 Score Calculation Fix - Change Log

**Date:** 2025-12-01
**Issues Fixed:**
1. CSV not syncing properly from server to local PC
2. F1 score only calculated for "moving" class, not for both "moving" and "stopped" classes

---

## Problem 1: CSV Not Syncing Properly

### Description
The experiments_log.csv was not properly updating on the local PC after running experiments on the remote server.

**Root Causes:**
1. CSV was being downloaded from `/home/seedoo/.../data/experiments_log.csv` (outside Docker) instead of `/app/data/experiments_log.csv` (inside Docker where experiment actually writes)
2. rsync was overwriting the entire local CSV file instead of appending new experiment results

### Solution

**File Modified:** `run_experiment.sh` (lines 834-852)

**What Changed:**
Instead of using rsync to download and overwrite the entire CSV file, the script now:
1. Retrieves ONLY the last line (newest experiment) from the Docker container's CSV using `docker exec llamafactory tail -n 1 /app/data/experiments_log.csv`
2. Appends that line to the local CSV file
3. Creates local CSV with header if it doesn't exist

**Code Added:**
```bash
log_info "Retrieving new experiment results from server CSV..."
# Get the last line from the CSV inside the Docker container (newest experiment)
# Then append it to the local CSV file (don't overwrite)
REMOTE_CSV_TAIL=$(ssh "$SERVER_SSH" "docker exec llamafactory tail -n 1 /app/data/experiments_log.csv" 2>/dev/null)

if [ -n "$REMOTE_CSV_TAIL" ]; then
    # Ensure local CSV exists
    if [ ! -f "$LOCAL_DIR/data/experiments_log.csv" ]; then
        log_warning "Local CSV does not exist, creating with header..."
        # Get the header from the remote CSV
        ssh "$SERVER_SSH" "docker exec llamafactory head -n 1 /app/data/experiments_log.csv" > "$LOCAL_DIR/data/experiments_log.csv"
    fi

    # Append the new experiment line to local CSV
    echo "$REMOTE_CSV_TAIL" >> "$LOCAL_DIR/data/experiments_log.csv"
    log_success "Appended new experiment to local CSV"
else
    log_warning "Could not retrieve experiment results from server CSV"
fi
```

**Why this fixes the issue:**
- Retrieves from the correct location (inside Docker container)
- Appends instead of overwrites, preserving all previous experiments
- Each experiment adds exactly one new line to the local CSV

---

## Problem 2: F1 Score Only Calculated for One Class

### Description
The F1 score in experiments_log.csv was showing incorrect values because it was only calculated for the "moving" class, not for both "moving" and "stopped" classes.

**Example of the problem:**
- Fine-tuned model gets 4/4 moving correct, 0/4 stopped correct
- Should have F1=0% (because it fails completely on stopped class)
- But was showing F1=66.67% (because it only measured "moving" class)

**Root Cause:**
The evaluation script (`evaluate_pipeline_simple.py`) DOES calculate per-class F1 scores (f1_moving and f1_stopped), but the experiment tracker (`experiment_tracker.py`) was NOT logging them to the CSV.

### Solution

#### Part 1: Added columns to CSV

**File Modified:** `experiment_tracker.py` (lines 151-172)

**What Changed:**
Added 4 new columns to the CSV:
- `base_model_f1_moving`
- `base_model_f1_stopped`
- `finetuned_model_f1_moving`
- `finetuned_model_f1_stopped`

**Code Added:**
```python
# Results - Overall
'base_model_accuracy',
'base_model_f1_score',
'base_model_f1_moving',       # NEW
'base_model_f1_stopped',      # NEW
'base_model_precision',
...
'finetuned_model_accuracy',
'finetuned_model_f1_score',
'finetuned_model_f1_moving',  # NEW
'finetuned_model_f1_stopped', # NEW
'finetuned_model_precision',
...
```

#### Part 2: Updated initialization to include new columns

**File Modified:** `experiment_tracker.py` (lines 366-387)

**What Changed:**
Initialize the new columns with empty values:

```python
# Results (empty initially)
'base_model_accuracy': '',
'base_model_f1_score': '',
'base_model_f1_moving': '',    # NEW
'base_model_f1_stopped': '',   # NEW
...
'finetuned_model_accuracy': '',
'finetuned_model_f1_score': '',
'finetuned_model_f1_moving': '',  # NEW
'finetuned_model_f1_stopped': '', # NEW
...
```

#### Part 3: Updated evaluation results logging

**File Modified:** `experiment_tracker.py` (lines 438-486)

**What Changed:**
When logging evaluation results, now also logs per-class F1 scores:

```python
# Process base model results
if base_results is not None:
    updates['base_model_accuracy'] = f"{base_results.get('accuracy', 0):.2f}"
    updates['base_model_f1_score'] = f"{base_results.get('f1_score', 0):.2f}"
    updates['base_model_f1_moving'] = f"{base_results.get('f1_moving', 0):.2f}"    # NEW
    updates['base_model_f1_stopped'] = f"{base_results.get('f1_stopped', 0):.2f}"  # NEW
    ...

# Process fine-tuned model results
if finetuned_results is not None:
    updates['finetuned_model_accuracy'] = f"{finetuned_results.get('accuracy', 0):.2f}"
    updates['finetuned_model_f1_score'] = f"{finetuned_results.get('f1_score', 0):.2f}"
    updates['finetuned_model_f1_moving'] = f"{finetuned_results.get('f1_moving', 0):.2f}"    # NEW
    updates['finetuned_model_f1_stopped'] = f"{finetuned_results.get('f1_stopped', 0):.2f}"  # NEW
    ...
```

**Why this fixes the issue:**
- Per-class F1 scores were already being calculated in `evaluate_pipeline_simple.py` (lines 274-277)
- Now they are properly logged to the CSV, giving visibility into model performance on both classes
- This allows proper assessment of model biases (e.g., always predicting "moving" or always predicting "stopped")

---

## Impact

### Before Fixes:
1. **CSV Issue**: Local CSV would not update with new experiment results from server
2. **F1 Issue**: F1 score was misleading - showed 66.67% even when model completely failed on one class

### After Fixes:
1. **CSV Issue**: Each experiment automatically appends new results to local CSV
2. **F1 Issue**: Both per-class F1 scores are logged, allowing proper model evaluation

---

## Files Modified

1. `run_experiment.sh`
   - Lines 834-852: CSV retrieval logic

2. `experiment_tracker.py`
   - Lines 151-172: CSV column definitions
   - Lines 366-387: Column initialization
   - Lines 438-449: Base model results logging
   - Lines 475-486: Fine-tuned model results logging

---

## Verification

To verify the fixes work:

1. Run an experiment on the server:
   ```bash
   ./run_experiment.sh --yes \
     --train-angles "0.0,15.0" \
     --test-angles "7.0,22.0,30.0,45.0" \
     --texture subtle_gray_stripes \
     --direction left \
     --speed-range "0.0,14.0" \
     --epochs 1
   ```

2. Check that local CSV has been updated with new experiment
3. Check that CSV includes both `f1_moving` and `f1_stopped` columns
4. Verify F1 scores make sense for both classes

---

## Notes

- The per-class F1 calculation was already correct in `evaluate_pipeline_simple.py`
- This fix only adds proper logging/tracking of those values
- The overall F1 score (`f1_score` column) uses `average='binary'` which measures the positive class (moving)
- The per-class F1 scores give separate measurements for both classes, which is more informative
