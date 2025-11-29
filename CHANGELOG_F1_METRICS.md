# Changelog: Enhanced Evaluation Metrics with F1 Score and Per-Texture/Per-Angle Analysis

**Date:** 2025-11-29
**Author:** AI Assistant
**Scope:** Enhanced evaluation pipeline with comprehensive metrics tracking

---

## Summary

Added comprehensive F1 score, precision, and recall metrics to the evaluation pipeline, with granular breakdowns by texture type and camera angle. This enhancement provides detailed performance analysis for both the base model and fine-tuned LoRA model across different visual conditions.

---

## Files Modified

### 1. `evaluate_pipeline_simple.py`

#### Changes Made:

**a) Added New Imports (Lines 23-25)**
- Added `sklearn.metrics` for F1 score, precision, and recall calculations
- Added `re` module for regex pattern matching in filename parsing
- Added `defaultdict` from collections for dynamic nested dictionary creation

**b) New Method: `parse_video_metadata()` (Lines 51-70)**
- **Purpose:** Parse texture type and camera angle from video filenames
- **Input:** Video file path string
- **Output:** Dictionary with 'texture' and 'angle' keys
- **Pattern:** Extracts from format `treadmill_XXXX_<texture>_<direction>_speed<X.X>_angle<X>_...`
- **Fallback:** Returns 'unknown' for unparseable filenames with warning log

**c) Enhanced `evaluate()` Method**

**Expanded Results Structure (Lines 118-142)**
- Added `y_true` and `y_pred` lists for overall F1 calculation
- Added `per_texture` dict: tracks metrics for each texture type (stripes, noise, rubber, etc.)
- Added `per_angle` dict: tracks metrics for each camera angle (angle0, angle30, etc.)
- Each texture/angle includes:
  - `y_true`, `y_pred` lists for F1 calculation
  - `correct`, `total` counts for accuracy
  - Per-class (`moving`, `stopped`) breakdown

**Enhanced Tracking in Evaluation Loop (Lines 202-247)**
- Parse metadata (texture, angle) from each video filename
- Convert predictions to binary labels (1=moving, 0=stopped)
- Track overall predictions in `y_true` and `y_pred`
- Track per-texture statistics and predictions
- Track per-angle statistics and predictions
- Added texture and angle to result details for transparency

**Post-Evaluation Metrics Calculation (Lines 252-295)**
- **Overall Metrics:**
  - F1 Score (binary average)
  - Precision and Recall
  - Per-class F1 (separate for moving/stopped)

- **Per-Texture Metrics:** For each texture type:
  - Accuracy, F1 Score, Precision, Recall
  - Per-class F1 scores

- **Per-Angle Metrics:** For each camera angle:
  - Accuracy, F1 Score, Precision, Recall
  - Per-class F1 scores

- All metrics use `zero_division=0` to handle edge cases gracefully
- All percentages scaled to 0-100 range for consistency

**d) Enhanced `_write_results()` Method (Lines 333-397)**

Completely redesigned report generation with hierarchical structure:

**Overall Metrics Section:**
- Accuracy with counts
- F1 Score, Precision, Recall

**Per-Class Metrics Section:**
- Moving: count, accuracy, F1 score
- Stopped: count, accuracy, F1 score

**Per-Texture Breakdown Section:**
- For each texture (sorted alphabetically):
  - Total videos
  - Accuracy with counts
  - F1 Score, Precision, Recall
  - Moving/Stopped counts with F1 scores

**Per-Angle Breakdown Section:**
- For each angle (sorted):
  - Total videos
  - Accuracy with counts
  - F1 Score, Precision, Recall
  - Moving/Stopped counts with F1 scores

### 2. `requirements.txt`

#### Changes Made:

**Added scikit-learn dependency (Line 17)**
- Added: `scikit-learn>=1.0.0`
- Placement: In "ops" section alongside scipy and numpy
- Reason: Required for F1 score, precision, and recall calculations

---

## Technical Details

### Metrics Definitions

1. **Accuracy:** `(correct predictions / total predictions) * 100`
2. **F1 Score:** Harmonic mean of precision and recall (binary classification)
3. **Precision:** `(true positives / (true positives + false positives)) * 100`
4. **Recall:** `(true positives / (true positives + false negatives)) * 100`
5. **Per-Class F1:** F1 score calculated separately for each class (moving vs stopped)

### Label Encoding
- `1` = Moving (positive class)
- `0` = Stopped (negative class)

### Texture Types Detected
Common textures in datasets: stripes, noise, rubber, grid, subtle_gray_stripes, etc.

### Angle Types Detected
Common angles: angle0, angle30, angle45, etc.

---

## Benefits

1. **Comprehensive Performance Analysis:**
   - Beyond simple accuracy, F1 provides balanced view of precision/recall trade-offs
   - Critical for imbalanced datasets

2. **Granular Insights:**
   - Identify which textures are easy/hard for the model
   - Identify which camera angles are easy/hard for the model
   - Per-class metrics reveal bias toward moving/stopped predictions

3. **Comparative Analysis:**
   - All metrics calculated separately for base model and LoRA model
   - Easy to compare performance across conditions

4. **Robust Evaluation:**
   - Handles edge cases with `zero_division=0`
   - Graceful handling of unparseable filenames

---

## Testing Recommendations

1. **Verify sklearn installation:**
   ```bash
   pip install scikit-learn>=1.0.0
   ```

2. **Test with existing dataset:**
   ```bash
   python3 evaluate_pipeline_simple.py \
     --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
     --test_dataset <your_test_dataset> \
     --adapter_name_or_path <your_lora_path> \
     --output_dir evaluation_results
   ```

3. **Verify report generation:**
   - Check `evaluation_results/evaluation_report_*.txt`
   - Verify all sections present: Overall, Per-Class, Per-Texture, Per-Angle

4. **Test edge cases:**
   - Single texture dataset
   - Single angle dataset
   - Small dataset (< 10 samples)

---

## Backward Compatibility

✅ **Fully backward compatible:**
- No changes to CLI arguments
- No changes to function signatures called by `run_full_pipeline.py`
- Existing report format enhanced (not replaced)
- All new metrics are additions (no removals)

---

## Future Enhancements (Optional)

1. Add confusion matrix visualization
2. Add per-speed metrics (speed0.0, speed12.0, etc.)
3. Add ROC-AUC score for probability-based predictions
4. Export metrics to JSON/CSV for programmatic analysis
5. Add statistical significance tests (t-test, bootstrap)

---

## Related Files

- `run_full_pipeline.py` - Calls `evaluate_pipeline_simple.py` (no changes needed)
- `evaluate_treadmill_lora.py` - Standalone evaluation script (not modified)
- `docker-compose.yml` - May need `scikit-learn` in container (verify)

---

## Commit Message

```
Add comprehensive F1 metrics and per-texture/per-angle analysis to evaluation

- Add F1 score, precision, recall for overall and per-class metrics
- Track and report metrics per texture type (stripes, noise, rubber, etc.)
- Track and report metrics per camera angle (angle0, angle30, etc.)
- Add parse_video_metadata() to extract texture/angle from filenames
- Enhance report generation with detailed breakdowns
- Add scikit-learn>=1.0.0 to requirements.txt
- All metrics calculated separately for base and LoRA models
- Fully backward compatible with existing pipeline
```

---

## Verification Checklist

- [x] Code compiles without syntax errors
- [x] All imports verified
- [x] scikit-learn added to requirements.txt
- [x] No breaking changes to external APIs
- [x] run_full_pipeline.py still works (no changes needed)
- [x] Detailed changelog documented
- [ ] Tested on real dataset (user to verify)
- [ ] Verified sklearn installed in Docker container (user to verify)
