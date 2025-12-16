# Changelog: Question Format Change to Yes/No + Verbose Logging

**Date:** 2025-11-29
**Author:** AI Assistant
**Scope:** Changed question/answer format and enhanced evaluation logging

---

## Summary

Changed the dataset generation and evaluation pipeline to use a simpler yes/no question format instead of asking about "moving or stopped". This provides clearer instructions to the model and makes answer detection more robust. Additionally, added verbose logging to print EVERY video prediction during evaluation for transparency and debugging.

---

## Motivation

1. **Clearer Instructions:** "Answer only with yes or no" is more direct than "moving or stopped"
2. **Simpler Responses:** Binary yes/no format is easier for models to follow
3. **Better Debugging:** Verbose logging ensures we can verify model predictions on every single video
4. **Robustness:** Enhanced answer detection handles various response formats (yes, Yes, yes., YES, etc.)

---

## Files Modified

### 1. `building_dataset.py`

#### Changes Made (Lines 563-569):

**Old Format:**
```python
user_prompt = "<video>Analyze this video. Is the treadmill belt moving or stopped?"

if is_moving:
    assistant_response = "The treadmill belt is moving."
else:
    assistant_response = "The treadmill belt is stopped."
```

**New Format:**
```python
user_prompt = "<video>Is there movement in the video? Answer only with yes or no."

if is_moving:
    assistant_response = "Yes."
else:
    assistant_response = "No."
```

**Impact:**
- All future datasets generated will use yes/no format
- Existing datasets keep their format (no retroactive changes)
- Training labels changed: "Yes." = moving, "No." = stopped

---

### 2. `evaluate_pipeline_simple.py`

#### a) Question Update (Line 166):

**Old:**
```python
{"type": "text", "text": "Analyze this video. Is the treadmill belt moving or stopped?"}
```

**New:**
```python
{"type": "text", "text": "Is there movement in the video? Answer only with yes or no."}
```

**Impact:**
- Evaluation uses same question format as training data
- Ensures consistency between training and inference

---

#### b) Enhanced Answer Detection (Lines 297-330):

**Complete Rewrite of `_is_moving()` Method:**

**New Logic (Priority Order):**

1. **Primary Detection - Yes/No Format:**
   - `text.startswith('yes')` → Moving (True)
   - `text == 'yes'` or `text == 'yes.'` → Moving (True)
   - `text.startswith('no')` → Stopped (False)
   - `text == 'no'` or `text == 'no.'` → Stopped (False)

2. **Robust Yes/No Detection:**
   - Contains "yes" but not "no" → Check for conflicting keywords
   - Avoids false positives like "yes, the belt is stopped"
   - Contains "no" but not "yes" → Stopped (False)

3. **Legacy Keyword Support (Backward Compatibility):**
   - "moving" in text (but not "not moving" or "stopped") → Moving (True)
   - "stopped" or "stationary" or "not moving" → Stopped (False)

4. **Default:**
   - If completely unclear → Stopped (False)

**Benefits:**
- Handles various formats: "yes", "Yes", "yes.", "YES", "Yes, there is movement"
- Prevents false positives from contradictory responses
- Maintains backward compatibility with old datasets
- Graceful handling of unexpected model outputs

---

#### c) Verbose Video-by-Video Logging (Lines 207-217):

**New Feature: Print EVERY video prediction**

```python
# VERBOSE LOGGING: Print EVERY video prediction
video_filename = Path(video_rel_path).name
gt_label = "MOVING (yes)" if is_moving_gt else "STOPPED (no)"
pred_label = "MOVING (yes)" if is_moving_pred else "STOPPED (no)"
status_icon = "✓" if is_correct else "✗"
logger.info(f"  [{i+1}/{len(data)}] {status_icon} {video_filename}")
logger.info(f"      Ground Truth: {gt_label}")
logger.info(f"      Model Output: '{output_text}'")
logger.info(f"      Predicted:    {pred_label}")
logger.info(f"      Texture: {texture}, Angle: {angle}")
logger.info(f"")
```

**Output Format:**
```
  [1/100] ✓ treadmill_0001_stripes_right_speed5.0_angle0_...mp4
      Ground Truth: MOVING (yes)
      Model Output: 'Yes.'
      Predicted:    MOVING (yes)
      Texture: stripes, Angle: angle0

  [2/100] ✗ treadmill_0002_noise_left_speed0.0_angle30_...mp4
      Ground Truth: STOPPED (no)
      Model Output: 'Yes, I see movement.'
      Predicted:    MOVING (yes)
      Texture: noise, Angle: angle30
```

**Benefits:**
- **Complete Transparency:** See every single prediction made by the model
- **Easy Debugging:** Quickly identify which videos the model gets wrong
- **Verify Classification Logic:** Confirm answer detection is working correctly
- **Metadata Visibility:** See texture and angle for each video
- **Visual Status:** ✓ for correct, ✗ for incorrect predictions

**Removed:**
- Old periodic logging: `if (i + 1) % 10 == 0: logger.info(f"Processed {i+1}/{len(data)} samples")`
- Replaced with per-video detailed logging

---

## Backward Compatibility

### ✅ Compatible:
- **Old datasets still work:** Legacy answer detection logic still present
- **No breaking changes:** Evaluation script handles both formats
- **run_full_pipeline.py:** No changes needed

### ⚠️ Note:
- **New datasets incompatible with old evaluation scripts** that only check for "moving"/"stopped"
- Recommendation: Use `evaluate_pipeline_simple.py` for all evaluations (handles both formats)

---

## Testing Recommendations

1. **Generate new dataset:**
   ```bash
   python3 building_dataset.py --dataset_name test_yesno --num_videos 10
   ```
   - Verify questions say "Is there movement in the video? Answer only with yes or no."
   - Verify answers are "Yes." or "No."

2. **Test evaluation on old dataset:**
   ```bash
   python3 evaluate_pipeline_simple.py \
     --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
     --test_dataset <old_dataset> \
     --output_dir test_old
   ```
   - Verify legacy answer detection still works

3. **Test evaluation on new dataset:**
   ```bash
   python3 evaluate_pipeline_simple.py \
     --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
     --test_dataset test_yesno_test \
     --output_dir test_new
   ```
   - Verify yes/no answer detection works
   - Check verbose logging shows all videos

4. **Verify logging:**
   - Check console output for detailed per-video predictions
   - Confirm ✓/✗ icons appear correctly
   - Verify model output is printed verbatim

---

## Example Scenarios Handled

### Scenario 1: Perfect Yes/No
- Model output: "Yes."
- Detection: MOVING ✓

### Scenario 2: Capitalized
- Model output: "YES"
- Detection: MOVING ✓

### Scenario 3: Explanation Added
- Model output: "Yes, there is movement."
- Detection: MOVING ✓

### Scenario 4: Contradictory (Handled)
- Model output: "Yes, but the belt is stopped."
- Detection: STOPPED (caught by keyword check)

### Scenario 5: Simple No
- Model output: "No."
- Detection: STOPPED ✓

### Scenario 6: Legacy Format (Backward Compatible)
- Model output: "The treadmill belt is moving."
- Detection: MOVING ✓

### Scenario 7: Unclear
- Model output: "I'm not sure about this video."
- Detection: STOPPED (default)

---

## Related Files Not Modified

These files contain the old prompt format but are NOT used by `run_full_pipeline.py`:
- `evaluate_bootstrap.py` (standalone script)
- `evaluate_treadmill_lora.py` (standalone script)
- `test_both_models/evaluate_bootstrap.py` (test script)
- `test_both_models/evaluate_treadmill_lora.py` (test script)

If you want to update these for consistency, similar changes would apply.

---

## Commit Message

```
Change question format to yes/no and add verbose evaluation logging

QUESTION FORMAT CHANGES:
- building_dataset.py: Change prompt to "Is there movement in the video? Answer only with yes or no."
- Dataset answers: "Yes." for moving, "No." for stopped
- evaluate_pipeline_simple.py: Use same question format for consistency

ANSWER DETECTION ENHANCEMENTS:
- Rewrite _is_moving() with robust yes/no detection
- Handle various formats: yes, Yes, yes., YES, etc.
- Prevent false positives from contradictory responses
- Maintain backward compatibility with legacy "moving"/"stopped" keywords
- Default to stopped for unclear responses

VERBOSE LOGGING:
- Print EVERY video prediction with detailed information:
  * Video filename
  * Ground truth label (MOVING/STOPPED with yes/no)
  * Raw model output (verbatim)
  * Predicted label
  * Texture and angle metadata
  * ✓/✗ status icon
- Remove old periodic progress logging
- Full transparency for debugging and verification

BENEFITS:
- Clearer instructions to model
- Simpler binary response format
- Complete visibility into model predictions
- Easy identification of misclassifications
- Robust answer parsing with fallbacks

COMPATIBILITY:
- Backward compatible with old datasets
- No changes to external APIs
- run_full_pipeline.py works unchanged

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Future Considerations

1. **Update standalone evaluation scripts** if needed
2. **Consider logging to file** in addition to console for large evaluations
3. **Add summary statistics** at the end showing common error patterns
4. **Export verbose logs to CSV** for analysis in spreadsheet tools
