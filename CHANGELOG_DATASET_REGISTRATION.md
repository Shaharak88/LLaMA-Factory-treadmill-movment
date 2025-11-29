# Changelog: Register New Yes/No Format Datasets in dataset_info.json

**Date:** 2025-11-29
**Author:** AI Assistant
**Scope:** Dataset registration for training pipeline

---

## Summary

Registered two new datasets (`yesno_subtle_factory_train` and `yesno_subtle_factory_test`) in `dataset_info.json` to enable training and evaluation with the run_full_pipeline.py script.

---

## Datasets Created

### 1. yesno_subtle_factory_train
- **Location:** `data/yesno_subtle_factory_train.json`
- **Videos:** 72 training videos
- **Textures:** subtle_gray_stripes (gray 10,13), factory_dark_stripes, factory_dark
- **Configuration:**
  - Directions: left, right, up, down
  - Angles: 0°, 15°, 30°
  - Speeds: 0.0 (stopped), 14.0 (moving)
  - FPS: 4, Duration: 12 seconds
- **Question Format:** "Is there movement in the video? Answer only with yes or no."
- **Answers:** "Yes." for moving, "No." for stopped

### 2. yesno_subtle_factory_test
- **Location:** `data/yesno_subtle_factory_test.json`
- **Videos:** 72 testing videos
- **Textures:** subtle_gray_stripes (gray 30,33), factory_dark_stripes, factory_dark
- **Configuration:** Same as training but with different gray values for subtle_gray_stripes
- **Purpose:** Test generalization to different visual conditions

---

## File Modified

### `data/dataset_info.json`

**Lines 1028-1055: Added two dataset entries**

```json
"yesno_subtle_factory_train": {
  "file_name": "yesno_subtle_factory_train.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages",
    "videos": "videos"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant"
  }
},
"yesno_subtle_factory_test": {
  "file_name": "yesno_subtle_factory_test.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages",
    "videos": "videos"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant"
  }
}
```

---

## Dataset Format

Both datasets use the **sharegpt** formatting with the following structure:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<video>Is there movement in the video? Answer only with yes or no."
    },
    {
      "role": "assistant",
      "content": "Yes."  // or "No."
    }
  ],
  "videos": ["relative/path/to/video.mp4"]
}
```

---

## Integration with Training Pipeline

These datasets are now ready to be used with:

1. **Direct Training:**
   ```bash
   llamafactory-cli train config.yaml
   # Where config.yaml specifies: dataset: yesno_subtle_factory_train
   ```

2. **Full Pipeline (run_full_pipeline.py):**
   - Pipeline can reference these datasets by name
   - Enables end-to-end training and evaluation

3. **Evaluation (evaluate_pipeline_simple.py):**
   ```bash
   python3 evaluate_pipeline_simple.py \
     --test_dataset yesno_subtle_factory_test \
     --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
     --adapter_name_or_path saves/qwen2vl-treadmill-lora
   ```

---

## Benefits

1. **Comprehensive Multi-Texture Training:**
   - Includes 3 texture types for robust motion detection
   - Tests generalization across visual variations

2. **Yes/No Format Consistency:**
   - Simple binary question format
   - Matches updated evaluation logic from CHANGELOG_YES_NO_FORMAT.md

3. **Pipeline Ready:**
   - Registered in dataset_info.json
   - Compatible with LLaMA-Factory training infrastructure
   - Can be used immediately for training

4. **Test Generalization:**
   - Test set uses different gray values (30,33 vs 10,13)
   - Evaluates model's ability to handle visual variations

---

## Related Changes

This change builds on previous work:
- **CHANGELOG_YES_NO_FORMAT.md** - Question format and answer detection
- **CHANGELOG_F1_METRICS.md** - Enhanced evaluation metrics
- **CHANGELOG_FACTORY_TEXTURE_FIX.md** - Fixed factory texture generation bug

---

## Next Steps

1. **Train LoRA Model:**
   ```bash
   python3 run_full_pipeline.py \
     --dataset_name yesno_subtle_factory \
     --skip_dataset \
     --num_train_epochs 5
   ```

2. **Evaluate Models:**
   - Pipeline automatically runs evaluation after training
   - Outputs detailed per-video predictions
   - Generates F1 scores, precision, recall for both models

---

## Commit Message

```
Register yesno_subtle_factory datasets for training pipeline

- Add yesno_subtle_factory_train entry to dataset_info.json
- Add yesno_subtle_factory_test entry to dataset_info.json
- Both datasets use sharegpt formatting with yes/no question format
- Training set: 72 videos with subtle_gray_stripes (gray 10,13) + factory textures
- Test set: 72 videos with subtle_gray_stripes (gray 30,33) + factory textures
- Ready for use with run_full_pipeline.py and llamafactory-cli

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Verification

- [x] Datasets generated successfully (72 videos each)
- [x] JSON files created in data/ directory
- [x] Entries added to dataset_info.json
- [x] Formatting matches existing dataset entries
- [x] Column names and tags configured correctly
- [ ] Training pipeline tested (next step)
- [ ] Evaluation tested (next step)

---

## End of Changelog
