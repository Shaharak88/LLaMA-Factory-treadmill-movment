# Pipeline Implementation Changelog

**Date:** 2025-11-26
**Author:** Claude Code
**Purpose:** Implement end-to-end pipeline for treadmill detection model training and evaluation

## Summary

Created a complete orchestration pipeline that automates dataset generation, model training, and evaluation for treadmill motion detection using Qwen2.5-VL. The pipeline includes:

1. **Dataset Generation** with automatic train/test split (90%/10%)
2. **LoRA Fine-tuning** on Qwen2.5-VL-3B-Instruct
3. **Evaluation** comparing base model vs fine-tuned model with accuracy metrics

---

## New Files Created

### 1. `run_full_pipeline.py` (Main Orchestration Script)

**Location:** `/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/run_full_pipeline.py`

**Purpose:** Master script that orchestrates the complete workflow from dataset generation through training to evaluation.

**Features:**
- Exposes all arguments from sub-scripts with sensible defaults
- Automatic train/test split (default 90%/10%)
- Only calls existing scripts (no implementation logic)
- Supports skipping individual steps (--skip_dataset, --skip_training, --skip_evaluation)
- Creates custom training configuration YAML automatically
- Designed to run inside Docker container

**Key Responsibilities:**
1. **Step 1 - Dataset Generation:**
   - Calls `building_dataset.py` twice to create separate train and test datasets
   - Uses different seeds for train/test to ensure diversity
   - Test set naming: `<dataset_name>_test`
   - Train set naming: `<dataset_name>_train`

2. **Step 2 - Model Training:**
   - Generates custom training YAML configuration
   - Calls training via `docker-compose run --rm llamafactory llamafactory-cli train <config>`
   - Uses only the training dataset (test set held out)
   - Saves LoRA adapter to specified output directory

3. **Step 3 - Model Evaluation:**
   - Calls `evaluate_pipeline.py` with test dataset
   - Evaluates both base model and fine-tuned model
   - Generates comparison report with accuracy metrics

**Arguments Exposed:**
- Dataset generation: All parameters from `building_dataset.py` (texture_type, speed_range, etc.)
- Training: LoRA rank, alpha, dropout, learning rate, epochs, batch size, etc.
- Evaluation: Max tokens, batch size, video processing parameters

**Usage Example:**
```bash
python3 run_full_pipeline.py \
  --dataset_name my_experiment \
  --num_videos 100 \
  --vary_parameters \
  --seed 42
```

---

### 2. `evaluate_pipeline.py` (Evaluation Script)

**Location:** `/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/evaluate_pipeline.py`

**Purpose:** Evaluates treadmill motion detection models using vLLM inference and calculates accuracy by comparing predictions to ground truth labels.

**Features:**
- Uses `vllm_infer.py` approach (scalable for synthetic data)
- Generates predictions in JSONL format
- Validates predictions against ground truth with exact match comparison
- Calculates overall accuracy and class-specific accuracy (moving vs stopped)
- Supports comparing base model vs fine-tuned model
- Generates detailed evaluation reports

**Key Components:**

1. **Inference Execution:**
   - Runs `scripts/vllm_infer.py` for both base and LoRA models
   - Outputs predictions to timestamped JSONL files
   - Supports all vLLM inference parameters (batch size, video processing, etc.)

2. **Prediction Parsing:**
   - Reads JSONL output from vllm_infer.py
   - Extracts `predict` and `label` fields
   - Handles JSON parsing errors gracefully

3. **Accuracy Calculation:**
   - Determines ground truth from label text ("moving" vs "stopped")
   - Determines prediction from predicted text
   - Exact match comparison using keyword detection
   - Tracks overall accuracy, moving accuracy, stopped accuracy
   - Stores detailed results for each prediction

4. **Report Generation:**
   - Creates comprehensive text report with:
     - Overall accuracy for both models
     - Class-specific accuracy breakdown
     - List of incorrect predictions
     - Model comparison and improvement percentage
   - Saves report to `evaluation_results_<timestamp>/evaluation_report_<timestamp>.txt`

**Label Detection Logic:**
- **Moving:** Prediction contains "moving" (but not "not moving")
- **Stopped:** Prediction contains "stopped", "stationary", or "not moving"
- Defaults to stopped if ambiguous

**Usage Example:**
```bash
python3 evaluate_pipeline.py \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora \
  --test_dataset my_experiment_test \
  --output_dir evaluation_results
```

---

### 3. `PIPELINE_CHANGELOG.md` (This File)

**Location:** `/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/PIPELINE_CHANGELOG.md`

**Purpose:** Documents all changes made to implement the pipeline.

---

## Design Decisions

### 1. Train/Test Split Strategy
- **Decision:** Generate separate datasets with different seeds
- **Rationale:**
  - Ensures test set is completely independent
  - No data leakage between train and test
  - Different seeds provide better diversity
  - Simpler than splitting a single dataset after generation

### 2. Evaluation Approach
- **Decision:** Use vllm_infer.py + accuracy calculation (Option B)
- **Rationale:**
  - More scalable for synthetic datasets
  - Works with JSON format used by LLaMA-Factory
  - Easier to extend to larger test sets
  - Consistent with training data format

### 3. Training Execution
- **Decision:** Call docker-compose directly
- **Rationale:**
  - Consistent with existing setup
  - Leverages existing Docker environment
  - Ensures proper GPU access and dependencies
  - Matches user's existing training workflow

### 4. Argument Exposure
- **Decision:** Expose all sub-script arguments with defaults
- **Rationale:**
  - Maximum flexibility for users
  - No hidden configuration
  - Easy to customize without modifying code
  - Clear documentation via --help

### 5. Modularity
- **Decision:** Keep scripts separate (not monolithic)
- **Rationale:**
  - Can run individual steps independently
  - Easier to debug and test
  - Supports skip flags for partial pipeline runs
  - Maintains separation of concerns

---

## Integration with Existing Code

### Files Used (Not Modified):
- `building_dataset.py` - Called for dataset generation
- `scripts/vllm_infer.py` - Called for inference
- `data/dataset_info.json` - Read for dataset configuration (auto-updated by building_dataset.py)
- `examples/train_qlora/qwen25vl_lora_sft.yaml` - Used as template for training config

### No Modifications Required:
All existing scripts work as-is. The pipeline scripts only orchestrate calls to existing tools.

---

## Testing Recommendations

### 1. Test Dataset Generation
```bash
python3 run_full_pipeline.py \
  --dataset_name test_pipeline \
  --num_videos 10 \
  --skip_training \
  --skip_evaluation
```

### 2. Test Training Only
```bash
python3 run_full_pipeline.py \
  --dataset_name test_pipeline \
  --skip_dataset \
  --skip_evaluation \
  --num_train_epochs 1
```

### 3. Test Evaluation Only
```bash
python3 evaluate_pipeline.py \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --test_dataset test_pipeline_test
```

### 4. Full Pipeline Test
```bash
python3 run_full_pipeline.py \
  --dataset_name full_test \
  --num_videos 20 \
  --num_train_epochs 1 \
  --seed 42
```

---

## Docker Usage

### Running Inside Container
```bash
docker exec llamafactory python3 /app/run_full_pipeline.py \
  --dataset_name docker_experiment \
  --num_videos 50 \
  --vary_parameters
```

### Running with docker-compose
The training step automatically uses docker-compose internally, so no special setup needed.

---

## Known Limitations

1. **Training Command:** Currently uses `docker-compose run`, which assumes Docker is available. May need adjustment for non-Docker environments.

2. **Hardcoded Paths:** Scripts assume they're in the LLaMA-Factory root directory.

3. **Error Recovery:** If one step fails, the entire pipeline stops. Consider adding checkpointing for long runs.

4. **Memory:** Large datasets may require significant memory for vLLM inference. Adjust batch_size if needed.

---

## Future Enhancements

1. **Checkpointing:** Save state between steps to allow resuming
2. **Metrics Dashboard:** Generate HTML/plots for evaluation results
3. **Hyperparameter Search:** Support grid search over training parameters
4. **Multi-run Comparison:** Track multiple experiments and compare results
5. **Video Preview:** Generate sample predictions with video thumbnails

---

## Files Modified

None. All changes are new files added to the repository.

---

## Verification Checklist

- [x] Scripts are executable (`chmod +x`)
- [x] `--help` output is clear and comprehensive
- [x] All arguments have sensible defaults
- [x] Scripts handle errors gracefully
- [x] Logging provides clear progress updates
- [x] Train/test split is properly implemented
- [x] Evaluation correctly compares predictions to ground truth
- [x] Reports are human-readable and informative
- [x] Docker compatibility maintained
- [x] No modifications to existing code required

---

## Conclusion

The pipeline implementation provides a complete, automated workflow for treadmill motion detection model development. It maintains separation of concerns, exposes all necessary configuration, and integrates seamlessly with existing LLaMA-Factory infrastructure.
