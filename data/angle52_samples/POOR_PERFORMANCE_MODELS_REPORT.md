======================================================================
POOR PERFORMANCE MODELS - COMPREHENSIVE ANALYSIS REPORT
======================================================================

Report Generated: 2025-11-30
Analysis Period: November 27, 2025
Number of Models Analyzed: 4 failed checkpoints

======================================================================
EXECUTIVE SUMMARY
======================================================================

This report documents 4 model checkpoints that performed 20% or worse
than the base model during evaluation. All failed models exhibit the
same pathological behavior: heavy bias toward predicting "moving" and
near-total inability to detect stopped treadmills.

Common Training Issues Identified:
- Insufficient training steps (6-80 steps)
- Limited texture diversity in training data
- Possible overfitting to motion patterns
- Direction-specific training bias

======================================================================
MODEL 1: qwen2vl-treadmill-lora-v2
======================================================================

PERFORMANCE DEGRADATION: -21.88%
Evaluation Date: November 27, 2025 at 10:49
Checkpoint Creation: November 27, 2025 at 10:47 (2 minutes before eval)

---
TRAINING CONFIGURATION
---
Training Dataset: high_quality_combo2_train
Base Model: Qwen/Qwen2.5-VL-3B-Instruct
Training Method: LoRA (Low-Rank Adaptation)

Training Hyperparameters:
  - Learning Rate: 5e-05
  - Batch Size: 1 (effective: 8 with gradient accumulation)
  - Gradient Accumulation Steps: 8
  - Optimizer: AdamW (betas=0.9,0.999, epsilon=1e-08)
  - LR Scheduler: Cosine with 10% warmup
  - Number of Epochs: 1
  - Total Steps: 6 (VERY LOW - Major Red Flag)
  - Training Loss: 0.1947
  - Mixed Precision: Native AMP
  - Seed: 42

Training Dataset Details:
  - Total Videos: 48
  - Moving: 24 (50%)
  - Stopped: 24 (50%)
  - Speed: 14.0 px/frame (fixed)
  - Textures: factory_dark (12), factory_dark_stripes (12),
              noise (12), stripes (12)
  - Directions: left (24), right (24)
  - Size: 34.37 MB (0.72 MB avg per video)
  - Generation Date: Nov 27 10:31

---
TEST CONFIGURATION
---
Test Dataset: high_quality_combo2_test

Test Dataset Details:
  - Total Videos: 32
  - Moving: 16 (50%)
  - Stopped: 16 (50%)
  - Speed: 14.0 px/frame (fixed)
  - Textures: factory_dark (8), factory_dark_stripes (8),
              noise (8), stripes (8)
  - Directions: left (16), right (16)
  - Size: 24.46 MB (0.76 MB avg per video)
  - Generation Date: Nov 27 10:32

---
EVALUATION RESULTS
---
Base Model Performance:
  - Accuracy: 81.25% (26/32 correct)
  - Moving: 14/16 (87.5%)
  - Stopped: 12/16 (75%)

Fine-Tuned Model Performance:
  - Accuracy: 59.38% (19/32 correct) ⚠️
  - Moving: 16/16 (100%) ✓
  - Stopped: 3/16 (18.75%) ✗ CATASTROPHIC FAILURE

Performance Change: -21.88%

---
ROOT CAUSE ANALYSIS
---
1. INSUFFICIENT TRAINING: Only 6 training steps is far too low
2. BIAS TOWARD MOTION: Model learned to always predict "moving"
3. STOPPED DETECTION COLLAPSE: Lost ability to detect stationary belts
4. TRAIN/TEST MISMATCH: Despite similar distributions, model failed

======================================================================
MODEL 2: qwen2vl-treadmill-lora-pipeline/checkpoint-80
======================================================================

PERFORMANCE DEGRADATION: -29.69%
Evaluation Date: November 27, 2025 at 17:13
Checkpoint Creation: November 27, 2025 at 17:01 (12 minutes before eval)

---
TRAINING CONFIGURATION
---
Training Dataset: preformat_right_only_20251130_141204
Base Model: Qwen/Qwen2.5-VL-3B-Instruct
Training Method: LoRA (Low-Rank Adaptation)

Training Hyperparameters:
  - Learning Rate: 5e-05
  - Batch Size: 1 (effective: 8 with gradient accumulation)
  - Gradient Accumulation Steps: 8
  - Optimizer: AdamW (betas=0.9,0.999, epsilon=1e-08)
  - LR Scheduler: Cosine with 10% warmup
  - Number of Epochs: 5
  - Checkpoint Step: 80
  - Mixed Precision: Native AMP
  - Seed: 42

Training Dataset Details:
  - Total Videos: 72
  - Moving: 36 (50%)
  - Stopped: 36 (50%)
  - Speed: 14.0 px/frame (fixed)
  - Textures: subtle_gray_stripes (72) - SINGLE TEXTURE ONLY
  - Directions: right (72) - SINGLE DIRECTION ONLY ⚠️
  - Size: 4.20 MB (0.06 MB avg per video)
  - Generation Date: Nov 30 12:12

---
TEST CONFIGURATION
---
Test Dataset: subtle_gray_challenge_test

Test Dataset Details:
  - Total Videos: 128
  - Moving: 64 (50%)
  - Stopped: 64 (50%)
  - Speed: 14.0 px/frame (fixed)
  - Textures: subtle_gray_stripes (128) - SINGLE TEXTURE
  - Directions: left (128) - OPPOSITE DIRECTION FROM TRAINING ⚠️
  - Size: 7.37 MB (0.06 MB avg per video)
  - Generation Date: Nov 27 16:40

---
EVALUATION RESULTS
---
Base Model Performance:
  - Accuracy: 100.00% (128/128 correct) ✓ PERFECT
  - Moving: 64/64 (100%)
  - Stopped: 64/64 (100%)

Fine-Tuned Model Performance:
  - Accuracy: 70.31% (90/128 correct) ⚠️
  - Moving: 64/64 (100%) ✓
  - Stopped: 26/64 (40.6%) ✗ SEVERE DEGRADATION

Performance Change: -29.69%

---
ROOT CAUSE ANALYSIS
---
1. DIRECTION MISMATCH: Trained on RIGHT only, tested on LEFT only
2. SINGLE TEXTURE OVERFITTING: No texture diversity in training
3. MOTION DETECTION BIAS: Model became biased toward predicting motion
4. GENERALIZATION FAILURE: Could not generalize to opposite direction
5. BASE MODEL WAS PERFECT: Fine-tuning made it significantly worse

======================================================================
MODEL 3: qwen2vl-treadmill-lora-pipeline/checkpoint-6 (First Eval)
======================================================================

PERFORMANCE DEGRADATION: -48.61% (WORST PERFORMER)
Evaluation Date: November 27, 2025 at 17:58
Checkpoint Creation: November 27, 2025 at 17:53 (5 minutes before eval)

---
TRAINING CONFIGURATION
---
Training Dataset: preformat_right_only_20251130_141204
Base Model: Qwen/Qwen2.5-VL-3B-Instruct
Training Method: LoRA (Low-Rank Adaptation)

Training Hyperparameters:
  - Learning Rate: 5e-05
  - Batch Size: 1 (effective: 8 with gradient accumulation)
  - Gradient Accumulation Steps: 8
  - Optimizer: AdamW (betas=0.9,0.999, epsilon=1e-08)
  - LR Scheduler: Cosine with 10% warmup
  - Number of Epochs: 5
  - Checkpoint Step: 6 (EXTREMELY LOW - Major Red Flag)
  - Mixed Precision: Native AMP
  - Seed: 42

Training Dataset Details:
  - Total Videos: 72
  - Moving: 36 (50%)
  - Stopped: 36 (50%)
  - Speed: 14.0 px/frame (fixed)
  - Textures: subtle_gray_stripes (72) - SINGLE TEXTURE ONLY
  - Directions: right (72) - SINGLE DIRECTION ONLY ⚠️
  - Size: 4.20 MB (0.06 MB avg per video)
  - Generation Date: Nov 30 12:12

---
TEST CONFIGURATION
---
Test Dataset: subtle_gray_20251127_193422_test

Test Dataset Details:
  - Total Videos: 72
  - Moving: 36 (50%)
  - Stopped: 36 (50%)
  - Speed: 14.0 px/frame (fixed)
  - Textures: subtle_gray_stripes (72) - SINGLE TEXTURE
  - Directions: right (72) - SAME DIRECTION AS TRAINING ✓
  - Size: 4.27 MB (0.06 MB avg per video)
  - Generation Date: Nov 27 17:44

---
EVALUATION RESULTS
---
Base Model Performance:
  - Accuracy: 100.00% (72/72 correct) ✓ PERFECT
  - Moving: 36/36 (100%)
  - Stopped: 36/36 (100%)

Fine-Tuned Model Performance:
  - Accuracy: 51.39% (37/72 correct) ✗ CATASTROPHIC
  - Moving: 36/36 (100%) ✓
  - Stopped: 1/36 (2.8%) ✗✗✗ COMPLETE FAILURE

Performance Change: -48.61% (WORST DEGRADATION)

---
ROOT CAUSE ANALYSIS
---
1. EXTREME UNDER-TRAINING: Only 6 checkpoint steps is catastrophically low
2. STOPPED DETECTION DESTROYED: Only 1/36 stopped videos detected
3. ALWAYS PREDICTS MOTION: Model learned to output "moving" for everything
4. EARLY CHECKPOINT FAILURE: This is an extremely early/unstable checkpoint
5. BASE MODEL PERFECT: Fine-tuning completely broke the model
6. SAME DISTRIBUTION: Even with matching train/test, model failed badly

======================================================================
MODEL 4: qwen2vl-treadmill-lora-pipeline/checkpoint-6 (Second Eval)
======================================================================

PERFORMANCE DEGRADATION: -35.16%
Evaluation Date: November 27, 2025 at 18:15
Checkpoint Creation: November 27, 2025 at 17:53 (22 minutes before eval)
Note: Same checkpoint as Model 3, different test dataset

---
TRAINING CONFIGURATION
---
Training Dataset: preformat_right_only_20251130_141204
[Same as Model 3 - see above]

---
TEST CONFIGURATION
---
Test Dataset: subtle_gray_train_20251127_193422
Note: This appears to be a TRAINING dataset used for testing

Test Dataset Details:
  - Total Videos: 128
  - Moving: 64 (50%)
  - Stopped: 64 (50%)
  - Speed: 14.0 px/frame (fixed)
  - Textures: subtle_gray_stripes (128) - SINGLE TEXTURE
  - Directions: right (128) - SAME DIRECTION AS TRAINING ✓
  - Size: 7.47 MB (0.06 MB avg per video)
  - Generation Date: Nov 27 17:42

---
EVALUATION RESULTS
---
Base Model Performance:
  - Accuracy: 98.44% (126/128 correct) ✓ NEAR PERFECT
  - Moving: 63/64 (98.4%)
  - Stopped: 63/64 (98.4%)

Fine-Tuned Model Performance:
  - Accuracy: 63.28% (81/128 correct) ⚠️
  - Moving: 64/64 (100%) ✓
  - Stopped: 17/64 (26.6%) ✗ SEVERE FAILURE

Performance Change: -35.16%

---
ROOT CAUSE ANALYSIS
---
1. SAME CHECKPOINT AS MODEL 3: checkpoint-6 consistently fails
2. TESTING ON TRAIN SET: Evaluated on what looks like training data
3. STILL FAILED BADLY: Even on similar distribution, severe degradation
4. STOPPED DETECTION COLLAPSE: Only 17/64 stopped videos detected
5. MOTION BIAS PERSISTS: Same pathological pattern as Model 3

======================================================================
CROSS-MODEL ANALYSIS
======================================================================

COMMON FAILURE PATTERNS:
1. All models detect moving belts perfectly (100% recall)
2. All models catastrophically fail at detecting stopped belts
3. All models were severely under-trained (6-80 steps)
4. All models show heavy bias toward predicting "moving"

TRAINING ISSUES:
- Models 2-4 trained on single direction only (right)
- Models 2-4 trained on single texture only (subtle_gray_stripes)
- Model 1 had only 6 total training steps (far too low)
- Checkpoint-6 (Models 3-4) is an extremely early/unstable checkpoint

DATASET ISSUES:
- Direction mismatch: Model 2 trained on right, tested on left
- Limited diversity: Single texture/direction training fails to generalize
- Test/train confusion: Model 4 tested on what appears to be training data

PERFORMANCE COMPARISON:
- Model 1: -21.88% (stopped: 3/16 = 18.75%)
- Model 2: -29.69% (stopped: 26/64 = 40.6%)
- Model 4: -35.16% (stopped: 17/64 = 26.6%)
- Model 3: -48.61% (stopped: 1/36 = 2.8%) ← WORST

======================================================================
RECOMMENDATIONS
======================================================================

IMMEDIATE ACTIONS:
1. NEVER use checkpoint-6 from any training run (too early/unstable)
2. INCREASE training steps significantly (minimum 100+ steps)
3. REQUIRE texture diversity in training data (4+ textures)
4. REQUIRE direction diversity in training data (2+ directions)
5. VALIDATE checkpoints at step 20, 40, 80, 160 before final selection

TRAINING IMPROVEMENTS:
1. Use balanced datasets with multiple textures and directions
2. Implement early stopping based on validation accuracy
3. Monitor stopped detection accuracy as primary metric
4. Add class weighting to prevent motion bias
5. Train for sufficient epochs (5+ epochs on diverse data)

EVALUATION IMPROVEMENTS:
1. Test on held-out data with different distributions than training
2. Report per-class metrics (not just overall accuracy)
3. Flag models with stopped detection < 80% as failures
4. Implement confusion matrix analysis
5. Test generalization across directions and textures

DATA COLLECTION:
1. Create diverse training sets with:
   - 4+ texture types
   - 4+ directions (left, right, up, down)
   - Multiple camera angles (0°, 15°, 30°, 45°, 60°)
   - Varied speeds (0.0 to 14.0 px/frame)
2. Ensure 50/50 balance of moving/stopped in all splits
3. Generate large training sets (500+ videos minimum)

======================================================================
APPENDIX: FILE LOCATIONS
======================================================================

EVALUATION REPORTS:
- Model 1: /app/evaluation_results_20251127_104659/evaluation_report_20251127_104754.txt
- Model 2: /app/evaluation_results_20251127/evaluation_report_20251127_170404.txt
- Model 3: /app/evaluation_results_20251127_174746/evaluation_report_20251127_175358.txt
- Model 4: /app/eval_train_finetuned/evaluation_report_20251127_180737.txt

MODEL CHECKPOINTS:
- Model 1: /app/saves/qwen2vl-treadmill-lora-v2/adapter_model.safetensors
- Model 2: /app/saves/qwen2vl-treadmill-lora-pipeline/checkpoint-80/adapter_model.safetensors
- Model 3: /app/saves/qwen2vl-treadmill-lora-pipeline/checkpoint-6/adapter_model.safetensors
- Model 4: [Same as Model 3]

TRAINING DATASETS:
- Model 1: /app/data/high_quality_combo2_train/
- Models 2-4: /app/data/preformat_right_only_20251130_141204/

TEST DATASETS:
- Model 1: /app/data/high_quality_combo2_test/
- Model 2: /app/data/subtle_gray_challenge_test/
- Model 3: /app/data/subtle_gray_test_20251127_193422/
- Model 4: /app/data/subtle_gray_train_20251127_193422/

======================================================================
END OF REPORT
======================================================================
