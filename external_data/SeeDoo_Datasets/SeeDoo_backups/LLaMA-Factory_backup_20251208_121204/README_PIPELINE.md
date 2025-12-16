# Treadmill Detection Pipeline

This directory contains a complete end-to-end pipeline for training and evaluating treadmill motion detection models using Qwen2.5-VL.

## Quick Start

### Full Pipeline (Dataset + Training + Evaluation)

```bash
python3 run_full_pipeline.py \
  --dataset_name my_experiment \
  --num_videos 100 \
  --vary_parameters \
  --seed 42
```

This single command will:
1. Generate 90 training videos + 10 test videos
2. Train a LoRA adapter on Qwen2.5-VL-3B
3. Evaluate both base and fine-tuned models
4. Generate accuracy comparison report

### Inside Docker Container (Recommended)

The pipeline is designed to run **inside the Docker container**:

```bash
docker exec llamafactory python3 /app/run_full_pipeline.py \
  --dataset_name docker_experiment \
  --num_videos 50 \
  --vary_parameters
```

### Outside Docker Container

If running from outside the container, use `--use_docker` flag:

```bash
python3 run_full_pipeline.py \
  --dataset_name local_experiment \
  --num_videos 50 \
  --use_docker
```

**Note:** By default, the pipeline assumes it's running **inside** the container. Only use `--use_docker` when running from the host machine.

## Pipeline Components

### 1. `run_full_pipeline.py` - Master Orchestration Script

Automates the complete workflow:
- **Dataset Generation**: Creates separate train/test sets (90%/10% split)
- **Model Training**: Fine-tunes LoRA adapter using docker-compose
- **Evaluation**: Compares base vs fine-tuned model on test set

**Key Features:**
- Exposes all sub-script arguments with defaults
- Supports skipping individual steps
- Only calls existing scripts (no implementation logic)
- Automatically creates training configs

**Common Arguments:**
```bash
--dataset_name          # Base name for datasets (required)
--num_videos           # Total videos to generate (default: 100)
--train_split          # Train ratio (default: 0.9 = 90% train, 10% test)
--vary_parameters      # Enable parameter variation for diversity
--seed                 # Random seed (default: 42)

# Skip flags
--skip_dataset         # Skip dataset generation
--skip_training        # Skip training
--skip_evaluation      # Skip evaluation

# Dataset parameters
--texture_type         # Comma-separated textures (e.g., stripes,noise,rubber)
--speed_range          # Speed range (default: 1.0,8.0)
--resolution          # Video resolution (default: 640x480)

# Training parameters
--lora_rank           # LoRA rank (default: 8)
--learning_rate       # Learning rate (default: 5.0e-5)
--num_train_epochs    # Epochs (default: 1)
```

### 2. `evaluate_pipeline.py` - Evaluation Script

Evaluates models using vLLM inference:
- Runs inference on both base and fine-tuned models
- Calculates accuracy by exact match comparison
- Generates detailed evaluation reports

**Usage:**
```bash
python3 evaluate_pipeline.py \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora \
  --test_dataset my_experiment_test
```

**Key Features:**
- Uses vllm_infer.py for scalable inference
- Validates predictions against ground truth
- Class-specific accuracy (moving vs stopped)
- Comparison reports with improvement metrics

### 3. `PIPELINE_CHANGELOG.md` - Documentation

Comprehensive documentation including:
- Design decisions and rationale
- Integration with existing code
- Testing recommendations
- Known limitations and future enhancements

## Usage Examples

### 1. Basic Pipeline

Generate 100 videos with varied parameters:

```bash
python3 run_full_pipeline.py \
  --dataset_name basic_test \
  --num_videos 100 \
  --vary_parameters
```

### 2. Custom Texture Combinations

Test specific texture types:

```bash
python3 run_full_pipeline.py \
  --dataset_name texture_test \
  --texture_type stripes,noise,rubber,grid \
  --speed_range 2.0,5.0
```

### 3. Skip Dataset Generation

Use existing datasets:

```bash
python3 run_full_pipeline.py \
  --dataset_name existing_exp \
  --skip_dataset \
  --num_videos 100
```

### 4. Only Evaluation

Evaluate an existing model:

```bash
python3 evaluate_pipeline.py \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter_name_or_path saves/qwen2vl-treadmill-lora \
  --test_dataset test_dataset_name
```

### 5. High-Quality Training

More epochs and larger dataset:

```bash
python3 run_full_pipeline.py \
  --dataset_name high_quality \
  --num_videos 500 \
  --vary_parameters \
  --num_train_epochs 3 \
  --learning_rate 1.0e-4
```

## Output Structure

After running the pipeline, you'll have:

```
LLaMA-Factory/
├── data/
│   ├── <dataset_name>_train/          # Training videos
│   ├── <dataset_name>_test/           # Test videos
│   ├── <dataset_name>_train.json      # Training dataset JSON
│   └── <dataset_name>_test.json       # Test dataset JSON
│
├── saves/
│   └── qwen2vl-treadmill-lora-pipeline/  # Trained LoRA adapter
│
└── evaluation_results_<timestamp>/
    ├── predictions_base_<timestamp>.jsonl      # Base model predictions
    ├── predictions_lora_<timestamp>.jsonl      # LoRA model predictions
    └── evaluation_report_<timestamp>.txt       # Comparison report
```

## Evaluation Report Format

The evaluation report includes:

```
==================================================================
TREADMILL MOTION DETECTION - EVALUATION REPORT
==================================================================

BASE MODEL RESULTS
------------------------------------------------------------------
Overall Accuracy: 65.00% (13/20)

Moving Videos:
  Total: 10
  Correct: 7
  Accuracy: 70.00%

Stopped Videos:
  Total: 10
  Correct: 6
  Accuracy: 60.00%

FINE-TUNED MODEL (LoRA) RESULTS
------------------------------------------------------------------
Overall Accuracy: 95.00% (19/20)

Moving Videos:
  Total: 10
  Correct: 10
  Accuracy: 100.00%

Stopped Videos:
  Total: 10
  Correct: 9
  Accuracy: 90.00%

==================================================================
COMPARISON
==================================================================

Base Model Accuracy:       65.00%
Fine-Tuned Model Accuracy: 95.00%
Improvement:               +30.00%

Status: ✓ Fine-tuning IMPROVED performance
```

## Train/Test Split

The pipeline ensures proper separation:

- **Training Set (90%)**: Used for LoRA fine-tuning and validation during training
- **Test Set (10%)**: Completely held out, only used in final evaluation
- **Different Seeds**: Train and test use different seeds for diversity

## Accuracy Calculation

The evaluation uses exact match comparison:

- **Moving Detection**: Prediction contains "moving" (but not "not moving")
- **Stopped Detection**: Prediction contains "stopped", "stationary", or "not moving"
- **Ground Truth**: Extracted from dataset labels

## Tips

1. **Start Small**: Test with 10-20 videos first to verify pipeline works
2. **Use --vary_parameters**: Creates more diverse and robust training data
3. **Adjust train_split**: Use 0.8 for smaller datasets to get more test samples
4. **Monitor Training**: Check tensorboard logs in the output directory
5. **Batch Size**: Reduce if you encounter OOM errors during evaluation

## Troubleshooting

### "No videos generated"
- Check that synthetic_data_generation.py is working
- Verify output directory permissions
- Check logs in `data/<dataset_name>/<dataset_name>_build_log.txt`

### "Training failed"
- Ensure Docker container has GPU access
- Check available VRAM (needs ~8GB for 4-bit quantization)
- Review training logs in the output directory

### "Evaluation timeout"
- Reduce --eval_batch_size for large test sets
- Check GPU memory availability
- Verify test dataset exists in dataset_info.json

## Advanced Usage

### Custom Training Configuration

Modify training parameters:

```bash
python3 run_full_pipeline.py \
  --dataset_name custom_train \
  --num_videos 200 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --learning_rate 1.0e-4 \
  --num_train_epochs 2 \
  --gradient_accumulation_steps 16
```

### Multiple Experiments

Run multiple experiments with different seeds:

```bash
for seed in 42 123 456; do
  python3 run_full_pipeline.py \
    --dataset_name experiment_seed_${seed} \
    --num_videos 100 \
    --seed ${seed}
done
```

## Requirements

- Python 3.8+
- Docker and docker-compose (for training)
- CUDA-compatible GPU (8GB+ VRAM recommended)
- All dependencies from LLaMA-Factory

## See Also

- `PIPELINE_CHANGELOG.md` - Detailed implementation notes
- `building_dataset.py` - Dataset generation script
- `scripts/vllm_infer.py` - Inference script
- `examples/train_qlora/qwen25vl_lora_sft.yaml` - Training config template
