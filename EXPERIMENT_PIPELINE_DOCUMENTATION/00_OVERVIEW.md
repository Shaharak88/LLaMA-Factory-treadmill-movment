# Experiment Pipeline Overview

## Purpose
This pipeline automates the entire workflow for training and evaluating a Qwen2.5-VL vision-language model to detect treadmill motion in synthetic videos. The system generates synthetic treadmill videos, trains a LoRA adapter, evaluates both base and fine-tuned models, and generates comprehensive HTML reports.

## Architecture

The pipeline consists of 5 main stages:

```
┌─────────────────────────────────────────────────────────────────┐
│                     1. ORCHESTRATION                            │
│                    (run_experiment.sh)                          │
│  - Syncs code to GPU server                                    │
│  - Coordinates all pipeline stages                             │
│  - Retrieves results back to local machine                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   2. DATA GENERATION                            │
│   building_dataset.py → synthetic_data_generation.py            │
│  - Generates synthetic treadmill videos                        │
│  - Creates train/test JSON datasets                            │
│  - Updates dataset_info.json                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    3. TRAINING                                  │
│    run_full_pipeline.py → llamafactory-cli                      │
│  - Trains LoRA adapter on Qwen2.5-VL                          │
│  - Logs experiment parameters (experiment_tracker.py)          │
│  - Saves model checkpoints                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    4. EVALUATION                                │
│              evaluate_pipeline_simple.py                        │
│  - Evaluates base model (Qwen2.5-VL)                          │
│  - Evaluates fine-tuned model (LoRA adapter)                   │
│  - Generates per-video predictions and metrics                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  5. REPORT GENERATION                           │
│           generate_experiment_report.py                         │
│  - Downloads videos from server                                │
│  - Generates interactive HTML report                           │
│  - Shows predictions, metrics, and video visualizations        │
└─────────────────────────────────────────────────────────────────┘
```

## Key Files

### Orchestration
- `run_experiment.sh` - Main entry point; coordinates entire pipeline

### Data Generation
- `building_dataset.py` - Dataset builder; orchestrates video generation
- `data/synthetic_treadmill/synthetic_data_generation.py` - Generates synthetic videos using Blender

### Training
- `run_full_pipeline.py` - Training orchestrator; calls LLaMA-Factory
- `experiment_tracker.py` - CSV-based experiment logging
- `llamafactory-cli` - LLaMA-Factory's CLI for training (external)

### Evaluation
- `evaluate_pipeline_simple.py` - Model evaluation with transformers

### Reporting
- `generate_experiment_report.py` - Interactive HTML report generator
- `dual_dataset_review.py` - Helper for downloading and processing videos (imported by report generator)

### Configuration
- `examples/train_qlora/qwen25vl_lora_pipeline_*.yaml` - Training configurations
- `data/dataset_info.json` - Dataset registry

### Data Storage
- `data/` - All datasets, videos, and metadata
- `data/{dataset_name}_train/` - Training videos
- `data/{dataset_name}_test/` - Test videos with evaluation results
- `saves/` - Trained model checkpoints
- `analytics/reports/` - Generated HTML reports

## Execution Flow

1. **User runs**: `./run_experiment.sh`
2. **Code sync**: rsync copies code to GPU server
3. **Docker execution on server**:
   - Stage 1: Generate dataset (building_dataset.py)
   - Stage 2: Train model (run_full_pipeline.py)
4. **Results retrieval**: rsync copies results back to local
5. **Local report generation**: generate_experiment_report.py creates HTML

## Key Design Decisions

### Why Docker?
- Consistent environment across local and remote machines
- GPU access via NVIDIA runtime
- Isolated dependencies

### Why rsync?
- Efficient file syncing (only changed files)
- Preserves directory structure
- Handles large video files

### Why separate train/test folders?
- Clear data separation
- Evaluation results stored with test data
- Easy to track which videos were used for testing

### Why CSV for experiment tracking?
- Human-readable and Excel-compatible
- Easy to aggregate results across experiments
- No database dependency

## Common Use Cases

### Run full experiment with default parameters
```bash
./run_experiment.sh
```

### Skip dataset generation (reuse existing dataset)
Add `--skip_dataset` to run_full_pipeline.py call in run_experiment.sh

### Change evaluation method
Edit `--eval_method` parameter (choices: 'yesno', 'moving_stopped')

### Modify training hyperparameters
Edit YAML config file or pass arguments to run_full_pipeline.py

## Output Locations

### On GPU Server (during execution)
- `/workspace/data/` - Datasets and videos
- `/workspace/saves/` - Model checkpoints
- `/workspace/evaluation_results/` - Evaluation outputs

### On Local Machine (after retrieval)
- `data/{dataset_name}_train/` - Training videos
- `data/{dataset_name}_test/` - Test videos
- `data/{dataset_name}_test/eval_{timestamp}/` - Evaluation results
- `saves/{model_name}/` - Model checkpoints
- `analytics/reports/` - HTML reports
- `experiments_log.csv` - Experiment tracking log

## Next Steps
- Read `01_ORCHESTRATION.md` for run_experiment.sh details
- Read `02_DATA_GENERATION.md` for video generation details
- Read `03_TRAINING.md` for training pipeline details
- Read `04_EVALUATION.md` for evaluation details
- Read `05_REPORTING.md` for report generation details
