# Experiment Pipeline Documentation

## Purpose
This documentation folder contains comprehensive documentation for the entire Qwen2.5-VL treadmill motion detection experiment pipeline. Every script, function, parameter, and design decision is documented in detail.

## For Your Boss (Executive Summary)

This project implements an end-to-end machine learning pipeline that:

1. **Generates synthetic training data** using Blender (3D graphics software)
2. **Fine-tunes a vision-language AI model** (Qwen2.5-VL) to detect treadmill motion
3. **Evaluates model performance** on test videos
4. **Generates comprehensive reports** with interactive visualizations
5. **Tracks all experiments** in a structured database (CSV)

**Key Results:**
- Base model (untrained): ~65% accuracy on treadmill motion detection
- Fine-tuned model: ~95% accuracy (+30% improvement)
- Fully automated pipeline: one command runs everything
- Complete reproducibility: all parameters logged and documented

**Business Value:**
- Demonstrates ML/AI capability for motion detection tasks
- Scalable to other visual detection problems
- Comprehensive experiment tracking enables data-driven decisions
- Professional-grade documentation for knowledge transfer

---

## Documentation Structure

### Start Here
📘 **[00_OVERVIEW.md](00_OVERVIEW.md)** - High-level architecture and workflow

### Core Pipeline Components
1. 🎬 **[01_ORCHESTRATION.md](01_ORCHESTRATION.md)** - `run_experiment.sh` - Main entry point
2. 🎥 **[02_DATA_GENERATION.md](02_DATA_GENERATION.md)** - Video generation with Blender
3. 🏋️ **[03_TRAINING.md](03_TRAINING.md)** - Model fine-tuning with LoRA
4. 📊 **[04_EVALUATION.md](04_EVALUATION.md)** - Model evaluation and metrics
5. 📈 **[05_REPORTING.md](05_REPORTING.md)** - HTML report generation
6. 📝 **[06_EXPERIMENT_TRACKING.md](06_EXPERIMENT_TRACKING.md)** - Experiment logging system

---

## Quick Navigation

### I want to...

**...understand the big picture**
→ Start with [00_OVERVIEW.md](00_OVERVIEW.md)

**...run an experiment**
→ Read [01_ORCHESTRATION.md](01_ORCHESTRATION.md) section "Usage"

**...generate custom videos**
→ Read [02_DATA_GENERATION.md](02_DATA_GENERATION.md) section "Parameter Interactions"

**...modify training parameters**
→ Read [03_TRAINING.md](03_TRAINING.md) section "Argument Parsing"

**...understand evaluation metrics**
→ Read [04_EVALUATION.md](04_EVALUATION.md) section "Metrics Calculation"

**...customize the HTML report**
→ Read [05_REPORTING.md](05_REPORTING.md) section "Report Customization"

**...analyze experiment results**
→ Read [06_EXPERIMENT_TRACKING.md](06_EXPERIMENT_TRACKING.md) section "Querying Experiments"

**...troubleshoot errors**
→ Check "Common Issues" sections in relevant documents

---

## How to Read This Documentation

### For First-Time Users
1. Read [00_OVERVIEW.md](00_OVERVIEW.md) - Get the big picture
2. Read [01_ORCHESTRATION.md](01_ORCHESTRATION.md) - Understand main workflow
3. Skim other documents to understand each component
4. Run your first experiment: `./run_experiment.sh`

### For Developers Modifying Code
1. Identify which component you're modifying
2. Read the corresponding detailed documentation
3. Understand function-by-function logic
4. Review "Parameter Interactions" and "Design Decisions" sections
5. Check "Common Issues" before making changes

### For Analysts/Researchers
1. Read [04_EVALUATION.md](04_EVALUATION.md) - Metrics and evaluation
2. Read [06_EXPERIMENT_TRACKING.md](06_EXPERIMENT_TRACKING.md) - Data analysis
3. Use Pandas/Excel to query `experiments_log.csv`
4. Generate custom reports from evaluation data

### For Managers/Stakeholders
1. Read "For Your Boss" section (above)
2. Read [00_OVERVIEW.md](00_OVERVIEW.md) "Architecture" section
3. Look at HTML reports in `analytics/reports/`
4. Review experiment trends in `experiments_log.csv`

---

## Documentation Philosophy

This documentation follows these principles:

### 1. Complete Context
Every parameter, function, and design decision is explained with:
- **What**: What does this do?
- **Why**: Why is it needed?
- **How**: How does it work?
- **When**: When should you use it?

### 2. Self-Sufficient
When you read the documentation again in 6 months, you should be able to:
- Understand exactly what each component does
- Modify parameters confidently
- Debug issues independently
- Extend functionality without breaking existing code

### 3. Multiple Levels
Each document provides:
- **High-level overview**: Architecture and flow
- **Medium-level details**: Function-by-function logic
- **Low-level specifics**: Parameter meanings, code snippets, examples

### 4. Real-World Examples
Every concept is illustrated with:
- Actual code snippets from the project
- Concrete parameter values
- Expected outputs
- Common issues and solutions

---

## Key Concepts

### Pipeline Stages
```
1. ORCHESTRATION (run_experiment.sh)
   ↓
2. DATA GENERATION (building_dataset.py + synthetic_data_generation.py)
   ↓
3. TRAINING (run_full_pipeline.py + llamafactory-cli)
   ↓
4. EVALUATION (evaluate_pipeline_simple.py)
   ↓
5. REPORTING (generate_experiment_report.py)
```

### Data Flow
```
Blender → Videos (MP4)
        ↓
        JSON datasets
        ↓
        Training (LoRA fine-tuning)
        ↓
        Model checkpoints
        ↓
        Evaluation metrics
        ↓
        HTML reports
        ↓
        experiments_log.csv
```

### Key Technologies
- **Blender**: 3D rendering for synthetic videos
- **PyTorch**: Deep learning framework
- **Transformers (Hugging Face)**: Model loading and inference
- **LLaMA-Factory**: Training framework
- **PEFT (LoRA)**: Parameter-efficient fine-tuning
- **BitsAndBytes**: Quantization for memory efficiency

### Key Files
| File | Purpose | Documentation |
|------|---------|---------------|
| `run_experiment.sh` | Main orchestrator | [01_ORCHESTRATION.md](01_ORCHESTRATION.md) |
| `building_dataset.py` | Dataset builder | [02_DATA_GENERATION.md](02_DATA_GENERATION.md) |
| `synthetic_data_generation.py` | Video generator | [02_DATA_GENERATION.md](02_DATA_GENERATION.md) |
| `run_full_pipeline.py` | Training orchestrator | [03_TRAINING.md](03_TRAINING.md) |
| `evaluate_pipeline_simple.py` | Model evaluation | [04_EVALUATION.md](04_EVALUATION.md) |
| `generate_experiment_report.py` | Report generator | [05_REPORTING.md](05_REPORTING.md) |
| `experiment_tracker.py` | Experiment logging | [06_EXPERIMENT_TRACKING.md](06_EXPERIMENT_TRACKING.md) |
| `experiments_log.csv` | Experiment database | [06_EXPERIMENT_TRACKING.md](06_EXPERIMENT_TRACKING.md) |

---

## Glossary

**LoRA (Low-Rank Adaptation):** Efficient fine-tuning method that trains small adapter matrices instead of the full model.

**DoRA (Weight-Decomposed LoRA):** Improved variant of LoRA with better performance.

**QLoRA:** Quantized LoRA - combines 4-bit quantization with LoRA for memory efficiency.

**Quantization:** Reducing model precision (e.g., 32-bit → 4-bit) to save memory.

**Fine-tuning:** Training a pre-trained model on a specific task.

**Synthetic data:** Computer-generated data (vs real-world captured data).

**F1 Score:** Harmonic mean of precision and recall, better than accuracy for imbalanced datasets.

**Evaluation method:** Format of the evaluation prompt (e.g., "yesno" vs "moving_stopped").

**Base model:** Original pre-trained model before fine-tuning.

**Fine-tuned model:** Model after training with LoRA adapter.

**Checkpoint:** Saved model state during training.

**Epoch:** One complete pass through the training dataset.

**Batch size:** Number of samples processed together.

**Gradient accumulation:** Simulating larger batch sizes by accumulating gradients over multiple steps.

**BFloat16:** 16-bit floating point format optimized for ML (vs standard Float16).

**Greedy decoding:** Always picking the most likely next token (deterministic).

**Sampling:** Randomly picking next token based on probabilities (non-deterministic).

---

## File Structure Reference

```
LLaMA-Factory/
├── run_experiment.sh                      # Main orchestrator
├── building_dataset.py                    # Dataset builder
├── run_full_pipeline.py                   # Training orchestrator
├── evaluate_pipeline_simple.py            # Evaluation script
├── generate_experiment_report.py          # Report generator
├── experiment_tracker.py                  # Experiment tracking
├── experiments_log.csv                    # Experiment database
│
├── data/                                  # All datasets and videos
│   ├── dataset_info.json                  # Dataset registry
│   ├── _exp_YYYYMMDD_HHMMSS_train/       # Training videos
│   │   ├── treadmill_0000_*.mp4
│   │   ├── ...
│   │   └── metadata_*.csv
│   ├── _exp_YYYYMMDD_HHMMSS_test/        # Test videos
│   │   ├── treadmill_0000_*.mp4
│   │   ├── ...
│   │   ├── metadata_*.csv
│   │   └── eval_YYYYMMDD_HHMMSS/         # Evaluation results
│   │       ├── evaluation_metadata_*.txt
│   │       ├── evaluation_report_*.txt
│   │       ├── live_evaluation_log_*.txt
│   │       └── per_video_predictions_*.csv
│   ├── _exp_*_train.json                  # Train dataset JSON
│   ├── _exp_*_test.json                   # Test dataset JSON
│   └── synthetic_treadmill/
│       └── synthetic_data_generation.py   # Video generation script
│
├── saves/                                 # Model checkpoints
│   └── _exp_*_train_YYYYMMDD_HHMMSS/     # Model directory
│       ├── adapter_config.json
│       ├── adapter_model.bin             # LoRA weights (~50-100 MB)
│       └── checkpoint-*/                  # Intermediate checkpoints
│
├── analytics/reports/                     # HTML reports
│   └── _exp_*_dual_report_*.html
│
├── examples/train_qlora/                  # Training configs
│   └── qwen25vl_lora_pipeline_*.yaml
│
└── EXPERIMENT_PIPELINE_DOCUMENTATION/     # This documentation
    ├── README.md                          # This file
    ├── 00_OVERVIEW.md
    ├── 01_ORCHESTRATION.md
    ├── 02_DATA_GENERATION.md
    ├── 03_TRAINING.md
    ├── 04_EVALUATION.md
    ├── 05_REPORTING.md
    └── 06_EXPERIMENT_TRACKING.md
```

---

## Common Workflows

### Running a Full Experiment
```bash
./run_experiment.sh
```

### Running with Custom Parameters
Edit parameters in `run_experiment.sh`:
```bash
# In Stage 2: Building dataset
--num_videos 200                          # More videos
--train_view_angle "0,30"                 # Multiple angles

# In Stage 3: Running pipeline
--num_train_epochs 20                     # More epochs
--learning_rate 1e-4                      # Different LR
```

### Evaluating an Existing Model
```bash
python evaluate_pipeline_simple.py \
  --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
  --adapter_name_or_path "saves/_exp_20251207_143033_train_20251207_143500" \
  --test_dataset "_exp_20251207_143033_test" \
  --eval_method "yesno"
```

### Generating Report for Existing Evaluation
```bash
python generate_experiment_report.py \
  --dataset_name "_exp_20251207_143033" \
  --output_dir "analytics/reports" \
  --skip_video_download
```

### Analyzing Experiment Results
```python
import pandas as pd

# Load experiment log
df = pd.read_csv('experiments_log.csv')

# Find best experiment
best = df.loc[df['finetuned_model_accuracy'].idxmax()]
print(f"Best experiment: {best['experiment_id']}")
print(f"Accuracy: {best['finetuned_model_accuracy']}%")
print(f"Dataset: {best['dataset_name']}")
```

---

## Troubleshooting

### Pipeline fails at Stage X
1. Check the corresponding documentation file for that stage
2. Look in "Common Issues" section
3. Check logs on GPU server: `ssh seedoo@hetzner-gpu.tail9e6e7.ts.net`
4. Verify Docker container: `docker ps -a`

### Out of memory during training
1. Reduce `per_device_train_batch_size` to 1
2. Increase `gradient_accumulation_steps` (keeps effective batch size)
3. Use 4-bit quantization (if not already)
4. Reduce `cutoff_len` (shorter sequences)

### Evaluation results are poor
1. Check if fine-tuned model is actually better than base
2. Verify `eval_method` matches training format
3. Review live evaluation logs to see model outputs
4. Check per-texture/per-angle breakdowns for patterns

### Videos not showing in report
1. Verify video paths are relative (not absolute)
2. Check videos exist in `data/` folders
3. Open browser console for errors
4. Try opening HTML file from local file system (not HTTP)

### Experiment log not updating
1. Check file permissions on `experiments_log.csv`
2. Verify no other process has file open (Excel, etc.)
3. Check for exceptions in tracker code
4. Backup and regenerate CSV if corrupted

---

## Maintenance

### Regular Tasks

**Weekly:**
- Backup `experiments_log.csv`
- Review recent experiment results
- Clean up old evaluation folders (if disk space low)

**Monthly:**
- Archive old experiments
- Update documentation if code changed
- Review and clean up `saves/` directory

**Before Major Changes:**
- Backup entire project
- Document current best practices
- Run baseline experiment for comparison

---

## Getting Help

### In Order of Preference:

1. **Search this documentation** - Use Ctrl+F to find keywords
2. **Check experiments_log.csv** - See what worked in past experiments
3. **Review code comments** - Inline documentation in Python files
4. **Check Git history** - See why changes were made
5. **Ask the team** - Share specific error messages and experiment IDs

### When Asking for Help:

Include:
- Experiment ID (if applicable)
- Stage where error occurred
- Error message (full traceback)
- What you've already tried
- Relevant parameters/configuration

---

## Document Maintenance

This documentation was created on: **2025-12-07**

Last updated by Claude Sonnet 4.5 based on the actual codebase.

**How to update documentation:**
1. Read the code file you're documenting
2. Understand every parameter, function, and design decision
3. Update the corresponding .md file
4. Add examples and common issues
5. Update this README if adding new sections

**Documentation standards:**
- Use clear, concise language
- Provide concrete examples (not generic placeholders)
- Explain the "why" not just the "what"
- Include common issues and solutions
- Use consistent formatting and structure

---

## Credits

**Project:** Qwen2.5-VL Treadmill Motion Detection
**Documentation:** Comprehensive pipeline documentation
**Purpose:** Complete knowledge transfer and reproducibility
**Audience:** Developers, researchers, analysts, stakeholders

**Technologies:**
- Qwen2.5-VL-7B-Instruct (vision-language model)
- LLaMA-Factory (training framework)
- Blender (3D rendering)
- PyTorch + Transformers + PEFT
- Custom orchestration and evaluation scripts

---

## Appendix: Quick Command Reference

```bash
# Run full experiment
./run_experiment.sh

# Generate dataset only
python building_dataset.py --dataset_name "my_experiment" --num_videos 100

# Train only (with existing dataset)
python run_full_pipeline.py \
  --dataset_name "_exp_20251207_143033" \
  --num_train_epochs 10 \
  --skip_dataset

# Evaluate only (with existing model)
python evaluate_pipeline_simple.py \
  --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
  --adapter_name_or_path "saves/my_model" \
  --test_dataset "my_experiment_test"

# Generate report only
python generate_experiment_report.py \
  --dataset_name "my_experiment" \
  --skip_video_download

# Sync code to server
rsync -avz --exclude 'data/' --exclude 'saves/' ./ seedoo@hetzner-gpu.tail9e6e7.ts.net:/workspace/

# Download results from server
rsync -avz seedoo@hetzner-gpu.tail9e6e7.ts.net:/workspace/saves/my_model/ ./saves/my_model/

# View experiment log
cat experiments_log.csv | column -t -s','

# Analyze experiments (Python)
python -c "import pandas as pd; df = pd.read_csv('experiments_log.csv'); print(df.describe())"
```

---

**Ready to get started? Begin with [00_OVERVIEW.md](00_OVERVIEW.md)!**
