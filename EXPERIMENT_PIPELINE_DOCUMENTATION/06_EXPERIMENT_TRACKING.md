# Experiment Tracking System

## Purpose
The experiment tracking system logs all parameters, configurations, and results for every experiment in a single CSV file (`experiments_log.csv`). This enables:
- Complete reproducibility of experiments
- Easy comparison across experiments
- Historical tracking of model performance
- Analysis of parameter effects on results

## Location
`/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/experiment_tracker.py`

## CSV Output Location
`/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/experiments_log.csv`

---

## Architecture

```
ExperimentTracker (class)
    │
    ├──> __init__(csv_path)
    │    └──> _ensure_csv_exists() → Create CSV with headers
    │
    ├──> start_experiment(args, actual_model_path)
    │    ├──> _get_next_experiment_id()
    │    ├──> Build experiment_data dict (all parameters)
    │    ├──> _append_or_update_row() → Write to CSV
    │    └──> Return experiment_id
    │
    ├──> update_status(stage, status)
    │    └──> _update_column() → Update stage status
    │
    ├──> update_evaluation_results(base_results, finetuned_results)
    │    ├──> Extract metrics from results dicts
    │    ├──> Format per-texture/per-angle as JSON
    │    └──> _update_column() → Write metrics to CSV
    │
    ├──> update_dataset_comparison_report(report_path)
    │    └──> _update_column() → Add report link
    │
    └──> finalize_experiment()
         └──> Clear current_experiment_id
```

---

## CSV Structure

The CSV contains **196 columns** tracking every aspect of an experiment.

### Column Categories

**1. Experiment Metadata (3 columns)**
- `experiment_id`: Unique integer ID (auto-incremented)
- `timestamp`: Human-readable timestamp (YYYY-MM-DD HH:MM:SS)
- `run_timestamp`: Machine-readable timestamp (YYYYMMDD_HHMMSS)

**2. Dataset Configuration (5 columns)**
- `dataset_name`: Base dataset name (e.g., "_exp_20251207_143033")
- `train_dataset_name`: Full train dataset name with suffix
- `test_dataset_name`: Full test dataset name with suffix
- `num_videos`: Total number of videos generated
- `train_split`: Train/test split ratio (e.g., 0.8)
- `seed`: Random seed for reproducibility

**3. TRAINING Dataset Parameters (27 columns)**
All prefixed with `train_`:
- `train_texture_type`: Texture type (e.g., "subtle_gray_stripes")
- `train_direction`: Motion direction ("left", "right", "up", "down")
- `train_view_angle`: Camera angle (e.g., "0", "30", "52")
- `train_speed_range`: Speed range (e.g., "3.0,3.0")
- `train_resolution`: Video resolution (e.g., "640x480")
- `train_fps`: Frames per second (e.g., 30)
- `train_duration`: Video duration in seconds (e.g., 3)
- `train_brightness`: Brightness adjustment (-1.0 to 1.0)
- `train_contrast`: Contrast adjustment (0.5 to 1.5)
- `train_lighting_variation`: Random lighting enabled (True/False)
- `train_lighting_intensity`: Light intensity (0.5 to 2.0)
- `train_motion_blur`: Motion blur enabled (True/False)
- `train_camera_noise`: Camera noise level (0.0 to 1.0)
- `train_edge_width`: Belt edge width in pixels
- `train_distance`: Camera distance (depth simulation)
- `train_stripe_width`: Stripe width for striped textures
- `train_stripe_spacing`: Spacing between stripes
- `train_stripe_gray`: Grayscale value for stripes (0-255)
- `train_background_gray`: Grayscale value for background (0-255)
- `train_stripe_distance_variance`: Variance in stripe positions
- `train_add_object`: Object overlay enabled (True/False)
- `train_object_type`: Object type (e.g., "sphere", "cube")
- `train_object_position`: Object position
- `train_object_size`: Object size
- `train_num_objects`: Number of objects
- `train_add_blur`: Blur enabled (True/False)
- `train_blur_type`: Blur type (e.g., "gaussian", "motion")
- `train_blur_intensity`: Blur intensity
- `train_random_blur_variation`: Random blur variation
- `train_vary_parameters`: Random parameter variation enabled

**4. TEST Dataset Parameters (27 columns)**
Same as training parameters, but prefixed with `test_`:
- `test_texture_type`
- `test_direction`
- ... (all same parameters as training)

**Why track train AND test parameters separately?**
- Enables analysis of generalization (e.g., train on angle=0, test on angle=52)
- Documents exactly what conditions each model was tested on
- Critical for understanding performance differences

**5. Model Configuration (2 columns)**
- `model_name_or_path`: Base model (e.g., "Qwen/Qwen2.5-VL-7B-Instruct")
- `template`: Chat template (e.g., "qwen2_vl")

**6. LoRA Configuration (5 columns)**
- `lora_output_dir`: Model save directory
- `lora_rank`: LoRA rank (e.g., 32)
- `lora_alpha`: LoRA alpha (e.g., 64)
- `lora_dropout`: LoRA dropout rate (e.g., 0.05)
- `cutoff_len`: Max sequence length (e.g., 4096)

**7. Training Hyperparameters (11 columns)**
- `per_device_train_batch_size`: Batch size per GPU
- `gradient_accumulation_steps`: Gradient accumulation steps
- `learning_rate`: Learning rate
- `lr_scheduler_type`: LR scheduler (e.g., "cosine")
- `warmup_ratio`: Warmup ratio
- `bf16`: BFloat16 enabled (True/False)
- `fp16`: Float16 enabled (True/False)
- `quantization_bit`: Quantization bits (4 or 8)
- `logging_steps`: Logging frequency
- `save_steps`: Checkpoint save frequency
- `eval_steps`: Evaluation frequency

**8. Evaluation Configuration (6 columns)**
- `eval_method`: Evaluation format ("yesno" or "moving_stopped")
- `eval_max_new_tokens`: Max tokens in model response
- `eval_batch_size`: Evaluation batch size
- `eval_video_fps`: FPS for evaluation videos
- `eval_video_maxlen`: Max frames per video
- `gpu_memory_utilization`: GPU memory utilization (0.0 to 1.0)

**9. Pipeline Control (4 columns)**
- `skip_dataset`: Dataset generation skipped (True/False)
- `skip_training`: Training skipped (True/False)
- `skip_evaluation`: Evaluation skipped (True/False)
- `use_docker`: Docker used (True/False)

**10. Status Tracking (3 columns)**
- `dataset_status`: Dataset stage status (pending/in_progress/completed/failed/skipped)
- `training_status`: Training stage status
- `evaluation_status`: Evaluation stage status

**11. Results - Overall Base Model (10 columns)**
- `base_model_accuracy`: Accuracy percentage
- `base_model_f1_score`: F1 score percentage
- `base_model_f1_moving`: F1 score for "moving" class
- `base_model_f1_stopped`: F1 score for "stopped" class
- `base_model_precision`: Precision percentage
- `base_model_recall`: Recall percentage
- `base_model_moving_correct`: Correct "moving" predictions
- `base_model_moving_total`: Total "moving" samples
- `base_model_stopped_correct`: Correct "stopped" predictions
- `base_model_stopped_total`: Total "stopped" samples

**12. Results - Overall Fine-Tuned Model (10 columns)**
Same metrics as base model, but for fine-tuned:
- `finetuned_model_accuracy`
- `finetuned_model_f1_score`
- ... (all same metrics)

**13. Results - Detailed Breakdowns (4 columns)**
JSON strings containing per-texture and per-angle results:
- `base_model_per_texture_results`: JSON with texture breakdowns
- `base_model_per_angle_results`: JSON with angle breakdowns
- `finetuned_model_per_texture_results`: JSON with texture breakdowns
- `finetuned_model_per_angle_results`: JSON with angle breakdowns

**Example JSON structure:**
```json
{
    "subtle_gray_stripes": {
        "accuracy": 85.0,
        "f1_score": 84.2,
        "correct": 17,
        "total": 20
    },
    "factory_dark_stripes": {
        "accuracy": 90.0,
        "f1_score": 89.5,
        "correct": 18,
        "total": 20
    }
}
```

**14. Manual Notes (2 columns)**
- `notes_1`: User notes field 1
- `notes_2`: User notes field 2

**15. Model Information (2 columns)**
- `model_path`: Actual path to saved model
- `model_name`: Model name/identifier

**16. Key Training Parameters (2 columns)**
- `use_dora`: DoRA enabled (True/False)
- `num_train_epochs`: Number of training epochs

**17. Reports (1 column)**
- `dataset_comparison_report`: Path to HTML report

**Total: 196 columns**

---

## Usage in Pipeline

### 1. Initialization

```python
from experiment_tracker import ExperimentTracker

# Initialize tracker (creates CSV if doesn't exist)
tracker = ExperimentTracker('experiments_log.csv')
```

### 2. Start Experiment

```python
# At beginning of run_full_pipeline.py
experiment_id = tracker.start_experiment(
    args,
    actual_model_path=model_output_dir
)

logger.info(f"Started experiment {experiment_id}")
```

**What happens:**
1. Auto-increments experiment ID
2. Creates new row in CSV
3. Logs all args parameters
4. Sets initial statuses to "pending"
5. Returns experiment ID

### 3. Update Status

```python
# Before training
tracker.update_status('training', 'in_progress')

# After training
tracker.update_status('training', 'completed')

# If error
tracker.update_status('training', 'failed')

# If skipped
tracker.update_status('training', 'skipped')
```

**Status values:**
- `pending`: Not started yet
- `in_progress`: Currently running
- `completed`: Finished successfully
- `failed`: Encountered error
- `skipped`: User chose to skip

### 4. Update Evaluation Results

```python
# Parse evaluation results
from experiment_tracker import parse_evaluation_results
base_results, finetuned_results = parse_evaluation_results('evaluation_results')

# Update tracker
tracker.update_evaluation_results(base_results, finetuned_results)
```

**What happens:**
1. Extracts metrics from results dicts
2. Formats percentages (e.g., 85.0%)
3. Converts per-texture/per-angle to JSON strings
4. Updates all result columns in CSV

### 5. Update Report Link

```python
# After report generation
tracker.update_dataset_comparison_report(
    'analytics/reports/_exp_20251207_143033_dual_report_20251207_151022.html'
)
```

### 6. Finalize Experiment

```python
# At end of run_full_pipeline.py
tracker.finalize_experiment()
logger.info(f"Experiment {experiment_id} completed")
```

---

## Helper Functions

### 1. parse_evaluation_results()

Parses text evaluation report to extract metrics.

```python
def parse_evaluation_results(output_dir: str) -> tuple[Optional[Dict], Optional[Dict]]:
    """
    Parse detailed evaluation results from evaluation output directory.

    Args:
        output_dir: Path to evaluation output directory

    Returns:
        Tuple of (base_results_dict, finetuned_results_dict) with all metrics
    """
    # Find evaluation_report_*.txt
    report_files = list(Path(output_dir).glob('evaluation_report_*.txt'))
    report_file = max(report_files, key=lambda p: p.stat().st_mtime)

    # Read report
    with open(report_file, 'r') as f:
        content = f.read()

    # Extract base model section
    base_section = re.search(r'BASE MODEL\s*=+\s*(.*?)(?:FINE-TUNED MODEL|$)', content, re.DOTALL)

    # Extract fine-tuned model section
    finetuned_section = re.search(r'FINE-TUNED MODEL\s*=+\s*(.*?)(?:COMPARISON|$)', content, re.DOTALL)

    # Parse each section
    base_results = parse_model_section(base_section.group(1))
    finetuned_results = parse_model_section(finetuned_section.group(1))

    return base_results, finetuned_results
```

**Parsing logic:**
Uses regex to extract:
- Overall accuracy: `Accuracy:\s*([\d.]+)%\s*\((\d+)/(\d+)\)`
- F1 score: `F1 Score:\s*([\d.]+)%`
- Per-class metrics: `Moving:\s*(\d+)/(\d+)(?:.*?F1:\s*([\d.]+)%)?`
- Per-texture breakdowns
- Per-angle breakdowns

---

## Querying Experiments

### Using Pandas

```python
import pandas as pd

# Load experiment log
df = pd.read_csv('experiments_log.csv')

# View all experiments
print(df[['experiment_id', 'timestamp', 'dataset_name', 'finetuned_model_accuracy']])

# Filter by date
recent = df[df['timestamp'] > '2025-12-01']

# Filter by parameter
angle52_experiments = df[df['test_view_angle'] == '52']

# Sort by accuracy
best = df.sort_values('finetuned_model_accuracy', ascending=False).head(10)

# Compare train/test conditions
generalization = df[df['train_view_angle'] != df['test_view_angle']]

# Group by parameter
accuracy_by_angle = df.groupby('test_view_angle')['finetuned_model_accuracy'].mean()
```

### Using Excel

1. Open `experiments_log.csv` in Excel
2. Use filters to explore data
3. Create pivot tables for analysis
4. Generate charts comparing experiments

### SQL Queries (if imported to database)

```sql
-- Best performing model
SELECT experiment_id, dataset_name, finetuned_model_accuracy
FROM experiments_log
ORDER BY finetuned_model_accuracy DESC
LIMIT 1;

-- Average improvement by evaluation method
SELECT eval_method,
       AVG(finetuned_model_accuracy - base_model_accuracy) as avg_improvement
FROM experiments_log
GROUP BY eval_method;

-- Find experiments with specific parameters
SELECT experiment_id, dataset_name, finetuned_model_accuracy
FROM experiments_log
WHERE train_texture_type = 'subtle_gray_stripes'
  AND test_view_angle = '52'
  AND num_train_epochs = 10;
```

---

## Benefits of CSV Format

**Advantages:**
1. **Human-readable**: Open in any text editor or spreadsheet software
2. **Universal**: Works with Python, R, Excel, SQL databases
3. **Version control**: Git can track changes to CSV
4. **No dependencies**: No database server required
5. **Portable**: Single file contains all experiment history

**Disadvantages:**
1. **Concurrent writes**: Not safe for multiple simultaneous experiments
2. **Large file size**: Can grow large with many experiments
3. **Performance**: Slower than database for complex queries

**When to migrate to database:**
- More than 1000 experiments
- Multiple users running experiments simultaneously
- Need complex queries with joins
- Need transaction support

---

## Example Analyses

### 1. Parameter Effect Analysis

**Question:** How does LoRA rank affect accuracy?

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('experiments_log.csv')

# Group by LoRA rank
rank_analysis = df.groupby('lora_rank').agg({
    'finetuned_model_accuracy': ['mean', 'std', 'count']
})

print(rank_analysis)

# Plot
plt.bar(rank_analysis.index, rank_analysis['finetuned_model_accuracy']['mean'])
plt.xlabel('LoRA Rank')
plt.ylabel('Average Accuracy (%)')
plt.title('Effect of LoRA Rank on Accuracy')
plt.show()
```

### 2. Generalization Analysis

**Question:** How well do models generalize to different viewing angles?

```python
# Filter experiments with different train/test angles
generalization = df[df['train_view_angle'] != df['test_view_angle']]

# Compare accuracy
same_angle = df[df['train_view_angle'] == df['test_view_angle']]

print(f"Same angle accuracy: {same_angle['finetuned_model_accuracy'].mean():.2f}%")
print(f"Different angle accuracy: {generalization['finetuned_model_accuracy'].mean():.2f}%")
```

### 3. Training Duration Analysis

**Question:** How many epochs are needed?

```python
epochs_analysis = df.groupby('num_train_epochs')['finetuned_model_accuracy'].mean()

plt.plot(epochs_analysis.index, epochs_analysis.values, marker='o')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Training Epochs')
plt.grid(True)
plt.show()
```

### 4. Failure Analysis

**Question:** Which experiments failed and why?

```python
# Find failed experiments
failed = df[(df['training_status'] == 'failed') | (df['evaluation_status'] == 'failed')]

print(f"Failed experiments: {len(failed)}")
print(failed[['experiment_id', 'timestamp', 'training_status', 'evaluation_status']])
```

---

## Best Practices

### 1. Backup Regularly

```bash
# Create timestamped backup
cp experiments_log.csv experiments_log_backup_$(date +%Y%m%d).csv
```

### 2. Manual Notes

Add notes to experiments for context:
```python
# After experiment, manually edit CSV to add notes
# Or add programmatically:
tracker._update_column(experiment_id, 'notes_1', 'Testing new texture type')
```

### 3. Validate Data

```python
# Check for missing values
df.isnull().sum()

# Check for duplicates
df[df.duplicated(['dataset_name', 'model_path'])]

# Validate accuracy ranges
assert (df['finetuned_model_accuracy'] >= 0).all()
assert (df['finetuned_model_accuracy'] <= 100).all()
```

### 4. Archive Old Experiments

```python
# Split into active and archive
recent = df[df['timestamp'] > '2025-01-01']
old = df[df['timestamp'] <= '2025-01-01']

recent.to_csv('experiments_log.csv', index=False)
old.to_csv('experiments_log_archive.csv', index=False)
```

## Next Steps
- Read `07_CONFIGURATION_FILES.md` for YAML config details
- Explore `experiments_log.csv` to see real experiment data
- Use Pandas/Excel to analyze experiment trends
