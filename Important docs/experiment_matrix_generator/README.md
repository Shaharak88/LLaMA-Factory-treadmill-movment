# Experiment Matrix Generator

## Overview
This script generates an HTML heatmap matrix comparing F1 scores across different samplers and datasets. The matrix uses a color gradient (red=worst, yellow=middle, green=best) to visualize performance differences.

## Data Sources

### Primary Source: Excel File
- **Path**: `C:\Users\shaha\Desktop\SeeDoo_Datasets\SeeDoo_backups\all.xlsx`
- Contains experiment runs with columns: `experiment_id`, `model_name`, `dataset_name`, `finetuned_model_f1_score`, `num_train_epochs`, `adapter_type`, `notes_1`, etc.

### Secondary Source: CSV File (for Real World data)
- **Path**: `C:\Users\shaha\Desktop\Qwen2.5\LLaMA-Factory\data\experiments_log.csv`
- Used to extract real world test results (models trained on synthetic data, tested on real world data)

## Sampler Classification Logic

The script determines the sampler type using:
1. **Manual corrections** for specific experiment IDs (due to naming inconsistencies)
2. **Model name patterns** as fallback

### Manual Corrections (experiment_id -> sampler)
```python
corrections = {
    80: 'No shuffle',
    67: 'HF shuffle',
    77: 'Our random shuffle (iter fix)',
    81: 'Our random shuffle'
}
```

### Model Name Pattern Matching (in order)
| Pattern in model_name | Sampler |
|----------------------|---------|
| `balanced` | Balanced |
| `iter_count` | Our random shuffle (iter fix) |
| `random_sampler` | Our random shuffle |
| `sequential` | No shuffle |
| `hf_shuffle` | HF shuffle |
| `noshuffle` | No shuffle |
| (default) | HF shuffle |

## Dataset Filtering

Only datasets with **more than 1 run** are included in the matrix. This filters out single-run experiments that can't be compared.

## Dataset Name Mapping

| Internal Name | Display Name |
|--------------|--------------|
| `_exp_20251208_230500` | 64 distances |
| `_exp_20251209_dist_augment` | 74=64 +10 distances |
| `_exp_20251209_dist_augment_x5_combined` | 124=64 +60 distances |
| `real_world` | Real World |

## Real World Data (Hardcoded)

Real world results are from models trained on `_exp_20251208_230500` (64 distances) and evaluated on `real_world_test_my_treadmill_processed`:

| Sampler | F1 Score | Source Model |
|---------|----------|--------------|
| HF shuffle | 51.41 | 5epoch_dec9_loraplus |
| Our random shuffle | 53.58 | loraplus_hf_shuffle_5epochs_dec10 |
| Our random shuffle (iter fix) | missing | - |
| Balanced | 45.97 | loraplus_balanced_exp230500_dec10 |
| No shuffle | missing | - |

## Color Gradient Logic

The gradient goes from **Red (worst) -> Yellow (middle) -> Green (best)**, calculated per dataset column independently.

```python
def get_gradient_color(value, min_val, max_val):
    # Normalize to 0-1 (1 = best)
    normalized = (value - min_val) / (max_val - min_val)

    if normalized >= 0.5:
        # Yellow to Green (best half)
        ...
    else:
        # Red to Yellow (worst half)
        ...
```

## Output
- **HTML File**: `C:\Users\shaha\Desktop\SeeDoo_Datasets\SeeDoo_backups\experiment_matrix.html`
- Features: Responsive table, hover effects, color gradient heatmap, "missing" indicators

## Usage

```bash
python generate_experiment_matrix.py
```

Or run directly:
```bash
python3 << 'EOF'
# (paste script contents)
EOF
```

## Samplers Explained

| Sampler | Script Flag | Description |
|---------|-------------|-------------|
| HF shuffle | `hf_shuffle` | HuggingFace RandomSampler (per-epoch shuffle) |
| Our random shuffle | `random` | Custom random sampler (per-epoch shuffle) |
| Our random shuffle (iter fix) | `random` with iter fix | Custom sampler with iteration count fix |
| Balanced | `balanced` | 50% moving, 50% stopped per batch |
| No shuffle | `hf_sequential` | HuggingFace SequentialSampler (no shuffle) |
