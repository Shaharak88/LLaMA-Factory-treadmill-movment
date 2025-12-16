# Dataset Builder - Synthetic Treadmill Dataset Generator

**Automated pipeline for creating synthetic treadmill video datasets formatted for LLaMA-Factory training.**

## Overview

`building_dataset.py` is a comprehensive tool that automates the complete workflow of:
1. Generating synthetic treadmill videos with varied parameters
2. Creating ShareGPT-formatted JSON datasets
3. Automatically updating `dataset_info.json`
4. Generating detailed summary reports

## Key Features

- **50/50 Split**: Always generates equal moving vs stopped videos
- **Simple Prompts**: Compact, focused prompts ("Is the treadmill belt moving or stopped?")
- **Automatic Variation**: Diverse parameters for robust training
- **Reproducible**: Seed-based generation for exact replication
- **Validated**: Complete error checking and file validation
- **Docker Ready**: Works seamlessly in Docker containers

## Installation

No additional dependencies beyond those already required for `synthetic_data_generation.py`:

```bash
pip install opencv-python-headless "numpy<2.0.0"
```

## Quick Start

### Generate a Small Test Dataset (10 videos)

```bash
python building_dataset.py \
  --dataset_name test_dataset \
  --num_videos 10 \
  --vary_parameters \
  --seed 42
```

### Generate a Production Dataset (100 videos)

```bash
python building_dataset.py \
  --dataset_name synthetic_treadmill_100 \
  --num_videos 100 \
  --vary_parameters \
  --seed 123
```

### Generate Specific Texture Dataset (stripes only)

```bash
python building_dataset.py \
  --dataset_name synthetic_stripes_motion \
  --num_videos 50 \
  --texture_type stripes \
  --vary_parameters \
  --speed_range 2.0,6.0
```

## Docker Usage

### Run Inside Docker Container

```bash
# Interactive mode
docker exec -it llamafactory bash
cd /app
python3 building_dataset.py \
  --dataset_name docker_dataset \
  --num_videos 20 \
  --vary_parameters

# One-line execution
docker exec llamafactory python3 /app/building_dataset.py \
  --dataset_name my_dataset \
  --num_videos 50 \
  --vary_parameters
```

## Command-Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--dataset_name` | Name for the dataset | `synthetic_treadmill_v1` |

### Basic Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `data/<dataset_name>` | Output directory for videos |
| `--num_videos` | `10` | Total number of videos |
| `--train_split` | `0.8` | Training split ratio (future use) |
| `--moving_ratio` | `0.5` | Ratio of moving videos (always 50/50) |
| `--seed` | `42` | Random seed for reproducibility |

### Video Generation Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--vary_parameters` | `False` | Automatically vary parameters for diversity |
| `--texture_type` | `stripes` | Texture type: stripes, noise, rubber, grid, diamond_plate |
| `--direction` | `right` | Motion direction: left, right, up, down |
| `--speed_range` | `1.0,8.0` | Speed range as min,max for moving videos |
| `--resolution` | `640x480` | Video resolution as WxH |
| `--fps` | `30` | Frames per second |
| `--duration` | `5.0` | Video duration in seconds |

### Camera & Lighting Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--view_angle` | `0.0` | Camera viewing angle in degrees |
| `--brightness` | `0.0` | Brightness adjustment (-1.0 to 1.0) |
| `--contrast` | `1.0` | Contrast adjustment (0.5 to 2.0) |
| `--lighting_variation` | `none` | Lighting type: none, vignette, gradient_lr, gradient_tb, spotlight |
| `--lighting_intensity` | `0.5` | Lighting intensity (0.0 to 1.0) |
| `--motion_blur` | `0` | Motion blur amount (0 to 10) |
| `--camera_noise` | `0.0` | Camera noise level (0.0 to 1.0) |

## Dataset Format

### ShareGPT Format

Each entry in the generated JSON file follows this format:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<video>Analyze this video. Is the treadmill belt moving or stopped?"
    },
    {
      "role": "assistant",
      "content": "The treadmill belt is moving."
    }
  ],
  "videos": [
    "data/synthetic_treadmill_100/treadmill_0042_stripes_right_speed3.5_angle15_bright0.05_contr1.12_640x480_seed84.mp4"
  ]
}
```

### Prompt Templates

**User Prompt** (always the same):
```
<video>Analyze this video. Is the treadmill belt moving or stopped?
```

**Assistant Response** (binary):
- Moving: `"The treadmill belt is moving."`
- Stopped: `"The treadmill belt is stopped."`

## Output Files

After running the script, the following files are created:

```
LLaMA-Factory/
├── data/
│   ├── <dataset_name>/                  # Video files directory
│   │   ├── treadmill_0001_*.mp4
│   │   ├── treadmill_0002_*.mp4
│   │   ├── ...
│   │   ├── <dataset_name>_build_log.txt
│   │   └── <dataset_name>_summary.txt
│   │
│   ├── <dataset_name>.json             # Dataset JSON (ShareGPT format)
│   └── dataset_info.json               # Updated with new dataset entry
```

### Summary Report

The script generates a detailed summary report showing:

- Total videos generated
- Moving vs stopped split (always 50/50)
- Texture type distribution
- Direction distribution
- Speed ranges
- File sizes (total and average)
- Output file paths

Example summary:

```
======================================================================
Dataset Generation Summary: synthetic_treadmill_100
======================================================================

Generation Date: 2025-11-26 14:30:45
Seed: 42

Dataset Statistics:
----------------------------------------------------------------------
  Total Videos: 100
  Moving Videos: 50 (50.0%)
  Stopped Videos: 50 (50.0%)

  Speed Range: 1.23 - 7.89 px/frame

Texture Distribution:
  diamond_plate :  20 (20.0%)
  grid          :  20 (20.0%)
  noise         :  20 (20.0%)
  rubber        :  20 (20.0%)
  stripes       :  20 (20.0%)

Direction Distribution:
  down          :  25 (25.0%)
  left          :  25 (25.0%)
  right         :  25 (25.0%)
  up            :  25 (25.0%)

File Sizes:
  Total: 125.45 MB
  Average: 1.25 MB per video

Output Files:
----------------------------------------------------------------------
  Videos: data/synthetic_treadmill_100/
  Dataset JSON: data/synthetic_treadmill_100.json
  Dataset Info: data/dataset_info.json
  Log File: data/synthetic_treadmill_100/synthetic_treadmill_100_build_log.txt

======================================================================
Dataset ready for training!
======================================================================
```

## Usage Examples

### Example 1: Small Test Dataset

Generate 10 videos with default parameters for testing:

```bash
python building_dataset.py \
  --dataset_name test_10 \
  --num_videos 10 \
  --seed 42
```

### Example 2: Diverse Training Dataset

Generate 100 videos with varied parameters for robust training:

```bash
python building_dataset.py \
  --dataset_name training_100 \
  --num_videos 100 \
  --vary_parameters \
  --seed 123
```

### Example 3: High-Resolution Dataset

Generate high-resolution videos for production use:

```bash
python building_dataset.py \
  --dataset_name hd_dataset \
  --num_videos 50 \
  --resolution 1280x720 \
  --fps 60 \
  --duration 10.0 \
  --vary_parameters
```

### Example 4: Specific Conditions Dataset

Generate videos with specific texture and speed range:

```bash
python building_dataset.py \
  --dataset_name rubber_slow_motion \
  --num_videos 40 \
  --texture_type rubber \
  --speed_range 0.5,3.0 \
  --vary_parameters
```

### Example 5: Challenging Conditions Dataset

Generate videos with difficult lighting and angles:

```bash
python building_dataset.py \
  --dataset_name challenging_conditions \
  --num_videos 60 \
  --vary_parameters \
  --lighting_variation vignette \
  --lighting_intensity 0.8 \
  --camera_noise 0.4 \
  --motion_blur 2
```

## Training with Generated Dataset

After generating your dataset, you can use it for training:

### 1. Verify Dataset Registration

Check that your dataset appears in `dataset_info.json`:

```bash
grep "<dataset_name>" data/dataset_info.json
```

### 2. Update Training Configuration

Edit your training YAML file (e.g., `examples/train_qlora/qwen25vl_lora_sft.yaml`):

```yaml
dataset: <dataset_name>
```

### 3. Run Training

```bash
llamafactory-cli train examples/train_qlora/qwen25vl_lora_sft.yaml
```

## Parameter Variation Strategy

When `--vary_parameters` is enabled, the script automatically varies:

1. **Texture Types**: Cycles through all 5 types (stripes, noise, rubber, grid, diamond_plate)
2. **Motion Directions**: Cycles through all 4 directions (left, right, up, down)
3. **Speed**: Random uniform in specified range (default: 1.0-8.0 px/frame)
4. **View Angle**: Random uniform in range (-30° to +30°)
5. **Brightness**: Random uniform in range (-0.2 to +0.2)
6. **Contrast**: Random uniform in range (0.7 to 1.3)
7. **Lighting**: Cycles through all 5 types (none, vignette, gradient_lr, gradient_tb, spotlight)
8. **Lighting Intensity**: Random uniform in range (0.3 to 0.8)
9. **Motion Blur**: Added for high-speed videos (speed > 5.0)
10. **Camera Noise**: Random uniform in range (0.0 to 0.3)

This ensures maximum diversity in the training data.

## Best Practices

### Dataset Size Recommendations

| Use Case | Recommended Size | Duration | Total Size |
|----------|-----------------|----------|------------|
| Quick Test | 10 videos | ~1 min | ~10 MB |
| Development | 50 videos | ~5 min | ~50 MB |
| Training | 100-200 videos | ~10-20 min | ~100-200 MB |
| Production | 500-1000 videos | ~1-2 hours | ~500-1000 MB |

### Parameter Recommendations

**For General Training:**
```bash
--num_videos 100
--vary_parameters
--resolution 640x480
--fps 30
--duration 5.0
```

**For High-Quality Training:**
```bash
--num_videos 200
--vary_parameters
--resolution 1280x720
--fps 60
--duration 8.0
```

**For Challenging Conditions:**
```bash
--num_videos 100
--vary_parameters
--lighting_intensity 0.8
--camera_noise 0.3
--motion_blur 2
```

## Troubleshooting

### Issue: Videos Not Generated

**Symptoms**: Script completes but no videos in output directory

**Solutions**:
1. Check if `synthetic_data_generation.py` exists:
   ```bash
   ls data/synthetic_treadmill/synthetic_data_generation.py
   ```

2. Verify Python dependencies:
   ```bash
   python3 -c "import cv2, numpy; print('OK')"
   ```

3. Check disk space:
   ```bash
   df -h
   ```

### Issue: Dataset JSON Not Created

**Symptoms**: Videos exist but no JSON file

**Solutions**:
1. Check file permissions:
   ```bash
   ls -la data/
   ```

2. Verify video filenames contain speed information:
   ```bash
   ls data/<dataset_name>/*.mp4
   ```

### Issue: Training Can't Find Dataset

**Symptoms**: Training fails with "dataset not found"

**Solutions**:
1. Verify dataset_info.json was updated:
   ```bash
   cat data/dataset_info.json | grep "<dataset_name>"
   ```

2. Check video paths are relative:
   ```bash
   head -20 data/<dataset_name>.json
   ```

3. Ensure videos exist at specified paths:
   ```bash
   ls data/<dataset_name>/*.mp4 | wc -l
   ```

## Performance Notes

### Generation Speed

Approximate generation times on typical hardware:

| Resolution | FPS | Duration | Time per Video | 100 Videos |
|------------|-----|----------|----------------|------------|
| 640x480 | 30 | 5s | ~1-2 sec | ~2-3 min |
| 1280x720 | 30 | 5s | ~3-5 sec | ~5-8 min |
| 1920x1080 | 60 | 10s | ~15-20 sec | ~25-35 min |

### Disk Space

Approximate file sizes:

| Resolution | FPS | Duration | Size per Video | 100 Videos |
|------------|-----|----------|----------------|------------|
| 640x480 | 30 | 5s | ~1.0 MB | ~100 MB |
| 1280x720 | 30 | 5s | ~2.5 MB | ~250 MB |
| 1920x1080 | 60 | 10s | ~10 MB | ~1 GB |

## Advanced Usage

### Custom Configuration File

Create a YAML config file for complex setups:

```yaml
# dataset_config.yaml
dataset_name: my_custom_dataset
num_videos: 200
vary_parameters: true
seed: 42
resolution: 1280x720
fps: 60
duration: 8.0
speed_range: 2.0,10.0
```

Then run with config:
```bash
# Note: Config file support can be added if needed
python building_dataset.py @dataset_config.yaml
```

### Batch Generation

Generate multiple datasets with different configurations:

```bash
#!/bin/bash
# batch_generate.sh

datasets=(
  "stripes_dataset"
  "noise_dataset"
  "rubber_dataset"
)

for dataset in "${datasets[@]}"; do
  python building_dataset.py \
    --dataset_name "$dataset" \
    --num_videos 100 \
    --texture_type "${dataset%%_*}" \
    --vary_parameters
done
```

### Integration with Training Pipeline

Automated dataset generation + training:

```bash
#!/bin/bash
# generate_and_train.sh

DATASET_NAME="auto_$(date +%Y%m%d_%H%M%S)"

echo "Generating dataset: $DATASET_NAME"
python building_dataset.py \
  --dataset_name "$DATASET_NAME" \
  --num_videos 100 \
  --vary_parameters

echo "Starting training..."
sed -i "s/dataset: .*/dataset: $DATASET_NAME/" examples/train_qlora/qwen25vl_lora_sft.yaml
llamafactory-cli train examples/train_qlora/qwen25vl_lora_sft.yaml
```

## API Reference

### DatasetBuilder Class

Main class for dataset building operations.

#### Methods

- `__init__(args)`: Initialize with configuration
- `setup_logging()`: Configure file and console logging
- `create_directories()`: Create output directories
- `validate_prerequisites()`: Check required files exist
- `generate_video_configs()`: Create video parameter configurations
- `generate_videos(configs)`: Generate synthetic videos
- `create_dataset_json(video_files)`: Create ShareGPT JSON
- `update_dataset_info()`: Update dataset_info.json
- `generate_summary_report()`: Create summary report
- `build()`: Execute complete pipeline

### Helper Functions

- `parse_arguments()`: Parse command-line arguments
- `main()`: Entry point

## Contributing

To extend the dataset builder:

1. **Add New Prompt Templates**: Edit `_create_dataset_entry()`
2. **Add New Parameters**: Update `parse_arguments()` and `_generate_configs()`
3. **Add New Validation**: Extend `validate_prerequisites()`
4. **Add New Reports**: Modify `generate_summary_report()`

## License

Same as LLaMA-Factory project.

## Support

For issues, questions, or feature requests:
- Check troubleshooting section above
- Review log files in output directory
- Consult PROJECT_SUMMARY.md for project overview
- Check data/synthetic_treadmill/README.md for video generation details

## Related Documentation

- `PROJECT_SUMMARY.md` - Complete project overview
- `data/synthetic_treadmill/README.md` - Synthetic video generation
- `data/synthetic_treadmill/QUICK_START.md` - Quick reference
- `examples/train_qlora/qwen25vl_lora_sft.yaml` - Training configuration
- `evaluate_treadmill_lora.py` - Evaluation script
- `evaluate_bootstrap.py` - Bootstrap evaluation

---

**Last Updated**: 2025-11-26
**Version**: 1.0.0
**Author**: AI-Generated
