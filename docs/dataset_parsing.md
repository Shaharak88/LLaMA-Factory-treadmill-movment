# Dataset Parsing: Video Metadata Extraction

## Overview

The `extract_video_metadata.py` script is a utility for extracting structured metadata from video filenames in treadmill datasets. It connects to a remote server (via SSH and Docker), lists all `.mp4` files in a specified dataset directory, parses each filename to extract training parameters, and outputs a comprehensive CSV file with all metadata.

## Purpose

This script solves the problem of tracking and analyzing the exact parameters used to generate each synthetic treadmill video. Since all generation parameters are encoded in the video filenames, the script parses these structured names and converts them into a tabular format suitable for:

- **Dataset analysis**: Understanding parameter distributions across training/test sets
- **Experiment tracking**: Linking videos back to their generation parameters
- **Quality control**: Verifying dataset composition and parameter coverage
- **Model evaluation**: Analyzing which parameter combinations the model struggles with

## Functionality

### Core Operations

1. **Remote Connection**: Connects to a remote server via SSH and optionally executes commands inside a Docker container
2. **Video Discovery**: Lists all `.mp4` files in the specified dataset directory
3. **Filename Parsing**: Extracts structured parameters from standardized video filenames
4. **CSV Generation**: Outputs a timestamped CSV file with all parsed metadata

### Filename Format

The script expects videos to follow this naming convention:

```
treadmill_{index}_{texture}_{direction}_speed{speed}_angle{angle}_bright{brightness}_contr{contrast}[_obj_{object_info}][_blur_{blur_info}]_{width}x{height}_seed{seed}.mp4
```

Example:
```
treadmill_0000_subtle_gray_stripes_left_speed8.0_angle0_bright1.00_contr1.00_obj_boxx1_30_center_640x480_seed46.mp4
```

### Command-Line Interface

```bash
python extract_video_metadata.py <dataset_name> [OPTIONS]

Required Arguments:
  dataset_name              Name of dataset (looks in /app/data/{dataset_name}/)

Optional Arguments:
  --server                  SSH server address (default: seedoo@hetzner-gpu.tail9e6e7.ts.net)
  --dataset-base-path       Base path for datasets (default: /app/data)
  --docker-container        Docker container name (default: llamafactory)
  --output-dir              Local directory for CSV output (default: ./data)
```

## Output Structure

### CSV Columns

The script outputs a CSV file with the following 21 columns:

| Column | Description | Example Values | Data Type |
|--------|-------------|----------------|-----------|
| `video_name` | Full filename including .mp4 extension | `treadmill_0000_subtle_gray_stripes_left_speed8.0_angle0_bright1.00_contr1.00_640x480_seed46.mp4` | String |
| `index` | Sequential video index | `0000`, `0001`, `0042` | String (zero-padded) |
| `texture` | Belt texture type | `subtle_gray_stripes`, `factory_floor`, `concrete` | String |
| `direction` | Belt movement direction | `left`, `right`, `up`, `down`, `stationary` | String |
| `speed` | Belt speed in units/frame | `0.0`, `4.0`, `8.0`, `12.0` | Float (as string) |
| `angle` | Camera angle in degrees | `0`, `15`, `30`, `45` | Integer (as string) |
| `brightness` | Brightness multiplier | `0.80`, `1.00`, `1.20` | Float (as string) |
| `contrast` | Contrast multiplier | `0.80`, `1.00`, `1.20` | Float (as string) |
| `stripe_contrast` | Stripe-specific contrast (legacy) | `N/A` | String |
| `background_contrast` | Background-specific contrast (legacy) | `N/A` | String |
| `object_enabled` | Whether objects are present | `yes`, `no` | String |
| `object_type` | Type of object placed on belt | `box`, `cone`, `cylinder` | String |
| `num_objects` | Number of objects | `1`, `2`, `3` | Integer (as string) |
| `object_size` | Object size descriptor | `30`, `50`, `large`, `small` | String |
| `object_position` | Object placement | `center`, `left`, `right`, `random` | String |
| `blur_enabled` | Whether blur is applied | `yes`, `no` | String |
| `blur_type` | Type of blur effect | `gaussian`, `motion` | String |
| `blur_intensity` | Blur strength | `0.3`, `0.5`, `1.0` | Float (as string) |
| `blur_variation` | Whether blur varies per frame | `yes`, `no` | String |
| `resolution` | Video resolution | `640x480`, `1280x720` | String |
| `seed` | Random seed for reproducibility | `42`, `46`, `123` | Integer (as string) |

### Sample Output

```csv
video_name,index,texture,direction,speed,angle,brightness,contrast,stripe_contrast,background_contrast,object_enabled,object_type,num_objects,object_size,object_position,blur_enabled,blur_type,blur_intensity,blur_variation,resolution,seed
treadmill_0000_subtle_gray_stripes_left_speed0.0_angle0_bright1.00_contr1.00_obj_boxx1_30_center_640x480_seed42.mp4,0000,subtle_gray_stripes,left,0.0,0,1.00,1.00,N/A,N/A,yes,box,1,30,center,no,,,no,640x480,42
treadmill_0000_subtle_gray_stripes_up_speed4.0_angle0_bright1.00_contr1.00_obj_boxx1_30_center_640x480_seed44.mp4,0000,subtle_gray_stripes,up,4.0,0,1.00,1.00,N/A,N/A,yes,box,1,30,center,no,,,no,640x480,44
treadmill_0000_subtle_gray_stripes_down_speed6.0_angle0_bright1.00_contr1.00_obj_boxx1_30_center_640x480_seed45.mp4,0000,subtle_gray_stripes,down,6.0,0,1.00,1.00,N/A,N/A,yes,box,1,30,center,no,,,no,640x480,45
treadmill_0000_subtle_gray_stripes_left_speed8.0_angle0_bright1.00_contr1.00_obj_boxx1_30_center_640x480_seed46.mp4,0000,subtle_gray_stripes,left,8.0,0,1.00,1.00,N/A,N/A,yes,box,1,30,center,no,,,no,640x480,46
treadmill_0001_factory_floor_right_speed12.0_angle15_bright0.90_contr1.10_640x480_seed50.mp4,0001,factory_floor,right,12.0,15,0.90,1.10,N/A,N/A,no,,,,,,,,no,640x480,50
treadmill_0002_concrete_stationary_speed0.0_angle30_bright1.00_contr1.00_obj_conex2_40_random_blur_gaussian_0.5_640x480_seed60.mp4,0002,concrete,stationary,0.0,30,1.00,1.00,N/A,N/A,yes,cone,2,40,random,yes,gaussian,0.5,no,640x480,60
```

## Output File Naming

CSV files are automatically timestamped to prevent overwriting:

```
{dataset_name}_metadata_{timestamp}.csv
```

Example: `preformat_right_only_20251130_141204_metadata_20251202_143022.csv`

## Implementation Details

### Parsing Strategy

The script uses a sequential parsing approach:
1. Splits filename by underscore delimiters
2. Identifies fixed-position components (index, texture, direction)
3. Scans for keyword-prefixed parameters (speed, angle, bright, contr)
4. Handles optional components (objects, blur) with lookahead
5. Extracts resolution and seed from end of filename

### Error Handling

- **Missing parameters**: Columns left empty (blank string)
- **Parsing errors**: Warning printed, partial metadata saved
- **Connection failures**: Error message with stderr output
- **No videos found**: Early exit with informative message

### Multi-part Texture Names

The script correctly handles texture names with multiple words:
- `subtle_gray_stripes` (3 words)
- `factory_floor` (2 words)
- `concrete` (1 word)

It consumes underscore-separated parts until it encounters a direction keyword.

## Use Cases

### 1. Experiment Analysis
```bash
# Extract metadata for a training run
python extract_video_metadata.py exp_20251201_162923_train

# Analyze parameter distribution
import pandas as pd
df = pd.read_csv('data/exp_20251201_162923_train_metadata_20251202.csv')
print(df['speed'].value_counts())
print(df['texture'].value_counts())
```

### 2. Dataset Verification
Compare train/test splits to ensure no parameter leakage:
```python
train_df = pd.read_csv('train_metadata.csv')
test_df = pd.read_csv('test_metadata.csv')

# Check for seed overlap (should be none)
assert len(set(train_df['seed']) & set(test_df['seed'])) == 0
```

### 3. Model Performance Analysis
Join predictions with metadata to identify failure modes:
```python
predictions_df = pd.read_csv('model_predictions.csv')
metadata_df = pd.read_csv('dataset_metadata.csv')

results = predictions_df.merge(metadata_df, on='video_name')
error_by_speed = results.groupby('speed')['error'].mean()
```

## Limitations

1. **Filename Dependency**: Relies entirely on standardized naming convention
2. **Legacy Fields**: `stripe_contrast` and `background_contrast` always return `N/A` (not in current format)
3. **Remote Access Only**: Currently requires SSH connection (no local directory support)
4. **Docker Coupling**: Assumes videos are in Docker container by default

## Future Enhancements

- Add support for local directory parsing (no SSH required)
- Extract frame-level metadata from video content
- Validate parsed parameters against known ranges
- Generate summary statistics automatically
- Support custom filename formats via regex patterns
