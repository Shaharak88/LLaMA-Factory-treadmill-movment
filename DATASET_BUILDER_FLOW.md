# Dataset Builder - Architecture and Flow

## Executive Summary

The **Dataset Builder** (`building_dataset.py`) is an automated pipeline for generating synthetic treadmill video datasets formatted for LLaMA-Factory training. It orchestrates video generation, dataset formatting, and registration in a single, reproducible workflow.

**Key Capabilities:**
- Generates synthetic treadmill videos with diverse parameters
- Creates ShareGPT-formatted JSON datasets
- Automatically registers datasets in LLaMA-Factory
- Supports both random variation and combinatorial generation modes
- Provides comprehensive validation and reporting

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATASET BUILDER PIPELINE                    │
│                     (building_dataset.py)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │     1. INITIALIZATION & VALIDATION       │
        │  • Parse CLI arguments                   │
        │  • Create directories                    │
        │  • Setup logging                         │
        │  • Validate prerequisites                │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │     2. VIDEO CONFIG GENERATION           │
        │  • Random Mode: 50/50 moving/stopped     │
        │  • Combo Mode: All combinations          │
        │  • Parameter variation strategy          │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │     3. VIDEO GENERATION                  │
        │  • Call synthetic_data_generation.py     │
        │  • Generate videos with parameters       │
        │  • Track statistics                      │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │     4. DATASET FORMATTING                │
        │  • Create ShareGPT JSON format           │
        │  • Map videos to prompts/responses       │
        │  • Shuffle for training                  │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │     5. REGISTRATION & REPORTING          │
        │  • Update dataset_info.json              │
        │  • Generate summary report               │
        │  • Create logs                           │
        └──────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  READY TO TRAIN │
                    └─────────────────┘
```

---

## Detailed Pipeline Flow

### Stage 1: Initialization & Validation

```
┌────────────────────────────────────────────────────────────────────┐
│                    INITIALIZATION PHASE                             │
└────────────────────────────────────────────────────────────────────┘
    │
    ├─► Parse CLI Arguments
    │   ├─ Required: --dataset_name
    │   ├─ Optional: --num_videos, --seed, --vary_parameters
    │   └─ Video params: texture, direction, speed, etc.
    │
    ├─► Initialize DatasetBuilder
    │   ├─ Set paths (output_dir, dataset_json, dataset_info)
    │   ├─ Initialize random generator with seed
    │   └─ Setup statistics tracking
    │
    ├─► Create Directories
    │   └─ data/<dataset_name>/
    │
    ├─► Setup Logging
    │   ├─ Console logging
    │   └─ File logging: <dataset_name>_build_log.txt
    │
    └─► Validate Prerequisites
        ├─ Check synthetic_data_generation.py exists
        ├─ Check dataset_info.json exists
        └─ Verify Python3 availability
```

### Stage 2: Video Configuration Generation

```
┌────────────────────────────────────────────────────────────────────┐
│                   CONFIG GENERATION MODES                           │
└────────────────────────────────────────────────────────────────────┘
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
          ▼                                   ▼
    ┌──────────────┐                  ┌──────────────┐
    │ RANDOM MODE  │                  │  COMBO MODE  │
    │ (default)    │                  │ (if comma)   │
    └──────────────┘                  └──────────────┘
          │                                   │
          │                                   │
          ▼                                   ▼
    50/50 Split:                    All Combinations:
    ├─ num_moving = num_videos/2    ├─ Parse all comma-separated
    └─ num_stopped = num_videos/2   │   parameters
                                    ├─ Generate cartesian product
    For each video:                 │   of all values
    ├─ texture_type                 └─ Create configs for each
    ├─ direction                        combination
    ├─ speed (random in range)
    ├─ view_angle (random)          Example:
    ├─ brightness (random)          --texture_type stripes,noise
    ├─ contrast (random)            --speed_range 2.0,5.0
    ├─ lighting (cycled)            → 4 videos (2×2)
    ├─ motion_blur
    ├─ camera_noise
    └─ edge_width
```

### Stage 3: Video Generation

```
┌────────────────────────────────────────────────────────────────────┐
│                      VIDEO GENERATION                               │
└────────────────────────────────────────────────────────────────────┘
    │
    │ For each config in shuffled_configs:
    │
    ├─► Build Command
    │   │
    │   python3 synthetic_data_generation.py \
    │     --num_videos 1 \
    │     --output_dir data/<dataset_name> \
    │     --seed <config_seed> \
    │     --texture_type <texture> \
    │     --direction <direction> \
    │     --speed <speed> \
    │     --view_angle <angle> \
    │     --brightness <brightness> \
    │     --contrast <contrast> \
    │     --lighting_variation <lighting> \
    │     --lighting_intensity <intensity> \
    │     --motion_blur <blur> \
    │     --camera_noise <noise> \
    │     --edge_width <width> \
    │     [--stripe_width <width>]     # if subtle_gray_stripes
    │     [--stripe_spacing <spacing>]  # if subtle_gray_stripes
    │     [--stripe_gray <gray>]        # if subtle_gray_stripes
    │     [--background_gray <gray>]    # if subtle_gray_stripes
    │     [--add-object]                # if object placement enabled
    │     [--add-blur]                  # if camera blur enabled
    │
    ├─► Execute Subprocess
    │   ├─ Timeout: 300 seconds per video
    │   └─ Capture stdout/stderr
    │
    ├─► Verify Video Created
    │   ├─ Find generated .mp4 file
    │   └─ Check file size
    │
    └─► Update Statistics
        ├─ total_videos++
        ├─ moving_videos++ or stopped_videos++
        ├─ texture_distribution[texture]++
        ├─ direction_distribution[direction]++
        ├─ speed_range (min/max)
        └─ total_size_bytes += file_size
```

### Stage 4: Dataset Formatting (ShareGPT)

```
┌────────────────────────────────────────────────────────────────────┐
│                   SHAREGPT FORMATTING                               │
└────────────────────────────────────────────────────────────────────┘
    │
    │ For each generated video:
    │
    ├─► Determine Motion State
    │   ├─ Parse filename: speed<value>
    │   └─ is_moving = (speed > 0.0)
    │
    ├─► Create Relative Path
    │   └─ video_path = "data/<dataset_name>/<filename>.mp4"
    │
    └─► Create Dataset Entry
        │
        {
          "messages": [
            {
              "role": "user",
              "content": "<video>Is there movement in the video? Answer only with yes or no."
            },
            {
              "role": "assistant",
              "content": "yes"  // or "no" if stopped
            }
          ],
          "videos": ["data/<dataset_name>/treadmill_0001_*.mp4"]
        }
    │
    ├─► Shuffle Entries
    │   └─ Random shuffle for better training
    │
    └─► Write JSON File
        └─ data/<dataset_name>.json
```

### Stage 5: Registration & Reporting

```
┌────────────────────────────────────────────────────────────────────┐
│                  REGISTRATION & REPORTING                           │
└────────────────────────────────────────────────────────────────────┘
    │
    ├─► Update dataset_info.json
    │   │
    │   ├─ Create backup: dataset_info.json.backup
    │   │
    │   ├─ Add new entry:
    │   │   {
    │   │     "<dataset_name>": {
    │   │       "file_name": "<dataset_name>.json",
    │   │       "formatting": "sharegpt",
    │   │       "columns": {
    │   │         "messages": "messages",
    │   │         "videos": "videos"
    │   │       },
    │   │       "tags": {
    │   │         "role_tag": "role",
    │   │         "content_tag": "content",
    │   │         "user_tag": "user",
    │   │         "assistant_tag": "assistant"
    │   │       },
    │   │       "media_dir": "data/<dataset_name>"
    │   │     }
    │   │   }
    │   │
    │   └─ Write updated dataset_info.json
    │
    └─► Generate Summary Report
        │
        ├─ Dataset Statistics
        │   ├─ Total videos
        │   ├─ Moving vs stopped split
        │   └─ Speed range
        │
        ├─ Distributions
        │   ├─ Texture types
        │   └─ Motion directions
        │
        ├─ File Sizes
        │   ├─ Total size (MB)
        │   └─ Average per video
        │
        ├─ Output Files
        │   ├─ Videos directory
        │   ├─ Dataset JSON path
        │   ├─ Dataset info path
        │   └─ Log file path
        │
        └─ Write to files:
            ├─ <dataset_name>_summary.txt
            └─ <dataset_name>_build_log.txt
```

---

## Component Architecture

### DatasetBuilder Class

```
┌─────────────────────────────────────────────────────────────────┐
│                      DatasetBuilder                              │
├─────────────────────────────────────────────────────────────────┤
│  Attributes:                                                     │
│    • args: Command-line arguments                               │
│    • dataset_name: Name of dataset                              │
│    • output_dir: Video output directory                         │
│    • num_videos: Total videos to generate                       │
│    • seed: Random seed                                          │
│    • rng: Random number generator                               │
│    • stats: Statistics dictionary                               │
│    • paths: Project paths (root, data, scripts)                 │
├─────────────────────────────────────────────────────────────────┤
│  Methods:                                                        │
│    ┌──────────────────────────────────────────────────┐         │
│    │  setup_logging()                                 │         │
│    │  → Configure file + console logging              │         │
│    └──────────────────────────────────────────────────┘         │
│    ┌──────────────────────────────────────────────────┐         │
│    │  create_directories()                            │         │
│    │  → Create output directories                     │         │
│    └──────────────────────────────────────────────────┘         │
│    ┌──────────────────────────────────────────────────┐         │
│    │  validate_prerequisites()                        │         │
│    │  → Check required files and dependencies         │         │
│    └──────────────────────────────────────────────────┘         │
│    ┌──────────────────────────────────────────────────┐         │
│    │  generate_video_configs()                        │         │
│    │  → Create video parameter configurations         │         │
│    │  → Returns: (moving_configs, stopped_configs)    │         │
│    └──────────────────────────────────────────────────┘         │
│    ┌──────────────────────────────────────────────────┐         │
│    │  _generate_configs(count, is_moving)             │         │
│    │  → Generate N configs with parameters            │         │
│    │  → Handles random vs fixed modes                 │         │
│    └──────────────────────────────────────────────────┘         │
│    ┌──────────────────────────────────────────────────┐         │
│    │  _generate_all_combinations()                    │         │
│    │  → Parse comma-separated parameters              │         │
│    │  → Generate cartesian product                    │         │
│    └──────────────────────────────────────────────────┘         │
│    ┌──────────────────────────────────────────────────┐         │
│    │  generate_videos(configs)                        │         │
│    │  → Call synthetic_data_generation.py             │         │
│    │  → Generate videos via subprocess                │         │
│    │  → Returns: list of video file paths             │         │
│    └──────────────────────────────────────────────────┘         │
│    ┌──────────────────────────────────────────────────┐         │
│    │  create_dataset_json(video_files)                │         │
│    │  → Create ShareGPT-formatted JSON                │         │
│    │  → Map videos to prompts/responses               │         │
│    └──────────────────────────────────────────────────┘         │
│    ┌──────────────────────────────────────────────────┐         │
│    │  update_dataset_info()                           │         │
│    │  → Register dataset in dataset_info.json         │         │
│    └──────────────────────────────────────────────────┘         │
│    ┌──────────────────────────────────────────────────┐         │
│    │  generate_summary_report()                       │         │
│    │  → Create detailed summary report                │         │
│    └──────────────────────────────────────────────────┘         │
│    ┌──────────────────────────────────────────────────┐         │
│    │  build()                                         │         │
│    │  → Main orchestration method                     │         │
│    │  → Calls all stages in sequence                  │         │
│    └──────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Feature Breakdown

### 1. Parameter Variation Modes

```
┌───────────────────────────────────────────────────────────────┐
│                    PARAMETER MODES                             │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  MODE 1: Fixed Parameters (default)                           │
│  ────────────────────────────────────────────────────         │
│  • Uses exact values from CLI arguments                       │
│  • Speeds evenly distributed across range                     │
│  • Deterministic output                                       │
│  • Good for controlled experiments                            │
│                                                                │
│  Example:                                                      │
│    --texture_type stripes                                     │
│    --speed_range 2.0,5.0                                      │
│    → All videos use stripes, speeds: 2.0, 3.5, 5.0           │
│                                                                │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  MODE 2: Random Variation (--vary_parameters)                 │
│  ────────────────────────────────────────────────────         │
│  • Randomly varies all parameters                             │
│  • Cycles through texture/lighting types                      │
│  • Uniform random for continuous values                       │
│  • Good for diverse training data                             │
│                                                                │
│  Randomized Parameters:                                       │
│    • texture_type: random from 7 types                        │
│    • direction: random from 4 directions                      │
│    • speed: uniform(speed_min, speed_max)                     │
│    • view_angle: uniform(-30, 30)                             │
│    • brightness: uniform(-0.1, 0.2)                           │
│    • contrast: uniform(0.7, 1.3)                              │
│    • lighting_variation: random from 5 types                  │
│    • lighting_intensity: uniform(0.3, 0.8)                    │
│    • motion_blur: randint(1, 3) if speed > 5                  │
│    • camera_noise: uniform(0.0, 0.3)                          │
│    • edge_width: uniform(0.05, 0.15)                          │
│                                                                │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  MODE 3: Combinatorial (comma-separated values)               │
│  ────────────────────────────────────────────────────         │
│  • Generates ALL combinations of specified values             │
│  • Overrides --num_videos                                     │
│  • Deterministic and exhaustive                               │
│  • Good for systematic testing                                │
│                                                                │
│  Example:                                                      │
│    --texture_type stripes,noise,rubber                        │
│    --speed_range 2.0,5.0                                      │
│    --fps 15,30                                                │
│    → 12 videos (3 textures × 2 speeds × 2 fps)               │
│                                                                │
│  Combinable Parameters:                                       │
│    • texture_type, direction, speed_range                     │
│    • resolution, fps, duration                                │
│    • view_angle, brightness, contrast                         │
│    • lighting_variation, lighting_intensity                   │
│    • motion_blur, camera_noise, edge_width                    │
│    • stripe_width, stripe_spacing, stripe_gray                │
│    • background_gray, stripe_distance_variance                │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### 2. Video Generation Parameters

```
┌───────────────────────────────────────────────────────────────┐
│                  VIDEO PARAMETERS                              │
├───────────────────────────────────────────────────────────────┤
│  TEXTURE TYPES (7 options):                                   │
│    • stripes            - Classic black/white stripes         │
│    • noise              - Random noise pattern                │
│    • rubber             - Rubber-like texture                 │
│    • grid               - Grid pattern                        │
│    • diamond_plate      - Diamond plate pattern               │
│    • factory_dark       - Dark factory texture                │
│    • factory_dark_stripes - Dark factory with stripes         │
│    • subtle_gray_stripes - Subtle gray stripe pattern         │
├───────────────────────────────────────────────────────────────┤
│  MOTION PARAMETERS:                                           │
│    • direction: left, right, up, down                         │
│    • speed: 0.0 (stopped) or 1.0-8.0 px/frame (moving)       │
│    • motion_blur: 0-10 (blur amount)                          │
├───────────────────────────────────────────────────────────────┤
│  CAMERA PARAMETERS:                                           │
│    • view_angle: -30° to +30° (camera perspective)           │
│    • brightness: -1.0 to +1.0 (adjustment)                    │
│    • contrast: 0.5 to 2.0 (adjustment)                        │
│    • camera_noise: 0.0 to 1.0 (noise level)                   │
├───────────────────────────────────────────────────────────────┤
│  LIGHTING PARAMETERS:                                         │
│    • lighting_variation:                                      │
│        - none, vignette, gradient_lr,                         │
│          gradient_tb, spotlight                               │
│    • lighting_intensity: 0.0 to 1.0                           │
├───────────────────────────────────────────────────────────────┤
│  VIDEO FORMAT:                                                │
│    • resolution: WxH (default: 640x480)                       │
│    • fps: frames per second (default: 4)                      │
│    • duration: seconds (default: 12.0)                        │
├───────────────────────────────────────────────────────────────┤
│  BELT ENCLOSURE:                                              │
│    • edge_width: 0.0-1.0 (percentage of frame)                │
├───────────────────────────────────────────────────────────────┤
│  SUBTLE GRAY STRIPES (special texture):                       │
│    • stripe_width: pixel width                                │
│    • stripe_spacing: pixels between stripes                   │
│    • stripe_gray: 0-255 (stripe color)                        │
│    • background_gray: 0-255 (background color)                │
│    • stripe_distance_variance: spacing variation              │
├───────────────────────────────────────────────────────────────┤
│  OBJECT PLACEMENT (optional):                                 │
│    • add_object: enable/disable                               │
│    • object_type: box, circle, random                         │
│    • object_position: center, left, right, random             │
│    • object_size: small, medium, large                        │
│    • num_objects: count of objects                            │
├───────────────────────────────────────────────────────────────┤
│  CAMERA BLUR (optional):                                      │
│    • add_blur: enable/disable                                 │
│    • blur_type: motion, gaussian, random                      │
│    • blur_intensity: light, medium, heavy, or 0.0-1.0         │
│    • random_blur_variation: enable/disable                    │
└───────────────────────────────────────────────────────────────┘
```

### 3. Output Files Structure

```
LLaMA-Factory/
├── data/
│   ├── <dataset_name>/                          # Video directory
│   │   ├── treadmill_0001_stripes_right_speed3.5_*.mp4
│   │   ├── treadmill_0002_noise_left_speed0.0_*.mp4
│   │   ├── treadmill_0003_rubber_up_speed5.2_*.mp4
│   │   ├── ...
│   │   ├── <dataset_name>_build_log.txt         # Generation log
│   │   └── <dataset_name>_summary.txt           # Summary report
│   │
│   ├── <dataset_name>.json                      # ShareGPT JSON
│   │   [
│   │     {
│   │       "messages": [...],
│   │       "videos": ["data/<dataset_name>/treadmill_0001_*.mp4"]
│   │     },
│   │     ...
│   │   ]
│   │
│   └── dataset_info.json                        # LLaMA-Factory registry
│       {
│         "<dataset_name>": {
│           "file_name": "<dataset_name>.json",
│           "formatting": "sharegpt",
│           "columns": {...},
│           "tags": {...},
│           "media_dir": "data/<dataset_name>"
│         },
│         ...
│       }
│
└── building_dataset.py                          # Main script
```

### 4. ShareGPT Format Details

```
┌───────────────────────────────────────────────────────────────┐
│                    SHAREGPT FORMAT                             │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  Structure:                                                    │
│  ──────────                                                    │
│  [                                                             │
│    {                                                           │
│      "messages": [                                             │
│        {                                                       │
│          "role": "user",                                       │
│          "content": "<video>Is there movement in the video? Answer only with yes or no."
│        },                                                      │
│        {                                                       │
│          "role": "assistant",                                  │
│          "content": "yes"    # or "no"                         │
│        }                                                       │
│      ],                                                        │
│      "videos": [                                               │
│        "data/<dataset_name>/treadmill_0001_*.mp4"             │
│      ]                                                         │
│    },                                                          │
│    ...                                                         │
│  ]                                                             │
│                                                                │
├───────────────────────────────────────────────────────────────┤
│  Prompts:                                                      │
│  ─────────                                                     │
│  User: "<video>Is there movement in the video? Answer only with yes or no."
│                                                                │
│  Assistant (moving): "yes"                                     │
│  Assistant (stopped): "no"                                     │
│                                                                │
├───────────────────────────────────────────────────────────────┤
│  Design Rationale:                                            │
│  ─────────────────                                            │
│  • Simple binary classification task                          │
│  • Compact prompts reduce token usage                         │
│  • Yes/no answers are easy to evaluate                        │
│  • Consistent format across all samples                       │
│  • <video> tag marks video input location                     │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

---

## Usage Patterns

### Pattern 1: Quick Test Dataset

```bash
# Generate 10 videos for testing
python building_dataset.py \
  --dataset_name test_dataset \
  --num_videos 10 \
  --seed 42

Output: 5 moving + 5 stopped videos
```

### Pattern 2: Diverse Training Dataset

```bash
# Generate 100 videos with varied parameters
python building_dataset.py \
  --dataset_name training_100 \
  --num_videos 100 \
  --vary_parameters \
  --seed 123

Output: 50 moving + 50 stopped videos with random variations
```

### Pattern 3: Systematic Texture Testing

```bash
# Test all textures at specific speeds
python building_dataset.py \
  --dataset_name texture_test \
  --texture_type stripes,noise,rubber,grid,diamond_plate \
  --speed_range 0.0,3.0 \
  --direction right

Output: 10 videos (5 textures × 2 speeds)
```

### Pattern 4: Comprehensive Parameter Sweep

```bash
# Generate all combinations of parameters
python building_dataset.py \
  --dataset_name param_sweep \
  --texture_type stripes,noise \
  --speed_range 2.0,5.0 \
  --fps 15,30 \
  --duration 5.0,10.0 \
  --view_angle -15,0,15

Output: 24 videos (2×2×2×2×3 = 24 combinations)
```

### Pattern 5: Subtle Gray Stripes Experiment

```bash
# Test subtle gray stripe variations
python building_dataset.py \
  --dataset_name gray_stripes \
  --texture_type subtle_gray_stripes \
  --stripe_gray 110,115,120,125,130,135 \
  --background_gray 135,140,145 \
  --speed_range 0.0,3.0

Output: 36 videos (6×3×2 = 36 combinations)
```

---

## Integration with LLaMA-Factory

```
┌─────────────────────────────────────────────────────────────────┐
│              LLAMA-FACTORY TRAINING INTEGRATION                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │  1. Dataset Builder generates dataset    │
        │     • Videos: data/<dataset_name>/       │
        │     • JSON: data/<dataset_name>.json     │
        │     • Registered in dataset_info.json    │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │  2. Update training configuration        │
        │     Edit: examples/train_qlora/          │
        │           qwen25vl_lora_sft.yaml         │
        │                                           │
        │     dataset: <dataset_name>              │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │  3. Run LLaMA-Factory training           │
        │                                           │
        │     llamafactory-cli train \             │
        │       examples/train_qlora/              │
        │       qwen25vl_lora_sft.yaml             │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │  4. Model trained on video classification │
        │     • Learns: moving vs stopped           │
        │     • Input: video frames                 │
        │     • Output: yes/no                      │
        └──────────────────────────────────────────┘
```

---

## Error Handling & Validation

```
┌─────────────────────────────────────────────────────────────────┐
│                    VALIDATION CHECKS                             │
├─────────────────────────────────────────────────────────────────┤
│  Prerequisite Validation:                                       │
│    ✓ synthetic_data_generation.py exists                        │
│    ✓ dataset_info.json exists                                   │
│    ✓ Python3 is available                                       │
│    ✓ Output directory is writable                               │
│                                                                  │
│  Video Generation Validation:                                   │
│    ✓ Subprocess completes successfully (returncode == 0)        │
│    ✓ Generated video file exists                                │
│    ✓ Video file has non-zero size                               │
│    ✓ Timeout protection (300s per video)                        │
│                                                                  │
│  Dataset Validation:                                            │
│    ✓ At least one video generated                               │
│    ✓ Video filenames contain speed information                  │
│    ✓ JSON is valid ShareGPT format                              │
│    ✓ All video paths are relative                               │
│                                                                  │
│  Error Recovery:                                                │
│    • Continues on individual video failures                     │
│    • Logs all errors to build log                               │
│    • Creates backup of dataset_info.json                        │
│    • Provides detailed error messages                           │
│    • Graceful keyboard interrupt handling                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Statistics & Reporting

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRACKED STATISTICS                           │
├─────────────────────────────────────────────────────────────────┤
│  Video Counts:                                                  │
│    • total_videos                                               │
│    • moving_videos                                              │
│    • stopped_videos                                             │
│                                                                  │
│  Parameter Distributions:                                       │
│    • texture_distribution: {texture: count}                     │
│    • direction_distribution: {direction: count}                 │
│    • speed_range: {min, max}                                    │
│                                                                  │
│  File Metrics:                                                  │
│    • total_size_bytes                                           │
│    • average_size_per_video                                     │
│                                                                  │
│  Generation Metadata:                                           │
│    • generation_date                                            │
│    • seed_used                                                  │
│    • configuration_parameters                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      GENERATED REPORTS                           │
├─────────────────────────────────────────────────────────────────┤
│  <dataset_name>_build_log.txt:                                  │
│    • Timestamped log of all operations                          │
│    • Configuration parameters                                   │
│    • Per-video generation status                                │
│    • Error messages and warnings                                │
│    • Complete execution trace                                   │
│                                                                  │
│  <dataset_name>_summary.txt:                                    │
│    • Dataset statistics summary                                 │
│    • Moving vs stopped split                                    │
│    • Texture/direction distributions                            │
│    • Speed ranges                                               │
│    • File sizes                                                 │
│    • Output file locations                                      │
│    • Ready-to-train confirmation                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Performance Characteristics

### Generation Speed

| Configuration | Videos/min | 100 Videos Time |
|---------------|-----------|-----------------|
| 640x480, 4fps, 12s | ~40-50 | ~2-3 minutes |
| 640x480, 30fps, 5s | ~20-30 | ~3-5 minutes |
| 1280x720, 30fps, 5s | ~10-15 | ~7-10 minutes |
| 1920x1080, 60fps, 10s | ~3-5 | ~20-30 minutes |

### Disk Usage

| Configuration | Per Video | 100 Videos |
|---------------|-----------|------------|
| 640x480, 4fps, 12s | ~400-600 KB | ~40-60 MB |
| 640x480, 30fps, 5s | ~1.0 MB | ~100 MB |
| 1280x720, 30fps, 5s | ~2.5 MB | ~250 MB |
| 1920x1080, 60fps, 10s | ~10 MB | ~1 GB |

---

## Dependencies

```
External Scripts:
  └─ data/synthetic_treadmill/synthetic_data_generation.py
     ├─ Generates synthetic treadmill videos
     ├─ Requires: opencv-python-headless
     ├─ Requires: numpy<2.0.0
     └─ Called via subprocess

Configuration Files:
  └─ data/dataset_info.json
     ├─ LLaMA-Factory dataset registry
     ├─ Updated automatically
     └─ Backup created before modification

Python Standard Library:
  • argparse - CLI argument parsing
  • json - JSON file handling
  • logging - Logging functionality
  • os - File system operations
  • pathlib - Path manipulation
  • re - Regular expressions
  • shutil - File operations
  • subprocess - External command execution
  • random - Random number generation
  • itertools - Combinatorial operations
  • datetime - Timestamps
```

---

## Best Practices

### 1. Dataset Naming
- Use descriptive names: `treadmill_stripes_varied`, `test_100_rubber`
- Include date/version for experiments: `exp_20251201_v1`
- Avoid spaces and special characters

### 2. Seed Management
- Always set `--seed` for reproducibility
- Use different seeds for train/test splits
- Document seeds in experiment logs

### 3. Parameter Selection
- Start with `--vary_parameters` for general training
- Use combination mode for systematic testing
- Test with small `--num_videos` first (10-20)
- Scale up once validated

### 4. Docker Usage
- Videos generate faster in Docker (native environment)
- Mount output directory for persistence
- Use absolute paths inside container

### 5. Storage Management
- Monitor disk space before large generations
- Clean up test datasets regularly
- Compress or archive old datasets

---

## Related Documentation

- `README_DATASET_BUILDER.md` - User guide and examples
- `data/synthetic_treadmill/synthetic_data_generation.py` - Video generation
- `data/dataset_info.json` - Dataset registry
- `examples/train_qlora/qwen25vl_lora_sft.yaml` - Training config
- `evaluate_treadmill_lora.py` - Evaluation script

---

**Document Version**: 1.0
**Last Updated**: 2025-12-07
**Author**: AI-Generated
