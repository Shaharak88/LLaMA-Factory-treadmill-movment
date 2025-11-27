# Changelog: Subtle Gray Stripes Texture Implementation

**Date:** 2025-11-27
**Feature:** Low-contrast gray stripe texture for motion detection threshold testing

## Overview
Added `subtle_gray_stripes` texture type to support systematic testing of Qwen2.5-VL's motion detection capabilities at varying contrast levels. This texture enables researchers to find the contrast threshold below which the model can no longer reliably detect belt movement.

## Changes Made

### 1. synthetic_treadmill/synthetic_data_generation.py

#### Added New Method: `generate_subtle_gray_stripes()` (Lines 327-380)
- **Purpose:** Generate low-contrast gray stripe patterns for challenging motion detection
- **Key Features:**
  - Configurable stripe width (default: 10px, thinner than standard stripes)
  - Configurable stripe spacing (default: 60px, wider apart than standard)
  - Parameterized stripe gray level (0-255, default: 125)
  - Parameterized background gray level (0-255, default: 140)
  - Auto-orientation perpendicular to motion direction
  - Subtle noise addition for realism (±3 gray levels)

#### Updated `generate_texture()` Method (Lines 382-415)
- Added `subtle_gray_stripes` to texture type dispatcher
- Updated docstring to include new texture type

#### Updated `generate_video()` Method (Lines 860-877)
- Modified texture generation to pass additional parameters
- Added conditional parameter passing for stripe_width, stripe_spacing, stripe_gray, background_gray
- Ensures backward compatibility with existing texture types

#### Updated `parse_arguments()` Function (Lines 1012-1028)
- Added `subtle_gray_stripes` to texture_type choices
- Added 4 new command-line arguments:
  - `--stripe_width`: Stripe width in pixels (default: 10, supports comma-separated values)
  - `--stripe_spacing`: Spacing between stripes in pixels (default: 60, supports comma-separated values)
  - `--stripe_gray`: Stripe gray level 0-255 (default: 125, supports comma-separated values)
  - `--background_gray`: Background gray level 0-255 (default: 140, supports comma-separated values)

#### Updated `main()` Function (Lines 1171-1199)
- Added parsing of new parameters from command-line arguments
- Handles both single values and comma-separated lists
- Adds parameters to base_config dictionary

### 2. building_dataset.py

#### Updated `_generate_all_combinations()` Method (Lines 176-247)
- Added parsing of new parameters: stripe_widths, stripe_spacings, stripe_grays, background_grays
- Added parameters to itertools.product() for full combination generation
- Updated combo unpacking to include 4 new parameters
- Added parameters to config dictionary for each combination

#### Updated `_generate_configs()` Method (Lines 358-361)
- Added default values for new parameters in non-variation mode
- Ensures parameters are available even when not using --vary_parameters

#### Updated `generate_videos()` Method (Lines 404-412)
- Added conditional parameter passing to subprocess command
- Only adds parameters if present in config
- Maintains backward compatibility

#### Updated `generate_video_configs()` Method (Lines 267)
- Added new parameters to combination mode detection list
- Ensures combination mode activates when any new parameter has comma-separated values

#### Updated `parse_arguments()` Function (Lines 854-862)
- Added 4 new command-line arguments with same defaults as synthetic_data_generation.py
- Updated texture_type help text to include `subtle_gray_stripes`
- All parameters support comma-separated values for combination mode

### 3. run_full_pipeline.py

#### Updated `_build_dataset_command()` Method (Lines 197-209)
- Added conditional passing of new parameters to building_dataset.py
- Uses hasattr() to check parameter existence for backward compatibility
- Only passes parameters if they are defined in args

#### Updated `parse_arguments()` Function (Lines 493-501)
- Added 4 new command-line arguments to dataset_group
- Consistent defaults across all three scripts (10, 60, 125, 140)
- All parameters support comma-separated values

## Usage Examples

### Example 1: Single Training Video (Fixed Contrast)
```bash
python3 synthetic_treadmill/synthetic_data_generation.py \
  --texture_type subtle_gray_stripes \
  --stripe_gray 125 \
  --background_gray 140 \
  --stripe_width 10 \
  --stripe_spacing 60 \
  --num_videos 1
```

### Example 2: Test Dataset with Multiple Stripe Gray Values
```bash
python3 building_dataset.py \
  --dataset_name subtle_gray_test \
  --texture_type subtle_gray_stripes \
  --stripe_gray 110,115,120,125,130,135 \
  --background_gray 140 \
  --stripe_width 10 \
  --stripe_spacing 60
```
This generates 6 videos, one for each stripe gray value (110, 115, 120, 125, 130, 135).

### Example 3: Full Pipeline - Training with Fixed Contrast
```bash
python3 run_full_pipeline.py \
  --dataset_name treadmill_train \
  --num_videos 100 \
  --texture_type subtle_gray_stripes \
  --stripe_gray 125 \
  --background_gray 140 \
  --seed 42
```

### Example 4: Full Pipeline - Testing with Varied Contrasts
```bash
python3 run_full_pipeline.py \
  --dataset_name treadmill_test \
  --texture_type subtle_gray_stripes \
  --stripe_gray 110,115,120,130,135 \
  --background_gray 140 \
  --skip_training \
  --skip_evaluation
```

## Technical Details

### Stripe Pattern Design
- **Orientation:** Automatically perpendicular to motion direction (horizontal stripes for left/right motion, vertical for up/down)
- **Contrast:** User-controllable delta between stripe and background gray (e.g., delta=15 for stripe=125, bg=140)
- **Spacing:** Wider spacing (60px default) vs standard stripes (20px) to reduce visual density
- **Width:** Thinner stripes (10px default) vs standard stripes (20px)

### Parameter Combination Support
All new parameters support comma-separated values for systematic testing:
- `--stripe_gray 110,115,120,125,130,135` generates 6 variations
- Can combine with other parameters: `--texture_type subtle_gray_stripes,stripes --stripe_gray 110,125`
- Generates Cartesian product of all comma-separated parameters

### Backward Compatibility
- All new parameters are optional with sensible defaults
- Existing texture types unaffected
- Scripts check for parameter existence before using them
- No breaking changes to existing functionality

## Files Modified
1. `synthetic_treadmill/synthetic_data_generation.py` - 80 lines added/modified
2. `building_dataset.py` - 45 lines added/modified
3. `run_full_pipeline.py` - 25 lines added/modified

## Testing Recommendations
1. Generate single video to verify texture appearance
2. Test combination mode with multiple stripe_gray values
3. Verify parameter passing through full pipeline
4. Compare with existing stripe textures for contrast validation

## Future Enhancements
- Add visualization script to compare contrast levels
- Add automatic contrast measurement utilities
- Support for gradient gray transitions within stripes
