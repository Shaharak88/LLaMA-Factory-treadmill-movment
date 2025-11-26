# Dataset Builder Changelog

## 2025-11-26 - Parameter Combination Mode Feature

### Summary
Added support for comma-separated parameter values to generate all possible combinations of videos from a single command line.

### Changes Made

#### 1. Added itertools import
- **Location**: Line 35
- **Change**: Added `import itertools` to support generating all parameter combinations
- **Reason**: Required for `itertools.product()` to generate cartesian product of all parameter values

#### 2. New Method: `_parse_parameter_values()`
- **Location**: Lines 137-152
- **Purpose**: Parse comma-separated parameter values into lists
- **Parameters**:
  - `param_str`: Comma-separated values (e.g., "4,6,8" or "stripes,noise")
  - `param_type`: Type to convert values to (int, float, str)
- **Returns**: List of parsed values
- **Example**: `"4,6,8"` → `[4, 6, 8]`

#### 3. New Method: `_generate_all_combinations()`
- **Location**: Lines 154-235
- **Purpose**: Generate all parameter combinations based on comma-separated arguments
- **Functionality**:
  - Parses all parameters that might have multiple values
  - Uses `itertools.product()` to generate all combinations
  - Creates config dictionaries for each combination
  - Automatically determines if video is moving based on speed
- **Returns**: List of all parameter combination configs
- **Example**:
  - Input: `--texture_type stripes,noise --speed_range 2.0,5.0`
  - Output: 4 configs (2 textures × 2 speeds)

#### 4. Enhanced Method: `generate_video_configs()`
- **Location**: Lines 237-276
- **Changes**:
  - Added detection for comma-separated parameters
  - Automatically switches to combination mode when any parameter has multiple values
  - Falls back to original behavior when no combinations detected
  - Separates moving and stopped configs automatically
- **New Behavior**:
  - When combinations detected: generates exact number of videos based on all combinations
  - When no combinations: uses original 50/50 moving/stopped split

#### 5. Updated Argument Parser Types
- **Location**: Lines 802-823
- **Changed Parameters** (from int/float to str to accept comma-separated values):
  - `--fps`: `type=int` → `type=str`
  - `--duration`: `type=float` → `type=str`
  - `--view_angle`: `type=float` → `type=str`
  - `--brightness`: `type=float` → `type=str`
  - `--contrast`: `type=float` → `type=str`
  - `--lighting_intensity`: `type=float` → `type=str`
  - `--motion_blur`: `type=int` → `type=str`
  - `--camera_noise`: `type=float` → `type=str`
  - `--edge_width`: `type=float` → `type=str`
- **Reason**: String type allows both single values and comma-separated values

#### 6. Updated Help Text
- **Location**: Lines 726-771, 794-823
- **Changes**:
  - Added examples showing combination mode usage
  - Updated all parameter help text to mention comma-separated value support
  - Added Docker usage example with combinations
- **New Examples**:
  ```bash
  # 4 textures x 2 speeds = 8 videos
  python building_dataset.py \
    --dataset_name combo_dataset \
    --texture_type stripes,noise,rubber,grid \
    --speed_range 2.0,5.0

  # 2 fps x 2 durations x 2 textures = 8 videos
  python building_dataset.py \
    --dataset_name multi_combo \
    --fps 15,30 \
    --duration 3.0,5.0 \
    --texture_type stripes,noise
  ```

### Feature Capabilities

#### All Parameters Supporting Combinations
1. `--texture_type`: stripes,noise,rubber,grid,diamond_plate,factory_dark,factory_dark_stripes
2. `--direction`: left,right,up,down
3. `--speed_range`: Any comma-separated float values (e.g., 2.0,4.0,6.0)
4. `--fps`: Any comma-separated integers (e.g., 15,30,60)
5. `--duration`: Any comma-separated floats (e.g., 3.0,5.0,10.0)
6. `--resolution`: Any comma-separated WxH values (e.g., 640x480,800x600)
7. `--view_angle`: Any comma-separated angles (e.g., -15,0,15)
8. `--brightness`: Any comma-separated values (e.g., -0.2,0.0,0.2)
9. `--contrast`: Any comma-separated values (e.g., 0.8,1.0,1.2)
10. `--lighting_variation`: none,vignette,gradient_lr,gradient_tb,spotlight
11. `--lighting_intensity`: Any comma-separated values (e.g., 0.3,0.5,0.7)
12. `--motion_blur`: Any comma-separated integers (e.g., 0,1,2,3)
13. `--camera_noise`: Any comma-separated values (e.g., 0.0,0.1,0.2,0.3)
14. `--edge_width`: Any comma-separated values (e.g., 0.05,0.1,0.15)

#### Behavior
- **Combination Mode**: Activated when ANY parameter has comma-separated values
- **Video Count**: Automatically determined by number of combinations (ignores `--num_videos` in combination mode)
- **Moving/Stopped**: Automatically determined by speed value (speed > 0.0 = moving)
- **Original Mode**: Still works when no parameters have multiple values
- **Backward Compatible**: All existing commands work exactly as before

### Usage Examples

#### Example 1: 4 Videos (4 Textures)
```bash
docker exec llamafactory python3 /app/building_dataset.py \
  --dataset_name test_4_textures \
  --texture_type stripes,noise,rubber,grid \
  --speed_range 3.0 \
  --fps 30 \
  --duration 5.0
```
Result: 4 videos (1 of each texture, all moving at 3.0 speed)

#### Example 2: 8 Videos (4 Textures × 2 Speeds)
```bash
docker exec llamafactory python3 /app/building_dataset.py \
  --dataset_name test_8_videos \
  --texture_type stripes,noise,rubber,grid \
  --speed_range 0.0,5.0
```
Result: 8 videos (4 moving, 4 stopped)

#### Example 3: Complex Combinations
```bash
docker exec llamafactory python3 /app/building_dataset.py \
  --dataset_name complex_combo \
  --texture_type stripes,noise \
  --fps 15,30 \
  --duration 3.0,5.0 \
  --speed_range 2.0,5.0
```
Result: 2×2×2×2 = 16 videos

### Testing Plan
1. Test with 4 texture types as requested
2. Verify all videos are generated correctly
3. Check dataset JSON is created properly
4. Confirm dataset_info.json is updated

### Backward Compatibility
✅ All existing commands continue to work
✅ No breaking changes to existing functionality
✅ New feature is opt-in via comma-separated values

### Files Modified
1. `/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/building_dataset.py`
   - Added itertools import
   - Added `_parse_parameter_values()` method
   - Added `_generate_all_combinations()` method
   - Enhanced `generate_video_configs()` method
   - Updated argument parser types and help text
   - Added usage examples in epilog

### Files Created
1. `/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/DATASET_BUILDER_CHANGELOG.md` (this file)
