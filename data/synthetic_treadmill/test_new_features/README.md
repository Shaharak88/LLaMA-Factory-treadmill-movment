# Test Videos - New Features Demonstration

Generated: 2025-11-26

## Overview
This directory contains test videos demonstrating the new features added in version 1.4.0:
1. **Moving belt edges fix** - Belt enclosure now moves with the texture
2. **New dark factory textures** - Two new texture types for industrial conveyor belts

## Test Videos

### 1. Dark Factory Texture (No Stripes)

**File:** `treadmill_0000_factory_dark_right_speed3.0_angle0_bright0.00_contr1.00_640x480_seed42.mp4`
- **Texture Type:** `factory_dark`
- **Direction:** Right (moving right)
- **Speed:** 3.0 pixels/frame
- **View Angle:** 0° (frontal view)
- **Features:**
  - Very dark RGB (25, 25, 25) simulating real industrial rubber belts
  - Multi-octave noise for natural worn appearance
  - Sparse directional wear marks
  - Belt edges moving with texture ✓

---

**File:** `treadmill_0000_factory_dark_right_speed3.0_angle25_bright0.00_contr1.00_640x480_seed43.mp4`
- **Texture Type:** `factory_dark`
- **Direction:** Right (moving right)
- **Speed:** 3.0 pixels/frame
- **View Angle:** 25° (viewing from above)
- **Features:**
  - Same dark factory texture with perspective transform
  - Demonstrates how texture looks from angled camera view
  - Belt edges properly aligned and moving with perspective ✓

### 2. Dark Factory Texture with Wide Stripes

**File:** `treadmill_0000_factory_dark_stripes_right_speed3.0_angle0_bright0.00_contr1.00_640x480_seed44.mp4`
- **Texture Type:** `factory_dark_stripes`
- **Direction:** Right (moving right)
- **Speed:** 3.0 pixels/frame
- **View Angle:** 0° (frontal view)
- **Features:**
  - Dark RGB colors: (20, 20, 20) and (35, 35, 35)
  - Stripe width: 50 pixels (wider spacing than standard 20px)
  - Vertical stripes (perpendicular to rightward motion)
  - Clearly visible motion detection
  - Belt edges moving with stripes ✓

---

**File:** `treadmill_0000_factory_dark_stripes_right_speed3.0_angle25_bright0.00_contr1.00_640x480_seed45.mp4`
- **Texture Type:** `factory_dark_stripes`
- **Direction:** Right (moving right)
- **Speed:** 3.0 pixels/frame
- **View Angle:** 25° (viewing from above)
- **Features:**
  - Same wide dark stripes with perspective transform
  - Motion clearly visible from angled view
  - Belt edges properly transformed and moving ✓

### 3. Moving Belt Edges Demonstration

**File:** `treadmill_0000_stripes_right_speed4.0_angle0_bright0.00_contr1.00_640x480_seed46.mp4`
- **Texture Type:** `stripes` (standard)
- **Direction:** Right (moving right)
- **Speed:** 4.0 pixels/frame (faster for clear demonstration)
- **Edge Width:** 0.12 (wider enclosure for visibility)
- **Features:**
  - Standard striped texture with prominent belt enclosure
  - **KEY FIX:** Belt edges/enclosure now scroll with the texture
  - Previously edges were static overlays - now integrated into moving texture
  - Demonstrates realistic conveyor belt appearance ✓

## What to Look For

### Moving Belt Edges Fix (ALL videos)
Watch the belt edges (the dark frame around the texture):
- ✅ **CORRECT (New):** Edges move/scroll with the belt texture
- ❌ **INCORRECT (Old):** Edges would stay static while texture moved underneath

This fix makes the videos look like a real conveyor belt with visible edges, not a stationary frame with texture behind it.

### Dark Factory Textures
The new `factory_dark` and `factory_dark_stripes` textures:
- Much darker than standard textures (RGB 20-35 vs 80-120)
- Simulate real industrial/factory conveyor belts and equipment
- `factory_dark`: Solid dark with subtle wear patterns
- `factory_dark_stripes`: Dark with widely-spaced stripes (50px vs standard 20px)

### Angle Comparison
Compare the 0° vs 25° angle versions:
- 0° = frontal view (no perspective distortion)
- 25° = viewing from above (perspective transform applied)
- Both show proper belt edge alignment and motion

## Usage Examples

To generate similar videos:

```bash
# Dark factory (no stripes)
python3 synthetic_data_generation.py --texture_type factory_dark --direction right --speed 3.0

# Dark factory with wide stripes
python3 synthetic_data_generation.py --texture_type factory_dark_stripes --direction right --speed 3.0

# With camera angle
python3 synthetic_data_generation.py --texture_type factory_dark --view_angle 25 --direction right --speed 3.0

# Wider belt edges for visibility
python3 synthetic_data_generation.py --texture_type stripes --edge_width 0.12 --speed 4.0
```

## Technical Details

### Changes Made (v1.4.0)
1. **Belt Enclosure Fix:**
   - Moved `apply_belt_enclosure()` from frame loop to before seamless texture creation
   - Pipeline: texture → **enclosure** → seamless → motion → effects
   - Result: Edges are part of scrolling texture, not static overlay

2. **New Texture Generators:**
   - `generate_factory_dark()`: Solid dark texture with noise and wear marks
   - `generate_factory_dark_stripes()`: Dark texture with 50px wide stripes
   - Auto-orientation perpendicular to motion (like standard stripes)

## Verification Checklist

- [x] factory_dark texture renders correctly
- [x] factory_dark texture with angle renders correctly
- [x] factory_dark_stripes texture renders correctly
- [x] factory_dark_stripes texture with angle renders correctly
- [x] Belt edges move with texture (not static)
- [x] Wide stripes are clearly spaced (50px)
- [x] Stripes oriented perpendicular to motion
- [x] All videos generated inside Docker container
- [x] All videos playable and demonstrating correct behavior

## File Locations

- Source code: `/app/synthetic_treadmill/synthetic_data_generation.py`
- Data copy: `/app/data/synthetic_treadmill/synthetic_data_generation.py` (synced)
- Test videos: `/app/data/synthetic_treadmill/test_new_features/`
- Changelog: `/app/synthetic_treadmill/CHANGELOG.md`
