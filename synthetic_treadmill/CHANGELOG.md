# Changelog - Synthetic Treadmill Video Generator

## [1.5.0] - 2025-11-26 - Fixed Stationary Belt Enclosure

### Fixed
- **Critical Fix**: Belt enclosure now appears as a single, fixed outline that doesn't move
  - Previously: Enclosure was applied to base texture before seamless tiling, causing it to wrap around stripes and move with the pattern
  - Problem: Outline appeared to "wrap" around internal stripes, creating borders between them
  - Problem: Outline shifted/moved relative to the striped pattern
  - Now: Enclosure is applied AFTER motion is applied to each frame
  - Result: Single, fixed border around the entire belt object that stays stationary
  - Creates much more realistic appearance - frame stays fixed while belt moves inside it

### Changed
- Moved `apply_belt_enclosure()` from before seamless texture creation to within frame generation loop
- Updated video generation pipeline: texture → seamless → motion → **enclosure** → camera effects
- Enclosure is now applied per-frame instead of being part of the texture

### Technical Details
- Removed enclosure application at line ~812 (before `create_seamless_texture()`)
- Added enclosure application at line ~832 (after `apply_motion()` in frame loop)
- Ensures enclosure appears as fixed overlay rather than moving texture element

### Tested
Generated comprehensive test videos to verify the fix:
- **Stripes (no angle)**: Fixed border with vertical stripes moving right
- **Stripes (25° angle)**: Fixed border maintained with perspective transform
- **Factory dark stripes (no angle)**: Fixed border with dark industrial texture
- **Factory dark stripes (30° angle)**: Fixed border with steep perspective
- **Rubber texture (no angle)**: Fixed border with non-stripe texture (edge_width=0.12)
- **Grid texture (20° angle)**: Fixed border with grid pattern moving left
- All test videos located in: `data/synthetic_treadmill/test_enclosure_fix/`
- All videos confirmed: Single stationary outline, no wrapping, texture moves inside frame ✓

## [1.4.0] - 2025-11-26 - Moving Belt Edges & Dark Factory Textures

### Fixed
- **Critical Fix**: Belt enclosure edges now move with the treadmill texture
  - Previously, belt edges/boundaries were applied as static overlay after motion
  - Now applied to base texture BEFORE motion simulation
  - Belt edges and enclosure now correctly translate with the moving belt
  - Creates much more realistic appearance of actual conveyor belts

### Added
- **New texture type: `factory_dark`** - Dark/black factory conveyor belt (no stripes)
  - RGB base color: (25, 25, 25) - very dark like real industrial belts
  - Subtle texture variations simulating worn rubber surface
  - Multi-octave noise for natural appearance
  - Sparse directional wear marks mimicking real factory equipment
  - Usage: `--texture_type factory_dark`

- **New texture type: `factory_dark_stripes`** - Dark factory conveyor with widely-spaced stripes
  - RGB base colors: (20, 20, 20) and (35, 35, 35) for dark industrial look
  - Default stripe width: 50 pixels (wider spacing than standard stripes)
  - Auto-orientation perpendicular to motion (like standard stripes)
  - Minimal noise to maintain dark appearance
  - Usage: `--texture_type factory_dark_stripes`

### Enhanced
- Updated texture type choices in argument parser to include new types
- Updated `generate_varied_configs()` to include new factory textures in dataset variation
- Both new textures suitable for industrial/factory conveyor belt simulation

### Technical Details
- Moved `apply_belt_enclosure()` call from frame loop to before seamless texture creation
- Enclosure is now part of the scrolling texture instead of static overlay
- Updated video generation pipeline: texture → enclosure → seamless → motion → effects

## [1.3.0] - 2025-11-26 - Realistic Belt Enclosure

### Added
- **Belt enclosure rendering**: Videos now include realistic treadmill/conveyor frame
  - Side rails (left/right edges) showing belt boundaries
  - Top and bottom enclosure
  - Belt edge lines with 3D shading for depth perception
  - Corner bolts/screws for mechanical realism
  - New parameter: `--edge_width` (0.05 to 0.2, default: 0.1)
- Automatic application of belt overlay to all generated frames
- Gradient shading on enclosure for 3D depth effect

### Enhanced
- Videos now look much more realistic with visible belt structure
- Better simulation of actual treadmill/conveyor appearance
- Configurable edge width for different belt sizes

### Tested
- Moving stripes video with enclosure ✓
- Stationary rubber video with enclosure ✓
- Moving grid video with wider enclosure (edge_width=0.12) ✓

## [1.2.0] - 2025-11-26 - Stationary Video Support & Bug Fix

### Added
- **Stationary video generation**: `--speed 0` now generates non-moving treadmill videos
  - Perfect for binary classification training (moving vs stationary)
  - Works with all texture types
  - Example: `--speed 0 --texture_type stripes`

### Fixed
- Fixed bug where non-stripe textures (rubber, noise, grid, diamond_plate) failed with `motion_direction` parameter
- Added parameter filtering in `generate_texture()` to only pass `motion_direction` to stripe generator

### Tested
- Stationary videos with stripes texture ✓
- Stationary videos with rubber texture ✓
- All texture types now work correctly with motion_direction parameter

## [1.1.0] - 2025-11-26 - Stripe Orientation Fix

### Fixed
- **Critical Fix**: Automatic stripe orientation perpendicular to motion direction
  - Left/Right motion now generates vertical stripes
  - Up/Down motion now generates horizontal stripes
  - Ensures motion is always visible to human eye and CV models

### Changed
- Added `motion_direction` parameter to `generate_stripes()` function
- Updated video generator to pass motion direction to texture generator
- Enhanced documentation explaining perpendicular orientation requirement

### Tested
- All 4 motion directions (left, right, up, down) verified working
- Generated test videos confirm visible motion in all cases

## [1.0.0] - 2025-11-26 - Initial Release

### Added
- Complete synthetic treadmill/conveyor belt video generation system
- 5 texture types: stripes, noise, rubber, grid, diamond_plate
- 4 motion directions: left, right, up, down
- Variable speed control (pixels per frame)
- Camera effects: view angle, brightness, contrast
- Lighting variations: vignette, gradients, spotlight
- Motion blur and camera noise simulation
- 16 total degrees of freedom for diverse datasets
- Full reproducibility with seed-based generation
- Parameter-encoded filenames
- Docker container compatibility
- Comprehensive documentation (README.md, QUICK_START.md)
- Requirements file for easy installation

### Technical Features
- Seamless texture tiling for infinite scrolling
- Perspective transforms for camera angle simulation
- Multiple lighting patterns for realism
- Programmatic frame generation (NumPy + OpenCV)
- MP4 video output with configurable resolution and FPS

### Documentation
- Complete README with examples
- Quick start guide
- Usage examples for all features
- Docker deployment instructions
