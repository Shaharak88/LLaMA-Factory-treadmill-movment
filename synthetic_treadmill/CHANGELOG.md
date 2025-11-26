# Changelog - Synthetic Treadmill Video Generator

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
