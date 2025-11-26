# Synthetic Treadmill Video Dataset Generator

A comprehensive Python tool for generating synthetic videos of moving treadmill/conveyor-belt surfaces for computer vision applications. This tool provides full control over visual parameters to create diverse, reproducible datasets for training and testing motion detection models.

## Features

### 🎨 Multiple Texture Types
- **Stripes**: Classic conveyor belt segment pattern (automatically oriented perpendicular to motion direction for visibility)
- **Noise**: Rough rubber surface with Perlin-like noise
- **Rubber**: Textured surface with raised bumps/dimples
- **Grid**: Tiled pattern with grid lines
- **Diamond Plate**: Industrial metal tread plate pattern

**Important**: The script automatically orients stripe patterns perpendicular to the motion direction. For example, if the belt moves left/right, stripes will be vertical; if the belt moves up/down, stripes will be horizontal. This ensures motion is always visible to the human eye and to computer vision models.

### 🎬 Motion Control
- **Directions**: Left, Right, Up, Down
- **Variable Speed**: Configurable pixels per frame (0 to 20+)
  - **Speed 0**: Generates stationary (non-moving) videos - perfect for training binary classifiers!
  - **Speed 0.5-20+**: Moving belt at various speeds
- **Seamless Looping**: Infinite scrolling effect with no visible seams

### 📷 Camera Effects
- **View Angle**: Perspective transforms (-45° to +45°)
- **Brightness**: Global brightness adjustment
- **Contrast**: Contrast modification
- **Lighting Variations**:
  - Vignette (darkening at edges)
  - Gradients (left-to-right, top-to-bottom)
  - Spotlight (circular illumination)
- **Motion Blur**: Directional blur for high-speed simulation
- **Camera Noise**: Simulated sensor noise

### 🔧 Full Reproducibility
- Fixed random seed support
- Deterministic generation
- Parameter-encoded filenames

## Installation

### Prerequisites

The script requires OpenCV and NumPy. Install dependencies using pip:

```bash
pip install opencv-python numpy
```

Or using the provided requirements file:

```bash
pip install -r requirements.txt
```

### Docker Setup

If you're using the LLaMA-Factory Docker container:

```bash
# Install dependencies inside the container
docker exec llamafactory pip install opencv-python-headless "numpy<2.0.0"

# Copy the synthetic_treadmill directory to a mounted location
cp -r synthetic_treadmill data/

# Run the script inside the container
docker exec llamafactory python3 /app/data/synthetic_treadmill/synthetic_data_generation.py --help
```

## Usage

### Basic Usage

Generate a single video with default settings:

```bash
python synthetic_data_generation.py --num_videos 1
```

### Command-Line Arguments

#### Basic Parameters
- `--num_videos`: Number of videos to generate (default: 1)
- `--output_dir`: Output directory path (default: ./synthetic_videos)
- `--seed`: Random seed for reproducibility (default: 42)

#### Video Parameters
- `--fps`: Frames per second (default: 30)
- `--duration`: Video duration in seconds (default: 5.0)
- `--resolution`: Video resolution as WxH (default: 640x480)

#### Motion Parameters
- `--direction`: Belt motion direction [left, right, up, down] (default: right)
- `--speed`: Motion speed in pixels per frame (default: 2.0)

#### Texture Parameters
- `--texture_type`: Type of belt texture [stripes, noise, rubber, grid, diamond_plate] (default: stripes)
- `--background_color`: Background color as R,G,B (default: 80,80,80)

#### Camera/Lighting Parameters
- `--view_angle`: Camera viewing angle in degrees, -45 to 45 (default: 0.0)
- `--brightness`: Brightness adjustment, -1.0 to 1.0 (default: 0.0)
- `--contrast`: Contrast adjustment, 0.5 to 2.0 (default: 1.0)
- `--lighting_variation`: Type of lighting [none, vignette, gradient_lr, gradient_tb, spotlight] (default: none)
- `--lighting_intensity`: Intensity of lighting variation, 0.0 to 1.0 (default: 0.5)

#### Additional Effects
- `--motion_blur`: Motion blur amount in pixels, 0 to 10 (default: 0)
- `--camera_noise`: Camera sensor noise level, 0.0 to 1.0 (default: 0.0)

#### Variation Mode
- `--vary_parameters`: Automatically vary parameters across multiple videos

## Examples

### Example 1: Basic Video Generation

Generate 5 videos with default settings:

```bash
python synthetic_data_generation.py --num_videos 5
```

**Output:**
- 5 videos with stripe texture, rightward motion
- 640x480 resolution at 30 FPS
- 5 seconds duration each

### Example 1b: Stationary (Non-Moving) Videos

Generate stationary treadmill videos for binary classification training:

```bash
# Stationary stripes
python synthetic_data_generation.py --speed 0 --texture_type stripes

# Stationary rubber texture
python synthetic_data_generation.py --speed 0 --texture_type rubber
```

**Use Case**: Perfect for training "moving" vs "stationary" classifiers!

### Example 2: Specific Texture and Motion

Generate a rubber texture moving left at high speed:

```bash
python synthetic_data_generation.py \
  --texture_type rubber \
  --direction left \
  --speed 5.0 \
  --duration 10
```

### Example 3: Camera Perspective and Lighting

Generate video with tilted camera view and spotlight lighting:

```bash
python synthetic_data_generation.py \
  --view_angle 30 \
  --lighting_variation spotlight \
  --lighting_intensity 0.8 \
  --brightness 0.1
```

### Example 4: High-Speed Motion with Blur

Simulate fast-moving belt with motion blur:

```bash
python synthetic_data_generation.py \
  --speed 10 \
  --motion_blur 3 \
  --direction right
```

### Example 5: Diverse Dataset Generation

Generate 20 videos with automatically varied parameters:

```bash
python synthetic_data_generation.py \
  --num_videos 20 \
  --vary_parameters \
  --seed 42 \
  --output_dir ./treadmill_dataset
```

**What this does:**
- Cycles through all texture types
- Varies directions (left, right, up, down)
- Randomizes speeds (1.0 to 8.0 px/frame)
- Randomizes view angles (-30° to 30°)
- Applies different lighting conditions
- Adds motion blur for high-speed videos
- All with seed-based reproducibility

### Example 6: High-Resolution Dataset

Generate HD videos:

```bash
python synthetic_data_generation.py \
  --resolution 1920x1080 \
  --fps 60 \
  --duration 10 \
  --num_videos 10 \
  --vary_parameters
```

### Example 7: Specific Color Scheme

Generate videos with custom background color:

```bash
python synthetic_data_generation.py \
  --texture_type grid \
  --background_color 120,120,120 \
  --contrast 1.3 \
  --brightness 0.2
```

### Example 8: All Texture Types

Generate one video for each texture type:

```bash
# Stripes
python synthetic_data_generation.py --texture_type stripes --output_dir ./textures

# Noise
python synthetic_data_generation.py --texture_type noise --output_dir ./textures

# Rubber
python synthetic_data_generation.py --texture_type rubber --output_dir ./textures

# Grid
python synthetic_data_generation.py --texture_type grid --output_dir ./textures

# Diamond Plate
python synthetic_data_generation.py --texture_type diamond_plate --output_dir ./textures
```

### Example 9: Lighting Comparison

Generate videos with different lighting conditions:

```bash
# Vignette
python synthetic_data_generation.py --lighting_variation vignette --lighting_intensity 0.8

# Gradient left-to-right
python synthetic_data_generation.py --lighting_variation gradient_lr --lighting_intensity 0.6

# Gradient top-to-bottom
python synthetic_data_generation.py --lighting_variation gradient_tb --lighting_intensity 0.6

# Spotlight
python synthetic_data_generation.py --lighting_variation spotlight --lighting_intensity 0.9
```

### Example 10: Docker Container Execution

Run inside the Docker container with output to mounted directory:

```bash
docker exec llamafactory python3 /app/data/synthetic_treadmill/synthetic_data_generation.py \
  --num_videos 10 \
  --vary_parameters \
  --output_dir /app/data/synthetic_treadmill/output \
  --resolution 1280x720 \
  --fps 30 \
  --duration 5
```

Access the generated videos:

```bash
ls -lh data/synthetic_treadmill/output/
```

## Output Format

### Video Files

Videos are saved as MP4 files with descriptive filenames encoding the generation parameters:

```
treadmill_0000_stripes_right_speed2.0_angle0_bright0.00_contr1.00_640x480_seed42.mp4
```

Filename breakdown:
- `treadmill_0000`: Video index
- `stripes`: Texture type
- `right`: Motion direction
- `speed2.0`: Motion speed
- `angle0`: View angle
- `bright0.00`: Brightness factor
- `contr1.00`: Contrast factor
- `640x480`: Resolution
- `seed42`: Random seed

### Directory Structure

```
output_dir/
├── treadmill_0000_stripes_right_speed2.0_angle0_bright0.00_contr1.00_640x480_seed42.mp4
├── treadmill_0001_noise_left_speed3.5_angle15_bright-0.10_contr1.20_640x480_seed43.mp4
├── treadmill_0002_rubber_up_speed5.0_angle-20_bright0.05_contr0.90_640x480_seed44.mp4
└── ...
```

## Technical Details

### Architecture

The script is organized into four main classes:

1. **TreadmillTextureGenerator**: Generates various texture patterns
2. **TreadmillMotionSimulator**: Simulates belt motion with seamless tiling
3. **CameraEffectsProcessor**: Applies camera and lighting effects
4. **SyntheticVideoGenerator**: Orchestrates the complete generation pipeline

### Motion Simulation

Motion is simulated by:
1. Creating a 2x seamless texture (tiled horizontally or vertically)
2. Translating texture by `speed * frame_index` pixels
3. Wrapping at edges using modulo operations
4. Extracting viewport-sized region

This creates smooth, infinite scrolling motion without visible seams.

### Automatic Stripe Orientation

**Critical Design Decision**: The stripe texture generator automatically orients stripes perpendicular to the motion direction. This is essential because:

- **Parallel stripes are invisible**: If stripes run parallel to motion (e.g., horizontal stripes moving left/right), the human eye cannot perceive any movement, and computer vision models cannot detect motion features.
- **Perpendicular stripes create visible flow**: When stripes cross the direction of motion, optical flow is clearly visible and measurable.

**Implementation**:
- Left/Right motion → Vertical stripes (crossing the motion path)
- Up/Down motion → Horizontal stripes (crossing the motion path)

This ensures all generated videos have detectable motion patterns suitable for training vision models.

### Perspective Transform

View angle simulation uses OpenCV's perspective transform:
- Positive angles: Compress top, expand bottom (viewing from above)
- Negative angles: Expand top, compress bottom (viewing from below)
- Transform maintains aspect ratio while creating depth illusion

### Lighting Effects

Lighting is applied using multiplicative masks:
- **Vignette**: Radial gradient from center
- **Gradients**: Linear intensity falloff
- **Spotlight**: Gaussian illumination pattern

## Performance

### Generation Speed

Typical generation times (on modern CPU):
- 640x480 @ 30 FPS, 5 seconds: ~0.5-1.0 seconds
- 1280x720 @ 30 FPS, 5 seconds: ~2-3 seconds
- 1920x1080 @ 60 FPS, 10 seconds: ~10-15 seconds

### Optimization Tips

1. **Lower resolution**: Use 640x480 for rapid prototyping
2. **Lower FPS**: 15-20 FPS sufficient for many applications
3. **Shorter duration**: Generate 2-3 second clips for testing
4. **Batch generation**: Use `--vary_parameters` for efficient dataset creation

## Degrees of Freedom Summary

The script exposes the following controllable parameters (degrees of freedom):

### Spatial/Geometric (5)
1. Resolution (width × height)
2. View angle (camera tilt)
3. Direction (motion vector)
4. Speed (motion magnitude)
5. Texture scale (implicit in texture generation)

### Appearance (6)
6. Texture type (5 options)
7. Background color (RGB)
8. Brightness
9. Contrast
10. Lighting variation type (5 options)
11. Lighting intensity

### Temporal (2)
12. FPS
13. Duration

### Effects (2)
14. Motion blur
15. Camera noise

### Meta (1)
16. Random seed

**Total: 16 degrees of freedom**

This high-dimensional parameter space enables generation of extremely diverse synthetic datasets.

## Recommended Workflows

### Workflow 1: Quick Prototyping

```bash
# Generate small test dataset
python synthetic_data_generation.py \
  --num_videos 5 \
  --duration 2 \
  --fps 15 \
  --resolution 320x240 \
  --vary_parameters
```

### Workflow 2: Training Dataset

```bash
# Generate diverse training set
python synthetic_data_generation.py \
  --num_videos 100 \
  --duration 5 \
  --fps 30 \
  --resolution 640x480 \
  --vary_parameters \
  --seed 42 \
  --output_dir ./train_data
```

### Workflow 3: Validation Dataset

```bash
# Generate held-out validation set with different seed
python synthetic_data_generation.py \
  --num_videos 20 \
  --duration 5 \
  --fps 30 \
  --resolution 640x480 \
  --vary_parameters \
  --seed 999 \
  --output_dir ./val_data
```

### Workflow 4: Stress Testing

```bash
# Generate edge cases: high speed, extreme angles, low contrast
python synthetic_data_generation.py \
  --speed 15 \
  --view_angle 45 \
  --contrast 0.5 \
  --camera_noise 0.8 \
  --motion_blur 5 \
  --lighting_variation vignette \
  --lighting_intensity 0.9
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'cv2'"

**Solution**: Install OpenCV
```bash
pip install opencv-python
```

### Issue: Videos are too large

**Solution**: Reduce resolution, FPS, or duration
```bash
python synthetic_data_generation.py --resolution 320x240 --fps 15 --duration 3
```

### Issue: Motion appears choppy

**Solution**: Increase FPS or decrease speed
```bash
python synthetic_data_generation.py --fps 60 --speed 2.0
```

### Issue: Textures look repetitive

**Solution**: Use `--vary_parameters` or manually vary texture types
```bash
python synthetic_data_generation.py --num_videos 10 --vary_parameters
```

### Issue: Docker permission denied

**Solution**: Ensure output directory is in a mounted volume
```bash
# Use /app/data/ which is mounted from the host
docker exec llamafactory python3 /app/data/synthetic_treadmill/synthetic_data_generation.py \
  --output_dir /app/data/synthetic_treadmill/output
```

## Future Enhancements

Potential additions for even more realism:

1. **3D Rendering**: Use 3D engine for true perspective
2. **Object Placement**: Add random objects on belt
3. **Defect Simulation**: Scratches, tears, stains
4. **Shadow Casting**: Dynamic shadows from objects
5. **Reflections**: Specular highlights on rubber
6. **Temporal Noise**: Flickering lighting
7. **Compression Artifacts**: JPEG/video compression simulation
8. **Multiple Cameras**: Different viewpoints
9. **Real Texture Photos**: Use real treadmill images as base
10. **Physics Simulation**: Vibration, wobble effects

## Citation

If you use this tool in your research, please cite:

```
Synthetic Treadmill Video Dataset Generator
Generated by Claude (Anthropic)
November 2025
```

## License

This code is provided as-is for research and educational purposes.

## Support

For questions or issues:
1. Check the examples above
2. Review the `--help` output
3. Inspect the source code docstrings
4. Test with minimal parameters first

---

**Happy Dataset Generation! 🎬**
