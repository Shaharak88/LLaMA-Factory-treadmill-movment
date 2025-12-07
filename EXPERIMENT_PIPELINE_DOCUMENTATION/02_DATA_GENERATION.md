# Data Generation Pipeline

## Purpose
The data generation pipeline creates synthetic treadmill videos using Blender and formats them into training/testing datasets for the vision-language model.

## Architecture

```
building_dataset.py (orchestrator)
    │
    ├──> Validates arguments
    ├──> Calculates video splits
    ├──> Creates dataset directories
    │
    └──> Calls: synthetic_data_generation.py
         │
         ├──> Generates individual videos with Blender
         ├──> Applies textures, motion, angles
         ├──> Saves videos + metadata
         │
         └──> Returns: List of video paths + metadata
    │
    ├──> Formats data into JSON (LLaMA-Factory format)
    ├──> Saves: {dataset}_train.json, {dataset}_test.json
    └──> Updates: data/dataset_info.json
```

---

## Part 1: building_dataset.py

### Location
`/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/building_dataset.py`

### Purpose
Orchestrates the dataset generation process:
1. Parses command-line arguments
2. Splits videos into train/test sets
3. Calls synthetic_data_generation.py to create videos
4. Formats videos into JSON datasets
5. Updates dataset registry

### Key Functions

#### 1. `parse_arguments()`
Defines all command-line parameters for dataset generation.

**Dataset Configuration:**
- `--dataset_name`: Unique dataset identifier (e.g., "_exp_20251207_143033")
- `--num_videos`: Total number of videos to generate
- `--train_split`: Ratio for train/test split (default: 0.8 = 80% train, 20% test)
- `--seed`: Random seed for reproducibility

**Train Dataset Parameters** (prefix: `--train_`):
- `texture_type`: Treadmill texture (e.g., "subtle_gray_stripes", "factory_dark_stripes")
- `direction`: Motion direction ("left", "right", "up", "down")
- `view_angle`: Camera angle in degrees ("0", "30", "52", or comma-separated "0,30,52")
- `speed_range`: Speed range "min,max" (e.g., "3.0,3.0" = fixed speed, "2.0,4.0" = variable)
- `resolution`: Video resolution (e.g., "640x480")
- `fps`: Frames per second (default: 30)
- `duration`: Video duration in seconds (default: 3)
- `brightness`: Brightness adjustment (-1.0 to 1.0, default: 0.0)
- `contrast`: Contrast adjustment (0.5 to 1.5, default: 1.0)
- `lighting_variation`: Enable random lighting (boolean)
- `lighting_intensity`: Light intensity (0.5 to 2.0)
- `motion_blur`: Enable motion blur (boolean)
- `camera_noise`: Camera noise level (0.0 to 1.0)
- `edge_width`: Belt edge width in pixels (default: 10)
- `distance`: Camera distance (simulates depth, default: 3.0)
- `center_randomization`: Position randomization (0=centered, 1=randomized with 30% min visibility)
- `stripe_width`: Width of stripes for striped textures (default: None)
- `stripe_spacing`: Spacing between stripes (default: None)
- `stripe_gray`: Grayscale value for stripes (0-255, default: None)
- `background_gray`: Grayscale value for background (0-255, default: None)
- `stripe_distance_variance`: Variance in stripe positions (default: 0.0)
- `vary_parameters`: Randomly vary texture parameters (boolean)

**Test Dataset Parameters** (prefix: `--test_`):
Same parameters as train, but allows different configurations for testing generalization.

**Why separate train/test parameters?**
- Enables controlled experiments (e.g., train on angle=0, test on angle=52)
- Tests model generalization to unseen conditions
- Common pattern: train on simple, test on complex variations

#### 2. `generate_dataset(args)`
Main orchestration function that coordinates the entire dataset generation.

**Step-by-step logic:**

**Step 1: Calculate splits**
```python
total_videos = args.num_videos
train_count = int(total_videos * args.train_split)
test_count = total_videos - train_count
```
- Example: 100 videos, 0.8 split → 80 train, 20 test

**Step 2: Create output directories**
```python
train_output_dir = project_root / "data" / f"{args.dataset_name}_train"
test_output_dir = project_root / "data" / f"{args.dataset_name}_test"
train_output_dir.mkdir(parents=True, exist_ok=True)
test_output_dir.mkdir(parents=True, exist_ok=True)
```
- Creates separate folders for train/test videos
- Example: `data/_exp_20251207_143033_train/`, `data/_exp_20251207_143033_test/`

**Step 3: Generate training videos**
```python
train_metadata_path, train_videos = generate_videos(
    output_dir=train_output_dir,
    num_videos=train_count,
    split_type='train',
    args=args,
    seed=args.seed
)
```
Calls `generate_videos()` helper function which:
1. Builds command-line arguments for synthetic_data_generation.py
2. Executes Blender script via subprocess
3. Returns metadata CSV path and list of generated videos

**Step 4: Generate test videos**
```python
test_metadata_path, test_videos = generate_videos(
    output_dir=test_output_dir,
    num_videos=test_count,
    split_type='test',
    args=args,
    seed=args.seed + 1  # Different seed for test set
)
```
- Uses `seed + 1` to ensure different random generation than training set

**Step 5: Format datasets**
```python
train_json_path = format_dataset(train_videos, args, split_type='train')
test_json_path = format_dataset(test_videos, args, split_type='test')
```
Converts video metadata into LLaMA-Factory JSON format.

**Step 6: Update dataset registry**
```python
update_dataset_info(args.dataset_name, train_json_path, test_json_path)
```
Adds dataset to `data/dataset_info.json` for easy reference.

#### 3. `generate_videos(output_dir, num_videos, split_type, args, seed)`
Helper function that calls the Blender video generation script.

**Building the command:**
```python
cmd = [
    "python", str(video_gen_script),
    "--output_dir", str(output_dir),
    "--num_videos", str(num_videos),
    "--seed", str(seed)
]
```

**Adding split-specific parameters:**
```python
# Get parameters for this split (train or test)
prefix = f"{split_type}_"

# Add all parameters with prefix
if getattr(args, f"{prefix}texture_type", None):
    cmd.extend(["--texture_type", getattr(args, f"{prefix}texture_type")])
if getattr(args, f"{prefix}direction", None):
    cmd.extend(["--direction", getattr(args, f"{prefix}direction")])
# ... (continues for all parameters)
```

**Why use prefixes?**
- Single argparse namespace contains both train and test parameters
- Prefix helps differentiate: `args.train_texture_type` vs `args.test_texture_type`

**Executing the script:**
```python
result = subprocess.run(
    cmd,
    cwd=project_root,
    capture_output=True,
    text=True,
    check=True
)
```
- `capture_output=True`: Capture stdout/stderr
- `text=True`: Decode output as text
- `check=True`: Raise exception if script fails

**Parsing output:**
```python
# synthetic_data_generation.py prints: "METADATA_PATH: /path/to/metadata.csv"
metadata_path_line = [line for line in result.stdout.split('\n')
                      if line.startswith("METADATA_PATH:")]
metadata_path = metadata_path_line[0].split("METADATA_PATH:")[1].strip()
```

**Loading metadata:**
```python
import pandas as pd
df = pd.read_csv(metadata_path)
videos = []
for _, row in df.iterrows():
    videos.append({
        'filename': row['filename'],
        'speed': row['speed'],
        'texture': row['texture_type'],
        'angle': row['view_angle'],
        'is_moving': row['speed'] > 0
    })
```

#### 4. `format_dataset(videos, args, split_type)`
Converts video metadata into LLaMA-Factory's expected JSON format.

**LLaMA-Factory JSON structure:**
```json
[
    {
        "messages": [
            {
                "role": "user",
                "content": "Is there movement in the video? Answer only with yes or no."
            },
            {
                "role": "assistant",
                "content": "yes"
            }
        ],
        "videos": [
            "data/_exp_20251207_143033_train/treadmill_0000_subtle_gray_stripes_right_speed3.0_angle0_dist3.0_bright0.00_contr1.00_640x480_seed42.mp4"
        ]
    }
]
```

**Message formatting:**
```python
for video in videos:
    # Determine ground truth label
    is_moving = video['is_moving']
    answer = "yes" if is_moving else "no"

    formatted_data.append({
        "messages": [
            {
                "role": "user",
                "content": "Is there movement in the video? Answer only with yes or no."
            },
            {
                "role": "assistant",
                "content": answer
            }
        ],
        "videos": [
            f"data/{args.dataset_name}_{split_type}/{video['filename']}"
        ]
    })
```

**Why this format?**
- LLaMA-Factory expects chat-style format (user/assistant)
- `videos` array contains relative paths from project root
- Simple yes/no answers for binary classification

**Saving JSON:**
```python
json_path = project_root / "data" / f"{args.dataset_name}_{split_type}.json"
with open(json_path, 'w') as f:
    json.dump(formatted_data, f, indent=2)
```

#### 5. `update_dataset_info(dataset_name, train_json_path, test_json_path)`
Updates the dataset registry file for easy reference.

**dataset_info.json structure:**
```json
{
    "_exp_20251207_143033_train": {
        "file_name": "_exp_20251207_143033_train.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "videos": "videos"
        }
    },
    "_exp_20251207_143033_test": {
        "file_name": "_exp_20251207_143033_test.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "videos": "videos"
        }
    }
}
```

**Why use dataset_info.json?**
- LLaMA-Factory uses this file to locate datasets
- Maps dataset names to JSON files
- Specifies formatting type ("sharegpt" = chat format)
- Defines column mappings

**Update logic:**
```python
dataset_info_path = project_root / "data" / "dataset_info.json"

# Load existing registry
if dataset_info_path.exists():
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
else:
    dataset_info = {}

# Add train dataset
dataset_info[f"{dataset_name}_train"] = {
    "file_name": train_json_path.name,
    "formatting": "sharegpt",
    "columns": {
        "messages": "messages",
        "videos": "videos"
    }
}

# Add test dataset
dataset_info[f"{dataset_name}_test"] = {
    "file_name": test_json_path.name,
    "formatting": "sharegpt",
    "columns": {
        "messages": "messages",
        "videos": "videos"
    }
}

# Save updated registry
with open(dataset_info_path, 'w') as f:
    json.dump(dataset_info, f, indent=2)
```

---

## Part 2: synthetic_data_generation.py

### Location
`/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/data/synthetic_treadmill/synthetic_data_generation.py`

### Purpose
Generates synthetic treadmill videos using Blender's Python API (bpy).

**Key features:**
- Creates 3D treadmill model with customizable textures
- Animates belt motion with realistic speeds
- Configures camera angles and distances
- Applies lighting, motion blur, and noise effects
- Renders high-quality MP4 videos
- Saves metadata CSV for tracking

### Main Components

#### 1. Blender Scene Setup

**Initialization:**
```python
import bpy
import bmesh
import math
import random
from pathlib import Path

# Clear existing scene
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.engine = 'CYCLES'  # Use Cycles for realistic rendering
```

**Why Cycles renderer?**
- Physically accurate lighting and materials
- Supports motion blur and realistic shadows
- Better quality than Eevee (faster but less accurate)

#### 2. Treadmill Geometry Creation

**Function: `create_treadmill(texture_type, speed, direction, ...)`**

Creates 3D treadmill geometry with these components:
1. **Belt surface**: Animated plane with texture
2. **Belt enclosure**: Side walls and edges
3. **Materials**: Procedural or image-based textures

**Belt creation:**
```python
# Create belt plane
bpy.ops.mesh.primitive_plane_add(size=belt_width, location=(0, 0, 0))
belt = bpy.context.active_object
belt.name = "TreadmillBelt"

# Apply subdivision for smooth surface
modifier = belt.modifiers.new(name="Subsurf", type='SUBSURF')
modifier.levels = 2  # Subdivision level

# Apply texture material
material = create_material(texture_type, speed, direction, ...)
belt.data.materials.append(material)
```

**Enclosure creation:**
```python
# Create side walls
for side in ['left', 'right', 'front', 'back']:
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=wall_positions[side]
    )
    wall = bpy.context.active_object
    wall.name = f"Wall_{side}"
    wall.scale = wall_scales[side]

    # Apply dark material (contrast with belt)
    wall_material = create_enclosure_material(edge_width)
    wall.data.materials.append(wall_material)
```

**Why add enclosure?**
- Provides visual context (this is a treadmill, not just a floating plane)
- Helps model learn treadmill structure
- Adds edge_width parameter for visual variation

#### 3. Texture Generation

**Function: `create_material(texture_type, speed, direction, ...)`**

Supports multiple texture types:

**A. Subtle Gray Stripes:**
```python
# Parameters
stripe_width = 0.125      # Width of each stripe
stripe_spacing = 0.25     # Space between stripes
stripe_gray = 125         # Gray value for stripe (0-255)
background_gray = 140     # Gray value for background

# Blender material nodes
material = bpy.data.materials.new(name="SubtleGrayStripes")
material.use_nodes = True
nodes = material.node_tree.nodes
links = material.node_tree.links

# Create procedural stripes using Math nodes
wave_texture = nodes.new(type='ShaderNodeTexWave')
wave_texture.wave_type = 'SAW'
wave_texture.inputs['Scale'].default_value = 1.0 / stripe_spacing
wave_texture.inputs['Distortion'].default_value = 0.0

# Color ramp to create binary stripes
color_ramp = nodes.new(type='ShaderNodeValToRGB')
color_ramp.color_ramp.elements[0].position = stripe_width / stripe_spacing
color_ramp.color_ramp.elements[0].color = (stripe_gray/255, stripe_gray/255, stripe_gray/255, 1)
color_ramp.color_ramp.elements[1].color = (background_gray/255, background_gray/255, background_gray/255, 1)
```

**B. Factory Dark Stripes:**
```python
# Similar to subtle_gray_stripes but with different parameters
stripe_gray = 50          # Darker stripes
background_gray = 100     # Darker background
contrast_multiplier = 1.2 # Increased contrast
```

**C. Custom Image Textures:**
```python
# Load image from file
image_texture = nodes.new(type='ShaderNodeTexImage')
image_texture.image = bpy.data.images.load(texture_path)

# Apply to material
links.new(image_texture.outputs['Color'], bsdf.inputs['Base Color'])
```

**Texture animation (motion):**
```python
# Apply offset based on speed and direction
mapping_node = nodes.new(type='ShaderNodeMapping')

# Calculate offset per frame
offset_per_frame = speed * (1.0 / fps)

# Set keyframes for animation
for frame in range(1, total_frames + 1):
    if direction == 'right':
        mapping_node.inputs['Location'].default_value.x = frame * offset_per_frame
    elif direction == 'left':
        mapping_node.inputs['Location'].default_value.x = -frame * offset_per_frame
    # ... (similar for up/down)

    mapping_node.inputs['Location'].keyframe_insert(data_path="default_value", frame=frame)
```

**Why use procedural textures?**
- Full control over parameters (stripe width, color, spacing)
- No need for external image files
- Can vary parameters programmatically

#### 4. Camera Setup

**Function: `setup_camera(view_angle, distance)`**

Positions camera to capture treadmill from specified angle.

**Camera positioning:**
```python
# Create camera
bpy.ops.object.camera_add()
camera = bpy.context.active_object
camera.name = "Camera"

# Calculate position based on angle
angle_rad = math.radians(view_angle)
camera_distance = distance  # Distance from treadmill

# Position camera
camera.location.x = 0  # Center horizontally
camera.location.y = -camera_distance * math.cos(angle_rad)  # Distance backward
camera.location.z = camera_distance * math.sin(angle_rad)   # Height

# Point camera at treadmill
look_at = (0, 0, 0)  # Treadmill center
direction = mathutils.Vector(look_at) - camera.location
camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
```

**View angles explained:**
- **0°**: Frontal view (eye level)
- **30°**: Slightly elevated view
- **52°**: High angle view (bird's eye)

**Distance parameter:**
```python
# distance controls camera distance from treadmill
# Larger distance = smaller treadmill in frame (simulates depth)
# distance=1.0: Very close (treadmill fills frame)
# distance=3.0: Medium distance (default)
# distance=6.0: Far away (small treadmill)
```

**Why is distance parameter important?**
- Simulates real-world camera placement
- Tests model's ability to handle different scales
- Adds visual variation to dataset

#### 5. Lighting Setup

**Function: `setup_lighting(lighting_variation, lighting_intensity)`**

Configures scene lighting for realistic rendering.

**Default lighting:**
```python
# Create sun light (directional)
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
sun = bpy.context.active_object
sun.data.energy = 1.0 * lighting_intensity

# Create area lights (soft ambient)
bpy.ops.object.light_add(type='AREA', location=(-3, -3, 5))
area_light_1 = bpy.context.active_object
area_light_1.data.energy = 0.5 * lighting_intensity

bpy.ops.object.light_add(type='AREA', location=(3, -3, 5))
area_light_2 = bpy.context.active_object
area_light_2.data.energy = 0.5 * lighting_intensity
```

**Random lighting variation:**
```python
if lighting_variation:
    # Randomize light positions
    sun.location.x += random.uniform(-2, 2)
    sun.location.y += random.uniform(-2, 2)
    sun.location.z += random.uniform(-2, 2)

    # Randomize light intensities
    sun.data.energy *= random.uniform(0.8, 1.2)
    area_light_1.data.energy *= random.uniform(0.5, 1.5)
    area_light_2.data.energy *= random.uniform(0.5, 1.5)
```

**Why use multiple lights?**
- Simulates real-world lighting conditions
- Reduces harsh shadows
- Adds visual realism

#### 6. Rendering Configuration

**Function: `setup_render(resolution, fps, duration, motion_blur, camera_noise)`**

Configures Blender's render settings.

**Basic settings:**
```python
scene = bpy.context.scene
scene.render.resolution_x = width  # e.g., 640
scene.render.resolution_y = height  # e.g., 480
scene.render.fps = fps  # e.g., 30
scene.frame_start = 1
scene.frame_end = fps * duration  # e.g., 30 * 3 = 90 frames
```

**Motion blur:**
```python
if motion_blur:
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.5  # Shutter speed (0.5 = realistic)
```

**Why enable motion blur?**
- Simulates real camera behavior (moving objects blur)
- Tests model's robustness to blur
- More realistic videos

**Camera noise:**
```python
if camera_noise > 0:
    # Add noise compositor node
    scene.use_nodes = True
    tree = scene.node_tree

    noise_node = tree.nodes.new(type='CompositorNodeRGBCurve')
    # Configure noise parameters based on camera_noise value
```

**Video encoding:**
```python
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'HIGH'  # Quality
scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
```

#### 7. Video Rendering

**Function: `render_video(output_path)`**

Executes Blender rendering to generate video file.

```python
scene = bpy.context.scene
scene.render.filepath = str(output_path)

# Render all frames
bpy.ops.render.render(animation=True, write_still=False)
```

**What happens during rendering?**
1. Blender renders each frame (1 to frame_end)
2. Applies textures, lighting, motion blur, noise
3. Encodes frames into H264 MP4 video
4. Saves to output_path

**Rendering time:**
- Depends on resolution, frame count, and complexity
- Typical: 1-3 minutes per 3-second video (90 frames at 30 fps)
- GPU acceleration used if available

#### 8. Metadata Tracking

**Function: `save_metadata(videos, output_dir)`**

Saves CSV file with all video parameters for tracking and analysis.

**CSV structure:**
```csv
filename,texture_type,direction,speed,view_angle,resolution,fps,duration,distance,edge_width,stripe_width,stripe_spacing,stripe_gray,background_gray,brightness,contrast,lighting_variation,lighting_intensity,motion_blur,camera_noise,seed
treadmill_0000_subtle_gray_stripes_right_speed3.0_angle0_dist3.0_bright0.00_contr1.00_640x480_seed42.mp4,subtle_gray_stripes,right,3.0,0,640x480,30,3,3.0,15,0.125,0.25,125,140,0.0,1.0,False,1.0,False,0.0,42
```

**Why save metadata?**
- Track exact parameters used for each video
- Enable analysis of model performance by parameter
- Reproducibility (regenerate exact video with same params)
- Debugging (identify problematic parameter combinations)

#### 9. Main Execution Loop

**Function: `main()`**

Orchestrates generation of multiple videos.

```python
def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)

    videos = []
    for i in range(args.num_videos):
        # Generate unique filename
        filename = generate_filename(i, args)
        output_path = args.output_dir / filename

        # Vary parameters if requested
        if args.vary_parameters:
            # Randomly modify texture parameters
            current_stripe_width = random.uniform(0.1, 0.2)
            current_background_gray = random.randint(120, 160)
            # ... etc

        # Create treadmill scene
        create_treadmill(args.texture_type, args.speed, args.direction, ...)

        # Setup camera and lighting
        setup_camera(args.view_angle, args.distance)
        setup_lighting(args.lighting_variation, args.lighting_intensity)

        # Configure rendering
        setup_render(args.resolution, args.fps, args.duration, args.motion_blur, args.camera_noise)

        # Render video
        render_video(output_path)

        # Track metadata
        videos.append({
            'filename': filename,
            'texture_type': args.texture_type,
            'speed': args.speed,
            # ... all parameters
        })

        # Clean scene for next video
        bpy.ops.wm.read_factory_settings(use_empty=True)

    # Save metadata CSV
    metadata_path = save_metadata(videos, args.output_dir)

    # Print metadata path for building_dataset.py to parse
    print(f"METADATA_PATH: {metadata_path}")
```

### Filename Convention

Videos are named with all parameters encoded in filename:
```
treadmill_{index:04d}_{texture}_{direction}_speed{speed}_angle{angle}_dist{distance}_bright{brightness:.2f}_contr{contrast:.2f}_{resolution}_seed{seed}.mp4
```

Example:
```
treadmill_0042_subtle_gray_stripes_right_speed3.0_angle52_dist3.0_bright0.00_contr1.00_640x480_seed87.mp4
```

**Why encode parameters in filename?**
- Self-documenting (filename tells you everything about video)
- Easy to identify videos by parameters
- No need to consult metadata CSV for basic info
- Helps with debugging and analysis

### Parameter Interactions

**Speed + Direction:**
- `speed=3.0, direction="right"`: Belt moves right at 3.0 units/sec
- `speed=0.0`: Stationary belt (stopped treadmill)

**Texture + Speed:**
- Striped textures show motion clearly (stripes moving)
- Solid textures harder to detect motion (need subtle cues)

**View Angle + Distance:**
- High angle (52°) + far distance (6.0) = bird's eye view, small treadmill
- Low angle (0°) + close distance (1.0) = frontal view, large treadmill

**Lighting Variation + Motion Blur:**
- Both add realism but also difficulty
- Tests model robustness to real-world conditions

## Output Structure

After running building_dataset.py, you get:

```
data/
├── _exp_20251207_143033_train/
│   ├── treadmill_0000_*.mp4
│   ├── treadmill_0001_*.mp4
│   ├── ...
│   ├── treadmill_0079_*.mp4
│   └── metadata_20251207_143105.csv
├── _exp_20251207_143033_test/
│   ├── treadmill_0000_*.mp4
│   ├── ...
│   ├── treadmill_0019_*.mp4
│   └── metadata_20251207_143245.csv
├── _exp_20251207_143033_train.json
├── _exp_20251207_143033_test.json
└── dataset_info.json (updated)
```

## Next Steps
- Read `03_TRAINING.md` for how these datasets are used in training
- Read `04_EVALUATION.md` for how test videos are evaluated
