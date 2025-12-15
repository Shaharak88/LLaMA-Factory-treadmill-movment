# Dataset Generation Randomness Analysis Report

## Executive Summary

This report documents all sources of randomness in the synthetic treadmill dataset generation pipeline. The codebase uses **two different random number generators**:
- **Python's `random` module** (building_dataset.py)
- **NumPy's `np.random.RandomState`** (synthetic_data_generation.py)

Both are seeded for reproducibility, but they maintain separate random states.

---

## 1. building_dataset.py - Randomness Sources

### Libraries Used
- **Library**: Python standard library `random`
- **Import**: Line 34: `import random`

### Initialization
- **Location**: Line 66
- **Code**: `self.rng = random.Random(self.seed)`
- **Purpose**: Creates a seeded random number generator for reproducible randomization

### Randomness Occurrences

#### 1.1 Parameter Variation (when `--vary_parameters` flag is True)
**Location**: Lines 329-339 in `_generate_configs()` method

```python
config['texture_type'] = self.rng.choice(textures)           # Line 329
config['direction'] = self.rng.choice(directions)            # Line 330
config['speed'] = self.rng.uniform(speed_min, speed_max)    # Line 331
config['view_angle'] = self.rng.uniform(-30, 30)            # Line 332
config['brightness'] = self.rng.uniform(-0.2, 0.2)          # Line 333
config['contrast'] = self.rng.uniform(0.7, 1.3)             # Line 334
config['lighting_variation'] = self.rng.choice(lighting_types)  # Line 335
config['lighting_intensity'] = self.rng.uniform(0.3, 0.8)   # Line 336
config['motion_blur'] = self.rng.randint(1, 3)              # Line 337
config['camera_noise'] = self.rng.uniform(0.0, 0.3)         # Line 338
config['edge_width'] = self.rng.uniform(0.05, 0.15)         # Line 339
```

**Randomized Parameters**:
- Texture type: Random choice from 7 texture types
- Direction: Random choice from ['left', 'right', 'up', 'down']
- Speed: Uniform distribution within specified range
- View angle: Uniform distribution [-30°, 30°]
- Brightness: Uniform distribution [-0.2, 0.2]
- Contrast: Uniform distribution [0.7, 1.3]
- Lighting variation: Random choice from 5 lighting types
- Lighting intensity: Uniform distribution [0.3, 0.8]
- Motion blur: Random integer [1, 3] for high-speed videos
- Camera noise: Uniform distribution [0.0, 0.3]
- Edge width: Uniform distribution [0.05, 0.15]

#### 1.2 Speed Randomization (always occurs for moving videos)
**Location**: Line 344 in `_generate_configs()` method

```python
config['speed'] = self.rng.uniform(speed_min, speed_max) if is_moving else 0.0
```

**Impact**: Even when `vary_parameters=False`, moving video speeds are still randomized within the specified speed range.

#### 1.3 Dataset Entry Shuffling
**Location**: Line 527 in `create_dataset_json()` method

```python
self.rng.shuffle(dataset_entries)
```

**Purpose**: Randomizes the order of training examples in the JSON dataset file for better training diversity.

#### 1.4 Configuration Shuffling
**Location**: Line 701 in `build()` method

```python
self.rng.shuffle(all_configs)
```

**Purpose**: Randomizes the order in which videos are generated to avoid generating all moving videos first, then all stopped videos.

---

## 2. synthetic_data_generation.py - Randomness Sources

### Libraries Used
- **Library**: NumPy's random number generator
- **Import**: Line 18: `import numpy as np`

### Initialization (Multiple RNG Instances)

1. **TreadmillTextureGenerator**
   - **Location**: Line 44
   - **Code**: `self.rng = np.random.RandomState(seed)`

2. **ObjectPlacementGenerator**
   - **Location**: Line 559
   - **Code**: `self.rng = np.random.RandomState(seed)`

3. **CameraEffectsProcessor**
   - **Location**: Line 898
   - **Code**: `self.rng = np.random.RandomState(seed)`

4. **generate_varied_configs function**
   - **Location**: Line 1654
   - **Code**: `rng = np.random.RandomState(base_config['seed'])`

### Randomness Occurrences

#### 2.1 Texture Generation Randomness

##### A. Stripe Texture (generate_stripes)
**Location**: Line 91
```python
noise = self.rng.randint(-5, 5, texture.shape, dtype=np.int16)
```
**Purpose**: Adds subtle random noise [-5, 5] to each pixel for realistic texture variation

##### B. Noise Pattern Texture (generate_noise_pattern)
**Location**: Line 116
```python
noise = self.rng.randn(self.height // scale, self.width // scale)
```
**Purpose**: Generates Perlin-like multi-octave noise using normal distribution for rough rubber surface simulation

##### C. Rubber Pattern Texture (generate_rubber_pattern)
**Locations**: Lines 141-143
```python
cx = self.rng.randint(0, self.width)     # Random X position
cy = self.rng.randint(0, self.height)    # Random Y position
radius = self.rng.randint(3, 8)          # Random bump size
```
**Purpose**: Places circular bumps at random locations with random sizes to simulate rubber texture

##### D. Grid Pattern Texture (generate_grid_pattern)
**Location**: Line 185
```python
noise = self.rng.randint(-3, 3, texture.shape, dtype=np.int16)
```
**Purpose**: Adds subtle noise variation to grid pattern

##### E. Factory Dark Texture (generate_factory_dark)
**Multiple Randomness Sources**:

1. **Noise octaves** (Line 251):
```python
noise = self.rng.randn(self.height // scale, self.width // scale)
```

2. **Wear mark positions and properties** (Lines 258-270):
```python
x = self.rng.randint(0, self.width)              # Random X position
y = self.rng.randint(0, self.height)             # Random Y position
length = self.rng.randint(10, 30)                # Random length
angle = self.rng.uniform(0, np.pi)               # Random angle
brightness_shift = self.rng.uniform(3, 8)        # Random brightness
```

##### F. Factory Dark Stripes (generate_factory_dark_stripes)
**Location**: Line 322
```python
noise = self.rng.randint(-3, 3, texture.shape, dtype=np.int16)
```

##### G. Subtle Gray Stripes (generate_subtle_gray_stripes)
**Locations**: Lines 375, 392, 400

1. **Variable stripe spacing** (when stripe_distance_variance > 0):
```python
spacing = self.rng.normal(stripe_spacing, stripe_distance_variance)  # Line 375, 392
```

2. **Subtle noise**:
```python
noise = self.rng.randint(-3, 3, texture.shape, dtype=np.int16)  # Line 400
```

#### 2.2 Object Placement Randomness

##### A. Random Position Selection
**Location**: Lines 627-628
```python
x = self.rng.randint(usable_left, usable_right)
y = self.rng.randint(usable_top, usable_bottom)
```
**When**: Only when `position='random'` parameter is set

##### B. Random Object Type Selection
**Location**: Line 759
```python
current_type = self.rng.choice(['box', 'circle'])
```
**When**: Only when `object_type='random'` parameter is set

##### C. Random Color Selection
**Location**: Line 764
```python
color_name = self.rng.choice(color_names)
```
**Purpose**: Randomly selects object color from 8 available colors (red, blue, green, yellow, orange, purple, cyan, white)

##### D. Random Cross-Belt Position
**Locations**: Lines 773-789
```python
# For horizontal motion (left/right)
y = self.rng.randint(edge_height + size_pixels,
                     self.height - edge_height - size_pixels)

# For vertical motion (up/down)
x = self.rng.randint(edge_width + size_pixels,
                     self.width - edge_width - size_pixels)
```
**Purpose**: Objects are placed at random positions perpendicular to belt motion direction

#### 2.3 Camera Effects Randomness

##### A. Random Blur Type Selection
**Location**: Line 1139
```python
blur_type = self.rng.choice(['motion', 'gaussian'])
```
**When**: Only when `blur_type='random'` parameter is set

##### B. Camera Noise Generation
**Location**: Line 1167
```python
noise = self.rng.randn(frame.shape[0], frame.shape[1], frame.shape[2])
noise = noise * noise_level * 20
```
**Purpose**: Adds Gaussian noise to simulate camera sensor noise

##### C. Random Blur Intensity Variation (per-frame)
**Location**: Line 1392
```python
intensity_variation = self.effects.rng.uniform(-0.1, 0.1)
blur_intensity = max(0.0, min(1.0, blur_intensity + intensity_variation))
```
**When**: Only when `random_blur_variation=True` parameter is set
**Purpose**: Varies blur intensity slightly between frames for realistic camera focus variation

#### 2.4 Varied Configuration Generation

**Function**: `generate_varied_configs()` (Lines 1639-1696)

When `--vary_parameters` flag is used in the standalone script:

```python
config['speed'] = rng.uniform(1.0, 8.0)              # Line 1672
config['view_angle'] = rng.uniform(-30, 30)          # Line 1675
config['brightness'] = rng.uniform(-0.2, 0.2)        # Line 1678
config['contrast'] = rng.uniform(0.7, 1.3)           # Line 1681
config['lighting_intensity'] = rng.uniform(0.3, 0.8) # Line 1685
config['motion_blur'] = rng.randint(1, 4)            # Line 1689 (when speed > 5.0)
config['camera_noise'] = rng.uniform(0.0, 0.3)       # Line 1692
```

---

## 3. Seed Propagation and Reproducibility

### Seed Flow

1. **Base Seed**: Set via `--seed` argument (default: 42)

2. **Per-Video Seeds**:
   - In building_dataset.py: `seed = base_seed + config_index` (Line 229, 318)
   - Each video gets a unique seed offset from base seed

3. **Independent RNG States**:
   - Each class (TreadmillTextureGenerator, ObjectPlacementGenerator, CameraEffectsProcessor) maintains its own RandomState
   - All initialized with the same per-video seed, ensuring synchronized randomness within a video

### Reproducibility Guarantees

✅ **Reproducible**:
- Texture patterns (given same seed)
- Object colors and positions (given same seed)
- Camera noise patterns (given same seed)
- Parameter variations (given same seed and same vary_parameters setting)
- Dataset entry order (given same seed)
- Video generation order (given same seed)

⚠️ **Conditionally Random** (depends on flags):
- Parameter variation: Only random if `--vary_parameters` flag is True
- Speed variation: Always random for moving videos within speed_range
- Object placement: Only random if `--add-object` flag is True
- Blur variation: Only random if `--random-blur-variation` flag is True

---

## 4. Summary Table: All Random Operations

| Component | Location | Library | Random Operation | Distribution | Range/Options |
|-----------|----------|---------|------------------|--------------|---------------|
| **building_dataset.py** |
| Parameter variation | Line 329 | random | texture_type | choice | 7 texture types |
| Parameter variation | Line 330 | random | direction | choice | 4 directions |
| Parameter variation | Line 331 | random | speed | uniform | [speed_min, speed_max] |
| Parameter variation | Line 332 | random | view_angle | uniform | [-30, 30] degrees |
| Parameter variation | Line 333 | random | brightness | uniform | [-0.2, 0.2] |
| Parameter variation | Line 334 | random | contrast | uniform | [0.7, 1.3] |
| Parameter variation | Line 335 | random | lighting_variation | choice | 5 lighting types |
| Parameter variation | Line 336 | random | lighting_intensity | uniform | [0.3, 0.8] |
| Parameter variation | Line 337 | random | motion_blur | randint | [1, 3] |
| Parameter variation | Line 338 | random | camera_noise | uniform | [0.0, 0.3] |
| Parameter variation | Line 339 | random | edge_width | uniform | [0.05, 0.15] |
| Speed (always) | Line 344 | random | speed | uniform | [speed_min, speed_max] |
| Dataset order | Line 527 | random | shuffle | shuffle | N/A |
| Generation order | Line 701 | random | shuffle | shuffle | N/A |
| **synthetic_data_generation.py** |
| Stripe texture | Line 91 | numpy | pixel noise | randint | [-5, 5] per pixel |
| Noise texture | Line 116 | numpy | Perlin noise | randn | Normal distribution |
| Rubber texture | Line 141-143 | numpy | bump position/size | randint | Pos: full frame, Size: [3, 8] |
| Grid texture | Line 185 | numpy | pixel noise | randint | [-3, 3] per pixel |
| Factory dark | Line 251 | numpy | noise octaves | randn | Normal distribution |
| Factory dark | Line 258-270 | numpy | wear marks | randint/uniform | Multiple properties |
| Factory dark stripes | Line 322 | numpy | pixel noise | randint | [-3, 3] per pixel |
| Subtle gray stripes | Line 375, 392 | numpy | stripe spacing | normal | μ=stripe_spacing, σ=variance |
| Subtle gray stripes | Line 400 | numpy | pixel noise | randint | [-3, 3] per pixel |
| Object position | Line 627-628 | numpy | x, y position | randint | Within usable belt area |
| Object type | Line 759 | numpy | shape | choice | ['box', 'circle'] |
| Object color | Line 764 | numpy | color | choice | 8 colors |
| Object cross-position | Line 773-789 | numpy | perpendicular pos | randint | Edge to edge (minus margins) |
| Blur type | Line 1139 | numpy | blur selection | choice | ['motion', 'gaussian'] |
| Camera noise | Line 1167 | numpy | sensor noise | randn | Normal * noise_level * 20 |
| Blur variation | Line 1392 | numpy | intensity shift | uniform | [-0.1, 0.1] |
| Varied configs | Line 1672 | numpy | speed | uniform | [1.0, 8.0] |
| Varied configs | Line 1675 | numpy | view_angle | uniform | [-30, 30] |
| Varied configs | Line 1678 | numpy | brightness | uniform | [-0.2, 0.2] |
| Varied configs | Line 1681 | numpy | contrast | uniform | [0.7, 1.3] |
| Varied configs | Line 1685 | numpy | lighting_intensity | uniform | [0.3, 0.8] |
| Varied configs | Line 1689 | numpy | motion_blur | randint | [1, 4] |
| Varied configs | Line 1692 | numpy | camera_noise | uniform | [0.0, 0.3] |

---

## 5. Key Findings

### Critical Randomness Points

1. **Texture Realism**: Every texture includes randomized noise for realism, making each frame slightly unique even with the same seed
2. **Speed Variation**: Moving videos ALWAYS have randomized speeds (even with vary_parameters=False)
3. **Dual RNG Systems**: Python's random and NumPy's random are independent; both must be seeded
4. **Per-Frame Randomness**: Camera noise and blur variation can change frame-to-frame
5. **Conditional Randomness**: Many randomizations only occur when specific flags are enabled

### Potential Issues

1. **Speed Inconsistency**: Speed is always randomized for moving videos, which may make controlled experiments difficult
2. **Subtle Noise**: All textures have subtle random noise added, which slightly breaks exact reproducibility at the pixel level
3. **Multiple RNG States**: Three separate RandomState objects could lead to confusion about seed behavior

---

## 6. Recommendations

### For Exact Reproducibility
1. Always set `--seed` parameter explicitly
2. Ensure same version of NumPy (random number generation changed in NumPy 1.17+)
3. Document which flags were used (vary_parameters, add_object, add_blur, random_blur_variation)

### For Controlled Experiments
1. Consider adding a flag to disable speed randomization
2. Consider adding a flag to disable texture noise for pixel-perfect reproducibility
3. Document the seed offset calculation (base_seed + video_index)

---

## Report Generated
Date: 2024
Analysis Tool: Manual code review
Files Analyzed:
- building_dataset.py (885 lines)
- data/synthetic_treadmill/synthetic_data_generation.py (1818 lines)
