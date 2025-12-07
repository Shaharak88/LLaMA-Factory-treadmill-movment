# Distance Parameter Examples

This folder contains example videos demonstrating the **distance parameter** feature.

## What is the Distance Parameter?

The distance parameter controls how large the treadmill appears in the video, simulating depth perception:
- **distance = 1.0**: Normal size (treadmill fills the frame as usual)
- **distance > 1.0**: Treadmill appears smaller/further away

The treadmill is always **centered** in the frame, with dark background (RGB 50,50,50) filling the surrounding area.

## Examples Generated

Run `./generate_distance_examples.sh` from the project root to generate 4 example videos:

### Example 1: distance=1.0 (Baseline)
- **File**: `treadmill_0000_*_dist1.0_*.mp4`
- **Effect**: Normal size, treadmill fills frame
- **Visual**: Same as all previous experiments (no change)

### Example 2: distance=1.5
- **File**: `treadmill_0000_*_dist1.5_*.mp4`
- **Effect**: Treadmill scaled to 67% size (1/1.5)
- **Visual**: Slightly smaller, thin dark border around edges

### Example 3: distance=2.0
- **File**: `treadmill_0000_*_dist2.0_*.mp4`
- **Effect**: Treadmill scaled to 50% size (1/2)
- **Visual**: Half the size, moderate dark border, clearly centered

### Example 4: distance=3.0
- **File**: `treadmill_0000_*_dist3.0_*.mp4`
- **Effect**: Treadmill scaled to 33% size (1/3)
- **Visual**: Much smaller, large dark background, clearly distant

## What to Observe

When comparing the videos side-by-side:

1. **Treadmill Size**: Each video shows progressively smaller treadmill
2. **Centering**: Treadmill stays perfectly centered in all videos
3. **Background**: Dark gray background appears around scaled treadmill
4. **Motion**: Belt motion speed is the same in all videos (3.0 px/frame)
5. **Belt Details**: Stripe patterns remain visible even at smaller sizes

## Common Parameters

All examples use identical parameters except distance:
- Texture: subtle_gray_stripes
- Direction: right
- Speed: 3.0 px/frame
- View Angle: 0° (frontal view)
- Resolution: 640x480
- Duration: 5 seconds
- FPS: 4

## Use Cases

The distance parameter is useful for:
- Testing model robustness to scale variations
- Simulating realistic viewing distances
- Creating datasets with varied object sizes
- Studying model performance on smaller features
- Training on multi-scale treadmill detection

## Technical Details

- Scaling uses OpenCV `cv2.resize()` with `INTER_LINEAR` interpolation
- Maintains aspect ratio during scaling
- Applied AFTER all other effects (blur, noise, lighting, perspective)
- Background color: RGB(50, 50, 50) - dark gray
- Treadmill centering: `(frame_width - scaled_width) // 2` for X offset

## Integration with Experiments

Use in full experiments with combination mode:

```bash
# Single distance
./run_experiment.sh --train-distance "2.0" --test-distance "2.0"

# Multiple distances (creates all combinations)
./run_experiment.sh --train-distance "1.0,2.0,3.0" --test-distance "1.5,2.5"

# Combined with other parameters
./run_experiment.sh \
  --train-distance "1.0,2.0" \
  --train-angles "0.0,15.0,30.0" \
  --train-speed "2.0,4.0" \
  --epochs 5
```

This creates: 2 distances × 3 angles × 2 speeds = **12 video variations**
