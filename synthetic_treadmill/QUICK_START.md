# Quick Start Guide - Synthetic Treadmill Generator

## 🚀 Installation (5 seconds)

### Option A: Local Machine
```bash
pip install opencv-python numpy
```

### Option B: Docker Container (LLaMA-Factory)
```bash
# Install dependencies in container
docker exec llamafactory pip install opencv-python-headless "numpy<2.0.0"

# Copy script to mounted directory
cp -r synthetic_treadmill data/
```

## 📽️ Generate Your First Video (10 seconds)

### Local
```bash
cd synthetic_treadmill
python synthetic_data_generation.py --num_videos 1
```

### Docker
```bash
docker exec llamafactory python3 /app/data/synthetic_treadmill/synthetic_data_generation.py \
  --num_videos 1 \
  --output_dir /app/data/synthetic_treadmill/output
```

**Output**: `./synthetic_videos/treadmill_0000_*.mp4`

## 🎯 Common Use Cases

### 1. Quick Test Dataset (30 seconds)
```bash
python synthetic_data_generation.py \
  --num_videos 5 \
  --duration 2 \
  --fps 15 \
  --vary_parameters
```

### 2. Training Dataset (5 minutes)
```bash
python synthetic_data_generation.py \
  --num_videos 50 \
  --duration 5 \
  --vary_parameters \
  --seed 42
```

### 3. High-Speed Motion
```bash
python synthetic_data_generation.py \
  --speed 10 \
  --motion_blur 3 \
  --direction right
```

### 4. Different Textures
```bash
# Try each texture type
for texture in stripes noise rubber grid diamond_plate; do
  python synthetic_data_generation.py --texture_type $texture
done
```

### 5. Camera Angles
```bash
# Different viewing angles
python synthetic_data_generation.py --view_angle 30  # Top view
python synthetic_data_generation.py --view_angle -30 # Bottom view
```

## 🎨 Key Parameters

| Parameter | Values | Example |
|-----------|--------|---------|
| `--texture_type` | stripes, noise, rubber, grid, diamond_plate | `--texture_type rubber` |
| `--direction` | left, right, up, down | `--direction left` |
| `--speed` | 0.5 to 20+ | `--speed 5.0` |
| `--view_angle` | -45 to 45 | `--view_angle 30` |
| `--lighting_variation` | none, vignette, gradient_lr, gradient_tb, spotlight | `--lighting_variation vignette` |
| `--resolution` | WxH | `--resolution 1920x1080` |
| `--fps` | 10 to 60+ | `--fps 30` |
| `--duration` | seconds | `--duration 5` |

## 📁 Output Location

### Local
```bash
ls -lh ./synthetic_videos/
```

### Docker
```bash
# Inside container
docker exec llamafactory ls -lh /app/data/synthetic_treadmill/output/

# On host
ls -lh data/synthetic_treadmill/output/
```

## 🔍 Verify Generated Videos

```bash
# Check file size
ls -lh synthetic_videos/

# Get video info
ffprobe synthetic_videos/treadmill_0000_*.mp4

# Play video (if you have a player)
vlc synthetic_videos/treadmill_0000_*.mp4
```

## 💡 Pro Tips

1. **Start Small**: Use `--duration 2 --fps 15` for quick testing
2. **Vary Parameters**: Always use `--vary_parameters` for datasets
3. **Set Seed**: Use `--seed 42` for reproducible results
4. **Check Output**: Verify first video before generating large batches
5. **Docker Paths**: Always use `/app/data/` paths for Docker output

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Import error | `pip install opencv-python numpy` |
| No output files | Check `--output_dir` exists and is writable |
| Videos too large | Reduce `--resolution` or `--duration` |
| Motion too fast | Decrease `--speed` or increase `--fps` |
| Docker permission | Use `/app/data/` mounted directory |

## 📚 Full Documentation

See [README.md](README.md) for complete documentation and examples.

## 🎓 Example Workflow

```bash
# Step 1: Generate small test batch
python synthetic_data_generation.py \
  --num_videos 3 \
  --duration 2 \
  --vary_parameters

# Step 2: Review outputs
ls -lh synthetic_videos/

# Step 3: Generate full dataset
python synthetic_data_generation.py \
  --num_videos 100 \
  --duration 5 \
  --vary_parameters \
  --seed 42 \
  --output_dir ./full_dataset

# Step 4: Check dataset statistics
echo "Total videos: $(ls full_dataset/*.mp4 | wc -l)"
du -sh full_dataset/
```

---

**Need help?** Run: `python synthetic_data_generation.py --help`
