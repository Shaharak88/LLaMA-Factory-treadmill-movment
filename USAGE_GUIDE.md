# Video Dataset Review Tool - Usage Guide

## What This Tool Does

**One command** to:
1. 📥 Download videos from server Docker container
2. 🎬 Re-encode to H.264 (browser-compatible format)
3. 📊 Generate interactive HTML report with working video playback

## Quick Start

```bash
python3 download_and_review_dataset.py DATASET_NAME
```

### Example:
```bash
python3 download_and_review_dataset.py yesno_subtle_factory_train
```

## What Happens

```
✅ Lists videos on server
✅ Downloads 72 videos from Docker container
✅ Re-encodes all to H.264 baseline profile (yuv420p)
✅ Generates metadata CSV
✅ Creates HTML report: yesno_subtle_factory_train_report.html
```

## View Report

Simply **double-click** the generated HTML file:
```
yesno_subtle_factory_train_report.html
```

Videos will play directly in your browser! 🎉

## Command Options

```bash
# Skip download (use existing local videos)
python3 download_and_review_dataset.py DATASET_NAME --skip-download

# Skip re-encoding (if already done)
python3 download_and_review_dataset.py DATASET_NAME --skip-reencode

# Custom output filename
python3 download_and_review_dataset.py DATASET_NAME --output my_report.html
```

## Server Configuration

The script connects to:
- **Server:** `seedoo@hetzner-gpu.tail9e6e7.ts.net`
- **Container:** `llamafactory`
- **Data path:** `/app/data/`

## File Locations

After running, you'll have:

```
data/
├── videos/
│   └── DATASET_NAME/
│       ├── video1.mp4 (re-encoded H.264)
│       ├── video2.mp4
│       └── _original_videos_backup/  (original videos)
├── DATASET_NAME_metadata_TIMESTAMP.csv
└── DATASET_NAME_report.html
```

## Video Re-encoding

All videos are automatically re-encoded to:
- **Codec:** H.264
- **Profile:** Constrained Baseline
- **Pixel Format:** yuv420p
- **Fast Start:** Enabled

This ensures **100% browser compatibility** (Chrome, Firefox, Edge, Safari).

## Troubleshooting

### Videos won't download
```bash
# Check available datasets on server
ssh seedoo@hetzner-gpu.tail9e6e7.ts.net "docker exec llamafactory ls /app/data/ | grep train"
```

### Videos won't play in browser
- ✅ They should! Videos are re-encoded to H.264 baseline profile
- Try opening the HTML in a different browser (Chrome recommended)
- Check browser console for errors (F12 → Console)

### Re-encoding is slow
- Normal! Re-encoding 72 videos takes ~2-5 minutes
- Original videos are backed up in `_original_videos_backup/`

### FFmpeg not found
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

## Example: Complete Workflow

```bash
# 1. Download and process dataset from server
python3 download_and_review_dataset.py yesno_subtle_factory_train

# Output:
# ✅ Downloaded 72 videos
# ✅ Re-encoded 72/72 videos
# ✅ Generated: yesno_subtle_factory_train_report.html

# 2. Open report
# Double-click: yesno_subtle_factory_train_report.html

# 3. Videos play in browser! 🎉
```

## Advanced Usage

### Re-process Existing Dataset
```bash
# Re-encode and regenerate report (skip download)
python3 download_and_review_dataset.py DATASET_NAME --skip-download
```

### Generate Report Only (No Re-encoding)
```bash
# Use existing videos as-is
python3 download_and_review_dataset.py DATASET_NAME --skip-download --skip-reencode
```

## Technical Details

### Video Format
- **Container:** MP4
- **Video Codec:** H.264 (libx264)
- **Profile:** Baseline (level 3.0)
- **Pixel Format:** yuv420p
- **Audio:** Removed (not needed for treadmill videos)
- **Fast Start:** Enabled for streaming

### File Paths
The HTML report uses **absolute file:// paths**:
```
file:///C:/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/data/videos/DATASET_NAME/video.mp4
```

This works perfectly when opening HTML files directly on Windows.

## Report Features

The generated HTML report includes:

### Tabs:
1. **Overview** - Statistics and distributions
2. **Dataset Defaults** - Most common parameter values
3. **By Texture** - Videos grouped by texture type
4. **By Direction** - up/down/left/right grouping
5. **By Speed** - Speed range grouping
6. **By Angle** - Camera angle grouping
7. **Objects** - Object type grouping
8. **Blur Effects** - Blur settings grouping
9. **All Videos** - Searchable grid with all videos

### Features:
- ✅ Video playback with controls
- ✅ Full metadata display
- ✅ Search/filter functionality
- ✅ Responsive design
- ✅ Distribution charts
- ✅ Under/over-representation alerts

## That's It!

One command to download, process, and review video datasets! 🚀

```bash
python3 download_and_review_dataset.py DATASET_NAME
```
