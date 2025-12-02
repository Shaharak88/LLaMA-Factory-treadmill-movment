# Video Report Generation - Troubleshooting Guide

This document details all issues encountered during video report generation and their solutions.

---

## Issue 1: Videos Not Playing in HTML Reports

### Problem
Videos displayed in the generated HTML reports but would not play in the browser. The video player showed a blank screen or error message when attempting to playback.

### Root Cause
The videos downloaded from the server were encoded with **mpeg4 codec**, which is not well-supported for HTML5 video playback in modern browsers. Browsers require **H.264 (h264) codec** for reliable MP4 video playback.

### Detection
Check video codec with:
```bash
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 video.mp4
```

Expected output: `h264`
Problem output: `mpeg4`

### Solution
Automatically convert all videos from mpeg4 to h264 codec after downloading using ffmpeg:

```bash
ffmpeg -i input.mp4 -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -c:a copy output.mp4
```

**Parameters explained:**
- `-c:v libx264`: Use H.264 codec for video
- `-preset fast`: Balance between encoding speed and compression
- `-crf 23`: Constant Rate Factor for quality (18-28 range, 23 is good default)
- `-pix_fmt yuv420p`: Pixel format for maximum browser compatibility
- `-c:a copy`: Copy audio stream without re-encoding

### Implementation
Added `convert_videos_to_h264()` function to `complete_dataset_review.py` that:
1. Checks if conversion is needed by probing the first video
2. Converts all videos in parallel to a temporary directory
3. Replaces original videos with converted versions
4. Provides progress feedback during conversion

---

## Issue 2: Report Organization

### Problem
Initially, HTML reports were generated in the project root directory, making it difficult to find and manage multiple report files.

### Solution (Attempted and Reverted)
We attempted to organize reports into `analytics/reports/` directory but this change was reverted due to complications.

### Current State
Reports are generated in the project root. Consider manual organization or using the `--output` parameter to specify custom locations.

---

## Issue 3: Incomplete Workflow

### Problem
The workflow required multiple manual steps:
1. Run `complete_dataset_review.py` to download videos and extract metadata
2. Manually run `generate_enhanced_video_report.py` on the metadata CSV
3. Manually convert videos if codec issues occurred

### Solution
Integrated all steps into a single workflow in `complete_dataset_review.py`:
1. **Step 1**: Extract metadata from server
2. **Step 2**: Download videos from server
3. **Step 2.5**: Auto-convert videos to H.264 codec
4. **Step 3**: Auto-generate enhanced video report
5. **Step 4**: Generate basic HTML report (fallback)

Now users only need to run:
```bash
python3 complete_dataset_review.py DATASET_NAME
```

---

## Complete Workflow Usage

### One-Command Solution
```bash
python3 complete_dataset_review.py _exp_20251202_172211_train
```

This single command:
- Connects to the remote server
- Finds the dataset in the Docker container
- Downloads all videos via SCP
- Extracts metadata and saves to CSV
- **Converts videos to H.264 codec for browser compatibility**
- **Generates enhanced HTML report with analytics**
- Generates fallback basic HTML report

### Output Files
- `data/DATASET_NAME_metadata_TIMESTAMP.csv` - Video metadata
- `data/videos/DATASET_NAME/` - Downloaded videos (H.264 codec)
- `DATASET_NAME_TIMESTAMP_enhanced_report.html` - Interactive report with analytics
- `dataset_report.html` - Basic fallback report

---

## Browser Compatibility

### Supported Browsers
Videos with H.264 codec work in:
- ✅ Chrome / Chromium
- ✅ Microsoft Edge
- ✅ Firefox
- ✅ Safari

### Viewing Reports
**Method 1: Direct Open** (Recommended)
- Simply double-click the HTML file
- Videos should play with absolute `file://` paths

**Method 2: Local Web Server** (If direct open fails)
```bash
python3 -m http.server 8000
```
Then open: `http://localhost:8000/REPORT.html`

---

## Technical Details

### Video Specifications
- **Codec**: H.264 (libx264)
- **Pixel Format**: yuv420p
- **Container**: MP4
- **Quality**: CRF 23 (good balance)
- **Audio**: Copy (no re-encoding)

### File Size Impact
H.264 encoding typically reduces file size compared to mpeg4:
- Original (mpeg4): ~37KB per video
- Converted (h264): ~5-6KB per video
- **Savings**: ~85% reduction

### Conversion Performance
- 80 videos converted in ~30 seconds
- Sequential processing (can be parallelized further if needed)
- Atomic replacement (temp directory approach prevents corruption)

---

## Troubleshooting Commands

### Check if videos need conversion
```bash
for f in data/videos/DATASET/*.mp4; do
    ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "$f";
done | sort | uniq -c
```

### Manually convert a single video
```bash
ffmpeg -i input.mp4 -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -c:a copy output.mp4
```

### Batch convert all videos in a directory
```bash
find data/videos/DATASET -name "*.mp4" -type f | xargs -P 8 -I {} bash -c \
    'basename=$(basename "{}") && ffmpeg -i "{}" -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -c:a copy "OUTPUT_DIR/$basename" -y -loglevel error'
```

### Verify HTML report videos are playable
Open browser console (F12) while viewing report and check for:
- File not found errors
- Codec errors
- Security/CORS errors

---

## Future Improvements

### Potential Enhancements
1. **Parallel video conversion** - Use multiprocessing for faster conversion
2. **Progress bars** - Visual feedback during long operations
3. **Codec detection at source** - Fix codec during video generation instead of post-processing
4. **Resume capability** - Skip already converted videos
5. **Quality presets** - Allow users to choose conversion quality (fast/balanced/high)

### Server-Side Solution
The ideal solution would be to generate videos with H.264 codec directly on the server during synthetic data generation, eliminating the need for post-download conversion.

---

## Summary

The main issue preventing video playback was the **mpeg4 codec incompatibility** with modern browsers. The solution is **automatic H.264 conversion** integrated into the download workflow. This is now handled transparently by the `complete_dataset_review.py` script.

**Key Takeaway**: Always ensure videos are encoded with H.264 codec for web-based HTML5 video playback.
