# Local Video Reports with Working Playback

## Problem Solved

The original `complete_dataset_review.py` generated HTML reports with **relative paths** that didn't work when opening the HTML file directly on Windows. Videos would not play because browsers couldn't resolve the relative paths correctly.

## Solution

The new `generate_local_video_report.py` generates HTML reports with **absolute `file://` paths** that work perfectly when opening HTML files directly on Windows.

---

## Quick Start (Windows)

### Option 1: Double-Click Batch File
Simply double-click: **`GENERATE_VIDEO_REPORT.bat`**

This will:
1. Find the most recent metadata CSV
2. Generate an HTML report with working video paths
3. Automatically open the report in your browser

### Option 2: Command Line
```bash
python generate_local_video_report.py data/YOUR_METADATA_FILE.csv
```

Example:
```bash
python generate_local_video_report.py data/_exp_20251201_172008_train_metadata_20251202_123827.csv
```

---

## How It Works

### 1. **Absolute File Paths**
Instead of relative paths like:
```
data/videos/_exp_20251201_172008_train/video.mp4
```

The new script uses absolute Windows paths:
```
file:///C:/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/data/videos/_exp_20251201_172008_train/video.mp4
```

### 2. **Automatic Video Discovery**
The script automatically searches for videos in multiple locations:
- `data/videos/{dataset_name}/`
- `data/{dataset_name}/`
- Current directory

### 3. **Missing Video Detection**
If videos are not found on disk, they are:
- Marked with a red "FILE NOT FOUND" badge
- Displayed with reduced opacity and dashed border
- Listed in a warning section at the top

---

## Complete Workflow

### Step 1: Extract Metadata from Server (if needed)
```bash
python complete_dataset_review.py YOUR_DATASET_NAME
```

This downloads videos and creates a metadata CSV file.

### Step 2: Generate Local Report
```bash
python generate_local_video_report.py data/YOUR_DATASET_metadata_*.csv
```

Or just double-click: **`GENERATE_VIDEO_REPORT.bat`**

### Step 3: View Report
Double-click the generated HTML file in `analytics/reports/` directory (e.g., `analytics/reports/_exp_20251201_172008_train_enhanced_report.html`)

Videos will play directly in your browser!

---

## File Locations

### Script Files
- **`generate_local_video_report.py`** - Main Python script
- **`GENERATE_VIDEO_REPORT.bat`** - Windows batch file for easy generation
- **`complete_dataset_review.py`** - Original script (downloads videos + metadata)

### Data Files
- **`data/*_metadata_*.csv`** - Metadata CSV files
- **`data/videos/{dataset}/`** - Downloaded video files
- **`analytics/reports/*_enhanced_report.html`** - Generated HTML reports (auto-created in analytics/reports/)

---

## Script Arguments

```bash
python generate_local_video_report.py METADATA_CSV [--output OUTPUT_HTML]
```

### Arguments:
- `METADATA_CSV` (required): Path to metadata CSV file
- `--output` (optional): Output HTML filename (auto-generated if not specified)

### Examples:
```bash
# Auto-generate output name
python generate_local_video_report.py data/_exp_20251201_172008_train_metadata_20251202_123827.csv

# Specify output name
python generate_local_video_report.py data/metadata.csv --output my_report.html
```

---

## Features

### Report Tabs:
1. **Overview** - Statistical summary and distributions
2. **Dataset Defaults** - Most common parameter values
3. **By Texture** - Videos grouped by texture
4. **By Direction** - Videos grouped by direction (up/down/left/right)
5. **By Speed** - Videos grouped by speed ranges
6. **By Angle** - Videos grouped by camera angle
7. **Objects** - Videos grouped by object type
8. **Blur Effects** - Videos grouped by blur settings
9. **All Videos** - Searchable grid of all videos

### Video Cards Show:
- Video playback with controls
- All metadata parameters
- Absolute file path (for debugging)
- Missing file indicators (if applicable)

---

## Troubleshooting

### Videos Still Don't Play?

1. **Check File Paths**
   - Open the HTML file
   - Right-click a video card → Inspect
   - Look at the `<source src="file:///C:/...">` path
   - Copy the path and check if the file exists at that location

2. **Browser Security**
   - Some browsers block local file access
   - **Chrome**: Works well
   - **Firefox**: May require enabling local file access
   - **Edge**: Works well

3. **Files Missing?**
   - Check if videos were downloaded to `data/videos/{dataset}/`
   - Re-run `complete_dataset_review.py` to download videos

4. **WSL Path Issues?**
   - The script auto-converts `/mnt/c/` to `C:/`
   - If videos are on a different drive (D:, E:), the script handles it

---

## Differences from Original Script

| Feature | `complete_dataset_review.py` | `generate_local_video_report.py` |
|---------|------------------------------|----------------------------------|
| Downloads videos | ✅ Yes | ❌ No (uses already downloaded) |
| Extracts metadata | ✅ Yes | ❌ No (uses existing CSV) |
| Video paths | 🔴 Relative (broken on Windows) | ✅ Absolute `file://` (works!) |
| Server access | ✅ Required | ❌ Not required |
| Speed | 🐢 Slow (downloads from server) | ⚡ Fast (local only) |

---

## Example Output

```
======================================================================
📹 GENERATE ENHANCED VIDEO REPORT
======================================================================
Input CSV: data/_exp_20251201_172008_train_metadata_20251202_123827.csv
Output HTML: analytics/reports/_exp_20251201_172008_train_enhanced_report.html
======================================================================

📋 Loading metadata...
📊 Analyzing dataset...
🎨 Generating HTML report...

✅ Generated: analytics/reports/_exp_20251201_172008_train_enhanced_report.html

📹 Video Statistics:
   Total: 9
   Found: 9
   Missing: 0

======================================================================
✅ COMPLETE!
======================================================================
📄 Report: analytics/reports/_exp_20251201_172008_train_enhanced_report.html
======================================================================

🎬 TO VIEW THE REPORT:

   Simply double-click the file: analytics/reports/_exp_20251201_172008_train_enhanced_report.html
   Or open it in your browser directly

   The videos will play using absolute file:// paths
======================================================================
```

---

## Notes

- Videos must already be downloaded to your PC
- The script works in WSL and directly in Windows
- Generated HTML files are self-contained and portable (as long as video paths remain valid)
- Each video card shows the absolute path for easy debugging

---

## Quick Reference

### Generate Report (Windows):
```
Double-click: GENERATE_VIDEO_REPORT.bat
```

### Generate Report (Command Line):
```bash
python generate_local_video_report.py data/*_metadata_*.csv
```

### View Report:
```
Double-click: analytics/reports/*_enhanced_report.html
```

That's it! 🎉
