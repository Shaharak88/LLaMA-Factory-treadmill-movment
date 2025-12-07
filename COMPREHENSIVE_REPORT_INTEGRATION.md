# Comprehensive Experiment Report Integration

## Overview

This document describes the integration of full dataset analytics from `dual_dataset_review.py` into the experiment reporting pipeline, resulting in a comprehensive HTML report that combines model performance metrics with complete dataset analysis.

**Date:** December 7, 2025
**Commit:** `6af8471`
**Files Modified:** `generate_experiment_report.py`

---

## What Changed

### Previous Behavior

Before this change:
- HTML report showed only basic dataset comparison (video counts, simple plots)
- Model performance metrics displayed but limited dataset analytics
- Users needed to run `dual_dataset_review.py` separately for full dataset analysis
- CSV stored relative file paths

### New Behavior

After this change:
- **Single comprehensive HTML report** with 20 interactive tabs
- Full integration of all `dual_dataset_review.py` analytics
- Model performance in dedicated tab (first tab)
- Complete train/test dataset analysis with interactive visualizations
- CSV stores clickable `file://` URLs
- **New unique HTML generated for each evaluation** (timestamped, never overwrites)

---

## Report Generation

### How It Works

**Every evaluation generates a new unique HTML report:**

```
analytics/reports/
├── _exp_20251207_125853_dual_report_20251207_130143.html  ← First eval
├── _exp_20251207_125853_dual_report_20251207_142055.html  ← Second eval (same dataset)
├── _exp_20251208_101523_dual_report_20251208_110234.html  ← Different experiment
└── _exp_20251208_101523_dual_report_20251208_155521.html  ← Re-evaluation
```

**Report Naming Convention:**
```
{dataset_name}_dual_report_{YYYYMMDD_HHMMSS}.html
```

**Properties:**
- ✅ Unique timestamp prevents overwrites
- ✅ Self-contained (all plots embedded as base64)
- ✅ Standalone HTML (no external dependencies)
- ✅ Linked in CSV for experiment tracking

**CSV Tracking:**
Each report is tracked in `experiments_log.csv`:
```csv
experiment_id,dataset_name,dataset_comparison_report
42,_exp_20251207_125853,file:///mnt/c/.../reports/_exp_20251207_125853_dual_report_20251207_130143.html
```

If you re-evaluate the same experiment, it generates a **new report with new timestamp** and **updates the CSV row** with the latest report URL.

---

## Report Structure

The generated HTML report contains **20 interactive tabs** organized as follows:

### 1. 🎯 Model Performance Tab (Tab 1)

**Content:**
- Base Model metrics (Accuracy, F1 Score)
- Fine-Tuned Model metrics (Accuracy, F1 Score)
- Non-default hyperparameters display

**Features:**
- Clean, focused view of model comparison
- Highlights only hyperparameters that differ from defaults
- Color-coded gradient cards for visual appeal

---

### 2. 🔵 Train Dataset Tabs (Tabs 2-10)

#### Tab 2: Train Overview
- Total video count
- Unique textures, directions, speed variations
- Parameter distribution bar charts
- Under-represented/over-represented group alerts

#### Tab 3: Train Defaults
- Table showing most common value for each parameter
- Count and percentage for each default value
- Covers all 20+ parameters (texture, direction, speed, angle, stripe gray, etc.)

#### Tab 4: Train Texture
- Videos grouped by texture type
- Up to 5 sample videos per texture with playback
- Full metadata display for each video

#### Tab 5: Train Direction
- Videos grouped by direction (up, down, left, right, stationary)
- Sample videos with interactive playback

#### Tab 6: Train Speed
- Videos grouped by speed ranges (stationary, slow, medium, fast)
- Speed distribution visualization

#### Tab 7: Train Angle
- Videos grouped by viewing angle
- Angle distribution across dataset

#### Tab 8: Train Objects
- Videos grouped by object presence/type
- Shows object-enabled vs no-objects

#### Tab 9: Train Blur
- Videos grouped by blur type
- Shows blur-enabled vs no-blur

#### Tab 10: Train All Videos
- Searchable catalog of all training videos
- Full metadata display for each video
- Interactive video playback
- Real-time search filtering

---

### 3. 🟣 Test Dataset Tabs (Tabs 11-19)

**Structure:** Identical to Train Dataset tabs (Tabs 2-10)

- Tab 11: Test Overview
- Tab 12: Test Defaults
- Tab 13: Test Texture
- Tab 14: Test Direction
- Tab 15: Test Speed
- Tab 16: Test Angle
- Tab 17: Test Objects
- Tab 18: Test Blur
- Tab 19: Test All Videos

---

### 4. 📊 Distribution Comparison Tab (Tab 20)

**Content:**
- Train vs Test video counts
- Train/Test ratio calculation
- **PDF/KDE Distribution Plots:**
  - Speed distribution comparison
  - Angle distribution comparison
  - Stripe gray values (conditional - only for subtle_gray textures)
  - Background gray values (conditional - only for subtle_gray textures)
- **Categorical Comparison Charts:**
  - Texture distribution (side-by-side bars)
  - Direction distribution (side-by-side bars)

**Features:**
- All plots embedded as base64-encoded images
- High-quality matplotlib visualizations
- Kernel Density Estimation (KDE) for continuous variables
- Side-by-side bar charts for categorical variables

---

## Technical Implementation

### Key Methods

#### 1. `generate_comparison_plots(train_metadata, test_metadata)`

Generates all comparison visualizations:
- Speed and angle PDF/KDE plots using `generate_pdf_plot()`
- Conditional stripe/background gray plots (only for subtle gray textures)
- Texture and direction categorical comparisons using `generate_categorical_comparison()`
- Returns dictionary of plot_name → base64 image data

#### 2. `generate_model_performance_tab_html(experiment)`

Creates model performance tab content:
- Extracts metrics from CSV (accuracy, F1 scores)
- Identifies non-default hyperparameters
- Formats display names (snake_case → Title Case)
- Returns HTML for performance tab

#### 3. `_build_comprehensive_html_report(...)`

Builds complete HTML with all tabs:
- Prepares JSON data for JavaScript
- Samples videos (max 5 per group)
- Generates comparison plots HTML
- Constructs full HTML structure with CSS and JavaScript
- Returns complete HTML string

#### 4. `generate_dual_report(experiment)`

Main orchestration method:
- Downloads videos (if not present locally)
- Extracts/loads metadata
- Analyzes distributions and groups videos
- Generates all plots
- Builds HTML report
- Creates timestamped filename
- **Returns `file://` URL** (not relative path)

**Timestamping Logic:**
```python
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
report_filename = f"{dataset_name}_dual_report_{timestamp}.html"
report_path = self.reports_dir / report_filename
```

### JavaScript Functions

All JavaScript functions from `dual_dataset_review.py` integrated:

```javascript
switchTab(event, tabName)           // Tab navigation with lazy loading
renderVideoCard(video, baseUrl)     // Video card with metadata
renderDistributionCharts(stats, id) // Bar charts for distributions
renderRepresentationAlerts(stats)   // Under/over-represented warnings
renderDefaultsTable(metadata, id)   // Defaults table generation
renderGroupTab(tabId, category)     // Grouped video views
renderAllVideos(dataset)            // Full video catalog
filterTrainVideos()                 // Search filtering for train
filterTestVideos()                  // Search filtering for test
```

### CSS Enhancements

Added comprehensive styling:
- Tab navigation with hover effects
- Color-coded tabs (performance = orange, train = blue, test = purple, comparison = gradient)
- Responsive grid layouts
- Card-based designs with hover animations
- Gradient backgrounds and shadows
- Mobile-friendly media queries

---

## File Path Format Change

### Before
```csv
dataset_comparison_report,analytics/reports/report_20251207_130143.html
```

### After
```csv
dataset_comparison_report,file:///mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/analytics/reports/report_20251207_130143.html
```

**Benefits:**
- ✅ Clickable in Excel, LibreOffice, and most CSV viewers
- ✅ Direct browser opening without path resolution
- ✅ Works across different systems with absolute paths
- ✅ No ambiguity about file location

---

## Integration with Pipeline

### Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│ SERVER: run_full_pipeline.py                               │
├─────────────────────────────────────────────────────────────┤
│ Step 1: Generate train/test datasets                       │
│ Step 2: Train model with LoRA                              │
│ Step 3: Evaluate base and fine-tuned models                │
└─────────────────────────────────────────────────────────────┘
                          ↓
                    rsync results
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ LOCAL PC: run_experiment.sh                                │
├─────────────────────────────────────────────────────────────┤
│ Phase 4: Retrieve results & sync CSV                       │
│ Phase 5: Generate comprehensive HTML report ← NEW          │
│   ├─ Check if videos exist locally                         │
│   ├─ Download videos if needed (smart download)            │
│   ├─ Extract/load metadata                                 │
│   ├─ Analyze distributions                                 │
│   ├─ Generate comparison plots                             │
│   ├─ Build comprehensive HTML (20 tabs)                    │
│   ├─ Create timestamped filename                           │
│   └─ Update CSV with file:// URL                           │
└─────────────────────────────────────────────────────────────┘
```

### Smart Video Download

The report generator intelligently handles video downloads:

```python
def ensure_videos_downloaded(dataset_name):
    if check_videos_downloaded(dataset_name):
        logger.info(f"✓ Videos already downloaded")
        return local_path

    logger.info(f"Downloading videos...")
    download_videos(...)
    convert_videos_to_h264(...)
    return local_path
```

**Benefits:**
- No re-download if videos already exist locally
- Automatic H.264 conversion for browser compatibility
- Progress logging for transparency
- Efficient bandwidth usage

---

## Analytics Included

### From dual_dataset_review.py

All analytics from `dual_dataset_review.py` are now integrated:

#### 1. **Statistical Analysis**
- Total video counts
- Unique value counts per parameter
- Distribution analysis with frequency counts
- Average videos per parameter value

#### 2. **Representation Detection**
- Under-represented groups (< 50% of average)
- Over-represented groups (> 50% of average)
- Expected vs actual counts
- Ratio calculations

#### 3. **Defaults Analysis**
- Most common value for each parameter
- Count and percentage for each default
- Covers 20+ parameters:
  - Core: texture, direction, speed, angle
  - Visual: brightness, contrast, stripe_gray, background_gray
  - Advanced: objects, blur, resolution, seed

#### 4. **Grouped Visualizations**
- Videos grouped by texture type
- Videos grouped by direction
- Videos grouped by speed ranges
- Videos grouped by viewing angle
- Videos grouped by object presence
- Videos grouped by blur type
- Max 5 samples per group for performance

#### 5. **Interactive Features**
- Video playback with HTML5 player
- Full metadata display per video
- Search/filter functionality
- Lazy loading for performance
- Responsive grid layouts

---

## Usage

### Command Line

```bash
# Generate report for specific experiment
python3 generate_experiment_report.py --experiment-id 42 --csv-path data/experiments_log.csv

# Or by dataset name
python3 generate_experiment_report.py --dataset-name _exp_20251207_125853 --csv-path data/experiments_log.csv
```

### Via Pipeline

The report is automatically generated in phase 5 of `run_experiment.sh`:

```bash
./run_experiment.sh --texture subtle_gray_stripes --epochs 5
```

After server-side execution and result retrieval, the script automatically:
1. Generates the comprehensive HTML report with unique timestamp
2. Updates the CSV with the `file://` URL
3. Displays the report URL in terminal output

### Opening Reports

**From Terminal Output:**
```
✅ Comprehensive report generated!

Report URL: file:///mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/analytics/reports/_exp_20251207_125853_dual_report_20251207_130223.html

To view: Click the link or paste it in your browser
```

**From CSV:**
1. Open `data/experiments_log.csv` in Excel/LibreOffice
2. Find the `dataset_comparison_report` column
3. Click the `file://` URL (Ctrl+Click in some applications)
4. Report opens in your default browser

---

## Video Metadata Display

Each video card shows complete metadata:

```
┌────────────────────────────────┐
│  [Video Player]                │
│  treadmill_0042_...            │
├────────────────────────────────┤
│ Index: 0042                    │
│ Texture: subtle_gray_stripes   │
│ Direction: left                │
│ Speed: 8.5                     │
│ Angle: 30.0°                   │
│ Brightness: 0.0                │
│ Contrast: 1.0                  │
│ Stripe Gray: 125               │
│ Background Gray: 140           │
│ Resolution: 640x480            │
│ Seed: 42                       │
└────────────────────────────────┘
```

---

## Report Management

### Multiple Reports Per Experiment

Since each evaluation generates a **new timestamped report**, you can:

1. **Compare different evaluation runs** of the same experiment
2. **Track changes** over time (e.g., after model improvements)
3. **Keep historical records** without overwrites
4. **Debug evaluation issues** by comparing reports

**Example Timeline:**
```
_exp_20251207_125853/
├── First evaluation  → report_20251207_130143.html
├── Bug fix           → (no new report)
├── Re-evaluation     → report_20251207_142055.html
└── Final evaluation  → report_20251207_155521.html
```

**CSV always points to the LATEST report** for that experiment.

### Cleanup Old Reports

If you want to remove old reports:

```bash
# List all reports for an experiment
ls analytics/reports/_exp_20251207_125853_dual_report_*.html

# Keep only the latest, remove older ones
cd analytics/reports
ls -t _exp_20251207_125853_dual_report_*.html | tail -n +2 | xargs rm

# Or keep last N reports (e.g., 5)
ls -t _exp_20251207_125853_dual_report_*.html | tail -n +6 | xargs rm
```

---

## Browser Compatibility

### Supported Browsers
- ✅ Google Chrome / Chromium
- ✅ Mozilla Firefox
- ✅ Microsoft Edge
- ✅ Safari (macOS)

### Features
- HTML5 video playback (H.264 codec)
- CSS Grid layouts
- Flexbox for responsive design
- JavaScript ES6+ features
- Sticky tab navigation
- Smooth scrolling

### Video Compatibility
Videos are automatically converted to H.264 codec for maximum browser compatibility:
- Original: MPEG-4 from Blender
- Converted: H.264 (libx264, yuv420p)
- Result: Plays in all modern browsers

---

## Performance Optimizations

### 1. Lazy Loading
- Video grids not rendered until tab is opened
- Prevents initial page load slowdown
- Improves perceived performance

### 2. Video Sampling
- Max 5 videos per group in grouped views
- Full catalog available in "All Videos" tab
- Reduces initial HTML size

### 3. Base64 Plot Embedding
- All plots embedded as data URIs
- No external image file dependencies
- Single self-contained HTML file

### 4. Smart Video Download
- Checks local existence before downloading
- Downloads only if missing
- Reuses existing videos across report generations

---

## File Locations

```
LLaMA-Factory/
├── generate_experiment_report.py          # Main script (MODIFIED)
├── dual_dataset_review.py                 # Original analytics script (UNCHANGED)
├── experiment_tracker.py                  # CSV tracking (UNCHANGED)
├── run_experiment.sh                      # Pipeline orchestrator (UNCHANGED)
├── data/
│   ├── experiments_log.csv               # Tracking CSV (stores file:// URLs)
│   ├── _exp_TIMESTAMP_train/             # Training videos
│   ├── _exp_TIMESTAMP_test/              # Test videos
│   └── _exp_TIMESTAMP_train_metadata_*.csv  # Metadata CSVs
└── analytics/
    └── reports/
        ├── _exp_TIMESTAMP_dual_report_TIMESTAMP1.html  # First eval
        ├── _exp_TIMESTAMP_dual_report_TIMESTAMP2.html  # Second eval
        └── _exp_TIMESTAMP_dual_report_TIMESTAMP3.html  # Third eval
```

---

## Troubleshooting

### Issue: Videos don't play in browser

**Cause:** Videos not converted to H.264
**Solution:** Script automatically converts. If manual conversion needed:
```bash
cd data/_exp_TIMESTAMP_train
ffmpeg -i input.mp4 -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -c:a copy output.mp4
```

### Issue: Report URL not clickable in CSV

**Cause:** CSV viewer doesn't recognize `file://` URLs
**Solution:**
- Excel: Ctrl+Click or right-click → "Open Hyperlink"
- LibreOffice: Ctrl+Click
- Manual: Copy URL and paste in browser address bar

### Issue: Missing comparison plots

**Cause:** Insufficient data for plot generation
**Solution:** Check that both train and test datasets have:
- At least 2 videos with valid speed values
- At least 2 videos with valid angle values
- Check logs for plot generation warnings

### Issue: Metadata CSV not found

**Cause:** Metadata not extracted from server
**Solution:** Script automatically extracts. If manual extraction needed:
```bash
python3 dual_dataset_review.py _exp_TIMESTAMP
```

### Issue: Multiple reports consuming disk space

**Cause:** Each evaluation creates new timestamped report
**Solution:** Periodically clean up old reports (see "Report Management" section above)

---

## Dependencies

### Python Packages

Required (already in environment):
```
matplotlib >= 3.5.0    # For plot generation
scipy >= 1.7.0         # For KDE (Kernel Density Estimation)
numpy >= 1.21.0        # For numerical operations
```

### External Tools

Required (should be installed):
```
ffmpeg                 # For video conversion
ffprobe                # For video codec detection
rsync                  # For file synchronization
ssh                    # For server communication
```

---

## Future Enhancements

### Potential Improvements

1. **Export Capabilities**
   - Export statistics to CSV
   - Export plots as separate images
   - PDF report generation

2. **Advanced Filtering**
   - Multi-parameter filtering
   - Range-based filters
   - Saved filter presets

3. **Comparison Modes**
   - Compare multiple experiments
   - Diff view between experiments
   - Historical trend analysis

4. **Performance Metrics Expansion**
   - Per-class F1 scores (moving/stopped)
   - Confusion matrices
   - ROC curves
   - Loss curves over epochs

5. **Interactive Plots**
   - Plotly integration for zoom/pan
   - Interactive distribution adjustments
   - Click-to-filter on plots

6. **Report Management UI**
   - Web interface to browse all reports
   - Automatic cleanup of old reports
   - Report comparison tool

---

## Related Documentation

- `DEPLOYMENT.md` - Server deployment and configuration
- `README_PIPELINE_USAGE.md` - Pipeline usage guide
- `HOW_TO_VIEW_VIDEOS.md` - Video viewing instructions
- `dual_dataset_review.py` - Original analytics implementation

---

## Summary

This integration provides a **comprehensive, self-contained HTML report** that combines:
- ✅ Model performance metrics (base vs fine-tuned)
- ✅ Complete dataset analytics (20 tabs total)
- ✅ Interactive video playback
- ✅ Distribution comparisons (PDF plots, categorical charts)
- ✅ Searchable video catalogs
- ✅ Clickable file:// URLs in CSV
- ✅ Unique timestamped reports (never overwrites)

**Result:** No need to run separate scripts - everything is in one place, automatically generated as part of the experiment pipeline. Each evaluation creates a new timestamped report for historical tracking.

---

**Last Updated:** December 7, 2025
**Version:** 1.0
**Author:** AI-Generated with Claude Code
