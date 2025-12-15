# Video Dataset Report Generator

## Overview

The `generate_video_report.py` script creates an interactive HTML report for reviewing video datasets. It provides an intuitive web interface with inline video playback, statistical analysis, and organized tabs for efficient dataset review.

## Features

### 1. Interactive Video Playback
- Inline HTML5 video players with play/pause controls
- Preloaded metadata for fast seeking
- Hover effects and responsive design

### 2. Statistical Analysis
- Automatic distribution analysis for all parameters
- Identification of under-represented groups (< 50% of average)
- Identification of over-represented groups (> 150% of average)
- Visual bar charts for all parameter distributions

### 3. Organized Tabs
- **Overview**: Statistics, distributions, and representation alerts
- **By Texture**: Videos grouped by belt texture type
- **By Direction**: Videos grouped by movement direction
- **By Speed**: Videos grouped into speed ranges (stationary, slow, medium, fast)
- **By Angle**: Videos grouped by camera angle
- **Objects**: Videos grouped by object presence/type
- **Blur Effects**: Videos grouped by blur type
- **All Videos**: Searchable grid of all videos

### 4. Rich Metadata Display
Each video card shows:
- Video filename
- Texture type
- Movement direction
- Speed value
- Camera angle
- Brightness and contrast
- Object information (type and count)
- Blur settings
- Resolution
- Random seed

### 5. Search and Filter
- Real-time search across all video metadata
- Filter by any parameter value
- Instant results

## Usage

### Basic Usage

```bash
# Generate report from metadata CSV
python generate_video_report.py data/dataset_metadata.csv
```

This creates `video_dataset_report.html` in the current directory.

### Custom Output Path

```bash
python generate_video_report.py data/dataset_metadata.csv --output reports/my_report.html
```

### With Remote Video URLs

If your videos are hosted remotely or in a specific directory:

```bash
python generate_video_report.py data/dataset_metadata.csv --video-url "http://localhost:8000/data/videos"
```

Or for local files with absolute path:

```bash
python generate_video_report.py data/dataset_metadata.csv --video-url "file:///mnt/data/videos"
```

### Complete Workflow Example

```bash
# Step 1: Extract metadata from server dataset
python extract_video_metadata.py my_experiment_train

# Step 2: Generate HTML report
python generate_video_report.py data/my_experiment_train_metadata_20251202_120000.csv --output reports/train_review.html

# Step 3: Serve videos locally (if needed)
cd data/my_experiment_train
python -m http.server 8000

# Step 4: Open the report in your browser
# The report will load videos from http://localhost:8000/
```

## Command-Line Options

```
usage: generate_video_report.py [-h] [--output OUTPUT] [--video-url VIDEO_URL] csv_path

positional arguments:
  csv_path              Path to metadata CSV file

optional arguments:
  -h, --help            Show help message and exit
  --output OUTPUT       Output HTML file path (default: video_dataset_report.html)
  --video-url VIDEO_URL Base URL for video files (leave empty for relative paths)
```

## Report Structure

### Overview Tab

Shows high-level statistics:
- Total number of videos
- Unique values per parameter
- Distribution bar charts for all parameters
- Alerts for under/over-represented groups

Example alerts:
```
⚠️ Under-represented Groups
- direction: stationary - 50 videos (expected ~150, ratio: 0.33x)
- speed: 12.0 - 75 videos (expected ~125, ratio: 0.60x)

📈 Over-represented Groups
- texture: subtle_gray_stripes - 300 videos (expected ~150, ratio: 2.0x)
```

### Grouped Tabs

Each tab shows sample videos (up to 5) from each group within that category:
- Clear group headers with video counts
- Visual separation between groups
- All metadata displayed per video

### All Videos Tab

- Complete dataset grid
- Search box for filtering
- All videos accessible for review

## Output Format

The generated HTML report is:
- **Self-contained**: All JavaScript and CSS embedded
- **Responsive**: Works on desktop, tablet, and mobile
- **Modern design**: Gradient backgrounds, smooth transitions
- **Accessibility**: Semantic HTML, keyboard navigation
- **Lightweight**: ~50KB base size + metadata JSON

## Configuration for Different Scenarios

### Local Videos (Same Directory as Report)

```bash
python generate_video_report.py metadata.csv
# Videos should be in same directory as HTML file
```

### Videos in Subdirectory

```bash
python generate_video_report.py metadata.csv --video-url "./videos"
```

### Remote Server Videos

```bash
python generate_video_report.py metadata.csv --video-url "https://cdn.example.com/datasets/exp123"
```

### Docker Container Videos

```bash
# Start local file server from Docker volume
docker exec llamafactory bash -c "cd /app/data/my_dataset && python3 -m http.server 8888"

# Generate report pointing to that server
python generate_video_report.py metadata.csv --video-url "http://localhost:8888"
```

## Analysis Capabilities

### Distribution Analysis

The script automatically calculates:
- Frequency counts for each parameter value
- Number of unique values per parameter
- Average samples per value
- Deviation from expected distribution

### Representation Thresholds

- **Under-represented**: < 50% of average frequency
- **Over-represented**: > 150% of average frequency
- **Ratio**: Actual count / expected count

Example:
```
If average is 100 videos per direction:
- "left": 120 videos → Normal (ratio: 1.2x)
- "stationary": 40 videos → Under-represented (ratio: 0.4x)
- "right": 180 videos → Over-represented (ratio: 1.8x)
```

## Browser Compatibility

Tested and working on:
- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+

Requires:
- HTML5 video support
- ES6 JavaScript (template literals, arrow functions)
- CSS Grid and Flexbox

## Performance Considerations

### Large Datasets (1000+ videos)

For datasets with many videos:
- Initial page load may take 2-3 seconds
- Consider splitting into train/test reports separately
- Use the search function to filter before playing videos
- Video preload is set to "metadata" only (not full video)

### Video Loading

- Videos load on-demand (not preloaded)
- Only metadata is fetched initially
- Playback starts when user clicks play
- Multiple videos can be played simultaneously

## Troubleshooting

### Videos Not Loading

**Issue**: Video player shows blank/error

**Solutions**:
1. Check `--video-url` parameter is correct
2. Ensure video files are accessible from HTML file location
3. Check browser console for CORS errors
4. Verify video filenames match exactly with CSV

**CORS Fix** (for remote videos):
```bash
# Python simple server with CORS
python -m http.server 8000 --bind 0.0.0.0
```

### Large CSV Files

**Issue**: Report generation is slow

**Solution**: The script handles up to 10,000 videos efficiently. For larger datasets:
```bash
# Split CSV first
head -n 5001 large_dataset.csv > part1.csv  # Header + 5000 rows
tail -n +5002 large_dataset.csv > part2_data.csv
head -n 1 large_dataset.csv > part2.csv
cat part2_data.csv >> part2.csv

# Generate separate reports
python generate_video_report.py part1.csv --output report_part1.html
python generate_video_report.py part2.csv --output report_part2.html
```

### Search Not Working

**Issue**: Search box doesn't filter videos

**Solution**: Ensure you're on the "All Videos" tab - search only works there.

## Integration with Experiment Workflow

### Complete Pipeline

```bash
# 1. Generate synthetic videos
python data/synthetic_treadmill/synthetic_data_generation.py --preset my_experiment

# 2. Extract metadata from generated videos
python extract_video_metadata.py my_experiment_train
python extract_video_metadata.py my_experiment_test

# 3. Generate review reports
python generate_video_report.py data/my_experiment_train_metadata_*.csv --output reports/train_report.html
python generate_video_report.py data/my_experiment_test_metadata_*.csv --output reports/test_report.html

# 4. Review before training
# Open reports in browser, verify distribution, check sample videos

# 5. Train model if dataset looks good
./run_experiment.sh my_experiment

# 6. Generate prediction report (future feature)
# Could extend to include model predictions alongside ground truth
```

## Future Enhancements

Potential features for next versions:
- Side-by-side comparison of train vs test distributions
- Model prediction overlays on videos
- Confusion matrix integration
- Export filtered subsets to new CSV
- Batch video download functionality
- Annotation capabilities
- Timeline scrubbing with parameter visualization
- Video quality metrics (brightness histogram, motion analysis)

## Examples

### Review Before Training

```bash
python generate_video_report.py data/exp_20251201_train_metadata.csv \
    --output reports/pre_training_review.html \
    --video-url "http://localhost:8000"
```

### Compare Multiple Experiments

```bash
# Generate reports for each experiment
for exp in exp1 exp2 exp3; do
    python generate_video_report.py data/${exp}_metadata.csv \
        --output reports/${exp}_report.html
done

# Open all reports in browser tabs for comparison
```

### Quality Assurance Check

```bash
# Generate report and check for issues
python generate_video_report.py data/new_dataset_metadata.csv

# Look for:
# - Under-represented parameter combinations
# - Missing parameter values
# - Unexpected video counts
# - Visual artifacts in sample videos
```

## File Size and Storage

### HTML Report Size

Base HTML: ~50KB
+ Metadata JSON: ~1-2KB per video
+ Total: ~50KB + (1.5KB × number_of_videos)

Example sizes:
- 100 videos: ~200KB
- 500 videos: ~800KB
- 1000 videos: ~1.5MB
- 5000 videos: ~7.5MB

### Video References

The HTML report only stores video filenames, not the actual video data. Video files remain separate and are loaded on-demand via the video player.

## See Also

- [Dataset Parsing Documentation](./dataset_parsing.md) - Details on `extract_video_metadata.py`
- [Pipeline Usage](../README_PIPELINE_USAGE.md) - End-to-end experiment workflow
- [Dataset Builder](../README_DATASET_BUILDER.md) - Synthetic video generation guide
