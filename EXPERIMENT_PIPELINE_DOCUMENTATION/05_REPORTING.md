# Report Generation

## Purpose
The report generation system creates interactive HTML reports that visualize experiment results, including:
- Model performance metrics and comparisons
- Per-texture and per-angle breakdowns
- Video previews with predictions
- Interactive charts and tables
- Dataset statistics and distribution

## Location
`/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/generate_experiment_report.py`

---

## Architecture

```
generate_experiment_report.py
    │
    ├──> parse_arguments()
    ├──> Load dataset_info.json
    │
    ├──> Load evaluation results:
    │    ├──> Read evaluation_report_*.txt
    │    ├──> Parse metrics (base and fine-tuned)
    │    └──> Extract per-texture/per-angle breakdowns
    │
    ├──> Download videos from server (optional):
    │    ├──> dual_dataset_review.py functions
    │    ├──> rsync from remote to local
    │    └──> Store in data/ folders
    │
    ├──> Load per-video predictions:
    │    ├──> Read per_video_predictions_*.csv
    │    └──> Match videos with metadata
    │
    ├──> Generate HTML report:
    │    ├──> Overall metrics section
    │    ├──> Comparison charts (base vs fine-tuned)
    │    ├──> Per-texture breakdown tables
    │    ├──> Per-angle breakdown tables
    │    ├──> Video grid with predictions
    │    │    ├──> Embedded video players
    │    │    ├──> Prediction badges (correct/incorrect)
    │    │    └──> Metadata overlays
    │    └──> Dataset statistics
    │
    └──> Save HTML to analytics/reports/
```

---

## Key Components

### 1. Report Input Data

The report generator requires several inputs:

**1. Evaluation Report (text)**
```
evaluation_results/evaluation_report_{timestamp}.txt
or
data/{dataset}_test/eval_{timestamp}/evaluation_report_{timestamp}.txt
```

Contains:
- Overall metrics (accuracy, F1, precision, recall)
- Per-class metrics (moving, stopped)
- Per-texture breakdowns
- Per-angle breakdowns
- Base vs fine-tuned comparison

**2. Per-Video Predictions (CSV)**
```
data/{dataset}_test/eval_{timestamp}/per_video_predictions_base_{timestamp}.csv
data/{dataset}_test/eval_{timestamp}/per_video_predictions_finetuned_{timestamp}.csv
```

CSV columns:
- `video_path`: Relative path to video file
- `prediction`: Model prediction ("moving" or "stopped")
- `label`: Ground truth label
- `speed`: Video speed parameter
- `correct`: Boolean indicating correct prediction
- `model_output`: Raw model output text

**3. Video Files**
```
data/{dataset}_train/*.mp4
data/{dataset}_test/*.mp4
```

Actual video files for embedding in report.

**4. Dataset Info**
```
data/dataset_info.json
```

Maps dataset names to JSON files.

### 2. HTML Report Structure

**Overall Layout:**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Experiment Report: {dataset_name}</title>
    <style>
        /* Embedded CSS for styling */
        /* Bootstrap-like responsive grid */
        /* Custom video player styling */
    </style>
    <script>
        /* JavaScript for interactivity */
        /* Filtering, sorting */
        /* Video playback controls */
    </script>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <h1>Experiment Report: {dataset_name}</h1>
        <p>Generated: {timestamp}</p>

        <!-- Overall Metrics Section -->
        <section id="overall-metrics">
            <h2>Overall Performance</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Base Model</h3>
                    <p>Accuracy: 65.00%</p>
                    <p>F1 Score: 63.33%</p>
                </div>
                <div class="metric-card">
                    <h3>Fine-Tuned Model</h3>
                    <p>Accuracy: 95.00%</p>
                    <p>F1 Score: 95.00%</p>
                </div>
                <div class="metric-card improvement">
                    <h3>Improvement</h3>
                    <p>+30.00%</p>
                </div>
            </div>
        </section>

        <!-- Comparison Charts -->
        <section id="charts">
            <h2>Performance Comparison</h2>
            <canvas id="accuracy-chart"></canvas>
            <canvas id="f1-chart"></canvas>
        </section>

        <!-- Per-Texture Breakdown -->
        <section id="per-texture">
            <h2>Per-Texture Performance</h2>
            <table class="breakdown-table">
                <thead>
                    <tr>
                        <th>Texture</th>
                        <th>Base Accuracy</th>
                        <th>Fine-Tuned Accuracy</th>
                        <th>Improvement</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>subtle_gray_stripes</td>
                        <td>65.00%</td>
                        <td>95.00%</td>
                        <td class="positive">+30.00%</td>
                    </tr>
                </tbody>
            </table>
        </section>

        <!-- Video Grid -->
        <section id="videos">
            <h2>Video Predictions</h2>
            <div class="filters">
                <button onclick="filterVideos('all')">All</button>
                <button onclick="filterVideos('correct')">Correct</button>
                <button onclick="filterVideos('incorrect')">Incorrect</button>
            </div>
            <div class="video-grid">
                <div class="video-card" data-correct="true">
                    <video controls>
                        <source src="data/.../treadmill_0042_....mp4" type="video/mp4">
                    </video>
                    <div class="video-info">
                        <span class="badge correct">✓ Correct</span>
                        <p>Prediction: Moving</p>
                        <p>Ground Truth: Moving</p>
                        <p>Speed: 3.0</p>
                        <p>Angle: 52°</p>
                    </div>
                </div>
                <!-- More video cards... -->
            </div>
        </section>

        <!-- Dataset Statistics -->
        <section id="stats">
            <h2>Dataset Statistics</h2>
            <ul>
                <li>Total Videos: 100 (80 train, 20 test)</li>
                <li>Train Texture: subtle_gray_stripes</li>
                <li>Test Texture: subtle_gray_stripes</li>
                <li>Train Angle: 0°</li>
                <li>Test Angle: 52°</li>
            </ul>
        </section>
    </div>
</body>
</html>
```

### 3. Key Features

**1. Interactive Video Grid**

Each video card shows:
- Video player with controls
- Prediction status badge (✓ Correct / ✗ Incorrect)
- Base model prediction
- Fine-tuned model prediction
- Ground truth label
- Metadata (speed, angle, texture)

**JavaScript filtering:**
```javascript
function filterVideos(filter) {
    const cards = document.querySelectorAll('.video-card');
    cards.forEach(card => {
        if (filter === 'all') {
            card.style.display = 'block';
        } else if (filter === 'correct') {
            card.style.display = card.dataset.correct === 'true' ? 'block' : 'none';
        } else if (filter === 'incorrect') {
            card.style.display = card.dataset.correct === 'false' ? 'block' : 'none';
        }
    });
}
```

**2. Performance Charts**

Uses Chart.js or similar library for visualizations:
- Bar charts comparing base vs fine-tuned
- Line charts showing metrics by texture/angle
- Confusion matrices

**3. Responsive Design**

CSS grid/flexbox for responsive layout:
```css
.video-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 20px;
}

@media (max-width: 768px) {
    .video-grid {
        grid-template-columns: 1fr;
    }
}
```

**4. Color-Coded Metrics**

Visual indicators for performance:
```css
.positive { color: green; }
.negative { color: red; }
.neutral { color: gray; }

.badge.correct {
    background: green;
    color: white;
}

.badge.incorrect {
    background: red;
    color: white;
}
```

### 4. Video Path Handling

**Challenge:** Videos are stored locally, but HTML needs relative paths.

**Solution:**
```python
def get_relative_video_path(video_path, report_path):
    """
    Calculate relative path from report to video.

    Report: analytics/reports/experiment_report.html
    Video:  data/_exp_20251207_143033_test/treadmill_0042_....mp4
    Result: ../../data/_exp_20251207_143033_test/treadmill_0042_....mp4
    """
    report_dir = Path(report_path).parent
    video_full = Path(video_path)
    relative = os.path.relpath(video_full, report_dir)
    return relative
```

**Why relative paths?**
- HTML can be moved without breaking video links
- Works when opened directly in browser
- No need for web server

### 5. Report Generation Function

**Pseudocode:**
```python
def generate_report(dataset_name, output_dir, skip_video_download):
    # 1. Load evaluation results
    evaluation_results = load_evaluation_report(dataset_name)
    base_metrics = evaluation_results['base']
    finetuned_metrics = evaluation_results['finetuned']

    # 2. Load per-video predictions
    base_predictions = pd.read_csv(f'per_video_predictions_base_*.csv')
    finetuned_predictions = pd.read_csv(f'per_video_predictions_finetuned_*.csv')

    # 3. Download videos if needed
    if not skip_video_download:
        download_videos_from_server(dataset_name)

    # 4. Build HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Experiment Report: {dataset_name}</title>
        {generate_css()}
    </head>
    <body>
        <div class="container">
            {generate_header(dataset_name)}
            {generate_metrics_section(base_metrics, finetuned_metrics)}
            {generate_charts(base_metrics, finetuned_metrics)}
            {generate_texture_breakdown(evaluation_results)}
            {generate_angle_breakdown(evaluation_results)}
            {generate_video_grid(base_predictions, finetuned_predictions)}
            {generate_dataset_stats(dataset_name)}
        </div>
        {generate_javascript()}
    </body>
    </html>
    """

    # 5. Save HTML
    report_path = Path(output_dir) / f"{dataset_name}_dual_report_{timestamp}.html"
    with open(report_path, 'w') as f:
        f.write(html)

    print(f"Report saved to: {report_path}")
    return report_path
```

### 6. Video Preview Grid

**Generating video cards:**
```python
def generate_video_card(video_path, base_pred, finetuned_pred, ground_truth):
    relative_path = get_relative_video_path(video_path, report_path)

    is_correct_base = (base_pred == ground_truth)
    is_correct_finetuned = (finetuned_pred == ground_truth)

    return f"""
    <div class="video-card" data-correct="{is_correct_finetuned}">
        <video controls width="100%" preload="metadata">
            <source src="{relative_path}" type="video/mp4">
        </video>
        <div class="video-info">
            <div class="predictions">
                <div class="prediction-row">
                    <strong>Base:</strong>
                    <span class="badge {'correct' if is_correct_base else 'incorrect'}">
                        {base_pred}
                    </span>
                </div>
                <div class="prediction-row">
                    <strong>Fine-Tuned:</strong>
                    <span class="badge {'correct' if is_correct_finetuned else 'incorrect'}">
                        {finetuned_pred}
                    </span>
                </div>
                <div class="prediction-row">
                    <strong>Ground Truth:</strong>
                    <span class="badge ground-truth">{ground_truth}</span>
                </div>
            </div>
            <div class="metadata">
                <p><strong>Speed:</strong> {metadata['speed']}</p>
                <p><strong>Angle:</strong> {metadata['angle']}</p>
                <p><strong>Texture:</strong> {metadata['texture']}</p>
            </div>
        </div>
    </div>
    """
```

### 7. Usage

**From run_experiment.sh:**
```bash
python generate_experiment_report.py \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "analytics/reports" \
  --skip_video_download  # Videos already downloaded
```

**Manual usage:**
```bash
# Generate report for specific experiment
python generate_experiment_report.py \
  --dataset_name "_exp_20251207_143033" \
  --output_dir "analytics/reports"

# Download videos from server first
python generate_experiment_report.py \
  --dataset_name "_exp_20251207_143033" \
  --output_dir "analytics/reports" \
  --remote_user "seedoo" \
  --remote_host "hetzner-gpu.tail9e6e7.ts.net" \
  --remote_project_root "/workspace"
```

### 8. Output Location

**Report saved to:**
```
analytics/reports/{dataset_name}_dual_report_{timestamp}.html
```

**Example:**
```
analytics/reports/_exp_20251207_143033_dual_report_20251207_151022.html
```

**Opening report:**
```bash
# Windows
start analytics/reports/*_dual_report_*.html

# Mac
open analytics/reports/*_dual_report_*.html

# Linux
xdg-open analytics/reports/*_dual_report_*.html
```

### 9. Report Customization

**Modifying report style:**
Edit CSS section in `generate_experiment_report.py`:
```python
def generate_css():
    return """
    <style>
        :root {
            --primary-color: #007bff;
            --success-color: #28a745;
            --danger-color: #dc3545;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Custom styles here */
    </style>
    """
```

**Adding new sections:**
```python
def generate_custom_section(data):
    return f"""
    <section id="custom-section">
        <h2>Custom Analysis</h2>
        <!-- Your content here -->
    </section>
    """

# In main HTML generation:
html = f"""
    ...
    {generate_custom_section(custom_data)}
    ...
"""
```

### 10. Advanced Features

**1. Video Comparison**
Side-by-side comparison of same video across different experiments:
```html
<div class="comparison-grid">
    <div class="comparison-item">
        <h4>Experiment 1</h4>
        <video src="exp1/video.mp4" controls></video>
        <p>Prediction: Moving (✓)</p>
    </div>
    <div class="comparison-item">
        <h4>Experiment 2</h4>
        <video src="exp2/video.mp4" controls></video>
        <p>Prediction: Stopped (✗)</p>
    </div>
</div>
```

**2. Error Analysis**
Group incorrect predictions by common patterns:
```python
def group_errors(predictions):
    errors = predictions[predictions['correct'] == False]

    # Group by texture
    errors_by_texture = errors.groupby('texture').size()

    # Group by angle
    errors_by_angle = errors.groupby('angle').size()

    # Group by speed
    errors_by_speed = errors.groupby('speed').size()

    return {
        'texture': errors_by_texture,
        'angle': errors_by_angle,
        'speed': errors_by_speed
    }
```

**3. Interactive Filtering**
Filter videos by multiple criteria:
```javascript
function filterVideos() {
    const texture = document.getElementById('texture-filter').value;
    const angle = document.getElementById('angle-filter').value;
    const correctness = document.getElementById('correctness-filter').value;

    document.querySelectorAll('.video-card').forEach(card => {
        const show = (
            (texture === 'all' || card.dataset.texture === texture) &&
            (angle === 'all' || card.dataset.angle === angle) &&
            (correctness === 'all' || card.dataset.correct === correctness)
        );
        card.style.display = show ? 'block' : 'none';
    });
}
```

---

## Statistical Failure Analysis Tab (New Feature - 2025-12-07)

### Purpose

The **Failure Analysis** tab provides automatic statistical analysis of model failures, identifying which experimental features (distance, angle, blur, object presence, etc.) significantly affect model accuracy. This helps diagnose *why* a model fails and guides future experiments.

### Key Capabilities

1. **Dynamic Feature Detection**: Automatically extracts ALL features from video filenames (e.g., `dist`, `angle`, `blur`, `object`, `stripe`, `bg`)
2. **Statistical Significance Testing**: Chi-squared and Fisher's exact tests identify which features significantly affect accuracy
3. **Per-Feature Failure Examples**: Shows video examples grouped by significant features with worst-performing values

### How It Works

**1. Feature Extraction from Filenames:**
```
treadmill_0042_subtle_gray_stripes_stripe226_bg120_left_speed14.0_angle30_dist5.0.mp4
                                    ↓
Extracted features: stripe=226, bg=120, speed=14.0, angle=30, dist=5.0
```

The system uses regex to find all `name+number` patterns and analyzes each one.

**2. Statistical Tests:**
- **Chi-squared test**: Tests if accuracy differs significantly across ALL values of a feature
- **Fisher's exact test**: Pairwise comparison between specific feature values
- **Significance threshold**: p < 0.01 (configurable)

**3. Dynamic Conclusions:**
| Scenario | Example Conclusion |
|----------|-------------------|
| One significant feature | "DIST is the ONLY statistically significant factor affecting model accuracy." |
| Multiple significant | "Multiple significant factors found: DIST AND ANGLE all affect model accuracy." |
| None significant | "No statistically significant differences detected in any feature." |

### Tab Contents

The Failure Analysis tab displays:

1. **Key Finding Box**: Summary conclusion from statistical analysis
2. **Statistically Significant Factors**: List of all factors that affect accuracy with p-values
3. **Chi-Squared Tests Table**: Shows Chi² statistic and p-value for each feature
4. **Fisher's Exact Test Table**: Significant pairwise comparisons
5. **Accuracy Tables by Feature**: Shows accuracy breakdown for ALL detected features, with "SIGNIFICANT" badges on relevant tables
6. **Failed Examples by Significant Feature**: For each significant feature, shows videos grouped by worst-performing values

### Example Output

```
Key Finding: DIST is the ONLY statistically significant factor affecting model accuracy.

Significant Factors:
- dist: Chi2=160.00, p=1.46e-33
- dist (moving): Chi2=160.00, p=1.46e-33

Accuracy by dist (SIGNIFICANT):
| dist | Label   | Correct | Total | Accuracy |
|------|---------|---------|-------|----------|
| 1.0  | moving  | 30      | 32    | 93.8%    |
| 1.0  | stopped | 31      | 32    | 96.9%    |
| 5.0  | moving  | 0       | 32    | 0.0%     |  ← Model fails at far distances
| 5.0  | stopped | 32      | 32    | 100.0%   |

Failed Examples by DIST (Worst Performing Values):
├── dist = 10.0 (Accuracy: 50.0%, 64 samples)
│   └── [4 video examples with playback]
├── dist = 8.0 (Accuracy: 50.0%, 64 samples)
│   └── [4 video examples with playback]
└── dist = 5.0 (Accuracy: 50.0%, 64 samples)
    └── [4 video examples with playback]
```

### Implementation Files

| File | Function |
|------|----------|
| `analytics/analyze_evaluation_failures.py` | `analyze_predictions()` - runs statistical analysis |
| `generate_experiment_report.py` | `generate_failure_analysis_tab_html()` - generates HTML |

### Adding New Features to Analysis

The analysis is fully automatic. To analyze new features:

1. **Encode feature in filename**: Use pattern `featureName` + `value` (e.g., `blur0.5`, `object1`)
2. **Generate videos with the feature**
3. **Run experiment** - analysis will automatically detect and test the new feature

**Skip list**: Features like `seed`, `treadmill`, `x`, `mp` are skipped (configured in `SKIP_FEATURES` in `analyze_evaluation_failures.py`)

### Using Results

The failure analysis helps you:
- **Identify model weaknesses**: Which features cause failures?
- **Plan next experiments**: Should you train with more distance variation? Add blur augmentation?
- **Compare experiments**: Did training on feature X improve robustness to feature Y?

## Next Steps
- Read `06_EXPERIMENT_TRACKING.md` for how experiments are logged
- See examples in `analytics/reports/` folder
