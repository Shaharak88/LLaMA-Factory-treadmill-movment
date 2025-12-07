# Evaluation Failure Analysis Tool

Analyzes per-video prediction results from treadmill motion detection experiments to identify which features contribute to model failures using statistical significance testing.

## Overview

This tool examines evaluation CSV files and performs:
1. **Accuracy breakdown** by feature (distance, angle, stripe color, background color)
2. **Chi-squared tests** to determine if accuracy differs across feature values
3. **Pairwise Fisher's exact tests** for specific feature value comparisons
4. **Statistical significance filtering** (default: p < 0.01)

## Usage

```bash
# Basic usage (uses default CSV filename)
python analyze_evaluation_failures.py per_video_predictions_finetuned_20251207.csv

# With custom p-value threshold
python analyze_evaluation_failures.py results.csv --threshold 0.05
```

## Input Format

The CSV file must have these columns:
- `video_path` - Path to video file (contains encoded feature values)
- `prediction` - Model's prediction
- `label` - Ground truth label ("moving" or "stopped")
- `correct` - Whether prediction was correct ("True" or "False")

Video filenames must encode features in this format:
```
treadmill_0000_subtle_gray_stripes_stripe226_bg120_left_speed14.0_angle30_dist5.0_...mp4
```

Extracted features:
- `dist1.0`, `dist3.0`, etc. - Distance from camera
- `angle0`, `angle30`, etc. - Viewing angle
- `stripe226`, `stripe229`, etc. - Stripe gray value (0-255)
- `bg120`, `bg124`, etc. - Background gray value (0-255)

## Output

### 1. Accuracy Tables
Shows accuracy breakdown for each feature value and label combination:

```
======================================================================
ACCURACY BY DISTANCE
======================================================================
Distance     Label      Correct    Total      Accuracy     Error
----------------------------------------------------------------------
1.0          moving     32         32         100.00%       0.00%
1.0          stopped    29         32          90.63%       9.37%
3.0          moving     0          32           0.00%     100.00%
...
```

### 2. Chi-Squared Test Results
Tests whether accuracy is independent of feature value:

```
--- CHI-SQUARED TESTS (Overall feature significance) ---
Feature                        Chi2        p-value    Significant
----------------------------------------------------------------------
Distance                     160.00       1.46e-33      YES ***
Angle                          0.00       1.00e+00           no
```

### 3. Fisher's Exact Test Results
Pairwise comparisons showing only significant results (p < threshold):

```
--- PAIRWISE FISHER'S EXACT TESTS (p < 0.01 only) ---
Feature         Val1     Val2     Acc1     Acc2      p-value
----------------------------------------------------------------------
Distance-mov     1.0      3.0   100.0%     0.0%     1.23e-15 ***
```

### 4. Summary
Concludes which factors significantly affect model accuracy.

## Statistical Methods

### Chi-Squared Test
- Tests null hypothesis: accuracy is independent of feature value
- Rejection (p < threshold) means the feature significantly affects accuracy
- Requires at least 2 groups with non-zero samples

### Fisher's Exact Test
- Used for 2x2 contingency tables comparing two specific feature values
- Exact test (not asymptotic approximation)
- More reliable for small sample sizes

## Error Handling

Per project requirements:
- No silent exceptions
- All errors reported with full stack trace
- Input validation with explicit error messages
- Single outermost exception handler

## Example Output

```
CONCLUSION
======================================================================

  DISTANCE is the ONLY statistically significant factor.
  Other features (angle, stripe, bg) show NO significant effect.
```

## Dependencies

- Python 3.7+
- scipy (for statistical tests)
- numpy
