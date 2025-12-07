#!/usr/bin/env python3
"""
Evaluation Failure Analysis Tool
================================

Analyzes per-video prediction results from treadmill motion detection experiments
to identify which features (distance, angle, stripe color, background color)
contribute to model failures.

This tool performs:
1. Accuracy breakdown by each feature (distance, angle, stripe, bg)
2. Chi-squared tests to determine if accuracy differs across feature values
3. Pairwise Fisher's exact tests for specific feature value comparisons
4. Statistical significance filtering (p < 0.01 threshold)

Usage:
    python analyze_evaluation_failures.py <csv_file>

Input CSV format:
    Expected columns: video_path, prediction, label, speed, correct, model_output

    Video filenames must contain feature values in this format:
    - Distance: dist1.0, dist3.0, etc.
    - Angle: angle0, angle30, etc.
    - Stripe gray: stripe226, stripe227, etc.
    - Background gray: bg120, bg121, etc.
    - Speed: speed0.0, speed14.0, etc.

Example:
    python analyze_evaluation_failures.py per_video_predictions_finetuned_20251207.csv

Output:
    - Accuracy tables by each feature
    - Chi-squared test results for overall feature significance
    - Fisher's exact test results for pairwise comparisons (p < 0.01 only)
    - Summary of statistically significant factors

Author: Generated for treadmill motion detection project
Date: December 2024
"""

import argparse
import csv
import re
import sys
import traceback
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from scipy.stats import chi2_contingency, fisher_exact
import numpy as np


# =============================================================================
# Feature Extraction
# =============================================================================

# Features to skip in analysis (not experimental parameters)
SKIP_FEATURES = {'seed', 'treadmill', 'x', 'mp'}


def extract_all_features(path: str) -> dict:
    """
    Dynamically extract ALL experimental features from a video filename.

    Automatically detects patterns like: name123, name12.34
    Uses the exact feature names from the filename (e.g., 'dist' not 'distance').

    Args:
        path: Video file path containing encoded feature values

    Returns:
        Dictionary with feature names as keys (exactly as in filename)
        and their numeric values.

    Example:
        >>> extract_all_features("treadmill_stripe226_bg120_left_speed14.0_angle30_dist5.0.mp4")
        {'stripe': 226, 'bg': 120, 'speed': 14.0, 'angle': 30, 'dist': 5.0}

        >>> extract_all_features("video_blur0.5_object1_contr1.2.mp4")
        {'blur': 0.5, 'object': 1, 'contr': 1.2}
    """
    features = {}

    # Pattern: word characters followed by number (int or float)
    # Matches: dist5.0, angle30, stripe226, blur0.5, object1, contr1.00
    # The (?<![a-zA-Z]) ensures we don't match partial words
    pattern = r'(?<![a-zA-Z])([a-zA-Z]+)(\d+\.?\d*)'

    for match in re.finditer(pattern, path):
        key = match.group(1).lower()
        value_str = match.group(2)

        # Skip non-experimental features
        if key in SKIP_FEATURES:
            continue

        # Skip if value string is empty
        if not value_str:
            continue

        # Convert to float or int
        if '.' in value_str:
            value = float(value_str)
        else:
            value = int(value_str)

        features[key] = value

    return features


# =============================================================================
# Statistical Tests
# =============================================================================

def compute_chi2_test(data: dict, label_filter: str = None) -> tuple:
    """
    Compute chi-squared test for independence between feature values and accuracy.

    Tests null hypothesis: accuracy is independent of feature value.
    Rejection (p < threshold) indicates the feature significantly affects accuracy.

    Args:
        data: Dictionary mapping (feature_value, label) -> {total, correct}
        label_filter: If provided, only include rows with this label ("moving"/"stopped")

    Returns:
        Tuple of (chi2_statistic, p_value, skip_reason)
        If test cannot be performed, returns (None, None, reason_string)
    """
    # Build contingency table: rows = feature values, cols = [correct, incorrect]
    filtered_data = {}
    for key, val in data.items():
        feat_val, label = key
        if label_filter and label != label_filter:
            continue
        if feat_val not in filtered_data:
            filtered_data[feat_val] = {'correct': 0, 'incorrect': 0}
        filtered_data[feat_val]['correct'] += val['correct']
        filtered_data[feat_val]['incorrect'] += val['total'] - val['correct']

    # Need at least 2 groups to compare
    if len(filtered_data) < 2:
        return None, None, "Less than 2 groups"

    # Create contingency table as numpy array
    table = []
    for feat_val in sorted(filtered_data.keys()):
        table.append([filtered_data[feat_val]['correct'], filtered_data[feat_val]['incorrect']])
    table = np.array(table)

    # Validate table - chi2 requires non-zero marginals
    if np.any(table.sum(axis=0) == 0):
        return None, None, "Zero column sum (all correct or all incorrect)"
    if np.any(table.sum(axis=1) == 0):
        return None, None, "Zero row sum (empty group)"

    # Run chi-squared test
    chi2, p_value, dof, expected = chi2_contingency(table)
    return chi2, p_value, None


def compute_pairwise_fisher(data: dict, label_filter: str = None) -> list:
    """
    Compute pairwise Fisher's exact tests between all pairs of feature values.

    Fisher's exact test is used for 2x2 contingency tables and is exact
    (not based on asymptotic approximation like chi-squared).

    Args:
        data: Dictionary mapping (feature_value, label) -> {total, correct}
        label_filter: If provided, only include rows with this label

    Returns:
        List of tuples: (val1, val2, p_value, acc1, acc2, n1, n2)
        Each tuple represents a pairwise comparison between two feature values
    """
    # Group data by feature value
    filtered_data = {}
    for key, val in data.items():
        feat_val, label = key
        if label_filter and label != label_filter:
            continue
        if feat_val not in filtered_data:
            filtered_data[feat_val] = {'correct': 0, 'total': 0}
        filtered_data[feat_val]['correct'] += val['correct']
        filtered_data[feat_val]['total'] += val['total']

    results = []
    feat_vals = sorted(filtered_data.keys())

    # Test all pairs of feature values
    for val1, val2 in combinations(feat_vals, 2):
        d1 = filtered_data[val1]
        d2 = filtered_data[val2]

        # Skip empty groups with explicit warning
        if d1['total'] == 0 or d2['total'] == 0:
            print(f"WARNING: Skipping Fisher test for {val1} vs {val2} - empty group")
            continue

        # Build 2x2 contingency table
        table = [
            [d1['correct'], d1['total'] - d1['correct']],
            [d2['correct'], d2['total'] - d2['correct']]
        ]

        # Run Fisher's exact test
        odds_ratio, p_value = fisher_exact(table)
        acc1 = (d1['correct'] / d1['total']) * 100
        acc2 = (d2['correct'] / d2['total']) * 100
        results.append((val1, val2, p_value, acc1, acc2, d1['total'], d2['total']))

    return results


# =============================================================================
# Output Formatting
# =============================================================================

def print_feature_analysis(data: dict, feature_name: str, label_filter: str = None):
    """
    Print accuracy breakdown table for a specific feature.

    Args:
        data: Dictionary mapping (feature_value, label) -> {total, correct}
        feature_name: Name of the feature for display
        label_filter: If provided, only show rows with this label
    """
    print(f"\n{'='*70}")
    title = f"ACCURACY BY {feature_name.upper()}"
    if label_filter:
        title += f" (only {label_filter} videos)"
    print(title)
    print("="*70)
    print(f"{feature_name:<12} {'Label':<10} {'Correct':<10} {'Total':<10} {'Accuracy':<12} {'Error':<12}")
    print("-"*70)

    for key in sorted(data.keys()):
        val, label = key
        if label_filter and label != label_filter:
            continue
        d = data[key]
        if d['total'] == 0:
            print(f"WARNING: {feature_name}={val}, label={label} has 0 total samples!")
            continue
        acc = (d['correct'] / d['total']) * 100
        err = 100 - acc
        print(f"{val:<12} {label:<10} {d['correct']:<10} {d['total']:<10} {acc:>6.2f}%      {err:>6.2f}%")


# =============================================================================
# Module API for Integration
# =============================================================================

def analyze_predictions(csv_path: str, p_threshold: float = 0.01) -> dict:
    """
    Analyze per-video predictions and return structured results.

    This function dynamically extracts ALL features from video filenames
    and performs statistical analysis on each. Feature names are taken
    exactly as they appear in filenames (e.g., 'dist' not 'distance').

    Args:
        csv_path: Path to per-video predictions CSV file
        p_threshold: P-value threshold for significance (default: 0.01)

    Returns:
        Dictionary with analysis results including accuracy breakdowns,
        statistical tests, significant factors, and failed video examples
        grouped by significant features.

    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Dynamic data structures - will be populated based on features found
    by_feature = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))
    all_features_found = set()

    # Store all rows and failed videos
    failed_videos = []
    all_rows = []

    # Read and parse CSV
    row_count = 0
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            features = extract_all_features(row['video_path'])
            label = row['label']
            correct = row['correct'] == 'True'
            prediction = row.get('prediction', '')

            # Track all features found
            all_features_found.update(features.keys())

            # Store row data with all extracted features
            row_data = {
                'video_path': row['video_path'],
                'label': label,
                'prediction': prediction,
                'correct': correct,
                'features': features,
                'model_output': row.get('model_output', '')
            }
            all_rows.append(row_data)

            # Store failed videos
            if not correct:
                failed_videos.append(row_data)

            # Aggregate by each feature + label dynamically
            for feat_name, feat_value in features.items():
                by_feature[feat_name][(feat_value, label)]['total'] += 1
                by_feature[feat_name][(feat_value, label)]['correct'] += int(correct)

    # Build accuracy by feature data (dynamic)
    def build_accuracy_list(data_dict):
        result = []
        for key in sorted(data_dict.keys()):
            val, label = key
            d = data_dict[key]
            if d['total'] > 0:
                acc = (d['correct'] / d['total']) * 100
                result.append({
                    'value': val,
                    'label': label,
                    'correct': d['correct'],
                    'total': d['total'],
                    'accuracy': round(acc, 2)
                })
        return result

    accuracy_by_feature = {}
    for feat_name in sorted(all_features_found):
        accuracy_by_feature[feat_name] = build_accuracy_list(by_feature[feat_name])

    # Run statistical tests dynamically for each feature
    chi2_tests = []
    significant_chi2_features = []  # Track which features are significant

    for feat_name in sorted(all_features_found):
        data = by_feature[feat_name]

        # Test overall (all labels)
        chi2, p_value, skip_reason = compute_chi2_test(data, None)
        if chi2 is not None:
            is_sig = p_value < p_threshold
            chi2_tests.append({
                'feature': feat_name,
                'chi2': round(chi2, 2),
                'p_value': p_value,
                'significant': is_sig
            })
            if is_sig:
                significant_chi2_features.append(feat_name)
        else:
            chi2_tests.append({
                'feature': feat_name,
                'chi2': None,
                'p_value': None,
                'significant': False,
                'skip_reason': skip_reason
            })

        # Test for moving only
        chi2, p_value, skip_reason = compute_chi2_test(data, "moving")
        if chi2 is not None:
            is_sig = p_value < p_threshold
            chi2_tests.append({
                'feature': f"{feat_name} (moving)",
                'chi2': round(chi2, 2),
                'p_value': p_value,
                'significant': is_sig
            })
            if is_sig and feat_name not in significant_chi2_features:
                significant_chi2_features.append(feat_name)

    # Run pairwise Fisher tests dynamically
    fisher_tests = []
    for feat_name in sorted(all_features_found):
        data = by_feature[feat_name]

        # Test overall
        pairwise_results = compute_pairwise_fisher(data, None)
        for val1, val2, p_value, acc1, acc2, n1, n2 in pairwise_results:
            is_sig = p_value < p_threshold
            fisher_tests.append({
                'feature': feat_name,
                'val1': val1,
                'val2': val2,
                'acc1': round(acc1, 1),
                'acc2': round(acc2, 1),
                'n1': n1,
                'n2': n2,
                'p_value': p_value,
                'significant': is_sig
            })

        # Test for moving only
        pairwise_results = compute_pairwise_fisher(data, "moving")
        for val1, val2, p_value, acc1, acc2, n1, n2 in pairwise_results:
            is_sig = p_value < p_threshold
            fisher_tests.append({
                'feature': f"{feat_name} (moving)",
                'val1': val1,
                'val2': val2,
                'acc1': round(acc1, 1),
                'acc2': round(acc2, 1),
                'n1': n1,
                'n2': n2,
                'p_value': p_value,
                'significant': is_sig
            })

    # Build significant factors list from chi2 tests
    significant_factors = []
    for test in chi2_tests:
        if test['significant']:
            significant_factors.append(
                f"{test['feature']}: Chi2={test['chi2']:.2f}, p={test['p_value']:.2e}"
            )

    # Build dynamic conclusion based on significant features
    sig_base_features = [f for f in significant_chi2_features]  # Features without "(moving)" suffix

    if len(sig_base_features) == 0:
        conclusion = "No statistically significant differences detected in any feature."
    elif len(sig_base_features) == 1:
        conclusion = f"{sig_base_features[0].upper()} is the ONLY statistically significant factor affecting model accuracy. Other features show NO significant effect."
    else:
        features_str = ", ".join(sig_base_features[:-1]) + " and " + sig_base_features[-1]
        conclusion = f"Multiple significant factors found: {features_str.upper()} all affect model accuracy."

    # Group failed videos by significant features for examples
    failed_by_feature = {}
    for feat_name in significant_chi2_features:
        failed_by_feature[feat_name] = defaultdict(list)
        for video in failed_videos:
            feat_value = video['features'].get(feat_name)
            if feat_value is not None:
                failed_by_feature[feat_name][feat_value].append(video)

    # For each significant feature, get worst performing values (lowest accuracy)
    worst_values_by_feature = {}
    for feat_name in significant_chi2_features:
        # Get accuracy for each value of this feature
        value_accuracies = []
        for item in accuracy_by_feature.get(feat_name, []):
            # Combine moving and stopped for overall accuracy per value
            pass
        # Sort by accuracy ascending to get worst values
        feat_acc = accuracy_by_feature.get(feat_name, [])
        # Group by value and compute overall accuracy
        by_value = defaultdict(lambda: {'correct': 0, 'total': 0})
        for item in feat_acc:
            by_value[item['value']]['correct'] += item['correct']
            by_value[item['value']]['total'] += item['total']

        value_accs = []
        for val, counts in by_value.items():
            if counts['total'] > 0:
                acc = (counts['correct'] / counts['total']) * 100
                value_accs.append((val, acc, counts['total']))

        # Sort by accuracy (worst first)
        value_accs.sort(key=lambda x: x[1])
        worst_values_by_feature[feat_name] = value_accs[:3]  # Top 3 worst values

    # Build examples for each significant feature
    significant_feature_examples = {}
    for feat_name in significant_chi2_features:
        examples = []
        worst_values = worst_values_by_feature.get(feat_name, [])
        for val, acc, total in worst_values:
            # Get up to 4 failed videos for this value
            failed_for_value = failed_by_feature[feat_name].get(val, [])[:4]
            if failed_for_value:
                examples.append({
                    'value': val,
                    'accuracy': round(acc, 1),
                    'total': total,
                    'videos': failed_for_value
                })
        significant_feature_examples[feat_name] = examples

    # Sort all failed videos by first significant feature value (descending)
    if significant_chi2_features:
        sort_feat = significant_chi2_features[0]
        failed_videos.sort(
            key=lambda x: (x['features'].get(sort_feat, 0) if x['features'].get(sort_feat) else 0),
            reverse=True
        )

    return {
        'row_count': row_count,
        'features_found': sorted(all_features_found),
        'accuracy_by_feature': accuracy_by_feature,
        'chi2_tests': chi2_tests,
        'fisher_tests': fisher_tests,
        'significant_factors': significant_factors,
        'significant_features': significant_chi2_features,
        'significant_feature_examples': significant_feature_examples,
        'conclusion': conclusion,
        'failed_videos': failed_videos[:50],
        'p_threshold': p_threshold
    }


# =============================================================================
# Main Analysis (CLI)
# =============================================================================

def main():
    """Main entry point for failure analysis."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Analyze evaluation failures by feature subgroups with statistical significance testing'
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        default='per_video_predictions_finetuned_20251207_153447.csv',
        help='Path to per-video predictions CSV file'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.01,
        help='P-value threshold for significance (default: 0.01)'
    )
    args = parser.parse_args()

    csv_path = args.csv_file
    P_THRESHOLD = args.threshold

    # Verify input file exists
    if not Path(csv_path).exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    # Data structures for aggregation by feature
    by_distance = defaultdict(lambda: {'total': 0, 'correct': 0})
    by_angle = defaultdict(lambda: {'total': 0, 'correct': 0})
    by_stripe = defaultdict(lambda: {'total': 0, 'correct': 0})
    by_bg = defaultdict(lambda: {'total': 0, 'correct': 0})
    by_label = defaultdict(lambda: {'total': 0, 'correct': 0})

    # For far distances only (where failures typically occur)
    by_angle_far = defaultdict(lambda: {'total': 0, 'correct': 0})
    by_stripe_far = defaultdict(lambda: {'total': 0, 'correct': 0})
    by_bg_far = defaultdict(lambda: {'total': 0, 'correct': 0})

    # Cross-feature analysis
    by_stripe_bg = defaultdict(lambda: {'total': 0, 'correct': 0})
    by_angle_dist = defaultdict(lambda: {'total': 0, 'correct': 0})

    # Read and parse CSV
    row_count = 0
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            features = extract_all_features(row['video_path'])
            label = row['label']
            correct = row['correct'] == 'True'

            dist = features['distance']
            angle = features['angle']
            stripe = features['stripe']
            bg = features['bg']

            # Validate all required features were extracted
            if dist is None:
                print(f"ERROR: Could not extract distance from row {row_count}: {row['video_path']}")
                sys.exit(1)
            if angle is None:
                print(f"ERROR: Could not extract angle from row {row_count}: {row['video_path']}")
                sys.exit(1)
            if stripe is None:
                print(f"ERROR: Could not extract stripe from row {row_count}: {row['video_path']}")
                sys.exit(1)
            if bg is None:
                print(f"ERROR: Could not extract bg from row {row_count}: {row['video_path']}")
                sys.exit(1)

            # Aggregate by individual feature + label
            by_distance[(dist, label)]['total'] += 1
            by_distance[(dist, label)]['correct'] += int(correct)

            by_angle[(angle, label)]['total'] += 1
            by_angle[(angle, label)]['correct'] += int(correct)

            by_stripe[(stripe, label)]['total'] += 1
            by_stripe[(stripe, label)]['correct'] += int(correct)

            by_bg[(bg, label)]['total'] += 1
            by_bg[(bg, label)]['correct'] += int(correct)

            by_label[label]['total'] += 1
            by_label[label]['correct'] += int(correct)

            # Aggregate for far distances only (dist > 1.0)
            if dist > 1.0:
                by_angle_far[(angle, label)]['total'] += 1
                by_angle_far[(angle, label)]['correct'] += int(correct)

                by_stripe_far[(stripe, label)]['total'] += 1
                by_stripe_far[(stripe, label)]['correct'] += int(correct)

                by_bg_far[(bg, label)]['total'] += 1
                by_bg_far[(bg, label)]['correct'] += int(correct)

            # Cross-feature analysis for moving videos at far distances
            if dist > 1.0 and label == 'moving':
                by_stripe_bg[(stripe, bg)]['total'] += 1
                by_stripe_bg[(stripe, bg)]['correct'] += int(correct)

                by_angle_dist[(angle, dist)]['total'] += 1
                by_angle_dist[(angle, dist)]['correct'] += int(correct)

    print(f"Processed {row_count} rows from {csv_path}")

    # Print header
    print("=" * 70)
    print("COMPREHENSIVE SUBGROUP FAILURE ANALYSIS")
    print(f"Input: {csv_path}")
    print(f"Significance threshold: p < {P_THRESHOLD}")
    print("=" * 70)

    # Print accuracy breakdown by each feature
    print_feature_analysis(by_distance, "Distance")
    print_feature_analysis(by_angle, "Angle")
    print_feature_analysis(by_stripe, "Stripe")
    print_feature_analysis(by_bg, "BG")

    # Statistical Significance Testing
    print("\n" + "=" * 70)
    print(f"STATISTICAL SIGNIFICANCE ANALYSIS (p < {P_THRESHOLD})")
    print("=" * 70)
    print("\nTests performed:")
    print("  - Chi-squared: Tests if accuracy differs across ALL values of a feature")
    print("  - Fisher's exact: Pairwise comparison between specific feature values")
    print("-" * 70)

    significant_results = []

    # Define features to test
    feature_tests = [
        ("Distance", by_distance, None),
        ("Distance (moving only)", by_distance, "moving"),
        ("Distance (stopped only)", by_distance, "stopped"),
        ("Angle", by_angle, None),
        ("Angle (moving only)", by_angle, "moving"),
        ("Stripe", by_stripe, None),
        ("Stripe (moving only)", by_stripe, "moving"),
        ("BG", by_bg, None),
        ("BG (moving only)", by_bg, "moving"),
    ]

    # Run chi-squared tests
    print("\n--- CHI-SQUARED TESTS (Overall feature significance) ---")
    print(f"{'Feature':<30} {'Chi2':>10} {'p-value':>15} {'Significant':>12}")
    print("-" * 70)

    for name, data, label_filter in feature_tests:
        chi2, p_value, skip_reason = compute_chi2_test(data, label_filter)
        if chi2 is not None:
            sig = "YES ***" if p_value < P_THRESHOLD else "no"
            print(f"{name:<30} {chi2:>10.2f} {p_value:>15.2e} {sig:>12}")
            if p_value < P_THRESHOLD:
                significant_results.append(('chi2', name, chi2, p_value))
        else:
            print(f"{name:<30} {'N/A':>10} {'N/A':>15} {'(' + skip_reason + ')':>20}")

    # Run pairwise Fisher tests
    print(f"\n--- PAIRWISE FISHER'S EXACT TESTS (p < {P_THRESHOLD} only) ---")
    print(f"{'Feature':<15} {'Val1':>8} {'Val2':>8} {'Acc1':>8} {'Acc2':>8} {'p-value':>12}")
    print("-" * 70)

    pairwise_tests = [
        ("Distance", by_distance, None),
        ("Distance-mov", by_distance, "moving"),
        ("Angle", by_angle, None),
        ("Angle-mov", by_angle, "moving"),
        ("Stripe", by_stripe, None),
        ("Stripe-mov", by_stripe, "moving"),
        ("BG", by_bg, None),
        ("BG-mov", by_bg, "moving"),
    ]

    found_significant = False
    for name, data, label_filter in pairwise_tests:
        pairwise_results = compute_pairwise_fisher(data, label_filter)
        for val1, val2, p_value, acc1, acc2, n1, n2 in pairwise_results:
            if p_value < P_THRESHOLD:
                found_significant = True
                print(f"{name:<15} {val1:>8} {val2:>8} {acc1:>7.1f}% {acc2:>7.1f}% {p_value:>12.2e} ***")
                significant_results.append(('fisher', name, val1, val2, p_value, acc1, acc2))

    if not found_significant:
        print(f"  No significant pairwise differences found (p < {P_THRESHOLD})")

    # Final Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: STATISTICALLY SIGNIFICANT DIFFERENCES (p < {P_THRESHOLD})")
    print("=" * 70)

    if significant_results:
        print("\nSignificant factors affecting model accuracy:")
        for result in significant_results:
            if result[0] == 'chi2':
                print(f"  - {result[1]}: Chi2={result[2]:.2f}, p={result[3]:.2e}")
            else:
                print(f"  - {result[1]}: {result[2]} vs {result[3]} ({result[5]:.1f}% vs {result[6]:.1f}%), p={result[4]:.2e}")
    else:
        print("\nNo statistically significant differences found.")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    distance_sig = any(r[1].startswith('Distance') for r in significant_results if r[0] == 'chi2')
    other_sig = any(not r[1].startswith('Distance') for r in significant_results)

    if distance_sig and not other_sig:
        print("\n  DISTANCE is the ONLY statistically significant factor.")
        print("  Other features (angle, stripe, bg) show NO significant effect.")
    elif significant_results:
        print("\n  Multiple significant factors found - see above.")
    else:
        print("\n  No significant differences detected in any feature.")


if __name__ == '__main__':
    # Outermost exception handler - always log full stack trace on error
    try:
        main()
    except Exception as e:
        print("\n" + "=" * 70)
        print("FATAL ERROR")
        print("=" * 70)
        print(f"Exception: {type(e).__name__}: {e}")
        print("\nFull stack trace:")
        traceback.print_exc()
        sys.exit(1)
