#!/usr/bin/env python3
"""
Analyze failure patterns in treadmill motion detection evaluation results.
Examines ALL features from video filenames with statistical significance testing.
"""
import csv
import re
import sys
import traceback
from collections import defaultdict
from itertools import combinations
from scipy.stats import chi2_contingency, fisher_exact
import numpy as np


def extract_all_features(path):
    """Extract all features from video filename."""
    features = {}

    # Distance
    match = re.search(r'dist(\d+\.\d+)', path)
    features['distance'] = float(match.group(1)) if match else None

    # Angle
    match = re.search(r'angle(\d+)', path)
    features['angle'] = int(match.group(1)) if match else None

    # Stripe gray value
    match = re.search(r'stripe(\d+)', path)
    features['stripe'] = int(match.group(1)) if match else None

    # Background gray value
    match = re.search(r'bg(\d+)', path)
    features['bg'] = int(match.group(1)) if match else None

    # Speed (from filename, not column)
    match = re.search(r'speed(\d+\.\d+)', path)
    features['speed'] = float(match.group(1)) if match else None

    return features


def compute_chi2_test(data, label_filter=None):
    """
    Compute chi-squared test for independence between feature values and accuracy.
    Returns (chi2, p_value) or (None, None) if test cannot be performed.
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

    if len(filtered_data) < 2:
        return None, None, "Less than 2 groups"

    # Create contingency table
    table = []
    for feat_val in sorted(filtered_data.keys()):
        table.append([filtered_data[feat_val]['correct'], filtered_data[feat_val]['incorrect']])

    table = np.array(table)

    # Validate table before running chi2
    if np.any(table.sum(axis=0) == 0):
        return None, None, "Zero column sum (all correct or all incorrect)"
    if np.any(table.sum(axis=1) == 0):
        return None, None, "Zero row sum (empty group)"

    # Run chi-squared test - let it raise if there's an error
    chi2, p_value, dof, expected = chi2_contingency(table)
    return chi2, p_value, None


def compute_pairwise_fisher(data, label_filter=None):
    """
    Compute pairwise Fisher's exact tests between all pairs of feature values.
    Returns list of (val1, val2, p_value, acc1, acc2, n1, n2) tuples.
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

    for val1, val2 in combinations(feat_vals, 2):
        d1 = filtered_data[val1]
        d2 = filtered_data[val2]

        # Skip if either group is empty
        if d1['total'] == 0 or d2['total'] == 0:
            print(f"WARNING: Skipping Fisher test for {val1} vs {val2} - empty group")
            continue

        # 2x2 contingency table
        table = [
            [d1['correct'], d1['total'] - d1['correct']],
            [d2['correct'], d2['total'] - d2['correct']]
        ]

        # Run Fisher's exact test - let it raise if there's an error
        odds_ratio, p_value = fisher_exact(table)
        acc1 = (d1['correct'] / d1['total']) * 100
        acc2 = (d2['correct'] / d2['total']) * 100
        results.append((val1, val2, p_value, acc1, acc2, d1['total'], d2['total']))

    return results


def print_feature_analysis(data, feature_name, label_filter=None):
    """Print analysis for a specific feature."""
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


def main():
    csv_path = "per_video_predictions_finetuned_20251207_153447.csv"

    # Data structures for ALL features
    by_distance = defaultdict(lambda: {'total': 0, 'correct': 0})
    by_angle = defaultdict(lambda: {'total': 0, 'correct': 0})
    by_stripe = defaultdict(lambda: {'total': 0, 'correct': 0})
    by_bg = defaultdict(lambda: {'total': 0, 'correct': 0})
    by_label = defaultdict(lambda: {'total': 0, 'correct': 0})

    # Combined features (for far distances only - where failures occur)
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

            # Validate extracted features
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

            # By individual feature + label
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

            # For far distances only (where failures occur)
            if dist > 1.0:
                by_angle_far[(angle, label)]['total'] += 1
                by_angle_far[(angle, label)]['correct'] += int(correct)

                by_stripe_far[(stripe, label)]['total'] += 1
                by_stripe_far[(stripe, label)]['correct'] += int(correct)

                by_bg_far[(bg, label)]['total'] += 1
                by_bg_far[(bg, label)]['correct'] += int(correct)

            # Cross-feature (for moving videos at far distances)
            if dist > 1.0 and label == 'moving':
                by_stripe_bg[(stripe, bg)]['total'] += 1
                by_stripe_bg[(stripe, bg)]['correct'] += int(correct)

                by_angle_dist[(angle, dist)]['total'] += 1
                by_angle_dist[(angle, dist)]['correct'] += int(correct)

    print(f"Processed {row_count} rows from CSV")

    # Print comprehensive results
    print("=" * 70)
    print("COMPREHENSIVE SUBGROUP FAILURE ANALYSIS")
    print("Model: dist_gen_1to10_left_gray_dora_dec7")
    print("Training: Distance 1.0 only")
    print("=" * 70)

    # 1. By Distance
    print_feature_analysis(by_distance, "Distance")

    # 2. By Angle (all)
    print_feature_analysis(by_angle, "Angle")

    # 3. By Stripe gray value (all)
    print_feature_analysis(by_stripe, "Stripe")

    # 4. By Background gray value (all)
    print_feature_analysis(by_bg, "BG")

    # Statistical Significance Testing
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE ANALYSIS (p < 0.01)")
    print("=" * 70)
    print("\nTests performed:")
    print("  - Chi-squared: Tests if accuracy differs across ALL values of a feature")
    print("  - Fisher's exact: Pairwise comparison between specific feature values")
    print("-" * 70)

    P_THRESHOLD = 0.01
    significant_results = []

    # Test each feature
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

    # Pairwise Fisher tests
    print("\n--- PAIRWISE FISHER'S EXACT TESTS (p < 0.01 only) ---")
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
        print("  No significant pairwise differences found (p < 0.01)")

    # Final Summary
    print("\n" + "=" * 70)
    print("SUMMARY: STATISTICALLY SIGNIFICANT DIFFERENCES (p < 0.01)")
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

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Check if distance is the only significant factor
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
    # Outermost exception handler - log full stack trace
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
