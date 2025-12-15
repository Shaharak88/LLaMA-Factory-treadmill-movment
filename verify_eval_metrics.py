#!/usr/bin/env python3
"""
Comprehensive verification of evaluation metrics.
Checks:
1. Prediction parsing from raw model output
2. Correctness flag (prediction == label)
3. Accuracy and F1 calculations
"""
import pandas as pd
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def parse_model_output_moving(text):
    """Parse model output to determine if it says 'moving' or 'stopped'"""
    text_lower = text.lower()

    # Check for explicit statements
    if 'belt is moving' in text_lower or 'treadmill is moving' in text_lower:
        return 'moving'
    if 'belt is stopped' in text_lower or 'treadmill is stopped' in text_lower:
        return 'stopped'
    if 'belt is not moving' in text_lower or 'treadmill is not moving' in text_lower:
        return 'stopped'

    # Additional patterns
    if 'is moving' in text_lower:
        return 'moving'
    if 'is stopped' in text_lower or 'not moving' in text_lower:
        return 'stopped'

    return 'unknown'

def verify_csv(csv_path, model_name):
    """Verify all aspects of a predictions CSV"""
    print(f"\n{'='*80}")
    print(f"VERIFYING: {model_name}")
    print(f"{'='*80}\n")

    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")

    # Step 1: Verify prediction parsing
    print("\n--- Step 1: Verifying Prediction Parsing ---")
    df['parsed_prediction'] = df['model_output'].apply(parse_model_output_moving)

    mismatches = df[df['prediction'] != df['parsed_prediction']]
    if len(mismatches) > 0:
        print(f"⚠️  WARNING: {len(mismatches)} mismatches between 'prediction' column and parsed output!")
        for idx, row in mismatches.head(10).iterrows():
            print(f"\n  Row {idx}:")
            print(f"    Video: {row['video_path']}")
            print(f"    Stored prediction: {row['prediction']}")
            print(f"    Parsed from output: {row['parsed_prediction']}")
            print(f"    Raw output: {row['model_output']}")
    else:
        print("✓ All predictions correctly parsed from model output")

    # Step 2: Verify 'correct' column
    print("\n--- Step 2: Verifying 'Correct' Column ---")
    df['computed_correct'] = df['prediction'] == df['label']

    correct_mismatches = df[df['correct'] != df['computed_correct']]
    if len(correct_mismatches) > 0:
        print(f"⚠️  WARNING: {len(correct_mismatches)} mismatches in 'correct' column!")
        for idx, row in correct_mismatches.head(10).iterrows():
            print(f"\n  Row {idx}:")
            print(f"    Prediction: {row['prediction']}, Label: {row['label']}")
            print(f"    Stored correct: {row['correct']}")
            print(f"    Should be: {row['computed_correct']}")
    else:
        print("✓ All 'correct' flags are accurate")

    # Step 3: Calculate metrics manually
    print("\n--- Step 3: Calculating Metrics ---")

    # Use the stored correct column (should match our computation)
    total = len(df)
    correct_count = df['correct'].sum()
    accuracy = correct_count / total

    # Per-class counts
    moving_df = df[df['label'] == 'moving']
    stopped_df = df[df['label'] == 'stopped']

    moving_total = len(moving_df)
    moving_correct = moving_df['correct'].sum()
    moving_incorrect = moving_total - moving_correct

    stopped_total = len(stopped_df)
    stopped_correct = stopped_df['correct'].sum()
    stopped_incorrect = stopped_total - stopped_correct

    print(f"\nOverall:")
    print(f"  Total samples: {total}")
    print(f"  Correct: {correct_count}")
    print(f"  Incorrect: {total - correct_count}")
    print(f"  Accuracy: {accuracy*100:.2f}% ({correct_count}/{total})")

    print(f"\nMoving class:")
    print(f"  Total: {moving_total}")
    print(f"  Correct: {moving_correct}")
    print(f"  Incorrect: {moving_incorrect}")
    print(f"  Class Accuracy: {(moving_correct/moving_total)*100:.2f}%")

    print(f"\nStopped class:")
    print(f"  Total: {stopped_total}")
    print(f"  Correct: {stopped_correct}")
    print(f"  Incorrect: {stopped_incorrect}")
    print(f"  Class Accuracy: {(stopped_correct/stopped_total)*100:.2f}%")

    # Calculate confusion matrix components
    df['pred_binary'] = (df['prediction'] == 'moving').astype(int)
    df['label_binary'] = (df['label'] == 'moving').astype(int)

    # True Positive: predicted moving, actually moving
    tp = len(df[(df['prediction'] == 'moving') & (df['label'] == 'moving')])
    # False Negative: predicted stopped, actually moving
    fn = len(df[(df['prediction'] == 'stopped') & (df['label'] == 'moving')])
    # False Positive: predicted moving, actually stopped
    fp = len(df[(df['prediction'] == 'moving') & (df['label'] == 'stopped')])
    # True Negative: predicted stopped, actually stopped
    tn = len(df[(df['prediction'] == 'stopped') & (df['label'] == 'stopped')])

    print(f"\nConfusion Matrix:")
    print(f"  TP (Moving→Moving): {tp}")
    print(f"  FN (Stopped→Moving): {fn}")
    print(f"  FP (Moving→Stopped): {fp}")
    print(f"  TN (Stopped→Stopped): {tn}")
    print(f"  Sanity check: {tp + fn + fp + tn} == {total} ? {tp + fn + fp + tn == total}")

    # Calculate F1, Precision, Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nMetrics (for 'moving' class as positive):")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall: {recall*100:.2f}%")
    print(f"  F1 Score: {f1*100:.2f}%")

    # Using sklearn for verification
    f1_sklearn = f1_score(df['label_binary'], df['pred_binary'])
    precision_sklearn = precision_score(df['label_binary'], df['pred_binary'])
    recall_sklearn = recall_score(df['label_binary'], df['pred_binary'])

    print(f"\nSklearn verification:")
    print(f"  Precision: {precision_sklearn*100:.2f}%")
    print(f"  Recall: {recall_sklearn*100:.2f}%")
    print(f"  F1 Score: {f1_sklearn*100:.2f}%")

    # Show some incorrect predictions
    print("\n--- Incorrect Predictions ---")
    incorrect = df[~df['correct']]
    print(f"Total incorrect: {len(incorrect)}")

    if len(incorrect) > 0:
        print("\nSample incorrect predictions:")
        for idx, row in incorrect.head(15).iterrows():
            print(f"\n  Video: {row['video_path'].split('/')[-1]}")
            print(f"    Predicted: {row['prediction']}, Actual: {row['label']}, Speed: {row['speed']}")
            print(f"    Model output: {row['model_output']}")

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'correct': correct_count,
        'total': total,
        'moving_correct': moving_correct,
        'moving_total': moving_total,
        'stopped_correct': stopped_correct,
        'stopped_total': stopped_total,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

def main():
    # Verify base model
    base_metrics = verify_csv('/app/data/_exp_20251204_171014_test/eval_20251204_161151/per_video_predictions_base_20251204_161151.csv', 'BASE MODEL')

    # Verify finetuned model
    finetuned_metrics = verify_csv('/app/data/_exp_20251204_171014_test/eval_20251204_161151/per_video_predictions_finetuned_20251204_161151.csv', 'FINE-TUNED MODEL')

    # Compare with reported values
    print(f"\n{'='*80}")
    print("COMPARISON WITH REPORTED VALUES")
    print(f"{'='*80}\n")

    print("BASE MODEL:")
    print(f"  Reported Accuracy: 83.59% (321/384)")
    print(f"  Calculated: {base_metrics['accuracy']*100:.2f}% ({base_metrics['correct']}/{base_metrics['total']})")
    acc_match = abs(base_metrics['correct'] - 321) == 0
    print(f"  Match: {'✓ YES' if acc_match else '✗ NO - DISCREPANCY!'}")

    print(f"\n  Reported F1: 83.11%")
    print(f"  Calculated: {base_metrics['f1']*100:.2f}%")
    f1_match = abs(base_metrics['f1']*100 - 83.11) < 1.0
    print(f"  Match: {'✓ YES' if f1_match else '✗ NO - DISCREPANCY!'}")

    print(f"\n  Reported Moving: 193/256")
    print(f"  Calculated: {base_metrics['moving_correct']}/{base_metrics['moving_total']}")
    moving_match = base_metrics['moving_correct'] == 193 and base_metrics['moving_total'] == 256
    print(f"  Match: {'✓ YES' if moving_match else '✗ NO - DISCREPANCY!'}")

    print(f"\n  Reported Stopped: 128/128")
    print(f"  Calculated: {base_metrics['stopped_correct']}/{base_metrics['stopped_total']}")
    stopped_match = base_metrics['stopped_correct'] == 128 and base_metrics['stopped_total'] == 128
    print(f"  Match: {'✓ YES' if stopped_match else '✗ NO - DISCREPANCY!'}")

    print("\n" + "="*80)
    print("FINE-TUNED MODEL:")
    print(f"  Reported Accuracy: 99.22% (381/384)")
    print(f"  Calculated: {finetuned_metrics['accuracy']*100:.2f}% ({finetuned_metrics['correct']}/{finetuned_metrics['total']})")
    ft_acc_match = abs(finetuned_metrics['correct'] - 381) == 0
    print(f"  Match: {'✓ YES' if ft_acc_match else '✗ NO - DISCREPANCY!'}")

    print(f"\n  Reported F1: 99.12%")
    print(f"  Calculated: {finetuned_metrics['f1']*100:.2f}%")
    ft_f1_match = abs(finetuned_metrics['f1']*100 - 99.12) < 1.0
    print(f"  Match: {'✓ YES' if ft_f1_match else '✗ NO - DISCREPANCY!'}")

    print(f"\n  Reported Moving: 256/256")
    print(f"  Calculated: {finetuned_metrics['moving_correct']}/{finetuned_metrics['moving_total']}")
    ft_moving_match = finetuned_metrics['moving_correct'] == 256 and finetuned_metrics['moving_total'] == 256
    print(f"  Match: {'✓ YES' if ft_moving_match else '✗ NO - DISCREPANCY!'}")

    print(f"\n  Reported Stopped: 125/128")
    print(f"  Calculated: {finetuned_metrics['stopped_correct']}/{finetuned_metrics['stopped_total']}")
    ft_stopped_match = finetuned_metrics['stopped_correct'] == 125 and finetuned_metrics['stopped_total'] == 128
    print(f"  Match: {'✓ YES' if ft_stopped_match else '✗ NO - DISCREPANCY!'}")

    # Summary
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)

    all_match = (acc_match and f1_match and moving_match and stopped_match and
                 ft_acc_match and ft_f1_match and ft_moving_match and ft_stopped_match)

    if all_match:
        print("✓ ALL METRICS VERIFIED CORRECTLY!")
        print("  The reported accuracy and F1 scores match the raw predictions.")
    else:
        print("✗ DISCREPANCIES FOUND!")
        print("  Some metrics do not match. Review the details above.")

if __name__ == "__main__":
    main()
