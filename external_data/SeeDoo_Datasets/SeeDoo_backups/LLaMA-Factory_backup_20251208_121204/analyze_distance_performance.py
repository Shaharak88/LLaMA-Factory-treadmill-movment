#!/usr/bin/env python3
"""
Analyze per-distance performance from evaluation CSV files.
"""
import pandas as pd
import re
from pathlib import Path

def extract_distance(video_path):
    """Extract distance value from video filename."""
    match = re.search(r'dist(\d+\.\d+)', video_path)
    if match:
        return float(match.group(1))
    return None

def analyze_per_distance(csv_path, model_name):
    """Analyze accuracy per distance from evaluation CSV."""
    df = pd.read_csv(csv_path)

    # Extract distance from video paths
    df['distance'] = df['video_path'].apply(extract_distance)

    # Group by distance and calculate metrics
    results = []
    for distance in sorted(df['distance'].unique()):
        dist_df = df[df['distance'] == distance]
        total = len(dist_df)
        correct = dist_df['correct'].sum()
        accuracy = (correct / total * 100) if total > 0 else 0
        error_rate = 100 - accuracy

        results.append({
            'Distance': distance,
            'Total': total,
            'Correct': correct,
            'Accuracy': f"{accuracy:.2f}%",
            'Error Rate': f"{error_rate:.2f}%"
        })

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))

    # Overall metrics
    overall_correct = df['correct'].sum()
    overall_total = len(df)
    overall_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0
    print(f"\n{'='*60}")
    print(f"Overall: {overall_correct}/{overall_total} = {overall_accuracy:.2f}%")
    print(f"{'='*60}\n")

    return result_df

if __name__ == '__main__':
    eval_dir = Path('/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/data/eval_20251207_153447')

    print("\n" + "="*60)
    print("Distance Generalization Analysis")
    print("Model: dist_gen_1to10_left_gray_dora_dec7")
    print("Training Distance: 1.0")
    print("Test Distances: 1.0, 3.0, 5.0, 8.0, 10.0")
    print("="*60)

    # Analyze base model
    base_csv = eval_dir / 'per_video_predictions_base_20251207_153447.csv'
    if base_csv.exists():
        base_results = analyze_per_distance(base_csv, "Base Model (Qwen2-VL-2B)")

    # Analyze fine-tuned model
    finetuned_csv = eval_dir / 'per_video_predictions_finetuned_20251207_153447.csv'
    if finetuned_csv.exists():
        finetuned_results = analyze_per_distance(finetuned_csv, "Fine-tuned Model (dist_gen_1to10_left_gray_dora_dec7)")
