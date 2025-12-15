#!/usr/bin/env python3
"""
Parameter Comparison Matrix Generator

Generates an HTML heatmap comparing F1 scores across different training parameter configurations.
Color gradient: Red (worst) -> Yellow (middle) -> Green (best)

Usage:
    python generate_parameter_matrix.py
"""

import pandas as pd


def get_gradient_color(value, min_val, max_val):
    """Calculate gradient color: Red (worst) -> Yellow (middle) -> Green (best)."""
    if max_val == min_val:
        return "rgb(34, 197, 94)"  # Green if all values are the same

    # Normalize to 0-1 (1 = best, 0 = worst)
    normalized = (value - min_val) / (max_val - min_val)

    if normalized >= 0.5:
        # Yellow to Green (best half)
        t = (normalized - 0.5) * 2
        r = int(250 + (34 - 250) * t)
        g = int(204 + (197 - 204) * t)
        b = int(21 + (94 - 21) * t)
    else:
        # Red to Yellow (worst half)
        t = normalized * 2
        r = int(220 + (250 - 220) * t)
        g = int(38 + (204 - 38) * t)
        b = int(38 + (21 - 38) * t)

    return f"rgb({r}, {g}, {b})"


def main():
    # Configuration
    csv_path = '/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/data/experiments_log.csv'
    output_path = '/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/parameter_comparison_matrix.html'

    # Load data
    df = pd.read_csv(csv_path)

    # Filter out rows with missing experiment_id (empty rows)
    df = df[df['experiment_id'].notna()]

    # Get last 4 experiments
    last_4 = df.tail(4).copy()

    # Extract key info for each experiment
    experiments = []
    for _, row in last_4.iterrows():
        exp_info = {
            'experiment_id': row['experiment_id'],
            'model_name': row['model_name'],
            'train_resolution': row['train_resolution'],
            'train_fps': row['train_fps'],
            'batch_size': row['per_device_train_batch_size'],
            'grad_acc': row['gradient_accumulation_steps'],
            'effective_batch': row['per_device_train_batch_size'] * row['gradient_accumulation_steps'],
            'f1_score': row['finetuned_model_f1_score'],
            'accuracy': row['finetuned_model_accuracy'],
            'f1_moving': row['finetuned_model_f1_moving'],
            'f1_stopped': row['finetuned_model_f1_stopped'],
            'dataset': row['dataset_name']
        }
        experiments.append(exp_info)

    # Calculate min/max for gradient
    f1_scores = [exp['f1_score'] for exp in experiments]
    min_f1 = min(f1_scores)
    max_f1 = max(f1_scores)

    # Generate HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Training Parameter Comparison Matrix</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        table {
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
            margin: 0 auto;
        }
        th {
            background-color: #4a5568;
            color: white;
            padding: 14px 16px;
            text-align: center;
            font-weight: 600;
            font-size: 0.95em;
        }
        th.config-header {
            text-align: left;
            min-width: 180px;
        }
        td {
            padding: 12px 16px;
            border-bottom: 1px solid #e2e8f0;
            text-align: center;
            font-size: 1em;
        }
        td.config-name {
            text-align: left;
            background-color: #f7fafc;
            font-weight: 600;
            font-size: 0.9em;
        }
        td.score {
            color: #000;
            text-shadow: 1px 1px 0px rgba(255,255,255,0.5);
            font-weight: 700;
            font-size: 1.2em;
        }
        td.param {
            font-weight: 500;
            color: #2d3748;
        }
        tr:hover td.config-name {
            background-color: #edf2f7;
        }
        .legend {
            text-align: center;
            margin: 20px 0;
            color: #666;
        }
        .gradient-bar {
            display: inline-block;
            width: 200px;
            height: 20px;
            background: linear-gradient(to right, rgb(220, 38, 38), rgb(250, 204, 21), rgb(34, 197, 94));
            border-radius: 4px;
            vertical-align: middle;
            margin: 0 10px;
        }
        .info-box {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 20px auto;
            max-width: 800px;
        }
        .info-box h3 {
            margin-top: 0;
            color: #4a5568;
        }
        .info-box ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        .info-box li {
            margin: 5px 0;
            color: #2d3748;
        }
        .param-name {
            font-size: 0.85em;
            color: #718096;
            display: block;
        }
    </style>
</head>
<body>
    <h1>Training Parameter Comparison Matrix</h1>
    <div class="subtitle">Dataset: _exp_20251209_dist_augment_x5_combined (124 distances)</div>

    <div class="legend">
        Worst <span class="gradient-bar"></span> Best
    </div>

    <div class="info-box">
        <h3>Parameter Variations</h3>
        <ul>
            <li><strong>Resolution:</strong> 180x180 vs 256x256 pixels</li>
            <li><strong>FPS:</strong> 2 vs 4 frames per second</li>
            <li><strong>Batch Size:</strong> 8, 14, or 16</li>
            <li><strong>Gradient Accumulation:</strong> 1 or 2 steps</li>
            <li><strong>Effective Batch Size:</strong> batch_size × grad_accumulation</li>
        </ul>
    </div>

    <table>
        <tr>
            <th class="config-header">Configuration</th>
            <th>Resolution</th>
            <th>FPS</th>
            <th>Batch Size</th>
            <th>Grad Acc</th>
            <th>Effective<br>Batch</th>
            <th>F1 Score</th>
            <th>Accuracy</th>
            <th>F1 Moving</th>
            <th>F1 Stopped</th>
        </tr>
"""

    # Add rows for each experiment
    for exp in experiments:
        color = get_gradient_color(exp['f1_score'], min_f1, max_f1)

        # Create a short config name
        config_name = f"Exp {exp['experiment_id']}"

        html += f"""        <tr>
            <td class="config-name">{config_name}<br><span class="param-name">{exp['model_name']}</span></td>
            <td class="param">{exp['train_resolution']}</td>
            <td class="param">{exp['train_fps']}</td>
            <td class="param">{exp['batch_size']}</td>
            <td class="param">{exp['grad_acc']}</td>
            <td class="param"><strong>{exp['effective_batch']}</strong></td>
            <td class="score" style="background-color: {color};">{exp['f1_score']:.2f}</td>
            <td class="param">{exp['accuracy']:.2f}</td>
            <td class="param">{exp['f1_moving']:.2f}</td>
            <td class="param">{exp['f1_stopped']:.2f}</td>
        </tr>
"""

    html += """    </table>

    <div class="info-box" style="margin-top: 30px;">
        <h3>Key Findings</h3>
        <ul>
"""

    # Find best configuration
    best_exp = max(experiments, key=lambda x: x['f1_score'])
    worst_exp = min(experiments, key=lambda x: x['f1_score'])

    html += f"""            <li><strong>Best Configuration:</strong> Exp {best_exp['experiment_id']} - {best_exp['train_resolution']}, {best_exp['train_fps']} FPS, Batch {best_exp['batch_size']}×{best_exp['grad_acc']} = {best_exp['effective_batch']} (F1: {best_exp['f1_score']:.2f})</li>
            <li><strong>Worst Configuration:</strong> Exp {worst_exp['experiment_id']} - {worst_exp['train_resolution']}, {worst_exp['train_fps']} FPS, Batch {worst_exp['batch_size']}×{worst_exp['grad_acc']} = {worst_exp['effective_batch']} (F1: {worst_exp['f1_score']:.2f})</li>
            <li><strong>F1 Score Range:</strong> {min_f1:.2f} - {max_f1:.2f} (Δ {max_f1 - min_f1:.2f})</li>
        </ul>
    </div>
</body>
</html>
"""

    # Save output
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"HTML saved to: {output_path}")
    print(f"\nExperiment Summary:")
    print(f"{'Exp ID':<8} {'Resolution':<12} {'FPS':<5} {'Batch':<7} {'GradAcc':<9} {'Eff.Batch':<10} {'F1 Score':<10}")
    print("-" * 80)
    for exp in experiments:
        print(f"{exp['experiment_id']:<8} {exp['train_resolution']:<12} {exp['train_fps']:<5} {exp['batch_size']:<7} {exp['grad_acc']:<9} {exp['effective_batch']:<10} {exp['f1_score']:<10.2f}")


if __name__ == '__main__':
    main()
