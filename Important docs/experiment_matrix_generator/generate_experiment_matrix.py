#!/usr/bin/env python3
"""
Experiment Matrix Generator

Generates an HTML heatmap comparing F1 scores across samplers and datasets.
Color gradient: Red (worst) -> Yellow (middle) -> Green (best)

Usage:
    python generate_experiment_matrix.py
"""

import pandas as pd


def get_sampler(row):
    """Determine sampler type from experiment data."""
    exp_id = row['experiment_id']
    model_name = str(row['model_name']) if pd.notna(row['model_name']) else ''

    # Manual corrections for specific experiments (due to naming inconsistencies)
    corrections = {
        80: 'No shuffle',
        67: 'HF shuffle',
        77: 'Our random shuffle (iter fix)',
        81: 'Our random shuffle'
    }

    if exp_id in corrections:
        return corrections[exp_id]

    # Pattern matching on model name
    model_lower = model_name.lower()
    if 'balanced' in model_lower:
        return 'Balanced'
    elif 'iter_count' in model_lower:
        return 'Our random shuffle (iter fix)'
    elif 'random_sampler' in model_lower:
        return 'Our random shuffle'
    elif 'sequential' in model_lower:
        return 'No shuffle'
    elif 'hf_shuffle' in model_lower:
        return 'HF shuffle'
    elif 'noshuffle' in model_lower:
        return 'No shuffle'
    elif 'batch_sampler' in model_lower:
        return 'Batch sampler'
    else:
        return 'HF shuffle'


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
    xlsx_path = '/mnt/c/Users/shaha/Desktop/SeeDoo_Datasets/SeeDoo_backups/all.xlsx'
    output_path = '/mnt/c/Users/shaha/Desktop/SeeDoo_Datasets/SeeDoo_backups/experiment_matrix.html'

    # Load data
    df = pd.read_excel(xlsx_path)

    # Add sampler classification
    df['sampler'] = df.apply(get_sampler, axis=1)

    # Filter to datasets with more than 1 run
    dataset_counts = df['dataset_name'].value_counts()
    multi_run_datasets = sorted(dataset_counts[dataset_counts > 1].index)
    df_filtered = df[df['dataset_name'].isin(multi_run_datasets)]

    # Define samplers and dataset names
    samplers = [
        'HF shuffle',
        'Our random shuffle',
        'Our random shuffle (iter fix)',
        'Balanced',
        'No shuffle'
    ]

    dataset_names = {
        '_exp_20251208_230500': '64 distances',
        '_exp_20251209_dist_augment': '74=64 +10 distances',
        '_exp_20251209_dist_augment_x5_combined': '124=64 +60 distances',
        'real_world': 'Real World'
    }

    all_datasets = list(multi_run_datasets) + ['real_world']

    # Build results dictionary
    results = {}
    for sampler in samplers:
        results[sampler] = {}
        for dataset in multi_run_datasets:
            subset = df_filtered[
                (df_filtered['sampler'] == sampler) &
                (df_filtered['dataset_name'] == dataset)
            ]
            if len(subset) > 0:
                results[sampler][dataset] = subset['finetuned_model_f1_score'].max()
            else:
                results[sampler][dataset] = None

    # Add real world data (hardcoded from experiments_log.csv analysis)
    # Models trained on _exp_20251208_230500, tested on real_world_test_my_treadmill_processed
    results['HF shuffle']['real_world'] = 51.41          # 5epoch_dec9_loraplus
    results['Our random shuffle']['real_world'] = 53.58  # loraplus_hf_shuffle_5epochs_dec10
    results['Our random shuffle (iter fix)']['real_world'] = None  # missing
    results['Balanced']['real_world'] = 45.97            # loraplus_balanced_exp230500_dec10
    results['No shuffle']['real_world'] = None           # missing

    # Calculate min/max per dataset for gradient
    min_per_dataset = {}
    max_per_dataset = {}
    for dataset in all_datasets:
        scores = [results[s][dataset] for s in samplers if results[s].get(dataset) is not None]
        min_per_dataset[dataset] = min(scores) if scores else 0
        max_per_dataset[dataset] = max(scores) if scores else 100

    # Generate HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Sampler Comparison Matrix</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
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
        }
        th.sampler-header {
            text-align: left;
            min-width: 220px;
        }
        td {
            padding: 14px 16px;
            border-bottom: 1px solid #e2e8f0;
            text-align: center;
            font-weight: 700;
            font-size: 1.2em;
        }
        td.sampler-name {
            text-align: left;
            background-color: #f7fafc;
            font-weight: 600;
        }
        td.score {
            color: #000;
            text-shadow: 1px 1px 0px rgba(255,255,255,0.5);
        }
        tr:hover td.sampler-name {
            background-color: #edf2f7;
        }
        .missing {
            color: #a0aec0 !important;
            background-color: #f7fafc !important;
            font-style: italic;
            font-weight: normal;
            text-shadow: none !important;
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
    </style>
</head>
<body>
    <h1>Sampler Comparison - F1 Scores by Dataset</h1>
    <div class="legend">
        Worst <span class="gradient-bar"></span> Best
    </div>
    <table>
        <tr>
            <th class="sampler-header">Sampler</th>
"""

    # Add dataset headers
    for dataset in all_datasets:
        display_name = dataset_names.get(dataset, dataset)
        html += f'            <th>{display_name}</th>\n'

    html += '        </tr>\n'

    # Add rows for each sampler
    for sampler in samplers:
        html += f'        <tr>\n            <td class="sampler-name">{sampler}</td>\n'

        for dataset in all_datasets:
            f1 = results[sampler].get(dataset)

            if f1 is not None:
                color = get_gradient_color(f1, min_per_dataset[dataset], max_per_dataset[dataset])
                html += f'            <td class="score" style="background-color: {color};">{f1:.2f}</td>\n'
            else:
                html += '            <td class="missing">missing</td>\n'

        html += '        </tr>\n'

    html += """    </table>
</body>
</html>
"""

    # Save output
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"HTML saved to: {output_path}")


if __name__ == '__main__':
    main()
