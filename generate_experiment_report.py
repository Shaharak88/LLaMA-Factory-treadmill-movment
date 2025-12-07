#!/usr/bin/env python3
"""
Generate Experiment Report with Model Performance Metrics

This script:
1. Checks if dataset videos are already downloaded
2. If not, downloads and processes videos using dual_dataset_review.py logic
3. Extracts model performance metrics (F1, accuracy, hyperparameters) from CSV
4. Generates enhanced HTML report with model performance section

Usage:
    python3 generate_experiment_report.py --experiment-id <id> --csv-path <path>

Or:
    python3 generate_experiment_report.py --dataset-name <name> --csv-path <path>
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Import functions from dual_dataset_review.py
sys.path.insert(0, str(Path(__file__).parent))
from dual_dataset_review import (
    extract_metadata,
    download_videos,
    convert_videos_to_h264,
    load_metadata,
    analyze_distribution,
    group_videos,
    generate_pdf_plot,
    generate_categorical_comparison
)
from experiment_tracker import ExperimentTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentReportGenerator:
    """Generate experiment report with model performance metrics."""

    # Default hyperparameter values (for filtering non-default values)
    DEFAULT_PARAMS = {
        'num_train_epochs': 5,
        'lora_rank': 8,
        'lora_alpha': 16,
        'per_device_train_batch_size': 1,
        'gradient_accumulation_steps': 8,
        'learning_rate': 5e-5,
        'use_dora': False
    }

    def __init__(self, csv_path: str):
        """Initialize with CSV path."""
        self.csv_path = Path(csv_path)
        self.project_root = Path(__file__).parent
        self.local_data_dir = self.project_root / "data"
        self.reports_dir = self.project_root / "analytics" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Server configuration (same as dual_dataset_review.py)
        self.server = "seedoo@hetzner-gpu.tail9e6e7.ts.net"
        self.remote_base = "/app/data"
        self.docker_container = "llamafactory"

    def get_experiment_from_csv(self, experiment_id: Optional[int] = None,
                                dataset_name: Optional[str] = None) -> Optional[Dict]:
        """
        Get experiment row from CSV by experiment_id or dataset_name.

        Args:
            experiment_id: Experiment ID to find
            dataset_name: Dataset name to find

        Returns:
            Dict with experiment data or None if not found
        """
        if not self.csv_path.exists():
            logger.error(f"CSV file not found: {self.csv_path}")
            return None

        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if experiment_id and row.get('experiment_id') == str(experiment_id):
                    return row
                if dataset_name and row.get('dataset_name') == dataset_name:
                    return row

        return None

    def check_videos_downloaded(self, dataset_name: str) -> bool:
        """
        Check if videos for dataset are already downloaded locally.

        Args:
            dataset_name: Name of dataset (e.g., "_exp_20251202_172211_train")

        Returns:
            True if videos exist locally, False otherwise
        """
        video_dir = self.local_data_dir / dataset_name
        if not video_dir.exists():
            return False

        # Check if there are any mp4 files
        mp4_files = list(video_dir.glob('*.mp4'))
        return len(mp4_files) > 0

    def ensure_videos_downloaded(self, dataset_name: str) -> Path:
        """
        Ensure videos for dataset are downloaded. Download if not present.

        Args:
            dataset_name: Name of dataset

        Returns:
            Path to video directory
        """
        if self.check_videos_downloaded(dataset_name):
            logger.info(f"✓ Videos already downloaded for {dataset_name}")
            return self.local_data_dir / dataset_name

        logger.info(f"Downloading videos for {dataset_name}...")

        # Download videos
        video_dir = download_videos(
            dataset_name=dataset_name,
            server=self.server,
            remote_base=self.remote_base,
            local_dir=str(self.local_data_dir),
            docker_container=self.docker_container
        )

        # Convert to H264
        convert_videos_to_h264(video_dir)

        return Path(video_dir)

    def get_metadata(self, dataset_name: str) -> str:
        """
        Get or extract metadata CSV for dataset.

        Args:
            dataset_name: Name of dataset

        Returns:
            Path to metadata CSV file
        """
        # Look for existing metadata CSV
        metadata_files = list(self.local_data_dir.glob(f"{dataset_name}_metadata_*.csv"))

        if metadata_files:
            # Use the most recent one
            latest = max(metadata_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"✓ Using existing metadata: {latest}")
            return str(latest)

        # Extract metadata from server
        logger.info(f"Extracting metadata for {dataset_name}...")
        metadata_path = extract_metadata(
            dataset_name=dataset_name,
            server=self.server,
            remote_base=self.remote_base,
            output_dir=str(self.local_data_dir),
            docker_container=self.docker_container
        )

        return metadata_path

    def extract_non_default_params(self, experiment: Dict) -> Dict[str, any]:
        """
        Extract only non-default hyperparameter values from experiment.

        Args:
            experiment: Experiment row from CSV

        Returns:
            Dict of parameter_name -> value for non-default params
        """
        non_defaults = {}

        for param, default_value in self.DEFAULT_PARAMS.items():
            exp_value = experiment.get(param, '')

            # Convert to appropriate type
            if param == 'use_dora':
                exp_value = exp_value.lower() == 'true' if exp_value else False
            elif param == 'learning_rate':
                try:
                    exp_value = float(exp_value) if exp_value else default_value
                except:
                    exp_value = default_value
            else:
                try:
                    exp_value = int(exp_value) if exp_value else default_value
                except:
                    exp_value = default_value

            # Only include if different from default
            if exp_value != default_value:
                non_defaults[param] = exp_value

        return non_defaults

    def format_param_name(self, param: str) -> str:
        """Format parameter name for display."""
        # Convert snake_case to Title Case
        words = param.replace('_', ' ').split()
        return ' '.join(word.capitalize() for word in words)

    def generate_model_performance_html(self, experiment: Dict) -> str:
        """
        Generate HTML for model performance section.

        Args:
            experiment: Experiment row from CSV

        Returns:
            HTML string for model performance section
        """
        # Extract metrics
        base_acc = experiment.get('base_model_accuracy', 'N/A')
        base_f1 = experiment.get('base_model_f1_score', 'N/A')
        ft_acc = experiment.get('finetuned_model_accuracy', 'N/A')
        ft_f1 = experiment.get('finetuned_model_f1_score', 'N/A')

        # Extract non-default params
        non_default_params = self.extract_non_default_params(experiment)

        # Build params HTML
        params_html = ""
        if non_default_params:
            params_items = "".join([
                f'<div class="param-item"><span class="param-label">{self.format_param_name(k)}:</span> <span class="param-value">{v}</span></div>'
                for k, v in non_default_params.items()
            ])
            params_html = f'''
            <div class="hyperparams-section">
                <h4>Non-Default Hyperparameters</h4>
                <div class="params-grid">
                    {params_items}
                </div>
            </div>
            '''

        html = f'''
        <div class="model-performance-section">
            <h3>🎯 Model Performance Metrics</h3>

            <div class="metrics-grid">
                <div class="metric-card base-model">
                    <h4>Base Model</h4>
                    <div class="metric-row">
                        <span class="metric-label">Accuracy:</span>
                        <span class="metric-value">{base_acc}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">F1 Score:</span>
                        <span class="metric-value">{base_f1}%</span>
                    </div>
                </div>

                <div class="metric-card finetuned-model">
                    <h4>Fine-Tuned Model</h4>
                    <div class="metric-row">
                        <span class="metric-label">Accuracy:</span>
                        <span class="metric-value">{ft_acc}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">F1 Score:</span>
                        <span class="metric-value">{ft_f1}%</span>
                    </div>
                </div>
            </div>

            {params_html}
        </div>

        <style>
            .model-performance-section {{
                background: white;
                padding: 30px;
                margin-bottom: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .model-performance-section h3 {{
                margin-bottom: 20px;
                color: #667eea;
                font-size: 1.8em;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 20px;
                margin-bottom: 25px;
            }}
            .metric-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }}
            .metric-card h4 {{
                margin: 0 0 15px 0;
                font-size: 1.3em;
                opacity: 0.95;
            }}
            .metric-row {{
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                font-size: 1.1em;
            }}
            .metric-label {{
                font-weight: 500;
                opacity: 0.9;
            }}
            .metric-value {{
                font-weight: bold;
                font-size: 1.2em;
            }}
            .hyperparams-section {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }}
            .hyperparams-section h4 {{
                margin: 0 0 15px 0;
                color: #667eea;
                font-size: 1.2em;
            }}
            .params-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 12px;
            }}
            .param-item {{
                background: white;
                padding: 10px 15px;
                border-radius: 6px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .param-label {{
                font-weight: 600;
                color: #495057;
                font-size: 0.9em;
            }}
            .param-value {{
                color: #667eea;
                font-weight: bold;
                font-size: 1em;
            }}
        </style>
        '''

        return html

    def generate_dual_report(self, experiment: Dict) -> str:
        """
        Generate dual dataset comparison report with model performance.

        Args:
            experiment: Experiment row from CSV

        Returns:
            Path to generated HTML report
        """
        train_dataset = experiment.get('train_dataset_name', '')
        test_dataset = experiment.get('test_dataset_name', '')
        dataset_name = experiment.get('dataset_name', '')

        logger.info(f"Generating report for experiment: {dataset_name}")
        logger.info(f"  Train: {train_dataset}")
        logger.info(f"  Test: {test_dataset}")

        # Ensure videos are downloaded for both datasets
        logger.info("\n=== TRAIN DATASET ===")
        train_video_dir = self.ensure_videos_downloaded(train_dataset)
        train_metadata_csv = self.get_metadata(train_dataset)

        logger.info("\n=== TEST DATASET ===")
        test_video_dir = self.ensure_videos_downloaded(test_dataset)
        test_metadata_csv = self.get_metadata(test_dataset)

        # Load and analyze metadata
        logger.info("\n=== GENERATING REPORT ===")
        train_metadata = load_metadata(train_metadata_csv)
        test_metadata = load_metadata(test_metadata_csv)

        train_stats = analyze_distribution(train_metadata)
        test_stats = analyze_distribution(test_metadata)

        train_groups = group_videos(train_metadata)
        test_groups = group_videos(test_metadata)

        # Generate comparison plots
        comparison_plots = self._generate_comparison_plots(train_metadata, test_metadata)

        # Generate model performance HTML
        performance_html = self.generate_model_performance_html(experiment)

        # Generate full HTML report (incorporating dual_dataset_review logic)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"{dataset_name}_dual_report_{timestamp}.html"
        report_path = self.reports_dir / report_filename

        # Create HTML report with model performance section at the top
        html_content = self._build_html_report(
            performance_html=performance_html,
            train_metadata=train_metadata,
            test_metadata=test_metadata,
            train_stats=train_stats,
            test_stats=test_stats,
            train_groups=train_groups,
            test_groups=test_groups,
            comparison_plots=comparison_plots,
            train_video_dir=train_video_dir,
            test_video_dir=test_video_dir,
            dataset_name=dataset_name
        )

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"\n✅ Report generated: {report_path}")

        # Return relative path from project root
        return str(report_path.relative_to(self.project_root))

    def _generate_comparison_plots(self, train_metadata: List[Dict],
                                   test_metadata: List[Dict]) -> Dict[str, str]:
        """Generate comparison plots for continuous and categorical variables."""
        plots = {}

        # Speed comparison (continuous)
        train_speeds = [float(v['speed']) for v in train_metadata if v.get('speed')]
        test_speeds = [float(v['speed']) for v in test_metadata if v.get('speed')]

        if train_speeds and test_speeds:
            plots['speed'] = generate_pdf_plot(train_speeds, test_speeds, 'Speed', 'Speed (units/s)')

        # Angle comparison (continuous)
        train_angles = [float(v['angle']) for v in train_metadata if v.get('angle')]
        test_angles = [float(v['angle']) for v in test_metadata if v.get('angle')]

        if train_angles and test_angles:
            plots['angle'] = generate_pdf_plot(train_angles, test_angles, 'View Angle', 'Angle (degrees)')

        # Direction comparison (categorical)
        from collections import Counter
        train_directions = Counter(v['direction'] for v in train_metadata if v.get('direction'))
        test_directions = Counter(v['direction'] for v in test_metadata if v.get('direction'))

        if train_directions and test_directions:
            plots['direction'] = generate_categorical_comparison(train_directions, test_directions, 'Direction')

        # Texture comparison (categorical)
        train_textures = Counter(v['texture'] for v in train_metadata if v.get('texture'))
        test_textures = Counter(v['texture'] for v in test_metadata if v.get('texture'))

        if train_textures and test_textures:
            plots['texture'] = generate_categorical_comparison(train_textures, test_textures, 'Texture')

        return plots

    def _build_html_report(self, performance_html: str, train_metadata: List[Dict],
                          test_metadata: List[Dict], train_stats: Dict, test_stats: Dict,
                          train_groups: Dict, test_groups: Dict, comparison_plots: Dict,
                          train_video_dir: Path, test_video_dir: Path, dataset_name: str) -> str:
        """Build complete HTML report."""

        # Build comparison plots HTML
        plots_html = ""
        for plot_name, plot_data in comparison_plots.items():
            plots_html += f'''
            <div class="comparison-plot">
                <h4>{plot_name.capitalize()} Distribution Comparison</h4>
                <img src="{plot_data}" alt="{plot_name} comparison" style="max-width: 100%; height: auto;">
            </div>
            '''

        # Build the full HTML (simplified version - full implementation would include all tabs from dual_dataset_review)
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Report - {dataset_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .content {{
            padding: 30px;
        }}
        .comparison-plot {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .comparison-plot h4 {{
            margin-bottom: 15px;
            color: #667eea;
            font-size: 1.3em;
        }}
        .dataset-section {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .dataset-section h3 {{
            color: #667eea;
            margin-bottom: 15px;
        }}
        .stats-item {{
            padding: 8px 0;
            border-bottom: 1px solid #dee2e6;
        }}
        .stats-item:last-child {{
            border-bottom: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Experiment Report</h1>
            <p>Dataset: {dataset_name}</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        <div class="content">
            {performance_html}

            <div class="dataset-section">
                <h3>📈 Dataset Comparison Plots</h3>
                {plots_html}
            </div>

            <div class="dataset-section">
                <h3>📋 Training Dataset Overview</h3>
                <div class="stats-item">Total Videos: {train_stats['total_videos']}</div>
                <div class="stats-item">Video Directory: {train_video_dir}</div>
            </div>

            <div class="dataset-section">
                <h3>📋 Test Dataset Overview</h3>
                <div class="stats-item">Total Videos: {test_stats['total_videos']}</div>
                <div class="stats-item">Video Directory: {test_video_dir}</div>
            </div>

            <div class="dataset-section">
                <p><em>Note: Full dataset visualization with video playback available in dual_dataset_review.py</em></p>
            </div>
        </div>
    </div>
</body>
</html>'''

        return html


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate experiment report with model performance metrics'
    )

    parser.add_argument('--experiment-id', type=int,
                       help='Experiment ID from CSV')
    parser.add_argument('--dataset-name', type=str,
                       help='Dataset name (alternative to experiment-id)')
    parser.add_argument('--csv-path', type=str, default='data/experiments_log.csv',
                       help='Path to experiments CSV (default: data/experiments_log.csv)')

    args = parser.parse_args()

    if not args.experiment_id and not args.dataset_name:
        parser.error("Must provide either --experiment-id or --dataset-name")

    # Create generator
    generator = ExperimentReportGenerator(csv_path=args.csv_path)

    # Get experiment from CSV
    experiment = generator.get_experiment_from_csv(
        experiment_id=args.experiment_id,
        dataset_name=args.dataset_name
    )

    if not experiment:
        logger.error(f"Experiment not found in CSV")
        sys.exit(1)

    # Generate report
    report_path = generator.generate_dual_report(experiment)

    # Update CSV with report link
    tracker = ExperimentTracker(csv_path=args.csv_path)
    tracker.current_experiment_id = experiment.get('experiment_id')
    tracker.update_dataset_comparison_report(report_path)

    print(f"\n✅ Report generated: {report_path}")
    print(f"\nTo view: Open {report_path} in your browser")

    return report_path


if __name__ == '__main__':
    main()
