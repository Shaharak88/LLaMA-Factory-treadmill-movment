#!/usr/bin/env python3
"""
Generate Experiment Report with Full Dataset Analytics and Model Performance

This script:
1. Checks if dataset videos are already downloaded
2. If not, downloads and processes videos using dual_dataset_review.py logic
3. Extracts model performance metrics (F1, accuracy, hyperparameters) from CSV
4. Generates comprehensive HTML report with:
   - All train/test dataset analytics (overview, defaults, grouped views)
   - Distribution comparison plots
   - Model performance metrics tab
   - Interactive video playback

Usage:
    python3 generate_experiment_report.py --experiment-id <id> --csv-path <path>

Or:
    python3 generate_experiment_report.py --dataset-name <name> --csv-path <path>
"""

import argparse
import base64
import csv
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter
import logging

# Import functions from dual_dataset_review.py
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "analytics"))
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

# Import failure analysis function (optional - gracefully handle if scipy not available)
try:
    from analytics.analyze_evaluation_failures import analyze_predictions
    FAILURE_ANALYSIS_AVAILABLE = True
except ImportError:
    FAILURE_ANALYSIS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Failure analysis not available (scipy not installed)")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentReportGenerator:
    """Generate comprehensive experiment report with full dataset analytics and model performance."""

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

    def find_evaluation_results(self, dataset_name: str) -> Optional[Dict[str, Path]]:
        """
        Find evaluation result CSV files for a dataset.

        Looks for eval_* folders in the dataset directory and finds the most recent
        per_video_predictions CSV files for both base and finetuned models.

        Args:
            dataset_name: Name of dataset (e.g., "_exp_20251207_143033_test")

        Returns:
            Dict with 'base' and 'finetuned' paths to CSV files, or None if not found
        """
        dataset_dir = self.local_data_dir / dataset_name

        if not dataset_dir.exists():
            return None

        # Find all eval_* folders
        eval_folders = list(dataset_dir.glob('eval_*'))

        if not eval_folders:
            logger.warning(f"No evaluation folders found in {dataset_dir}")
            return None

        # Get the most recent eval folder
        latest_eval_folder = max(eval_folders, key=lambda p: p.stat().st_mtime)
        logger.info(f"Found evaluation folder: {latest_eval_folder}")

        # Look for prediction CSV files
        base_csvs = list(latest_eval_folder.glob('per_video_predictions_base_*.csv'))
        finetuned_csvs = list(latest_eval_folder.glob('per_video_predictions_finetuned_*.csv'))

        if not base_csvs or not finetuned_csvs:
            logger.warning(f"Prediction CSV files not found in {latest_eval_folder}")
            return None

        # Get the most recent of each
        base_csv = max(base_csvs, key=lambda p: p.stat().st_mtime)
        finetuned_csv = max(finetuned_csvs, key=lambda p: p.stat().st_mtime)

        logger.info(f"✓ Found base predictions: {base_csv.name}")
        logger.info(f"✓ Found finetuned predictions: {finetuned_csv.name}")

        return {
            'base': base_csv,
            'finetuned': finetuned_csv
        }

    def download_evaluation_results(self, dataset_name: str) -> Optional[Dict[str, Path]]:
        """
        Download evaluation results from server container.

        Uses docker cp via SSH to copy eval folders from container to local.

        Args:
            dataset_name: Name of dataset

        Returns:
            Dict with 'base' and 'finetuned' paths to CSV files, or None if download failed
        """
        logger.info(f"Downloading evaluation results for {dataset_name} from server container...")

        local_dataset_path = self.local_data_dir / dataset_name
        local_dataset_path.mkdir(parents=True, exist_ok=True)

        import subprocess
        import tempfile

        # Step 1: Find eval folders in container
        find_cmd = f"ssh {self.server} 'docker exec {self.docker_container} find /app/data/{dataset_name} -type d -name \"eval_*\" -maxdepth 1'"
        try:
            result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True, check=True)
            eval_folders = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]

            if not eval_folders:
                logger.warning(f"No eval folders found in container for {dataset_name}")
                return None

            logger.info(f"Found {len(eval_folders)} eval folder(s) in container")

            # Step 2: For each eval folder, docker cp it
            for container_path in eval_folders:
                eval_folder_name = os.path.basename(container_path)

                # Docker cp from container to temp location on server, then rsync to local
                # Simpler: use docker cp with output redirection
                local_eval_path = local_dataset_path / eval_folder_name

                # Create a temp dir on server, docker cp to it, then rsync
                cp_cmd = f"ssh {self.server} 'docker cp {self.docker_container}:{container_path} /tmp/{eval_folder_name} && tar czf - -C /tmp {eval_folder_name} && rm -rf /tmp/{eval_folder_name}' | tar xzf - -C {local_dataset_path}"

                subprocess.run(cp_cmd, shell=True, check=True, capture_output=True)
                logger.info(f"  ✓ Downloaded {eval_folder_name}")

            logger.info("✓ Download complete")
            return self.find_evaluation_results(dataset_name)

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download evaluation results: {e}")
            logger.error(f"stderr: {e.stderr}")
            return None

    def ensure_evaluation_results(self, dataset_name: str) -> Optional[Dict[str, Path]]:
        """
        Ensure evaluation results are available locally. Download if needed.

        Args:
            dataset_name: Name of dataset

        Returns:
            Dict with 'base' and 'finetuned' paths to CSV files, or None if not available
        """
        # First, check if results exist locally
        results = self.find_evaluation_results(dataset_name)

        if results:
            return results

        # If not, try to download from server
        logger.info(f"Evaluation results not found locally, attempting download...")
        return self.download_evaluation_results(dataset_name)

    def load_prediction_data(self, csv_paths: Dict[str, Path]) -> Dict[str, Dict]:
        """
        Load per-video prediction data from CSV files.

        Args:
            csv_paths: Dict with 'base' and 'finetuned' paths to CSV files

        Returns:
            Dict mapping video_filename -> {
                'label': ground truth,
                'base_raw': base model raw output,
                'base_eval': base model evaluated prediction,
                'finetuned_raw': finetuned model raw output,
                'finetuned_eval': finetuned model evaluated prediction
            }
        """
        prediction_data = {}

        # Load base model predictions
        logger.info(f"Loading base model predictions from {csv_paths['base']}")
        with open(csv_paths['base'], 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_path = row['video_path']
                # Extract just the filename from the path for matching
                video_filename = Path(video_path).name
                prediction_data[video_filename] = {
                    'label': row['label'],
                    'base_raw': row['model_output'],
                    'base_eval': row['prediction']
                }

        # Load finetuned model predictions
        logger.info(f"Loading finetuned model predictions from {csv_paths['finetuned']}")
        with open(csv_paths['finetuned'], 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_path = row['video_path']
                # Extract just the filename from the path for matching
                video_filename = Path(video_path).name
                if video_filename in prediction_data:
                    prediction_data[video_filename]['finetuned_raw'] = row['model_output']
                    prediction_data[video_filename]['finetuned_eval'] = row['prediction']
                else:
                    logger.warning(f"Video {video_filename} in finetuned CSV but not in base CSV")

        logger.info(f"✓ Loaded predictions for {len(prediction_data)} videos")
        return prediction_data

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

    def generate_failure_analysis_tab_html(self, analysis_results: Dict, test_video_dir: str) -> str:
        """
        Generate HTML for failure analysis tab content.

        Args:
            analysis_results: Dictionary from analyze_predictions() function
            test_video_dir: Relative path to test video directory for video playback

        Returns:
            HTML string for failure analysis tab content
        """
        if not analysis_results:
            return '<div class="alert alert-warning"><h3>⚠️ Failure Analysis Not Available</h3><p>Could not run failure analysis. Ensure evaluation results are available.</p></div>'

        # Build accuracy by distance table
        distance_rows = ""
        for item in analysis_results.get('accuracy_by_feature', {}).get('distance', []):
            acc_class = "high-acc" if item['accuracy'] >= 80 else ("mid-acc" if item['accuracy'] >= 50 else "low-acc")
            distance_rows += f"""
                <tr class="{acc_class}">
                    <td>{item['value']}</td>
                    <td>{item['label']}</td>
                    <td>{item['correct']}</td>
                    <td>{item['total']}</td>
                    <td><strong>{item['accuracy']:.1f}%</strong></td>
                </tr>
            """

        # Build chi-squared tests table
        chi2_rows = ""
        for test in analysis_results.get('chi2_tests', []):
            sig_class = "significant" if test.get('significant') else ""
            sig_marker = "✓ YES" if test.get('significant') else "no"
            chi2_val = f"{test['chi2']:.2f}" if test['chi2'] is not None else "N/A"
            p_val = f"{test['p_value']:.2e}" if test['p_value'] is not None else test.get('skip_reason', 'N/A')
            chi2_rows += f"""
                <tr class="{sig_class}">
                    <td>{test['feature']}</td>
                    <td>{chi2_val}</td>
                    <td>{p_val}</td>
                    <td><strong>{sig_marker}</strong></td>
                </tr>
            """

        # Build significant Fisher tests table (only significant ones)
        fisher_rows = ""
        sig_fisher_tests = [t for t in analysis_results.get('fisher_tests', []) if t.get('significant')]
        for test in sig_fisher_tests[:20]:  # Limit to 20 rows
            fisher_rows += f"""
                <tr class="significant">
                    <td>{test['feature']}</td>
                    <td>{test['val1']}</td>
                    <td>{test['val2']}</td>
                    <td>{test['acc1']:.1f}%</td>
                    <td>{test['acc2']:.1f}%</td>
                    <td>{test['p_value']:.2e}</td>
                </tr>
            """

        if not fisher_rows:
            fisher_rows = '<tr><td colspan="6" style="text-align: center; color: #666;">No significant pairwise differences found</td></tr>'

        # Build significant factors list
        sig_factors_html = ""
        for factor in analysis_results.get('significant_factors', []):
            sig_factors_html += f'<li class="sig-factor">{factor}</li>'

        if not sig_factors_html:
            sig_factors_html = '<li style="color: #666;">No statistically significant factors found</li>'

        # Build failed videos examples (up to 12)
        failed_videos = analysis_results.get('failed_videos', [])[:12]
        failed_videos_html = ""
        for video in failed_videos:
            video_filename = Path(video['video_path']).name
            video_url = f"{test_video_dir}/{video_filename}"
            failed_videos_html += f"""
                <div class="failed-video-card">
                    <video controls preload="metadata">
                        <source src="{video_url}" type="video/mp4">
                        Your browser does not support video.
                    </video>
                    <div class="failed-video-info">
                        <div class="failed-badge">FAILED</div>
                        <div class="video-detail"><span class="label">Label:</span> <span class="value truth">{video['label']}</span></div>
                        <div class="video-detail"><span class="label">Predicted:</span> <span class="value predicted">{video['prediction']}</span></div>
                        <div class="video-detail"><span class="label">Distance:</span> <span class="value">{video['distance']}</span></div>
                        <div class="video-detail"><span class="label">Angle:</span> <span class="value">{video['angle']}°</span></div>
                        <div class="video-detail"><span class="label">Model Output:</span> <span class="value model-output">{video['model_output'][:100]}...</span></div>
                    </div>
                </div>
            """

        conclusion = analysis_results.get('conclusion', 'No conclusion available.')
        p_threshold = analysis_results.get('p_threshold', 0.01)
        row_count = analysis_results.get('row_count', 0)

        html = f'''
        <h2>🔬 Statistical Failure Analysis</h2>
        <p style="color: #495057; margin-bottom: 20px;">
            Analyzed <strong>{row_count}</strong> predictions using statistical significance testing (p &lt; {p_threshold}).
        </p>

        <!-- Conclusion Box -->
        <div class="conclusion-box">
            <h3>📊 Key Finding</h3>
            <p>{conclusion}</p>
        </div>

        <!-- Significant Factors -->
        <div class="analysis-section">
            <h3>🎯 Statistically Significant Factors</h3>
            <ul class="sig-factors-list">
                {sig_factors_html}
            </ul>
        </div>

        <!-- Accuracy by Distance -->
        <div class="analysis-section">
            <h3>📏 Accuracy by Distance</h3>
            <p class="section-desc">Shows how model accuracy varies with camera distance from the treadmill.</p>
            <table class="analysis-table">
                <thead>
                    <tr>
                        <th>Distance</th>
                        <th>Label</th>
                        <th>Correct</th>
                        <th>Total</th>
                        <th>Accuracy</th>
                    </tr>
                </thead>
                <tbody>
                    {distance_rows}
                </tbody>
            </table>
        </div>

        <!-- Chi-Squared Tests -->
        <div class="analysis-section">
            <h3>📈 Chi-Squared Tests (Feature Significance)</h3>
            <p class="section-desc">Tests whether accuracy differs significantly across feature values. Significant results (p &lt; {p_threshold}) indicate the feature affects model accuracy.</p>
            <table class="analysis-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Chi²</th>
                        <th>p-value</th>
                        <th>Significant</th>
                    </tr>
                </thead>
                <tbody>
                    {chi2_rows}
                </tbody>
            </table>
        </div>

        <!-- Pairwise Fisher Tests -->
        <div class="analysis-section">
            <h3>🔍 Significant Pairwise Comparisons (Fisher's Exact Test)</h3>
            <p class="section-desc">Compares accuracy between specific pairs of feature values. Only significant differences shown (p &lt; {p_threshold}).</p>
            <table class="analysis-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Value 1</th>
                        <th>Value 2</th>
                        <th>Acc 1</th>
                        <th>Acc 2</th>
                        <th>p-value</th>
                    </tr>
                </thead>
                <tbody>
                    {fisher_rows}
                </tbody>
            </table>
        </div>

        <!-- Failed Video Examples -->
        <div class="analysis-section">
            <h3>🎬 Failed Prediction Examples</h3>
            <p class="section-desc">Sample videos where the model made incorrect predictions, sorted by distance (furthest first).</p>
            <div class="failed-videos-grid">
                {failed_videos_html if failed_videos_html else '<p style="color: #666;">No failed predictions found.</p>'}
            </div>
        </div>
        '''

        return html

    def generate_model_performance_tab_html(self, experiment: Dict) -> str:
        """
        Generate HTML for model performance tab content.

        Args:
            experiment: Experiment row from CSV

        Returns:
            HTML string for model performance tab content
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
        <h2>🎯 Model Performance Metrics</h2>

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
        '''

        return html

    def generate_comparison_plots(self, train_metadata: List[Dict],
                                 test_metadata: List[Dict]) -> Dict[str, str]:
        """
        Generate comparison plots for continuous and categorical variables.

        Args:
            train_metadata: Training dataset metadata
            test_metadata: Test dataset metadata

        Returns:
            Dict of plot_name -> base64 encoded image
        """
        plots = {}

        logger.info("\n🎨 Generating comparison charts...")

        # Speed distribution
        try:
            train_speeds = [float(v['speed']) for v in train_metadata
                          if v.get('speed', '').replace('.', '').replace('-', '').isdigit()]
            test_speeds = [float(v['speed']) for v in test_metadata
                         if v.get('speed', '').replace('.', '').replace('-', '').isdigit()]
            if train_speeds and test_speeds:
                plots['speed'] = generate_pdf_plot(train_speeds, test_speeds, 'Speed', 'Speed (units/sec)')
                logger.info("   ✅ Speed distribution plot")
        except Exception as e:
            logger.warning(f"   ⚠️  Speed plot failed: {e}")
            plots['speed'] = ""

        # Angle distribution
        try:
            train_angles = [float(v['angle']) for v in train_metadata
                          if v.get('angle', '').replace('.', '').replace('-', '').isdigit()]
            test_angles = [float(v['angle']) for v in test_metadata
                         if v.get('angle', '').replace('.', '').replace('-', '').isdigit()]
            if train_angles and test_angles:
                plots['angle'] = generate_pdf_plot(train_angles, test_angles, 'View Angle', 'Angle (degrees)')
                logger.info("   ✅ Angle distribution plot")
        except Exception as e:
            logger.warning(f"   ⚠️  Angle plot failed: {e}")
            plots['angle'] = ""

        # Check for subtle gray textures
        train_textures = [v.get('texture', '') for v in train_metadata]
        test_textures = [v.get('texture', '') for v in test_metadata]
        has_subtle_gray = any('subtle' in t.lower() and 'gray' in t.lower()
                            for t in train_textures + test_textures)

        if has_subtle_gray:
            # Stripe gray distribution
            try:
                train_stripe = [float(v['stripe_contrast']) for v in train_metadata
                              if v.get('stripe_contrast', 'N/A') not in ['N/A', '']
                              and v.get('stripe_contrast', '').replace('.', '').isdigit()]
                test_stripe = [float(v['stripe_contrast']) for v in test_metadata
                             if v.get('stripe_contrast', 'N/A') not in ['N/A', '']
                             and v.get('stripe_contrast', '').replace('.', '').isdigit()]
                if train_stripe and test_stripe:
                    plots['stripe_gray'] = generate_pdf_plot(train_stripe, test_stripe,
                                                            'Stripe Gray Values', 'Gray Value')
                    logger.info("   ✅ Stripe gray distribution plot")
            except Exception as e:
                logger.warning(f"   ⚠️  Stripe gray plot failed: {e}")

            # Background gray distribution
            try:
                train_bg = [float(v['background_contrast']) for v in train_metadata
                          if v.get('background_contrast', 'N/A') not in ['N/A', '']
                          and v.get('background_contrast', '').replace('.', '').isdigit()]
                test_bg = [float(v['background_contrast']) for v in test_metadata
                         if v.get('background_contrast', 'N/A') not in ['N/A', '']
                         and v.get('background_contrast', '').replace('.', '').isdigit()]
                if train_bg and test_bg:
                    plots['background_gray'] = generate_pdf_plot(train_bg, test_bg,
                                                                'Background Gray Values', 'Gray Value')
                    logger.info("   ✅ Background gray distribution plot")
            except Exception as e:
                logger.warning(f"   ⚠️  Background gray plot failed: {e}")

        # Texture comparison (categorical)
        try:
            train_texture_counter = Counter([v.get('texture', 'unknown') for v in train_metadata])
            test_texture_counter = Counter([v.get('texture', 'unknown') for v in test_metadata])
            plots['texture'] = generate_categorical_comparison(train_texture_counter,
                                                              test_texture_counter, 'Texture')
            logger.info("   ✅ Texture comparison chart")
        except Exception as e:
            logger.warning(f"   ⚠️  Texture comparison failed: {e}")
            plots['texture'] = ""

        # Direction comparison (categorical)
        try:
            train_direction_counter = Counter([v.get('direction', 'unknown') for v in train_metadata])
            test_direction_counter = Counter([v.get('direction', 'unknown') for v in test_metadata])
            plots['direction'] = generate_categorical_comparison(train_direction_counter,
                                                                test_direction_counter, 'Direction')
            logger.info("   ✅ Direction comparison chart")
        except Exception as e:
            logger.warning(f"   ⚠️  Direction comparison failed: {e}")
            plots['direction'] = ""

        return plots

    def generate_dual_report(self, experiment: Dict) -> str:
        """
        Generate comprehensive dual dataset report with full analytics and model performance.

        Args:
            experiment: Experiment row from CSV

        Returns:
            Path to generated HTML report (as file:// URL)
        """
        train_dataset = experiment.get('train_dataset_name', '')
        test_dataset = experiment.get('test_dataset_name', '')
        dataset_name = experiment.get('dataset_name', '')

        logger.info(f"Generating comprehensive report for experiment: {dataset_name}")
        logger.info(f"  Train: {train_dataset}")
        logger.info(f"  Test: {test_dataset}")

        # Ensure videos are downloaded for both datasets
        logger.info("\n=== TRAIN DATASET ===")
        train_video_dir = self.ensure_videos_downloaded(train_dataset)
        train_metadata_csv = self.get_metadata(train_dataset)

        logger.info("\n=== TEST DATASET ===")
        test_video_dir = self.ensure_videos_downloaded(test_dataset)
        test_metadata_csv = self.get_metadata(test_dataset)

        # Load evaluation results (predictions) for test dataset
        logger.info("\n=== LOADING EVALUATION RESULTS ===")
        test_prediction_data = {}
        failure_analysis_results = None
        test_eval_csvs = self.ensure_evaluation_results(test_dataset)
        if test_eval_csvs:
            test_prediction_data = self.load_prediction_data(test_eval_csvs)

            # Run failure analysis on finetuned predictions
            if FAILURE_ANALYSIS_AVAILABLE:
                logger.info("\n=== RUNNING FAILURE ANALYSIS ===")
                try:
                    failure_analysis_results = analyze_predictions(str(test_eval_csvs['finetuned']))
                    logger.info(f"✓ Failure analysis complete: {failure_analysis_results['row_count']} predictions analyzed")
                    logger.info(f"  Conclusion: {failure_analysis_results['conclusion'][:80]}...")
                except Exception as e:
                    logger.warning(f"⚠️  Failure analysis failed: {e}")
                    failure_analysis_results = None
            else:
                logger.warning("⚠️  Failure analysis not available (scipy not installed)")
        else:
            logger.warning("⚠️  No evaluation results found for test dataset. Per-video predictions will not be displayed.")

        # Load and analyze metadata
        logger.info("\n=== ANALYZING DATASETS ===")
        train_metadata = load_metadata(train_metadata_csv)
        test_metadata = load_metadata(test_metadata_csv)

        train_stats = analyze_distribution(train_metadata)
        test_stats = analyze_distribution(test_metadata)

        train_groups = group_videos(train_metadata)
        test_groups = group_videos(test_metadata)

        # Generate comparison plots
        comparison_plots = self.generate_comparison_plots(train_metadata, test_metadata)

        # Generate model performance HTML
        performance_html = self.generate_model_performance_tab_html(experiment)

        # Generate full HTML report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"{dataset_name}_dual_report_{timestamp}.html"
        report_path = self.reports_dir / report_filename

        # Calculate relative paths from HTML report to video directories
        # (same logic as dual_dataset_review.py for browser compatibility)
        output_dir = os.path.dirname(str(report_path.resolve()))
        train_video_url = os.path.relpath(str(train_video_dir.resolve()), output_dir)
        test_video_url = os.path.relpath(str(test_video_dir.resolve()), output_dir)

        # Generate failure analysis HTML
        failure_analysis_html = self.generate_failure_analysis_tab_html(failure_analysis_results, test_video_url)

        # Create HTML report
        html_content = self._build_comprehensive_html_report(
            performance_html=performance_html,
            failure_analysis_html=failure_analysis_html,
            train_metadata=train_metadata,
            test_metadata=test_metadata,
            train_stats=train_stats,
            test_stats=test_stats,
            train_groups=train_groups,
            test_groups=test_groups,
            comparison_plots=comparison_plots,
            train_video_dir=train_video_url,
            test_video_dir=test_video_url,
            dataset_name=dataset_name,
            test_prediction_data=test_prediction_data
        )

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"\n✅ Report generated: {report_path}")

        # Return as file:// URL
        absolute_path = report_path.resolve()
        file_url = f"file://{absolute_path}"

        return file_url

    def _build_comprehensive_html_report(
        self,
        performance_html: str,
        failure_analysis_html: str,
        train_metadata: List[Dict],
        test_metadata: List[Dict],
        train_stats: Dict,
        test_stats: Dict,
        train_groups: Dict,
        test_groups: Dict,
        comparison_plots: Dict,
        train_video_dir: str,
        test_video_dir: str,
        dataset_name: str,
        test_prediction_data: Dict[str, Dict] = None
    ) -> str:
        """Build complete comprehensive HTML report with all analytics, model performance, and failure analysis."""

        # Prepare JSON data
        train_metadata_json = json.dumps(train_metadata, indent=2)
        test_metadata_json = json.dumps(test_metadata, indent=2)
        train_stats_json = json.dumps(train_stats, indent=2)
        test_stats_json = json.dumps(test_stats, indent=2)
        test_prediction_data_json = json.dumps(test_prediction_data if test_prediction_data else {}, indent=2)

        # Sample videos (max 5 per group)
        train_samples = {}
        for category, category_groups in train_groups.items():
            train_samples[category] = {}
            for group_name, group_videos in category_groups.items():
                train_samples[category][group_name] = group_videos[:5]
        train_samples_json = json.dumps(train_samples, indent=2)

        test_samples = {}
        for category, category_groups in test_groups.items():
            test_samples[category] = {}
            for group_name, group_videos in category_groups.items():
                test_samples[category][group_name] = group_videos[:5]
        test_samples_json = json.dumps(test_samples, indent=2)

        # Calculate train/test ratio
        train_count = len(train_metadata)
        test_count = len(test_metadata)
        ratio = f"{train_count/test_count:.2f}" if test_count > 0 else "N/A"

        # Build comparison tab HTML
        comparison_html = f"""
            <div class="stats-comparison-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px;">
                <div class="stat-card">
                    <h3>Train Videos</h3>
                    <div class="value">{train_count}</div>
                </div>
                <div class="stat-card">
                    <h3>Test Videos</h3>
                    <div class="value">{test_count}</div>
                </div>
                <div class="stat-card">
                    <h3>Train/Test Ratio</h3>
                    <div class="value">{ratio}</div>
                </div>
            </div>

            <div class="pdf-plot" style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h3 style="color: #667eea; margin-bottom: 15px;">Speed Distribution (PDF)</h3>
                <img src="{comparison_plots.get('speed', '')}" style="width: 100%; max-width: 900px; display: block; margin: 0 auto;">
            </div>

            <div class="pdf-plot" style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h3 style="color: #667eea; margin-bottom: 15px;">Angle Distribution (PDF)</h3>
                <img src="{comparison_plots.get('angle', '')}" style="width: 100%; max-width: 900px; display: block; margin: 0 auto;">
            </div>
        """

        # Add conditional gray value plots
        if comparison_plots.get('stripe_gray'):
            comparison_html += f"""
            <div class="pdf-plot" style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h3 style="color: #667eea; margin-bottom: 15px;">Stripe Gray Values Distribution (PDF)</h3>
                <img src="{comparison_plots['stripe_gray']}" style="width: 100%; max-width: 900px; display: block; margin: 0 auto;">
            </div>
            """

        if comparison_plots.get('background_gray'):
            comparison_html += f"""
            <div class="pdf-plot" style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h3 style="color: #667eea; margin-bottom: 15px;">Background Gray Values Distribution (PDF)</h3>
                <img src="{comparison_plots['background_gray']}" style="width: 100%; max-width: 900px; display: block; margin: 0 auto;">
            </div>
            """

        comparison_html += f"""
            <div class="categorical-comparison" style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h3 style="color: #667eea; margin-bottom: 15px;">Texture Distribution Comparison</h3>
                <img src="{comparison_plots.get('texture', '')}" style="width: 100%; max-width: 900px; display: block; margin: 0 auto;">
            </div>

            <div class="categorical-comparison" style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h3 style="color: #667eea; margin-bottom: 15px;">Direction Distribution Comparison</h3>
                <img src="{comparison_plots.get('direction', '')}" style="width: 100%; max-width: 900px; display: block; margin: 0 auto;">
            </div>
        """

        # Build complete HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Experiment Report - {dataset_name}</title>
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
            max-width: 1600px;
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
        header .experiment-name {{ font-size: 1.2em; opacity: 0.9; font-family: monospace; }}
        .tabs {{
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
            overflow-x: auto;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .tab {{
            padding: 15px 20px;
            cursor: pointer;
            border: none;
            background: transparent;
            font-size: 0.9em;
            font-weight: 500;
            color: #495057;
            transition: all 0.3s;
            white-space: nowrap;
        }}
        .tab:hover {{ background: rgba(102, 126, 234, 0.1); color: #667eea; }}
        .tab.active {{ background: white; color: #667eea; border-bottom: 3px solid #667eea; }}
        .tab.performance-tab {{ background: linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, rgba(255, 152, 0, 0.15) 100%); font-weight: 600; }}
        .tab.train-tab {{ border-left: 3px solid #667eea; }}
        .tab.test-tab {{ border-left: 3px solid #764ba2; }}
        .tab.comparison-tab {{ background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); font-weight: 600; }}
        .tab-content {{ padding: 30px; display: none; }}
        .tab-content.active {{ display: block; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{ font-size: 1em; opacity: 0.9; margin-bottom: 10px; }}
        .stat-card .value {{ font-size: 2.5em; font-weight: bold; }}

        /* Model Performance Styles */
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

        .distribution-chart {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .distribution-chart h3 {{ margin-bottom: 15px; color: #667eea; }}
        .bar-item {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }}
        .bar-label {{
            width: 150px;
            font-size: 0.9em;
            color: #495057;
        }}
        .bar {{
            flex: 1;
            height: 25px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 4px;
            position: relative;
            margin-right: 10px;
        }}
        .bar-value {{
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            color: white;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .alert {{
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .alert-warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            color: #856404;
        }}
        .alert-info {{
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            color: #0c5460;
        }}
        .video-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }}
        .video-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        .video-card:hover {{ transform: translateY(-5px); }}
        .video-card video {{
            width: 100%;
            border-radius: 8px;
            margin-bottom: 15px;
            background: #000;
        }}
        .video-card h4 {{
            font-size: 0.9em;
            color: #667eea;
            margin-bottom: 10px;
            word-break: break-all;
        }}
        .video-metadata {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            font-size: 0.85em;
        }}
        .metadata-item {{
            display: flex;
            justify-content: space-between;
            padding: 5px;
            background: white;
            border-radius: 4px;
        }}
        .metadata-label {{ font-weight: 600; color: #495057; }}
        .metadata-value {{ color: #667eea; }}
        .group-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0 15px 0;
            display: flex;
            justify-content: space-between;
        }}
        .search-box {{
            width: 100%;
            padding: 12px;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            font-size: 1em;
            margin-bottom: 20px;
        }}
        .search-box:focus {{ outline: none; border-color: #667eea; }}
        @media (max-width: 768px) {{
            .video-grid {{ grid-template-columns: 1fr; }}
            .stats-grid {{ grid-template-columns: 1fr; }}
        }}

        /* Failure Analysis Tab Styles */
        .tab.failure-analysis-tab {{
            background: linear-gradient(135deg, rgba(220, 53, 69, 0.15) 0%, rgba(253, 126, 20, 0.15) 100%);
            font-weight: 600;
        }}
        .conclusion-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }}
        .conclusion-box h3 {{
            margin: 0 0 10px 0;
            font-size: 1.3em;
        }}
        .conclusion-box p {{
            margin: 0;
            font-size: 1.1em;
            line-height: 1.5;
        }}
        .analysis-section {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 25px;
            border-left: 4px solid #667eea;
        }}
        .analysis-section h3 {{
            color: #667eea;
            margin: 0 0 15px 0;
            font-size: 1.2em;
        }}
        .section-desc {{
            color: #6c757d;
            margin-bottom: 15px;
            font-size: 0.95em;
        }}
        .sig-factors-list {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .sig-factor {{
            background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(253, 126, 20, 0.1) 100%);
            padding: 12px 18px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #dc3545;
            font-family: monospace;
            font-size: 0.95em;
        }}
        .analysis-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .analysis-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 15px;
            text-align: left;
            font-weight: 600;
        }}
        .analysis-table td {{
            padding: 10px 15px;
            border-bottom: 1px solid #dee2e6;
        }}
        .analysis-table tr:last-child td {{
            border-bottom: none;
        }}
        .analysis-table tr.high-acc td {{
            background: rgba(40, 167, 69, 0.1);
        }}
        .analysis-table tr.mid-acc td {{
            background: rgba(255, 193, 7, 0.1);
        }}
        .analysis-table tr.low-acc td {{
            background: rgba(220, 53, 69, 0.1);
        }}
        .analysis-table tr.significant td {{
            background: rgba(220, 53, 69, 0.15);
            font-weight: 500;
        }}
        .failed-videos-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }}
        .failed-video-card {{
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border: 2px solid #dc3545;
        }}
        .failed-video-card video {{
            width: 100%;
            border-radius: 8px;
            margin-bottom: 12px;
            background: #000;
        }}
        .failed-video-info {{
            position: relative;
        }}
        .failed-badge {{
            position: absolute;
            top: -25px;
            right: 0;
            background: #dc3545;
            color: white;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: bold;
        }}
        .video-detail {{
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid #eee;
            font-size: 0.9em;
        }}
        .video-detail:last-child {{
            border-bottom: none;
        }}
        .video-detail .label {{
            color: #6c757d;
            font-weight: 500;
        }}
        .video-detail .value {{
            font-weight: 600;
        }}
        .video-detail .value.truth {{
            color: #28a745;
        }}
        .video-detail .value.predicted {{
            color: #dc3545;
        }}
        .video-detail .value.model-output {{
            font-family: monospace;
            font-size: 0.85em;
            color: #6c757d;
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Comprehensive Experiment Report</h1>
            <p class="experiment-name">{dataset_name}</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Train: {train_count} videos | Test: {test_count} videos</p>
        </header>

        <div class="tabs">
            <button class="tab performance-tab active" onclick="switchTab(event, 'performance')">🎯 Model Performance</button>
            <button class="tab failure-analysis-tab" onclick="switchTab(event, 'failure-analysis')">🔬 Failure Analysis</button>

            <button class="tab train-tab" onclick="switchTab(event, 'train-overview')">🔵 Train: Overview</button>
            <button class="tab train-tab" onclick="switchTab(event, 'train-defaults')">🔵 Train: Defaults</button>
            <button class="tab train-tab" onclick="switchTab(event, 'train-texture')">🔵 Train: Texture</button>
            <button class="tab train-tab" onclick="switchTab(event, 'train-direction')">🔵 Train: Direction</button>
            <button class="tab train-tab" onclick="switchTab(event, 'train-speed')">🔵 Train: Speed</button>
            <button class="tab train-tab" onclick="switchTab(event, 'train-angle')">🔵 Train: Angle</button>
            <button class="tab train-tab" onclick="switchTab(event, 'train-objects')">🔵 Train: Objects</button>
            <button class="tab train-tab" onclick="switchTab(event, 'train-blur')">🔵 Train: Blur</button>
            <button class="tab train-tab" onclick="switchTab(event, 'train-all')">🔵 Train: All Videos</button>

            <button class="tab test-tab" onclick="switchTab(event, 'test-overview')">🟣 Test: Overview</button>
            <button class="tab test-tab" onclick="switchTab(event, 'test-defaults')">🟣 Test: Defaults</button>
            <button class="tab test-tab" onclick="switchTab(event, 'test-texture')">🟣 Test: Texture</button>
            <button class="tab test-tab" onclick="switchTab(event, 'test-direction')">🟣 Test: Direction</button>
            <button class="tab test-tab" onclick="switchTab(event, 'test-speed')">🟣 Test: Speed</button>
            <button class="tab test-tab" onclick="switchTab(event, 'test-angle')">🟣 Test: Angle</button>
            <button class="tab test-tab" onclick="switchTab(event, 'test-objects')">🟣 Test: Objects</button>
            <button class="tab test-tab" onclick="switchTab(event, 'test-blur')">🟣 Test: Blur</button>
            <button class="tab test-tab" onclick="switchTab(event, 'test-all')">🟣 Test: All Videos</button>

            <button class="tab comparison-tab" onclick="switchTab(event, 'comparison')">📊 Distribution Comparison</button>
        </div>

        <!-- MODEL PERFORMANCE TAB -->
        <div id="performance" class="tab-content active">
            {performance_html}
        </div>

        <!-- FAILURE ANALYSIS TAB -->
        <div id="failure-analysis" class="tab-content">
            {failure_analysis_html}
        </div>

        <!-- TRAIN TABS -->
        <div id="train-overview" class="tab-content">
            <h2>📊 Train Dataset - Statistical Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Videos</h3>
                    <div class="value">{train_stats['total_videos']}</div>
                </div>
                <div class="stat-card">
                    <h3>Unique Textures</h3>
                    <div class="value">{train_stats['distributions'].get('texture', {}).get('unique_values', 0) if 'texture' in train_stats['distributions'] else 0}</div>
                </div>
                <div class="stat-card">
                    <h3>Directions</h3>
                    <div class="value">{train_stats['distributions'].get('direction', {}).get('unique_values', 0) if 'direction' in train_stats['distributions'] else 0}</div>
                </div>
                <div class="stat-card">
                    <h3>Speed Variations</h3>
                    <div class="value">{train_stats['distributions'].get('speed', {}).get('unique_values', 0) if 'speed' in train_stats['distributions'] else 0}</div>
                </div>
            </div>
            <div id="train-distribution-charts"></div>
            <div id="train-representation-alerts"></div>
        </div>

        <div id="train-defaults" class="tab-content">
            <h2>📋 Train Dataset - Defaults & Common Values</h2>
            <p style="margin-bottom: 20px; color: #495057;">Most common value for each parameter.</p>
            <div id="train-defaults-table"></div>
        </div>

        <div id="train-texture" class="tab-content"></div>
        <div id="train-direction" class="tab-content"></div>
        <div id="train-speed" class="tab-content"></div>
        <div id="train-angle" class="tab-content"></div>
        <div id="train-objects" class="tab-content"></div>
        <div id="train-blur" class="tab-content"></div>

        <div id="train-all" class="tab-content">
            <h2>Train: All Videos</h2>
            <input type="text" class="search-box" id="trainSearchBox"
                   placeholder="Search by name, texture, direction, speed..."
                   onkeyup="filterTrainVideos()">
            <div id="trainAllVideosGrid" class="video-grid"></div>
        </div>

        <!-- TEST TABS -->
        <div id="test-overview" class="tab-content">
            <h2>📊 Test Dataset - Statistical Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Videos</h3>
                    <div class="value">{test_stats['total_videos']}</div>
                </div>
                <div class="stat-card">
                    <h3>Unique Textures</h3>
                    <div class="value">{test_stats['distributions'].get('texture', {}).get('unique_values', 0) if 'texture' in test_stats['distributions'] else 0}</div>
                </div>
                <div class="stat-card">
                    <h3>Directions</h3>
                    <div class="value">{test_stats['distributions'].get('direction', {}).get('unique_values', 0) if 'direction' in test_stats['distributions'] else 0}</div>
                </div>
                <div class="stat-card">
                    <h3>Speed Variations</h3>
                    <div class="value">{test_stats['distributions'].get('speed', {}).get('unique_values', 0) if 'speed' in test_stats['distributions'] else 0}</div>
                </div>
            </div>
            <div id="test-distribution-charts"></div>
            <div id="test-representation-alerts"></div>
        </div>

        <div id="test-defaults" class="tab-content">
            <h2>📋 Test Dataset - Defaults & Common Values</h2>
            <p style="margin-bottom: 20px; color: #495057;">Most common value for each parameter.</p>
            <div id="test-defaults-table"></div>
        </div>

        <div id="test-texture" class="tab-content"></div>
        <div id="test-direction" class="tab-content"></div>
        <div id="test-speed" class="tab-content"></div>
        <div id="test-angle" class="tab-content"></div>
        <div id="test-objects" class="tab-content"></div>
        <div id="test-blur" class="tab-content"></div>

        <div id="test-all" class="tab-content">
            <h2>Test: All Videos</h2>
            <input type="text" class="search-box" id="testSearchBox"
                   placeholder="Search by name, texture, direction, speed..."
                   onkeyup="filterTestVideos()">
            <div id="testAllVideosGrid" class="video-grid"></div>
        </div>

        <!-- COMPARISON TAB -->
        <div id="comparison" class="tab-content">
            <h2>📊 Train vs Test Distribution Comparison</h2>
            {comparison_html}
        </div>
    </div>

    <script>
        const trainMetadata = {train_metadata_json};
        const testMetadata = {test_metadata_json};
        const trainStats = {train_stats_json};
        const testStats = {test_stats_json};
        const trainSamples = {train_samples_json};
        const testSamples = {test_samples_json};
        const trainVideoBaseUrl = '{train_video_dir}';
        const testVideoBaseUrl = '{test_video_dir}';
        const testPredictionData = {test_prediction_data_json};

        function switchTab(event, tabName) {{
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
            event.currentTarget.classList.add('active');

            // Lazy load videos
            if (tabName === 'train-all' && !document.getElementById('trainAllVideosGrid').innerHTML) {{
                renderAllVideos('train');
            }} else if (tabName === 'test-all' && !document.getElementById('testAllVideosGrid').innerHTML) {{
                renderAllVideos('test');
            }}
        }}

        function renderVideoCard(video, baseUrl, isTestVideo = false) {{
            const videoUrl = baseUrl ? `${{baseUrl}}/${{video.video_name}}` : video.video_name;

            // Get prediction data for test videos
            let predictionsHtml = '';
            if (isTestVideo && testPredictionData) {{
                // Use just the filename to look up predictions
                const predictions = testPredictionData[video.video_name];

                if (predictions) {{
                    predictionsHtml = `
                        <div style="margin-top: 15px; padding: 15px; background: #e8f5e9; border-radius: 8px; border-left: 4px solid #4caf50;">
                            <h5 style="margin: 0 0 10px 0; color: #2e7d32; font-size: 0.95em;">📊 Model Predictions</h5>
                            <div style="display: grid; gap: 8px; font-size: 0.85em;">
                                <div style="padding: 8px; background: white; border-radius: 4px;">
                                    <span style="font-weight: 600; color: #1976d2;">Ground Truth:</span>
                                    <span style="color: #1976d2; font-weight: bold; text-transform: uppercase;">${{predictions.label}}</span>
                                </div>
                                <div style="padding: 8px; background: white; border-radius: 4px;">
                                    <div style="font-weight: 600; color: #f57c00; margin-bottom: 4px;">Base Model:</div>
                                    <div style="padding-left: 10px;">
                                        <div style="margin-bottom: 3px;"><span style="font-weight: 500;">Raw:</span> <span style="color: #666;">${{predictions.base_raw}}</span></div>
                                        <div><span style="font-weight: 500;">Evaluated:</span> <span style="color: #f57c00; font-weight: bold; text-transform: uppercase;">${{predictions.base_eval}}</span></div>
                                    </div>
                                </div>
                                <div style="padding: 8px; background: white; border-radius: 4px;">
                                    <div style="font-weight: 600; color: #388e3c; margin-bottom: 4px;">Fine-Tuned Model:</div>
                                    <div style="padding-left: 10px;">
                                        <div style="margin-bottom: 3px;"><span style="font-weight: 500;">Raw:</span> <span style="color: #666;">${{predictions.finetuned_raw}}</span></div>
                                        <div><span style="font-weight: 500;">Evaluated:</span> <span style="color: #388e3c; font-weight: bold; text-transform: uppercase;">${{predictions.finetuned_eval}}</span></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                }}
            }}

            return `
                <div class="video-card">
                    <video controls preload="metadata">
                        <source src="${{videoUrl}}" type="video/mp4">
                        Your browser does not support video.
                    </video>
                    <h4>${{video.video_name}}</h4>
                    <div class="video-metadata">
                        <div class="metadata-item"><span class="metadata-label">Index:</span><span class="metadata-value">${{video.index || 'N/A'}}</span></div>
                        <div class="metadata-item"><span class="metadata-label">Texture:</span><span class="metadata-value">${{video.texture || 'N/A'}}</span></div>
                        <div class="metadata-item"><span class="metadata-label">Direction:</span><span class="metadata-value">${{video.direction || 'N/A'}}</span></div>
                        <div class="metadata-item"><span class="metadata-label">Speed:</span><span class="metadata-value">${{video.speed || 'N/A'}}</span></div>
                        <div class="metadata-item"><span class="metadata-label">Angle:</span><span class="metadata-value">${{video.angle || 'N/A'}}°</span></div>
                        <div class="metadata-item"><span class="metadata-label">Brightness:</span><span class="metadata-value">${{video.brightness || 'N/A'}}</span></div>
                        <div class="metadata-item"><span class="metadata-label">Contrast:</span><span class="metadata-value">${{video.contrast || 'N/A'}}</span></div>
                        <div class="metadata-item"><span class="metadata-label">Stripe Gray:</span><span class="metadata-value">${{video.stripe_contrast || 'N/A'}}</span></div>
                        <div class="metadata-item"><span class="metadata-label">Background Gray:</span><span class="metadata-value">${{video.background_contrast || 'N/A'}}</span></div>
                        <div class="metadata-item"><span class="metadata-label">Resolution:</span><span class="metadata-value">${{video.resolution || 'N/A'}}</span></div>
                        <div class="metadata-item"><span class="metadata-label">Seed:</span><span class="metadata-value">${{video.seed || 'N/A'}}</span></div>
                    </div>
                    ${{predictionsHtml}}
                </div>
            `;
        }}

        function renderDistributionCharts(stats, containerId) {{
            const container = document.getElementById(containerId);
            for (const [param, data] of Object.entries(stats.distributions)) {{
                const chartDiv = document.createElement('div');
                chartDiv.className = 'distribution-chart';
                const counts = data.counts;
                const maxCount = Math.max(...Object.values(counts));
                let barsHtml = '';
                for (const [value, count] of Object.entries(counts)) {{
                    const width = (count / maxCount) * 100;
                    barsHtml += `
                        <div class="bar-item">
                            <div class="bar-label">${{value || 'empty'}}</div>
                            <div class="bar" style="width: ${{width}}%">
                                <span class="bar-value">${{count}}</span>
                            </div>
                        </div>
                    `;
                }}
                chartDiv.innerHTML = `
                    <h3>${{param.charAt(0).toUpperCase() + param.slice(1)}} Distribution</h3>
                    <div>${{barsHtml}}</div>
                `;
                container.appendChild(chartDiv);
            }}
        }}

        function renderRepresentationAlerts(stats, containerId) {{
            const container = document.getElementById(containerId);
            if (stats.under_represented.length > 0) {{
                const div = document.createElement('div');
                div.className = 'alert alert-warning';
                div.innerHTML = `
                    <h3>⚠️ Under-represented Groups</h3>
                    <ul>
                        ${{stats.under_represented.map(item =>
                            `<li><strong>${{item.parameter}}: ${{item.value}}</strong> - ${{item.count}} videos (expected ~${{item.expected}})</li>`
                        ).join('')}}
                    </ul>
                `;
                container.appendChild(div);
            }}
            if (stats.over_represented.length > 0) {{
                const div = document.createElement('div');
                div.className = 'alert alert-info';
                div.innerHTML = `
                    <h3>📈 Over-represented Groups</h3>
                    <ul>
                        ${{stats.over_represented.map(item =>
                            `<li><strong>${{item.parameter}}: ${{item.value}}</strong> - ${{item.count}} videos (expected ~${{item.expected}})</li>`
                        ).join('')}}
                    </ul>
                `;
                container.appendChild(div);
            }}
        }}

        function renderDefaultsTable(metadata, containerId) {{
            const container = document.getElementById(containerId);
            const fields = ['index', 'texture', 'direction', 'speed', 'angle', 'brightness', 'contrast',
                'stripe_contrast', 'background_contrast', 'object_enabled', 'object_type',
                'num_objects', 'object_size', 'object_position', 'blur_enabled', 'blur_type',
                'blur_intensity', 'blur_variation', 'resolution', 'seed'];
            const defaults = {{}};
            fields.forEach(field => {{
                const counter = {{}};
                metadata.forEach(video => {{
                    const value = video[field] || 'N/A';
                    counter[value] = (counter[value] || 0) + 1;
                }});
                let maxCount = 0, mostCommon = 'N/A';
                for (const [value, count] of Object.entries(counter)) {{
                    if (count > maxCount) {{
                        maxCount = count;
                        mostCommon = value;
                    }}
                }}
                defaults[field] = {{
                    value: mostCommon,
                    count: maxCount,
                    percentage: ((maxCount / metadata.length) * 100).toFixed(1)
                }};
            }});
            let html = `
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <thead>
                            <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                                <th style="padding: 15px; text-align: left;">Parameter</th>
                                <th style="padding: 15px; text-align: left;">Most Common Value</th>
                                <th style="padding: 15px; text-align: center;">Count</th>
                                <th style="padding: 15px; text-align: center;">Percentage</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
            fields.forEach((field, i) => {{
                const bg = i % 2 === 0 ? '#ffffff' : '#f8f9fa';
                const def = defaults[field];
                html += `
                    <tr style="background: ${{bg}};">
                        <td style="padding: 12px; font-weight: 600; border-bottom: 1px solid #dee2e6;">
                            ${{field.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}}
                        </td>
                        <td style="padding: 12px; color: #667eea; font-family: monospace; border-bottom: 1px solid #dee2e6;">
                            ${{def.value}}
                        </td>
                        <td style="padding: 12px; text-align: center; border-bottom: 1px solid #dee2e6;">
                            ${{def.count}}
                        </td>
                        <td style="padding: 12px; text-align: center; border-bottom: 1px solid #dee2e6;">
                            ${{def.percentage}}%
                        </td>
                    </tr>
                `;
            }});
            html += '</tbody></table></div>';
            container.innerHTML = html;
        }}

        function renderGroupTab(tabId, categoryName, samples, videoBaseUrl, isTestVideo = false) {{
            const container = document.getElementById(tabId);
            const categoryData = samples[categoryName];
            let html = `<h2>${{categoryName.charAt(0).toUpperCase() + categoryName.slice(1)}} Groups</h2>`;
            for (const [groupName, videos] of Object.entries(categoryData)) {{
                html += `
                    <div class="group-header">
                        <h3>${{groupName}}</h3>
                        <span>${{videos.length}} samples</span>
                    </div>
                    <div class="video-grid">
                        ${{videos.map(video => renderVideoCard(video, videoBaseUrl, isTestVideo)).join('')}}
                    </div>
                `;
            }}
            container.innerHTML = html;
        }}

        function renderAllVideos(dataset) {{
            const metadata = dataset === 'train' ? trainMetadata : testMetadata;
            const baseUrl = dataset === 'train' ? trainVideoBaseUrl : testVideoBaseUrl;
            const gridId = dataset === 'train' ? 'trainAllVideosGrid' : 'testAllVideosGrid';
            const isTestVideo = dataset === 'test';
            document.getElementById(gridId).innerHTML = metadata.map(video => renderVideoCard(video, baseUrl, isTestVideo)).join('');
        }}

        function filterTrainVideos() {{
            const search = document.getElementById('trainSearchBox').value.toLowerCase();
            document.querySelectorAll('#trainAllVideosGrid .video-card').forEach(card => {{
                card.style.display = card.textContent.toLowerCase().includes(search) ? 'block' : 'none';
            }});
        }}

        function filterTestVideos() {{
            const search = document.getElementById('testSearchBox').value.toLowerCase();
            document.querySelectorAll('#testAllVideosGrid .video-card').forEach(card => {{
                card.style.display = card.textContent.toLowerCase().includes(search) ? 'block' : 'none';
            }});
        }}

        window.addEventListener('DOMContentLoaded', () => {{
            // Train dataset
            renderDistributionCharts(trainStats, 'train-distribution-charts');
            renderRepresentationAlerts(trainStats, 'train-representation-alerts');
            renderDefaultsTable(trainMetadata, 'train-defaults-table');
            renderGroupTab('train-texture', 'texture', trainSamples, trainVideoBaseUrl, false);
            renderGroupTab('train-direction', 'direction', trainSamples, trainVideoBaseUrl, false);
            renderGroupTab('train-speed', 'speed_range', trainSamples, trainVideoBaseUrl, false);
            renderGroupTab('train-angle', 'angle', trainSamples, trainVideoBaseUrl, false);
            renderGroupTab('train-objects', 'objects', trainSamples, trainVideoBaseUrl, false);
            renderGroupTab('train-blur', 'blur', trainSamples, trainVideoBaseUrl, false);

            // Test dataset
            renderDistributionCharts(testStats, 'test-distribution-charts');
            renderRepresentationAlerts(testStats, 'test-representation-alerts');
            renderDefaultsTable(testMetadata, 'test-defaults-table');
            renderGroupTab('test-texture', 'texture', testSamples, testVideoBaseUrl, true);
            renderGroupTab('test-direction', 'direction', testSamples, testVideoBaseUrl, true);
            renderGroupTab('test-speed', 'speed_range', testSamples, testVideoBaseUrl, true);
            renderGroupTab('test-angle', 'angle', testSamples, testVideoBaseUrl, true);
            renderGroupTab('test-objects', 'objects', testSamples, testVideoBaseUrl, true);
            renderGroupTab('test-blur', 'blur', testSamples, testVideoBaseUrl, true);
        }});
    </script>
</body>
</html>"""

        return html


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive experiment report with full analytics and model performance'
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

    # Generate comprehensive report
    report_url = generator.generate_dual_report(experiment)

    # Update CSV with report link (as file:// URL)
    tracker = ExperimentTracker(csv_path=args.csv_path)
    tracker.current_experiment_id = experiment.get('experiment_id')
    tracker.update_dataset_comparison_report(report_url)

    print(f"\n✅ Comprehensive report generated!")
    print(f"\nReport URL: {report_url}")
    print(f"\nTo view: Click the link or paste it in your browser")

    return report_url


if __name__ == '__main__':
    main()
