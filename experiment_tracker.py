#!/usr/bin/env python3
"""
Experiment Tracker - CSV-based experiment logging for treadmill detection pipeline

This module provides functionality to track all experiments in a CSV file, including:
- All dataset generation arguments
- All model configuration parameters
- All LoRA training hyperparameters
- Evaluation results
- Manual notes columns

Author: AI-Generated
Date: 2025-11-30
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Manages experiment tracking in a CSV file.
    """

    # Define all CSV columns
    CSV_COLUMNS = [
        # Experiment metadata
        'experiment_id',
        'timestamp',
        'run_timestamp',

        # Dataset configuration
        'dataset_name',
        'train_dataset_name',
        'test_dataset_name',
        'num_videos',
        'train_split',
        'seed',

        # TRAINING Dataset Parameters
        'train_texture_type',
        'train_direction',
        'train_view_angle',
        'train_speed_range',
        'train_resolution',
        'train_fps',
        'train_duration',
        'train_brightness',
        'train_contrast',
        'train_lighting_variation',
        'train_lighting_intensity',
        'train_motion_blur',
        'train_camera_noise',
        'train_edge_width',
        'train_stripe_width',
        'train_stripe_spacing',
        'train_stripe_gray',
        'train_background_gray',
        'train_stripe_distance_variance',
        'train_add_object',
        'train_object_type',
        'train_object_position',
        'train_object_size',
        'train_num_objects',
        'train_add_blur',
        'train_blur_type',
        'train_blur_intensity',
        'train_random_blur_variation',
        'train_vary_parameters',

        # TEST Dataset Parameters
        'test_texture_type',
        'test_direction',
        'test_view_angle',
        'test_speed_range',
        'test_resolution',
        'test_fps',
        'test_duration',
        'test_brightness',
        'test_contrast',
        'test_lighting_variation',
        'test_lighting_intensity',
        'test_motion_blur',
        'test_camera_noise',
        'test_edge_width',
        'test_stripe_width',
        'test_stripe_spacing',
        'test_stripe_gray',
        'test_background_gray',
        'test_stripe_distance_variance',
        'test_add_object',
        'test_object_type',
        'test_object_position',
        'test_object_size',
        'test_num_objects',
        'test_add_blur',
        'test_blur_type',
        'test_blur_intensity',
        'test_random_blur_variation',
        'test_vary_parameters',

        # Model configuration
        'model_name_or_path',
        'template',

        # LoRA configuration
        'lora_output_dir',
        'lora_rank',
        'lora_alpha',
        'lora_dropout',
        'cutoff_len',

        # Training hyperparameters
        'per_device_train_batch_size',
        'gradient_accumulation_steps',
        'learning_rate',
        'lr_scheduler_type',
        'warmup_ratio',
        'bf16',
        'fp16',
        'quantization_bit',
        'logging_steps',
        'save_steps',
        'eval_steps',

        # Evaluation configuration
        'eval_method',
        'eval_max_new_tokens',
        'eval_batch_size',
        'eval_video_fps',
        'eval_video_maxlen',
        'gpu_memory_utilization',

        # Pipeline control
        'skip_dataset',
        'skip_training',
        'skip_evaluation',
        'use_docker',

        # Status tracking
        'dataset_status',
        'training_status',
        'evaluation_status',

        # Results - Overall
        'base_model_accuracy',
        'base_model_f1_score',
        'base_model_f1_moving',
        'base_model_f1_stopped',
        'base_model_precision',
        'base_model_recall',
        'base_model_moving_correct',
        'base_model_moving_total',
        'base_model_stopped_correct',
        'base_model_stopped_total',

        'finetuned_model_accuracy',
        'finetuned_model_f1_score',
        'finetuned_model_f1_moving',
        'finetuned_model_f1_stopped',
        'finetuned_model_precision',
        'finetuned_model_recall',
        'finetuned_model_moving_correct',
        'finetuned_model_moving_total',
        'finetuned_model_stopped_correct',
        'finetuned_model_stopped_total',

        # Results - Detailed breakdowns (JSON format)
        'base_model_per_texture_results',
        'base_model_per_angle_results',
        'finetuned_model_per_texture_results',
        'finetuned_model_per_angle_results',

        # Manual notes
        'notes_1',
        'notes_2',

        # Model information (for tracking and re-evaluation)
        'model_path',
        'model_name',

        # Key training parameters (for easy reference)
        'use_dora',
        'num_train_epochs'
    ]

    def __init__(self, csv_path: str = 'experiments_log.csv'):
        """
        Initialize experiment tracker.

        Args:
            csv_path: Path to CSV file for logging experiments
        """
        self.csv_path = Path(csv_path)
        self.current_experiment_id = None
        self._ensure_csv_exists()

    def _ensure_csv_exists(self) -> None:
        """Create CSV file with headers if it doesn't exist."""
        if not self.csv_path.exists():
            logger.info(f"Creating new experiment log: {self.csv_path}")
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
                writer.writeheader()
            logger.info(f"  [OK] Experiment log created with {len(self.CSV_COLUMNS)} columns")
        else:
            logger.info(f"Using existing experiment log: {self.csv_path}")

    def _get_next_experiment_id(self) -> int:
        """
        Get the next experiment ID by reading existing CSV.

        Returns:
            int: Next experiment ID
        """
        if not self.csv_path.exists() or os.path.getsize(self.csv_path) == 0:
            return 1

        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    return 1
                # Get max experiment_id
                max_id = max(int(row['experiment_id']) for row in rows if row.get('experiment_id'))
                return max_id + 1
        except Exception as e:
            logger.warning(f"Error reading experiment IDs: {e}, defaulting to 1")
            return 1

    def start_experiment(self, args, actual_model_path: str = None) -> int:
        """
        Start a new experiment and log initial parameters.

        Args:
            args: Command-line arguments from argparse
            actual_model_path: The actual calculated model path (overrides args.lora_output_dir)
                              This ensures we log the real path, not the default argument value

        Returns:
            int: Experiment ID
        """
        experiment_id = self._get_next_experiment_id()
        self.current_experiment_id = experiment_id

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Prepare experiment data
        experiment_data = {
            # Metadata
            'experiment_id': experiment_id,
            'timestamp': timestamp,
            'run_timestamp': run_timestamp,

            # Dataset configuration
            'dataset_name': getattr(args, 'dataset_name', ''),
            'train_dataset_name': f"{getattr(args, 'dataset_name', '')}_train",
            'test_dataset_name': f"{getattr(args, 'dataset_name', '')}_test",
            'num_videos': getattr(args, 'num_videos', ''),
            'train_split': getattr(args, 'train_split', ''),
            'seed': getattr(args, 'seed', ''),

            # TRAINING Dataset Parameters (metadata)
            'train_texture_type': getattr(args, 'train_texture_type', ''),
            'train_direction': getattr(args, 'train_direction', ''),
            'train_view_angle': getattr(args, 'train_view_angle', ''),
            'train_speed_range': getattr(args, 'train_speed_range', ''),
            'train_resolution': getattr(args, 'train_resolution', ''),
            'train_fps': getattr(args, 'train_fps', ''),
            'train_duration': getattr(args, 'train_duration', ''),
            'train_brightness': getattr(args, 'train_brightness', ''),
            'train_contrast': getattr(args, 'train_contrast', ''),
            'train_lighting_variation': getattr(args, 'train_lighting_variation', ''),
            'train_lighting_intensity': getattr(args, 'train_lighting_intensity', ''),
            'train_motion_blur': getattr(args, 'train_motion_blur', ''),
            'train_camera_noise': getattr(args, 'train_camera_noise', ''),
            'train_edge_width': getattr(args, 'train_edge_width', ''),
            'train_stripe_width': getattr(args, 'train_stripe_width', ''),
            'train_stripe_spacing': getattr(args, 'train_stripe_spacing', ''),
            'train_stripe_gray': getattr(args, 'train_stripe_gray', ''),
            'train_background_gray': getattr(args, 'train_background_gray', ''),
            'train_stripe_distance_variance': getattr(args, 'train_stripe_distance_variance', ''),
            'train_add_object': getattr(args, 'train_add_object', ''),
            'train_object_type': getattr(args, 'train_object_type', ''),
            'train_object_position': getattr(args, 'train_object_position', ''),
            'train_object_size': getattr(args, 'train_object_size', ''),
            'train_num_objects': getattr(args, 'train_num_objects', ''),
            'train_add_blur': getattr(args, 'train_add_blur', ''),
            'train_blur_type': getattr(args, 'train_blur_type', ''),
            'train_blur_intensity': getattr(args, 'train_blur_intensity', ''),
            'train_random_blur_variation': getattr(args, 'train_random_blur_variation', ''),
            'train_vary_parameters': getattr(args, 'train_vary_parameters', ''),

            # TEST Dataset Parameters (metadata)
            'test_texture_type': getattr(args, 'test_texture_type', ''),
            'test_direction': getattr(args, 'test_direction', ''),
            'test_view_angle': getattr(args, 'test_view_angle', ''),
            'test_speed_range': getattr(args, 'test_speed_range', ''),
            'test_resolution': getattr(args, 'test_resolution', ''),
            'test_fps': getattr(args, 'test_fps', ''),
            'test_duration': getattr(args, 'test_duration', ''),
            'test_brightness': getattr(args, 'test_brightness', ''),
            'test_contrast': getattr(args, 'test_contrast', ''),
            'test_lighting_variation': getattr(args, 'test_lighting_variation', ''),
            'test_lighting_intensity': getattr(args, 'test_lighting_intensity', ''),
            'test_motion_blur': getattr(args, 'test_motion_blur', ''),
            'test_camera_noise': getattr(args, 'test_camera_noise', ''),
            'test_edge_width': getattr(args, 'test_edge_width', ''),
            'test_stripe_width': getattr(args, 'test_stripe_width', ''),
            'test_stripe_spacing': getattr(args, 'test_stripe_spacing', ''),
            'test_stripe_gray': getattr(args, 'test_stripe_gray', ''),
            'test_background_gray': getattr(args, 'test_background_gray', ''),
            'test_stripe_distance_variance': getattr(args, 'test_stripe_distance_variance', ''),
            'test_add_object': getattr(args, 'test_add_object', ''),
            'test_object_type': getattr(args, 'test_object_type', ''),
            'test_object_position': getattr(args, 'test_object_position', ''),
            'test_object_size': getattr(args, 'test_object_size', ''),
            'test_num_objects': getattr(args, 'test_num_objects', ''),
            'test_add_blur': getattr(args, 'test_add_blur', ''),
            'test_blur_type': getattr(args, 'test_blur_type', ''),
            'test_blur_intensity': getattr(args, 'test_blur_intensity', ''),
            'test_random_blur_variation': getattr(args, 'test_random_blur_variation', ''),
            'test_vary_parameters': getattr(args, 'test_vary_parameters', ''),

            # Model configuration
            'model_name_or_path': getattr(args, 'model_name_or_path', ''),
            'template': getattr(args, 'template', ''),

            # LoRA configuration
            # Use actual_model_path for lora_output_dir as well (fixes CSV logging bug)
            'lora_output_dir': actual_model_path or getattr(args, 'lora_output_dir', ''),
            'lora_rank': getattr(args, 'lora_rank', ''),
            'lora_alpha': getattr(args, 'lora_alpha', ''),
            'lora_dropout': getattr(args, 'lora_dropout', ''),
            'use_dora': getattr(args, 'use_dora', False),
            'cutoff_len': getattr(args, 'cutoff_len', ''),

            # Training hyperparameters
            'per_device_train_batch_size': getattr(args, 'per_device_train_batch_size', ''),
            'gradient_accumulation_steps': getattr(args, 'gradient_accumulation_steps', ''),
            'learning_rate': getattr(args, 'learning_rate', ''),
            'num_train_epochs': getattr(args, 'num_train_epochs', ''),
            'lr_scheduler_type': getattr(args, 'lr_scheduler_type', ''),
            'warmup_ratio': getattr(args, 'warmup_ratio', ''),
            'bf16': getattr(args, 'bf16', False),
            'fp16': getattr(args, 'fp16', False),
            'quantization_bit': getattr(args, 'quantization_bit', ''),
            'logging_steps': getattr(args, 'logging_steps', ''),
            'save_steps': getattr(args, 'save_steps', ''),
            'eval_steps': getattr(args, 'eval_steps', ''),

            # Evaluation configuration
            'eval_method': getattr(args, 'eval_method', 'yesno'),
            'eval_max_new_tokens': getattr(args, 'eval_max_new_tokens', ''),
            'eval_batch_size': getattr(args, 'eval_batch_size', ''),
            'eval_video_fps': getattr(args, 'eval_video_fps', ''),
            'eval_video_maxlen': getattr(args, 'eval_video_maxlen', ''),
            'gpu_memory_utilization': getattr(args, 'gpu_memory_utilization', ''),

            # Pipeline control
            'skip_dataset': getattr(args, 'skip_dataset', False),
            'skip_training': getattr(args, 'skip_training', False),
            'skip_evaluation': getattr(args, 'skip_evaluation', False),
            'use_docker': getattr(args, 'use_docker', False),

            # Status (initial)
            'dataset_status': 'pending' if not getattr(args, 'skip_dataset', False) else 'skipped',
            'training_status': 'pending' if not getattr(args, 'skip_training', False) else 'skipped',
            'evaluation_status': 'pending' if not getattr(args, 'skip_evaluation', False) else 'skipped',

            # Results (empty initially)
            'base_model_accuracy': '',
            'base_model_f1_score': '',
            'base_model_f1_moving': '',
            'base_model_f1_stopped': '',
            'base_model_precision': '',
            'base_model_recall': '',
            'base_model_moving_correct': '',
            'base_model_moving_total': '',
            'base_model_stopped_correct': '',
            'base_model_stopped_total': '',

            'finetuned_model_accuracy': '',
            'finetuned_model_f1_score': '',
            'finetuned_model_f1_moving': '',
            'finetuned_model_f1_stopped': '',
            'finetuned_model_precision': '',
            'finetuned_model_recall': '',
            'finetuned_model_moving_correct': '',
            'finetuned_model_moving_total': '',
            'finetuned_model_stopped_correct': '',
            'finetuned_model_stopped_total': '',

            'base_model_per_texture_results': '',
            'base_model_per_angle_results': '',
            'finetuned_model_per_texture_results': '',
            'finetuned_model_per_angle_results': '',

            # Manual notes (empty)
            'notes_1': '',
            'notes_2': '',

            # Model information
            # CRITICAL FIX: Use actual_model_path (calculated path) instead of args (default value)
            # This fixes the bug where CSV logged wrong paths like saves/qwen2vl-treadmill-lora-pipeline
            # instead of the actual saves/{model_name} path
            'model_path': actual_model_path or getattr(args, 'eval_model_path', '') or getattr(args, 'lora_output_dir', ''),
            'model_name': getattr(args, 'model_name', '')
        }

        # Write to CSV
        self._append_or_update_row(experiment_data)

        logger.info(f"[TRACKER] Started experiment {experiment_id}")
        logger.info(f"[TRACKER] Logged to: {self.csv_path.absolute()}")

        return experiment_id

    def update_status(self, stage: str, status: str) -> None:
        """
        Update the status of a pipeline stage.

        Args:
            stage: One of 'dataset', 'training', 'evaluation'
            status: One of 'pending', 'in_progress', 'completed', 'failed', 'skipped'
        """
        if self.current_experiment_id is None:
            logger.warning("[TRACKER] No active experiment to update")
            return

        column_name = f"{stage}_status"
        self._update_column(self.current_experiment_id, column_name, status)
        logger.info(f"[TRACKER] Updated {stage} status: {status}")

    def update_evaluation_results(self, base_results: Optional[Dict] = None,
                                   finetuned_results: Optional[Dict] = None) -> None:
        """
        Update evaluation results with detailed metrics.

        Args:
            base_results: Dictionary with base model evaluation results
            finetuned_results: Dictionary with fine-tuned model evaluation results
        """
        if self.current_experiment_id is None:
            logger.warning("[TRACKER] No active experiment to update")
            return

        updates = {}

        # Process base model results
        if base_results is not None:
            updates['base_model_accuracy'] = f"{base_results.get('accuracy', 0):.2f}"
            updates['base_model_f1_score'] = f"{base_results.get('f1_score', 0):.2f}"
            updates['base_model_f1_moving'] = f"{base_results.get('f1_moving', 0):.2f}"
            updates['base_model_f1_stopped'] = f"{base_results.get('f1_stopped', 0):.2f}"
            updates['base_model_precision'] = f"{base_results.get('precision', 0):.2f}"
            updates['base_model_recall'] = f"{base_results.get('recall', 0):.2f}"
            updates['base_model_moving_correct'] = str(base_results.get('moving', {}).get('correct', 0))
            updates['base_model_moving_total'] = str(base_results.get('moving', {}).get('total', 0))
            updates['base_model_stopped_correct'] = str(base_results.get('stopped', {}).get('correct', 0))
            updates['base_model_stopped_total'] = str(base_results.get('stopped', {}).get('total', 0))

            # Store detailed breakdowns as JSON
            if 'per_texture' in base_results:
                texture_summary = {}
                for tex, data in base_results['per_texture'].items():
                    texture_summary[tex] = {
                        'accuracy': round(data.get('accuracy', 0), 2),
                        'f1_score': round(data.get('f1_score', 0), 2),
                        'correct': data.get('correct', 0),
                        'total': data.get('total', 0)
                    }
                import json
                updates['base_model_per_texture_results'] = json.dumps(texture_summary)

            if 'per_angle' in base_results:
                angle_summary = {}
                for ang, data in base_results['per_angle'].items():
                    angle_summary[ang] = {
                        'accuracy': round(data.get('accuracy', 0), 2),
                        'f1_score': round(data.get('f1_score', 0), 2),
                        'correct': data.get('correct', 0),
                        'total': data.get('total', 0)
                    }
                updates['base_model_per_angle_results'] = json.dumps(angle_summary)

        # Process fine-tuned model results
        if finetuned_results is not None:
            updates['finetuned_model_accuracy'] = f"{finetuned_results.get('accuracy', 0):.2f}"
            updates['finetuned_model_f1_score'] = f"{finetuned_results.get('f1_score', 0):.2f}"
            updates['finetuned_model_f1_moving'] = f"{finetuned_results.get('f1_moving', 0):.2f}"
            updates['finetuned_model_f1_stopped'] = f"{finetuned_results.get('f1_stopped', 0):.2f}"
            updates['finetuned_model_precision'] = f"{finetuned_results.get('precision', 0):.2f}"
            updates['finetuned_model_recall'] = f"{finetuned_results.get('recall', 0):.2f}"
            updates['finetuned_model_moving_correct'] = str(finetuned_results.get('moving', {}).get('correct', 0))
            updates['finetuned_model_moving_total'] = str(finetuned_results.get('moving', {}).get('total', 0))
            updates['finetuned_model_stopped_correct'] = str(finetuned_results.get('stopped', {}).get('correct', 0))
            updates['finetuned_model_stopped_total'] = str(finetuned_results.get('stopped', {}).get('total', 0))

            # Store detailed breakdowns as JSON
            if 'per_texture' in finetuned_results:
                texture_summary = {}
                for tex, data in finetuned_results['per_texture'].items():
                    texture_summary[tex] = {
                        'accuracy': round(data.get('accuracy', 0), 2),
                        'f1_score': round(data.get('f1_score', 0), 2),
                        'correct': data.get('correct', 0),
                        'total': data.get('total', 0)
                    }
                import json
                updates['finetuned_model_per_texture_results'] = json.dumps(texture_summary)

            if 'per_angle' in finetuned_results:
                angle_summary = {}
                for ang, data in finetuned_results['per_angle'].items():
                    angle_summary[ang] = {
                        'accuracy': round(data.get('accuracy', 0), 2),
                        'f1_score': round(data.get('f1_score', 0), 2),
                        'correct': data.get('correct', 0),
                        'total': data.get('total', 0)
                    }
                updates['finetuned_model_per_angle_results'] = json.dumps(angle_summary)

        # Apply all updates
        for column, value in updates.items():
            self._update_column(self.current_experiment_id, column, value)

        logger.info(f"[TRACKER] Updated evaluation results")
        if base_results is not None:
            logger.info(f"  Base model: {base_results.get('accuracy', 0):.2f}% accuracy, {base_results.get('f1_score', 0):.2f}% F1")
        if finetuned_results is not None:
            logger.info(f"  Fine-tuned model: {finetuned_results.get('accuracy', 0):.2f}% accuracy, {finetuned_results.get('f1_score', 0):.2f}% F1")

    def _append_or_update_row(self, data: Dict[str, Any]) -> None:
        """
        Append a new row or update existing row in CSV.

        Args:
            data: Dictionary with column names as keys
        """
        # Read all existing rows
        rows = []
        if self.csv_path.exists() and os.path.getsize(self.csv_path) > 0:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        # Check if experiment_id exists
        experiment_id = str(data['experiment_id'])
        found = False
        for i, row in enumerate(rows):
            if row.get('experiment_id') == experiment_id:
                # Update existing row
                rows[i].update(data)
                found = True
                break

        if not found:
            # Append new row
            rows.append(data)

        # Write all rows back
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def _update_column(self, experiment_id: int, column_name: str, value: Any) -> None:
        """
        Update a specific column for an experiment.

        Args:
            experiment_id: Experiment ID to update
            column_name: Column name to update
            value: New value
        """
        # Read all rows
        rows = []
        if self.csv_path.exists() and os.path.getsize(self.csv_path) > 0:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        # Update the specific row
        experiment_id_str = str(experiment_id)
        for row in rows:
            if row.get('experiment_id') == experiment_id_str:
                row[column_name] = str(value)
                break

        # Write back
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def finalize_experiment(self) -> None:
        """Mark experiment as complete."""
        if self.current_experiment_id is None:
            logger.warning("[TRACKER] No active experiment to finalize")
            return

        logger.info(f"[TRACKER] Experiment {self.current_experiment_id} finalized")
        self.current_experiment_id = None


def parse_evaluation_results(output_dir: str) -> tuple[Optional[Dict], Optional[Dict]]:
    """
    Parse detailed evaluation results from evaluation output directory.

    Args:
        output_dir: Path to evaluation output directory

    Returns:
        Tuple of (base_results_dict, finetuned_results_dict) with all metrics
    """
    output_path = Path(output_dir)

    # Look for evaluation report
    report_files = list(output_path.glob('evaluation_report_*.txt'))
    if not report_files:
        logger.warning(f"[TRACKER] No evaluation report found in {output_dir}")
        return None, None

    # Read the most recent report
    report_file = max(report_files, key=lambda p: p.stat().st_mtime)

    try:
        with open(report_file, 'r') as f:
            content = f.read()

        import re

        def parse_model_section(section_text: str) -> Optional[Dict]:
            """Parse a model's results section."""
            results = {}

            # Overall metrics
            acc_match = re.search(r'Accuracy:\s*([\d.]+)%\s*\((\d+)/(\d+)\)', section_text)
            if acc_match:
                results['accuracy'] = float(acc_match.group(1))
                results['correct'] = int(acc_match.group(2))
                results['total'] = int(acc_match.group(3))

            f1_match = re.search(r'F1 Score:\s*([\d.]+)%', section_text)
            if f1_match:
                results['f1_score'] = float(f1_match.group(1))

            prec_match = re.search(r'Precision:\s*([\d.]+)%', section_text)
            if prec_match:
                results['precision'] = float(prec_match.group(1))

            rec_match = re.search(r'Recall:\s*([\d.]+)%', section_text)
            if rec_match:
                results['recall'] = float(rec_match.group(1))

            # Per-class metrics
            moving_match = re.search(r'Moving:\s*(\d+)/(\d+)', section_text)
            if moving_match:
                results['moving'] = {
                    'correct': int(moving_match.group(1)),
                    'total': int(moving_match.group(2))
                }

            stopped_match = re.search(r'Stopped:\s*(\d+)/(\d+)', section_text)
            if stopped_match:
                results['stopped'] = {
                    'correct': int(stopped_match.group(1)),
                    'total': int(stopped_match.group(2))
                }

            # Per-texture breakdown
            texture_section = re.search(r'PER-TEXTURE BREAKDOWN:(.*?)(?:PER-ANGLE|COMPARISON|$)', section_text, re.DOTALL)
            if texture_section:
                results['per_texture'] = {}
                texture_blocks = re.finditer(r'(\w+(?:_\w+)*):\s*\n\s*Total:\s*(\d+).*?\n\s*Accuracy:\s*([\d.]+)%\s*\((\d+)/\d+\)(?:.*?F1 Score:\s*([\d.]+)%)?', texture_section.group(1), re.DOTALL)
                for match in texture_blocks:
                    tex_name = match.group(1)
                    results['per_texture'][tex_name] = {
                        'total': int(match.group(2)),
                        'accuracy': float(match.group(3)),
                        'correct': int(match.group(4)),
                        'f1_score': float(match.group(5)) if match.group(5) else 0.0
                    }

            # Per-angle breakdown
            angle_section = re.search(r'PER-ANGLE BREAKDOWN:(.*?)(?:COMPARISON|$)', section_text, re.DOTALL)
            if angle_section:
                results['per_angle'] = {}
                angle_blocks = re.finditer(r'(angle\d+):\s*\n\s*Total:\s*(\d+).*?\n\s*Accuracy:\s*([\d.]+)%\s*\((\d+)/\d+\)(?:.*?F1 Score:\s*([\d.]+)%)?', angle_section.group(1), re.DOTALL)
                for match in angle_blocks:
                    ang_name = match.group(1)
                    results['per_angle'][ang_name] = {
                        'total': int(match.group(2)),
                        'accuracy': float(match.group(3)),
                        'correct': int(match.group(4)),
                        'f1_score': float(match.group(5)) if match.group(5) else 0.0
                    }

            return results if results else None

        # Split into base and fine-tuned sections
        base_section = re.search(r'BASE MODEL\s*=+\s*(.*?)(?:FINE-TUNED MODEL|$)', content, re.DOTALL | re.IGNORECASE)
        finetuned_section = re.search(r'FINE-TUNED MODEL\s*=+\s*(.*?)(?:COMPARISON|$)', content, re.DOTALL | re.IGNORECASE)

        base_results = parse_model_section(base_section.group(1)) if base_section else None
        finetuned_results = parse_model_section(finetuned_section.group(1)) if finetuned_section else None

        return base_results, finetuned_results

    except Exception as e:
        logger.warning(f"[TRACKER] Error parsing evaluation results: {e}")
        import traceback
        traceback.print_exc()
        return None, None
