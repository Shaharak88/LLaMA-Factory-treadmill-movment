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
        'vary_parameters',

        # Video generation parameters
        'texture_type',
        'direction',
        'speed_range',
        'resolution',
        'fps',
        'duration',
        'view_angle',
        'brightness',
        'contrast',
        'lighting_variation',
        'lighting_intensity',
        'motion_blur',
        'camera_noise',
        'edge_width',

        # Subtle gray stripes parameters
        'stripe_width',
        'stripe_spacing',
        'stripe_gray',
        'background_gray',
        'stripe_distance_variance',

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
        'num_train_epochs',
        'lr_scheduler_type',
        'warmup_ratio',
        'bf16',
        'fp16',
        'quantization_bit',
        'logging_steps',
        'save_steps',
        'eval_steps',

        # Evaluation configuration
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

        # Results
        'base_model_accuracy',
        'finetuned_model_accuracy',

        # Manual notes
        'notes_1',
        'notes_2'
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

    def start_experiment(self, args) -> int:
        """
        Start a new experiment and log initial parameters.

        Args:
            args: Command-line arguments from argparse

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
            'vary_parameters': getattr(args, 'vary_parameters', False),

            # Video generation parameters
            'texture_type': getattr(args, 'texture_type', ''),
            'direction': getattr(args, 'direction', ''),
            'speed_range': getattr(args, 'speed_range', ''),
            'resolution': getattr(args, 'resolution', ''),
            'fps': getattr(args, 'fps', ''),
            'duration': getattr(args, 'duration', ''),
            'view_angle': getattr(args, 'view_angle', ''),
            'brightness': getattr(args, 'brightness', ''),
            'contrast': getattr(args, 'contrast', ''),
            'lighting_variation': getattr(args, 'lighting_variation', ''),
            'lighting_intensity': getattr(args, 'lighting_intensity', ''),
            'motion_blur': getattr(args, 'motion_blur', ''),
            'camera_noise': getattr(args, 'camera_noise', ''),
            'edge_width': getattr(args, 'edge_width', ''),

            # Subtle gray stripes parameters
            'stripe_width': getattr(args, 'stripe_width', ''),
            'stripe_spacing': getattr(args, 'stripe_spacing', ''),
            'stripe_gray': getattr(args, 'stripe_gray', ''),
            'background_gray': getattr(args, 'background_gray', ''),
            'stripe_distance_variance': getattr(args, 'stripe_distance_variance', ''),

            # Model configuration
            'model_name_or_path': getattr(args, 'model_name_or_path', ''),
            'template': getattr(args, 'template', ''),

            # LoRA configuration
            'lora_output_dir': getattr(args, 'lora_output_dir', ''),
            'lora_rank': getattr(args, 'lora_rank', ''),
            'lora_alpha': getattr(args, 'lora_alpha', ''),
            'lora_dropout': getattr(args, 'lora_dropout', ''),
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
            'finetuned_model_accuracy': '',

            # Manual notes (empty)
            'notes_1': '',
            'notes_2': ''
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

    def update_evaluation_results(self, base_accuracy: Optional[float] = None,
                                   finetuned_accuracy: Optional[float] = None) -> None:
        """
        Update evaluation results.

        Args:
            base_accuracy: Accuracy of base model (0-100)
            finetuned_accuracy: Accuracy of fine-tuned model (0-100)
        """
        if self.current_experiment_id is None:
            logger.warning("[TRACKER] No active experiment to update")
            return

        updates = {}
        if base_accuracy is not None:
            updates['base_model_accuracy'] = f"{base_accuracy:.2f}"
        if finetuned_accuracy is not None:
            updates['finetuned_model_accuracy'] = f"{finetuned_accuracy:.2f}"

        for column, value in updates.items():
            self._update_column(self.current_experiment_id, column, value)

        logger.info(f"[TRACKER] Updated evaluation results")
        if base_accuracy is not None:
            logger.info(f"  Base model: {base_accuracy:.2f}%")
        if finetuned_accuracy is not None:
            logger.info(f"  Fine-tuned model: {finetuned_accuracy:.2f}%")

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


def parse_evaluation_results(output_dir: str) -> tuple[Optional[float], Optional[float]]:
    """
    Parse evaluation results from evaluation output directory.

    Args:
        output_dir: Path to evaluation output directory

    Returns:
        Tuple of (base_accuracy, finetuned_accuracy) or (None, None) if not found
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

        base_accuracy = None
        finetuned_accuracy = None

        # Parse accuracies from report
        # Look for patterns like "Base Model Accuracy: 85.50%"
        import re

        base_match = re.search(r'Base Model.*?Accuracy.*?:\s*([\d.]+)%', content, re.IGNORECASE)
        if base_match:
            base_accuracy = float(base_match.group(1))

        finetuned_match = re.search(r'(?:Fine-?tuned|LoRA).*?Accuracy.*?:\s*([\d.]+)%', content, re.IGNORECASE)
        if finetuned_match:
            finetuned_accuracy = float(finetuned_match.group(1))

        return base_accuracy, finetuned_accuracy

    except Exception as e:
        logger.warning(f"[TRACKER] Error parsing evaluation results: {e}")
        return None, None
