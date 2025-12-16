#!/usr/bin/env python3
"""
Evaluation Pipeline - Treadmill Motion Detection Model Evaluation

This script evaluates both the base Qwen2.5-VL model and a fine-tuned LoRA model
on a test dataset. It uses vllm_infer.py to generate predictions and calculates
accuracy by comparing predictions against ground truth labels.

The script:
1. Runs inference using vllm_infer.py for both base and fine-tuned models
2. Parses the JSONL output files
3. Compares predictions to ground truth labels
4. Calculates accuracy metrics
5. Generates detailed evaluation report

Author: AI-Generated
Date: 2025-11-26
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TreadmillEvaluator:
    """
    Evaluates treadmill motion detection models by comparing predictions to ground truth.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize evaluator.

        Args:
            args: Command-line arguments
        """
        self.args = args
        self.project_root = Path(__file__).parent
        self.vllm_script = self.project_root / "scripts" / "vllm_infer.py"

        # Output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp for this evaluation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Results storage
        self.results = {
            'base_model': None,
            'lora_model': None
        }

    def validate_prerequisites(self) -> None:
        """Validate that required files exist."""
        logger.info("Validating prerequisites...")

        if not self.vllm_script.exists():
            raise FileNotFoundError(f"vllm_infer.py not found: {self.vllm_script}")
        logger.info(f"  ✓ Found vllm_infer.py")

        dataset_info_path = self.project_root / "data" / "dataset_info.json"
        if not dataset_info_path.exists():
            raise FileNotFoundError(f"dataset_info.json not found: {dataset_info_path}")
        logger.info(f"  ✓ Found dataset_info.json")

    def run_inference(self, model_name: str, adapter_path: str = None) -> Path:
        """
        Run inference using vllm_infer.py.

        Args:
            model_name: Name identifier for the model ("base" or "lora")
            adapter_path: Path to LoRA adapter (None for base model)

        Returns:
            Path: Path to generated JSONL file
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Running inference: {model_name.upper()} MODEL")
        logger.info(f"{'='*70}")

        # Output file
        output_file = self.output_dir / f"predictions_{model_name}_{self.timestamp}.jsonl"

        # Build command
        cmd = [
            'python3',
            str(self.vllm_script),
            '--model_name_or_path', self.args.model_name_or_path,
            '--dataset', self.args.test_dataset,
            '--dataset_dir', self.args.dataset_dir,
            '--template', self.args.template,
            '--save_name', str(output_file),
            '--max_new_tokens', str(self.args.max_new_tokens),
            '--batch_size', str(self.args.batch_size),
            '--video_fps', str(self.args.video_fps),
            '--video_maxlen', str(self.args.video_maxlen),
            '--image_max_pixels', str(self.args.image_max_pixels),
            '--image_min_pixels', str(self.args.image_min_pixels),
            '--vllm_config', json.dumps({"gpu_memory_utilization": self.args.gpu_memory_utilization})
        ]

        # Add adapter if provided
        if adapter_path:
            cmd.extend(['--adapter_name_or_path', adapter_path])

        logger.info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode != 0:
                logger.error(f"Inference failed:\n{result.stderr}")
                raise RuntimeError(f"Inference failed for {model_name}")

            logger.info(f"✓ Inference completed successfully")
            logger.info(f"Output: {output_file}")

            return output_file

        except subprocess.TimeoutExpired:
            logger.error(f"Inference timed out for {model_name}")
            raise
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    def parse_predictions(self, jsonl_file: Path) -> List[Dict]:
        """
        Parse JSONL predictions file.

        Args:
            jsonl_file: Path to JSONL file

        Returns:
            List[Dict]: List of prediction entries
        """
        logger.info(f"Parsing predictions from: {jsonl_file.name}")

        predictions = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    predictions.append(entry)
                except json.JSONDecodeError as e:
                    logger.warning(f"  Line {line_num}: Failed to parse JSON - {e}")

        logger.info(f"  Parsed {len(predictions)} predictions")
        return predictions

    def evaluate_predictions(self, predictions: List[Dict]) -> Dict:
        """
        Evaluate predictions against ground truth labels.

        Args:
            predictions: List of prediction entries with 'predict' and 'label' fields

        Returns:
            Dict: Evaluation metrics and details
        """
        logger.info("Evaluating predictions...")

        results = {
            'total': 0,
            'correct': 0,
            'incorrect': 0,
            'moving': {'total': 0, 'correct': 0, 'incorrect': 0},
            'stopped': {'total': 0, 'correct': 0, 'incorrect': 0},
            'details': []
        }

        for idx, entry in enumerate(predictions):
            pred_text = entry.get('predict', '').lower().strip()
            label_text = entry.get('label', '').lower().strip()

            # Determine ground truth
            is_moving_gt = self._is_moving_label(label_text)

            # Determine prediction
            is_moving_pred = self._is_moving_prediction(pred_text)

            # Check correctness
            is_correct = (is_moving_gt == is_moving_pred)

            # Update statistics
            results['total'] += 1
            if is_correct:
                results['correct'] += 1
            else:
                results['incorrect'] += 1

            # Update class-specific stats
            class_key = 'moving' if is_moving_gt else 'stopped'
            results[class_key]['total'] += 1
            if is_correct:
                results[class_key]['correct'] += 1
            else:
                results[class_key]['incorrect'] += 1

            # Store details
            results['details'].append({
                'index': idx,
                'ground_truth': 'moving' if is_moving_gt else 'stopped',
                'prediction': 'moving' if is_moving_pred else 'stopped',
                'correct': is_correct,
                'pred_text': pred_text,
                'label_text': label_text
            })

        # Calculate accuracy
        results['accuracy'] = (results['correct'] / results['total'] * 100) if results['total'] > 0 else 0.0
        results['moving']['accuracy'] = (results['moving']['correct'] / results['moving']['total'] * 100) if results['moving']['total'] > 0 else 0.0
        results['stopped']['accuracy'] = (results['stopped']['correct'] / results['stopped']['total'] * 100) if results['stopped']['total'] > 0 else 0.0

        logger.info(f"  Overall Accuracy: {results['accuracy']:.2f}% ({results['correct']}/{results['total']})")
        logger.info(f"  Moving Accuracy: {results['moving']['accuracy']:.2f}% ({results['moving']['correct']}/{results['moving']['total']})")
        logger.info(f"  Stopped Accuracy: {results['stopped']['accuracy']:.2f}% ({results['stopped']['correct']}/{results['stopped']['total']})")

        return results

    def _is_moving_label(self, label_text: str) -> bool:
        """
        Determine if ground truth label indicates moving.

        Args:
            label_text: Ground truth label text

        Returns:
            bool: True if moving, False if stopped
        """
        # Check for positive indicators
        if 'moving' in label_text:
            return True
        elif 'stopped' in label_text or 'stationary' in label_text or 'not moving' in label_text:
            return False
        else:
            logger.warning(f"  Ambiguous ground truth label: '{label_text}' - assuming stopped")
            return False

    def _is_moving_prediction(self, pred_text: str) -> bool:
        """
        Determine if prediction indicates moving.

        Args:
            pred_text: Predicted text

        Returns:
            bool: True if moving, False if stopped
        """
        # Check for positive indicators (moving)
        if 'moving' in pred_text and 'not moving' not in pred_text:
            return True
        elif 'stopped' in pred_text or 'stationary' in pred_text or 'not moving' in pred_text:
            return False
        else:
            # Default to stopped if unclear
            return False

    def generate_report(self) -> None:
        """Generate comprehensive evaluation report."""
        logger.info("\n" + "="*70)
        logger.info("Generating evaluation report...")
        logger.info("="*70)

        report_path = self.output_dir / f"evaluation_report_{self.timestamp}.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("TREADMILL MOTION DETECTION - EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")

            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Dataset: {self.args.test_dataset}\n")
            f.write(f"Base Model: {self.args.model_name_or_path}\n")
            if self.args.adapter_name_or_path:
                f.write(f"LoRA Adapter: {self.args.adapter_name_or_path}\n")
            f.write("\n")

            # Base model results
            if self.results['base_model']:
                f.write("-"*70 + "\n")
                f.write("BASE MODEL RESULTS\n")
                f.write("-"*70 + "\n")
                self._write_model_results(f, self.results['base_model'])
                f.write("\n")

            # LoRA model results
            if self.results['lora_model']:
                f.write("-"*70 + "\n")
                f.write("FINE-TUNED MODEL (LoRA) RESULTS\n")
                f.write("-"*70 + "\n")
                self._write_model_results(f, self.results['lora_model'])
                f.write("\n")

            # Comparison
            if self.results['base_model'] and self.results['lora_model']:
                f.write("="*70 + "\n")
                f.write("COMPARISON\n")
                f.write("="*70 + "\n\n")

                base_acc = self.results['base_model']['accuracy']
                lora_acc = self.results['lora_model']['accuracy']
                improvement = lora_acc - base_acc

                f.write(f"Base Model Accuracy:       {base_acc:.2f}%\n")
                f.write(f"Fine-Tuned Model Accuracy: {lora_acc:.2f}%\n")
                f.write(f"Improvement:               {improvement:+.2f}%\n\n")

                if improvement > 0:
                    f.write("Status: ✓ Fine-tuning IMPROVED performance\n")
                elif improvement < 0:
                    f.write("Status: ✗ Fine-tuning DEGRADED performance\n")
                else:
                    f.write("Status: = No change in performance\n")
                f.write("\n")

            f.write("="*70 + "\n")
            f.write(f"Full report saved to: {report_path}\n")
            f.write("="*70 + "\n")

        # Print report to console
        with open(report_path, 'r') as f:
            print("\n" + f.read())

    def _write_model_results(self, f, results: Dict) -> None:
        """Write model results to file."""
        f.write(f"Overall Accuracy: {results['accuracy']:.2f}% ({results['correct']}/{results['total']})\n\n")

        f.write(f"Moving Videos:\n")
        f.write(f"  Total: {results['moving']['total']}\n")
        f.write(f"  Correct: {results['moving']['correct']}\n")
        f.write(f"  Accuracy: {results['moving']['accuracy']:.2f}%\n\n")

        f.write(f"Stopped Videos:\n")
        f.write(f"  Total: {results['stopped']['total']}\n")
        f.write(f"  Correct: {results['stopped']['correct']}\n")
        f.write(f"  Accuracy: {results['stopped']['accuracy']:.2f}%\n\n")

        # Show incorrect predictions
        incorrect = [d for d in results['details'] if not d['correct']]
        if incorrect:
            f.write(f"Incorrect Predictions ({len(incorrect)}):\n")
            for detail in incorrect[:10]:  # Show first 10
                f.write(f"  #{detail['index']}: GT={detail['ground_truth']}, Pred={detail['prediction']}\n")
                f.write(f"    Prediction: {detail['pred_text'][:100]}\n")
            if len(incorrect) > 10:
                f.write(f"  ... and {len(incorrect) - 10} more\n")

    def run(self) -> None:
        """Execute complete evaluation pipeline."""
        try:
            logger.info("="*70)
            logger.info("Starting Treadmill Model Evaluation")
            logger.info("="*70)

            # Validate
            self.validate_prerequisites()

            # Run base model inference
            logger.info("\n### STEP 1: Base Model Evaluation ###")
            base_predictions_file = self.run_inference("base", adapter_path=None)
            base_predictions = self.parse_predictions(base_predictions_file)
            self.results['base_model'] = self.evaluate_predictions(base_predictions)

            # Run LoRA model inference if adapter provided
            if self.args.adapter_name_or_path:
                logger.info("\n### STEP 2: Fine-Tuned Model Evaluation ###")
                lora_predictions_file = self.run_inference("lora", adapter_path=self.args.adapter_name_or_path)
                lora_predictions = self.parse_predictions(lora_predictions_file)
                self.results['lora_model'] = self.evaluate_predictions(lora_predictions)

            # Generate report
            logger.info("\n### STEP 3: Generate Report ###")
            self.generate_report()

            logger.info("\n" + "="*70)
            logger.info("✓ Evaluation completed successfully!")
            logger.info("="*70)

        except KeyboardInterrupt:
            logger.warning("\n\nEvaluation interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"\n\nError during evaluation: {e}", exc_info=True)
            sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate treadmill motion detection models using vllm inference',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Model arguments
    parser.add_argument('--model_name_or_path', type=str,
                       default='Qwen/Qwen2.5-VL-3B-Instruct',
                       help='Base model name or path')
    parser.add_argument('--adapter_name_or_path', type=str, default=None,
                       help='Path to LoRA adapter (optional, for comparing base vs fine-tuned)')

    # Dataset arguments
    parser.add_argument('--test_dataset', type=str, required=True,
                       help='Test dataset name (must be in dataset_info.json)')
    parser.add_argument('--dataset_dir', type=str, default='data',
                       help='Dataset directory (default: data)')

    # Inference arguments
    parser.add_argument('--template', type=str, default='qwen2_vl',
                       help='Template name (default: qwen2_vl)')
    parser.add_argument('--max_new_tokens', type=int, default=128,
                       help='Maximum new tokens to generate (default: 128)')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for inference (default: 1024)')
    parser.add_argument('--video_fps', type=float, default=2.0,
                       help='Video FPS for processing (default: 2.0)')
    parser.add_argument('--video_maxlen', type=int, default=128,
                       help='Maximum video length (default: 128)')
    parser.add_argument('--image_max_pixels', type=int, default=768*768,
                       help='Maximum image pixels (default: 768*768)')
    parser.add_argument('--image_min_pixels', type=int, default=32*32,
                       help='Minimum image pixels (default: 32*32)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8,
                       help='GPU memory utilization for vllm (default: 0.8)')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results (default: evaluation_results)')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    evaluator = TreadmillEvaluator(args)
    evaluator.run()


if __name__ == '__main__':
    main()
