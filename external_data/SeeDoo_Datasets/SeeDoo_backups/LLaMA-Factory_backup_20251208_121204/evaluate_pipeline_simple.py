#!/usr/bin/env python3
"""
Simple Evaluation Pipeline - Treadmill Motion Detection
Replaces vllm with standard transformers inference.

Author: AI-Generated
Date: 2025-11-26
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration as QwenModel
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import re
import csv
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SimpleEvaluator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.project_root = Path(__file__).parent
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load dataset info first to get test dataset folder name
        self.dataset_info_path = self.project_root / args.dataset_dir / "dataset_info.json"
        if not self.dataset_info_path.exists():
            raise FileNotFoundError(f"dataset_info.json not found at {self.dataset_info_path}")

        with open(self.dataset_info_path, 'r') as f:
            self.dataset_info = json.load(f)

        # Create timestamped evaluation folder inside the test dataset directory
        # AND also create the legacy output_dir for backward compatibility
        if args.test_dataset in self.dataset_info:
            # Extract test dataset folder from file_name (e.g., "_exp_20251203_153238_test.json" -> "_exp_20251203_153238_test")
            test_file = self.dataset_info[args.test_dataset]["file_name"]
            test_folder = test_file.replace('.json', '')
            test_dataset_dir = self.project_root / args.dataset_dir / test_folder

            # Create timestamped evaluation folder inside test dataset directory (primary location)
            eval_folder_name = f"eval_{self.timestamp}"
            self.output_dir = test_dataset_dir / eval_folder_name
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created evaluation output directory: {self.output_dir}")

            # ALSO create the legacy output_dir for backward compatibility with experiment tracker
            self.legacy_output_dir = Path(args.output_dir)
            self.legacy_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Legacy output directory (for tracker compatibility): {self.legacy_output_dir}")
        else:
            # Fallback to original behavior if dataset not found (shouldn't happen)
            self.output_dir = Path(args.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.legacy_output_dir = self.output_dir  # Same location
            logger.warning(f"Test dataset '{args.test_dataset}' not found in dataset_info.json, using default output_dir")

    def parse_video_metadata(self, video_path: str) -> Dict[str, str]:
        """
        Parse texture and angle from video filename.
        Expected format: treadmill_XXXX_<texture>_<direction>_speed<X.X>_angle<X>_...
        Texture can be multi-word with underscores (e.g., subtle_gray_stripes, factory_dark_stripes)

        Returns:
            Dict with 'texture' and 'angle' keys
        """
        filename = Path(video_path).name

        # Pattern: treadmill_XXXX_<texture>_<direction>_speed<X.X>_angle<X>_...
        # Capture texture (can have underscores) until we hit a known direction
        # Directions: left, right, up, down
        match = re.search(r'treadmill_\d+_(.+?)_(left|right|up|down)_speed([\d.]+)_angle(\d+)', filename)

        if match:
            texture = match.group(1)
            speed = match.group(3)
            angle = f"angle{match.group(4)}"
            return {'texture': texture, 'angle': angle, 'speed': speed}
        else:
            logger.warning(f"Could not parse metadata from filename: {filename}")
            return {'texture': 'unknown', 'angle': 'unknown', 'speed': 'unknown'}

    def get_evaluation_prompt(self) -> str:
        """
        Get the evaluation prompt based on eval_method.

        Returns:
            str: The prompt to use for evaluation
        """
        if self.args.eval_method == 'moving_stopped':
            return "Is the treadmill belt moving or stopped?"
        else:  # default: yesno
            return "Is there movement in the video? Answer only with yes or no."

    def load_model(self, model_path: str, adapter_path: str = None):
        """Load model and processor."""
        logger.info(f"Loading base model: {model_path}")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = QwenModel.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        if adapter_path:
            logger.info(f"Loading LoRA adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            model.eval()
            
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        return model, processor

    def load_test_data(self) -> List[Dict]:
        """Load test data from dataset file."""
        if self.args.test_dataset not in self.dataset_info:
            raise ValueError(f"Dataset '{self.args.test_dataset}' not found in dataset_info.json")
            
        filename = self.dataset_info[self.args.test_dataset]["file_name"]
        file_path = self.project_root / self.args.dataset_dir / filename
        
        logger.info(f"Loading test data from: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        return data

    def evaluate(self, model, processor, data: List[Dict], model_name: str, base_model_path: str, adapter_path: str = None) -> Dict:
        """Run evaluation loop."""
        logger.info(f"Evaluating {model_name} on {len(data)} samples...")
        logger.info(f"Using evaluation method: {self.args.eval_method}")

        # Get the prompt for this evaluation method
        evaluation_prompt = self.get_evaluation_prompt()
        logger.info(f"Evaluation prompt: '{evaluation_prompt}'")

        # Open live evaluation log file
        model_name_for_file = "base" if "Base" in model_name else "finetuned"
        live_log_path = self.output_dir / f"live_evaluation_log_{model_name_for_file}_{self.timestamp}.txt"
        live_log_file = open(live_log_path, 'w', encoding='utf-8')

        # Write header to log file and print to console
        header_lines = [
            "=" * 80,
            f"MODEL: {model_name}",
            f"BASE MODEL PATH: {base_model_path}"
        ]
        if adapter_path:
            header_lines.append(f"ADAPTER PATH: {adapter_path}")
        header_lines.extend([
            "=" * 80,
            ""
        ])

        header_text = "\n".join(header_lines)
        print(header_text)
        live_log_file.write(header_text + "\n")
        live_log_file.flush()

        results = {
            'total': 0,
            'correct': 0,
            'incorrect': 0,
            'moving': {'total': 0, 'correct': 0},
            'stopped': {'total': 0, 'correct': 0},
            'details': [],
            # For F1 score calculation
            'y_true': [],
            'y_pred': [],
            # Per-texture metrics
            'per_texture': defaultdict(lambda: {
                'y_true': [], 'y_pred': [],
                'correct': 0, 'total': 0,
                'moving': {'total': 0, 'correct': 0},
                'stopped': {'total': 0, 'correct': 0}
            }),
            # Per-angle metrics
            'per_angle': defaultdict(lambda: {
                'y_true': [], 'y_pred': [],
                'correct': 0, 'total': 0,
                'moving': {'total': 0, 'correct': 0},
                'stopped': {'total': 0, 'correct': 0}
            }),
            # Per-video CSV data
            'per_video_data': []
        }
        
        for i, item in enumerate(data):
            # Extract info
            video_rel_path = item['videos'][0]
            video_path = self.project_root / video_rel_path
            
            # Get GT label from assistant message
            assistant_msg = next(m for m in item['messages'] if m['role'] == 'assistant')
            gt_text = assistant_msg['content']
            is_moving_gt = self._is_moving(gt_text)
            
            # Prepare input with dynamic prompt based on eval_method
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": str(video_path),
                            "fps": 4.0, # Consistent with previous logic
                            "min_pixels": 224 * 224,
                            "max_pixels": 384 * 384
                        },
                        {"type": "text", "text": evaluation_prompt}
                    ]
                }
            ]
            
            # Inference
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False
                )
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Scoring
            is_moving_pred = self._is_moving(output_text)
            is_correct = (is_moving_gt == is_moving_pred)

            # Parse metadata (includes speed now)
            metadata = self.parse_video_metadata(video_rel_path)
            texture = metadata['texture']
            angle = metadata['angle']
            speed = metadata['speed']

            # LIVE LOGGING: Print AND write to file for EVERY video
            # Take classification directly from eval script's _is_moving() logic
            video_filename = Path(video_rel_path).name
            gt_label = "moving" if is_moving_gt else "stopped"
            parsed_classification = "moving" if is_moving_pred else "stopped"
            status_icon = "✓ CORRECT" if is_correct else "✗ INCORRECT"

            # Build the log entry (works for BOTH yesno and moving_stopped eval methods)
            log_lines = [
                f"[{i+1}/{len(data)}] {status_icon}",
                f"Video Name: {video_filename}",
                f"Video Path (Relative): {video_rel_path}",
                f"Video Path (Absolute): {video_path}",
                f"Speed: {speed}",
                f"Label (Ground Truth): {gt_label}",
                f"Model Answer (Raw): {output_text}",
                f"Parsed Classification: {parsed_classification}",
                "-" * 80,
                ""
            ]

            log_text = "\n".join(log_lines)

            # Print to console
            print(log_text)

            # Write to file
            live_log_file.write(log_text + "\n")
            live_log_file.flush()  # Ensure immediate write to disk

            # Labels for F1 calculation: 1 = moving, 0 = stopped
            y_true_label = 1 if is_moving_gt else 0
            y_pred_label = 1 if is_moving_pred else 0

            # Update overall stats
            results['total'] += 1
            results['y_true'].append(y_true_label)
            results['y_pred'].append(y_pred_label)
            if is_correct: results['correct'] += 1
            else: results['incorrect'] += 1

            cat = 'moving' if is_moving_gt else 'stopped'
            results[cat]['total'] += 1
            if is_correct: results[cat]['correct'] += 1

            # Update per-texture stats
            results['per_texture'][texture]['total'] += 1
            results['per_texture'][texture]['y_true'].append(y_true_label)
            results['per_texture'][texture]['y_pred'].append(y_pred_label)
            if is_correct: results['per_texture'][texture]['correct'] += 1
            results['per_texture'][texture][cat]['total'] += 1
            if is_correct: results['per_texture'][texture][cat]['correct'] += 1

            # Update per-angle stats
            results['per_angle'][angle]['total'] += 1
            results['per_angle'][angle]['y_true'].append(y_true_label)
            results['per_angle'][angle]['y_pred'].append(y_pred_label)
            if is_correct: results['per_angle'][angle]['correct'] += 1
            results['per_angle'][angle][cat]['total'] += 1
            if is_correct: results['per_angle'][angle][cat]['correct'] += 1

            results['details'].append({
                'index': i,
                'video': video_rel_path,
                'gt': 'moving' if is_moving_gt else 'stopped',
                'pred': 'moving' if is_moving_pred else 'stopped',
                'correct': is_correct,
                'output': output_text,
                'texture': texture,
                'angle': angle,
                'speed': speed
            })

            # Store per-video data for CSV export
            results['per_video_data'].append({
                'video_path': video_rel_path,
                'prediction': 'moving' if is_moving_pred else 'stopped',
                'label': 'moving' if is_moving_gt else 'stopped',
                'speed': speed,
                'correct': is_correct,
                'model_output': output_text
            })

        # Calculate overall accuracy and F1 scores
        results['accuracy'] = (results['correct'] / results['total'] * 100) if results['total'] > 0 else 0.0

        if len(results['y_true']) > 0:
            # Overall metrics - use macro average to account for both classes equally
            results['f1_score'] = f1_score(results['y_true'], results['y_pred'], average='macro', zero_division=0) * 100
            results['precision'] = precision_score(results['y_true'], results['y_pred'], average='macro', zero_division=0) * 100
            results['recall'] = recall_score(results['y_true'], results['y_pred'], average='macro', zero_division=0) * 100

            # Per-class F1 scores
            f1_per_class = f1_score(results['y_true'], results['y_pred'], average=None, zero_division=0)
            if len(f1_per_class) == 2:
                results['f1_stopped'] = f1_per_class[0] * 100  # class 0 = stopped
                results['f1_moving'] = f1_per_class[1] * 100   # class 1 = moving

            # Calculate per-texture metrics
            for texture, tex_data in results['per_texture'].items():
                if len(tex_data['y_true']) > 0:
                    tex_data['accuracy'] = (tex_data['correct'] / tex_data['total'] * 100)
                    tex_data['f1_score'] = f1_score(tex_data['y_true'], tex_data['y_pred'], average='macro', zero_division=0) * 100
                    tex_data['precision'] = precision_score(tex_data['y_true'], tex_data['y_pred'], average='macro', zero_division=0) * 100
                    tex_data['recall'] = recall_score(tex_data['y_true'], tex_data['y_pred'], average='macro', zero_division=0) * 100

                    # Per-class F1 for texture
                    f1_tex_class = f1_score(tex_data['y_true'], tex_data['y_pred'], average=None, zero_division=0)
                    if len(f1_tex_class) == 2:
                        tex_data['f1_stopped'] = f1_tex_class[0] * 100
                        tex_data['f1_moving'] = f1_tex_class[1] * 100

            # Calculate per-angle metrics
            for angle, ang_data in results['per_angle'].items():
                if len(ang_data['y_true']) > 0:
                    ang_data['accuracy'] = (ang_data['correct'] / ang_data['total'] * 100)
                    ang_data['f1_score'] = f1_score(ang_data['y_true'], ang_data['y_pred'], average='macro', zero_division=0) * 100
                    ang_data['precision'] = precision_score(ang_data['y_true'], ang_data['y_pred'], average='macro', zero_division=0) * 100
                    ang_data['recall'] = recall_score(ang_data['y_true'], ang_data['y_pred'], average='macro', zero_division=0) * 100

                    # Per-class F1 for angle
                    f1_ang_class = f1_score(ang_data['y_true'], ang_data['y_pred'], average=None, zero_division=0)
                    if len(f1_ang_class) == 2:
                        ang_data['f1_stopped'] = f1_ang_class[0] * 100
                        ang_data['f1_moving'] = f1_ang_class[1] * 100

        # Close live evaluation log file with completion message
        completion_msg = f"\n{'='*80}\nEvaluation complete! Log saved to: {live_log_path}\n{'='*80}\n"
        print(completion_msg)
        live_log_file.write(completion_msg)
        live_log_file.close()
        logger.info(f"Live evaluation log saved to: {live_log_path}")

        return results

    def _is_moving(self, text: str) -> bool:
        """
        Determine if text indicates moving.
        Handles multiple response formats:
        - Yes/No format: "yes" (moving) or "no" (stopped)
        - Moving/Stopped format: "moving", "stopped", "stationary"
        - Negations: "not moving", "isn't moving", "is not moving"
        - Complete sentences: "The treadmill belt is moving/stopped/stationary"

        This function must correctly parse responses from both evaluation methods
        (yesno and moving_stopped) as well as ground truth labels.
        """
        text = text.lower().strip()

        # First, check for explicit negative indicators (highest priority)
        # These override everything else
        negative_indicators = [
            'not moving',
            'isn\'t moving',
            'is not moving',
            'stopped',
            'stationary',
            'not in motion',
            'no movement'
        ]
        for indicator in negative_indicators:
            if indicator in text:
                return False

        # Check for "no" response (from yes/no format)
        # Only if we haven't already found negative indicators
        if text.startswith('no') or text == 'no' or text == 'no.':
            return False

        # More flexible "no" detection, but avoid "no" within words
        if ' no ' in f' {text} ' or text.endswith(' no'):
            return False

        # Now check for positive indicators (moving)
        # "moving" keyword (most common for moving_stopped eval method)
        if 'moving' in text:
            return True

        # "yes" response (from yes/no format)
        if text.startswith('yes') or text == 'yes' or text == 'yes.':
            return True

        # More flexible "yes" detection
        if ' yes ' in f' {text} ' or text.endswith(' yes'):
            return True

        # Check for motion-related keywords
        motion_keywords = ['in motion', 'is moving', 'belt is moving', 'movement']
        for keyword in motion_keywords:
            if keyword in text:
                return True

        # Default to stopped if completely ambiguous
        # This is safer than defaulting to moving
        logger.warning(f"Ambiguous response, defaulting to 'stopped': '{text}'")
        return False

    def save_per_video_csv(self, results: Dict, model_name: str) -> None:
        """
        Save per-video predictions to CSV file.

        Args:
            results: Evaluation results dictionary containing per_video_data
            model_name: Name of the model (e.g., "base" or "finetuned")
        """
        csv_path = self.output_dir / f"per_video_predictions_{model_name}_{self.timestamp}.csv"

        logger.info(f"Saving per-video predictions to: {csv_path}")

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['video_path', 'prediction', 'label', 'speed', 'correct', 'model_output']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results['per_video_data'])

        logger.info(f"  Saved {len(results['per_video_data'])} video predictions")

    def save_evaluation_metadata(self) -> None:
        """
        Save comprehensive evaluation metadata including all parameters and settings.
        This file documents every aspect of the evaluation run.
        """
        metadata_path = self.output_dir / f"evaluation_metadata_{self.timestamp}.txt"

        logger.info(f"Saving evaluation metadata to: {metadata_path}")

        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("EVALUATION METADATA - COMPLETE CONFIGURATION\n")
            f.write("="*80 + "\n\n")

            # Timestamp information
            f.write("TIMESTAMP INFORMATION:\n")
            f.write(f"  Evaluation Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Timestamp ID: {self.timestamp}\n\n")

            # Model configuration
            f.write("MODEL CONFIGURATION:\n")
            f.write(f"  Base Model Path: {self.args.model_name_or_path}\n")
            f.write(f"  Adapter Path: {self.args.adapter_name_or_path if self.args.adapter_name_or_path else 'None (Base Model Only)'}\n")
            f.write(f"  Template: {self.args.template}\n\n")

            # Dataset configuration
            f.write("DATASET CONFIGURATION:\n")
            f.write(f"  Test Dataset Name: {self.args.test_dataset}\n")
            f.write(f"  Dataset Directory: {self.args.dataset_dir}\n")
            if self.args.test_dataset in self.dataset_info:
                test_info = self.dataset_info[self.args.test_dataset]
                f.write(f"  Dataset File: {test_info['file_name']}\n")
                if 'formatting' in test_info:
                    f.write(f"  Dataset Formatting: {test_info['formatting']}\n")
                if 'columns' in test_info:
                    f.write(f"  Dataset Columns: {test_info['columns']}\n")
            f.write(f"  Dataset Info Path: {self.dataset_info_path}\n\n")

            # Evaluation configuration
            f.write("EVALUATION CONFIGURATION:\n")
            f.write(f"  Evaluation Method: {self.args.eval_method}\n")
            f.write(f"  Evaluation Prompt: \"{self.get_evaluation_prompt()}\"\n")
            f.write(f"  Max New Tokens: {self.args.max_new_tokens}\n")
            f.write(f"  Batch Size: {self.args.batch_size}\n\n")

            # Video processing configuration
            f.write("VIDEO PROCESSING CONFIGURATION:\n")
            f.write(f"  Video FPS: {self.args.video_fps}\n")
            f.write(f"  Video Max Length: {self.args.video_maxlen}\n")
            if self.args.image_max_pixels:
                f.write(f"  Image Max Pixels: {self.args.image_max_pixels}\n")
            if self.args.image_min_pixels:
                f.write(f"  Image Min Pixels: {self.args.image_min_pixels}\n")
            f.write(f"  Min Pixels (hardcoded): 224 * 224 = {224*224}\n")
            f.write(f"  Max Pixels (hardcoded): 384 * 384 = {384*384}\n")
            f.write(f"  FPS (hardcoded in inference): 4.0\n\n")

            # Quantization configuration
            f.write("QUANTIZATION CONFIGURATION:\n")
            f.write(f"  Load in 4-bit: True\n")
            f.write(f"  4-bit Compute Dtype: torch.float16\n")
            f.write(f"  4-bit Use Double Quant: True\n")
            f.write(f"  4-bit Quant Type: nf4\n\n")

            # GPU configuration
            f.write("GPU CONFIGURATION:\n")
            if self.args.gpu_memory_utilization:
                f.write(f"  GPU Memory Utilization: {self.args.gpu_memory_utilization}\n")
            else:
                f.write(f"  GPU Memory Utilization: Not specified (auto)\n")
            f.write(f"  Device Map: auto\n")
            f.write(f"  Low CPU Memory Usage: True\n\n")

            # Output configuration
            f.write("OUTPUT CONFIGURATION:\n")
            f.write(f"  Output Directory: {self.output_dir}\n")
            f.write(f"  Project Root: {self.project_root}\n\n")

            # Inference configuration
            f.write("INFERENCE CONFIGURATION:\n")
            f.write(f"  Sampling: False (do_sample=False, deterministic greedy decoding)\n")
            f.write(f"  Trust Remote Code: True\n\n")

            # Parsing logic information
            f.write("PARSING LOGIC:\n")
            f.write(f"  Evaluation Method: {self.args.eval_method}\n")
            if self.args.eval_method == 'yesno':
                f.write("  Expected Model Responses: 'yes' (moving) or 'no' (stopped)\n")
                f.write("  Parsing Strategy: Check for 'yes'/'no' keywords (case-insensitive)\n")
            else:  # moving_stopped
                f.write("  Expected Model Responses: 'The treadmill belt is moving.' or 'The treadmill belt is stopped.'\n")
                f.write("  Parsing Strategy: Check for 'moving'/'stopped' keywords (case-insensitive)\n")
            f.write("  Fallback: Default to 'stopped' if parsing is ambiguous\n\n")

            # Git information (if available)
            try:
                import subprocess
                git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                                   cwd=self.project_root,
                                                   stderr=subprocess.DEVNULL).decode().strip()
                git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                    cwd=self.project_root,
                                                    stderr=subprocess.DEVNULL).decode().strip()
                git_status = subprocess.check_output(['git', 'status', '--short'],
                                                    cwd=self.project_root,
                                                    stderr=subprocess.DEVNULL).decode().strip()

                f.write("GIT INFORMATION:\n")
                f.write(f"  Branch: {git_branch}\n")
                f.write(f"  Commit: {git_commit}\n")
                if git_status:
                    f.write(f"  Status: MODIFIED (uncommitted changes present)\n")
                    f.write(f"  Modified Files:\n")
                    for line in git_status.split('\n')[:10]:  # Show first 10 modified files
                        f.write(f"    {line}\n")
                else:
                    f.write(f"  Status: CLEAN (no uncommitted changes)\n")
                f.write("\n")
            except:
                f.write("GIT INFORMATION:\n")
                f.write("  Not available\n\n")

            # System information
            f.write("SYSTEM INFORMATION:\n")
            f.write(f"  Python Version: {sys.version.split()[0]}\n")
            f.write(f"  PyTorch Version: {torch.__version__}\n")
            f.write(f"  CUDA Available: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"  CUDA Version: {torch.version.cuda}\n")
                f.write(f"  GPU Count: {torch.cuda.device_count()}\n")
                for i in range(torch.cuda.device_count()):
                    f.write(f"  GPU {i}: {torch.cuda.get_device_name(i)}\n")
            f.write("\n")

            # Command line arguments (complete record)
            f.write("COMMAND LINE ARGUMENTS (COMPLETE):\n")
            for arg, value in vars(self.args).items():
                f.write(f"  --{arg}: {value}\n")
            f.write("\n")

            f.write("="*80 + "\n")
            f.write("END OF METADATA\n")
            f.write("="*80 + "\n")

        logger.info(f"  Saved complete evaluation metadata")

    def generate_report(self, base_results, lora_results=None):
        """Generate text report in both primary and legacy locations."""
        report_content = self._generate_report_content(base_results, lora_results)

        # Save to primary location (inside test dataset folder)
        report_path = self.output_dir / f"evaluation_report_{self.timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logger.info(f"Report saved to {report_path}")

        # ALSO save to legacy location for experiment tracker compatibility
        if hasattr(self, 'legacy_output_dir') and self.legacy_output_dir != self.output_dir:
            legacy_report_path = self.legacy_output_dir / f"evaluation_report_{self.timestamp}.txt"
            with open(legacy_report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Report also saved to legacy location: {legacy_report_path}")

        # Print report
        print(report_content)

    def _generate_report_content(self, base_results, lora_results=None) -> str:
        """Generate report content as a string."""
        lines = []
        lines.append("="*70)
        lines.append("TREADMILL MOTION DETECTION - SIMPLE EVALUATION REPORT")
        lines.append("="*70)
        lines.append("")
        lines.append(f"Date: {datetime.now()}")
        lines.append(f"Dataset: {self.args.test_dataset}")
        # Add adapter information if evaluating a fine-tuned model
        if lora_results and self.args.adapter_name_or_path:
            adapter_path = self.args.adapter_name_or_path
            adapter_name = Path(adapter_path).name if '/' in adapter_path else adapter_path
            lines.append(f"Fine-Tuned Model Adapter Path: {adapter_path}")
            lines.append(f"Fine-Tuned Model Adapter Name: {adapter_name}")
        lines.append("")

        # Base model section
        lines.extend(self._format_results("BASE MODEL", base_results))

        # Fine-tuned model section
        if lora_results:
            lines.extend(self._format_results("FINE-TUNED MODEL", lora_results))
            lines.append("="*70)
            lines.append("COMPARISON")
            lines.append("="*70)
            imp = lora_results['accuracy'] - base_results['accuracy']
            lines.append(f"Improvement: {imp:+.2f}%")

        return "\n".join(lines)

    def _format_results(self, title: str, results: dict) -> list:
        """Format results section as list of lines."""
        lines = []
        lines.append(f"{title}")
        lines.append("=" * len(title))
        lines.append("")

        # Overall metrics
        lines.append("OVERALL METRICS:")
        lines.append(f"  Accuracy:  {results['accuracy']:.2f}% ({results['correct']}/{results['total']})")
        if 'f1_score' in results:
            lines.append(f"  F1 Score:  {results['f1_score']:.2f}%")
            lines.append(f"  Precision: {results['precision']:.2f}%")
            lines.append(f"  Recall:    {results['recall']:.2f}%")

        # Per-class metrics
        lines.append("")
        lines.append("PER-CLASS METRICS:")
        moving_line = f"  Moving:  {results['moving']['correct']}/{results['moving']['total']}"
        if 'f1_moving' in results:
            moving_acc = (results['moving']['correct'] / results['moving']['total'] * 100) if results['moving']['total'] > 0 else 0
            moving_line += f" (Acc: {moving_acc:.2f}%, F1: {results['f1_moving']:.2f}%)"
        lines.append(moving_line)

        stopped_line = f"  Stopped: {results['stopped']['correct']}/{results['stopped']['total']}"
        if 'f1_stopped' in results:
            stopped_acc = (results['stopped']['correct'] / results['stopped']['total'] * 100) if results['stopped']['total'] > 0 else 0
            stopped_line += f" (Acc: {stopped_acc:.2f}%, F1: {results['f1_stopped']:.2f}%)"
        lines.append(stopped_line)
        lines.append("")

        # Per-texture breakdown
        if 'per_texture' in results and len(results['per_texture']) > 0:
            lines.append("PER-TEXTURE BREAKDOWN:")
            for texture in sorted(results['per_texture'].keys()):
                tex_data = results['per_texture'][texture]
                lines.append(f"  {texture}:")
                lines.append(f"    Total: {tex_data['total']} videos")
                lines.append(f"    Accuracy: {tex_data.get('accuracy', 0):.2f}% ({tex_data['correct']}/{tex_data['total']})")
                if 'f1_score' in tex_data:
                    lines.append(f"    F1 Score: {tex_data['f1_score']:.2f}%")
                    lines.append(f"    Precision: {tex_data['precision']:.2f}%")
                    lines.append(f"    Recall: {tex_data['recall']:.2f}%")
                moving_line = f"    Moving: {tex_data['moving']['correct']}/{tex_data['moving']['total']}"
                if 'f1_moving' in tex_data:
                    moving_line += f" (F1: {tex_data['f1_moving']:.2f}%)"
                lines.append(moving_line)
                stopped_line = f"    Stopped: {tex_data['stopped']['correct']}/{tex_data['stopped']['total']}"
                if 'f1_stopped' in tex_data:
                    stopped_line += f" (F1: {tex_data['f1_stopped']:.2f}%)"
                lines.append(stopped_line)
                lines.append("")

        # Per-angle breakdown
        if 'per_angle' in results and len(results['per_angle']) > 0:
            lines.append("PER-ANGLE BREAKDOWN:")
            for angle in sorted(results['per_angle'].keys()):
                ang_data = results['per_angle'][angle]
                lines.append(f"  {angle}:")
                lines.append(f"    Total: {ang_data['total']} videos")
                lines.append(f"    Accuracy: {ang_data.get('accuracy', 0):.2f}% ({ang_data['correct']}/{ang_data['total']})")
                if 'f1_score' in ang_data:
                    lines.append(f"    F1 Score: {ang_data['f1_score']:.2f}%")
                    lines.append(f"    Precision: {ang_data['precision']:.2f}%")
                    lines.append(f"    Recall: {ang_data['recall']:.2f}%")
                moving_line = f"    Moving: {ang_data['moving']['correct']}/{ang_data['moving']['total']}"
                if 'f1_moving' in ang_data:
                    moving_line += f" (F1: {ang_data['f1_moving']:.2f}%)"
                lines.append(moving_line)
                stopped_line = f"    Stopped: {ang_data['stopped']['correct']}/{ang_data['stopped']['total']}"
                if 'f1_stopped' in ang_data:
                    stopped_line += f" (F1: {ang_data['f1_stopped']:.2f}%)"
                lines.append(stopped_line)
                lines.append("")

        return lines

    def _write_results(self, f, title, results):
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")

        # Overall metrics
        f.write("OVERALL METRICS:\n")
        f.write(f"  Accuracy:  {results['accuracy']:.2f}% ({results['correct']}/{results['total']})\n")
        if 'f1_score' in results:
            f.write(f"  F1 Score:  {results['f1_score']:.2f}%\n")
            f.write(f"  Precision: {results['precision']:.2f}%\n")
            f.write(f"  Recall:    {results['recall']:.2f}%\n")

        # Per-class metrics
        f.write("\nPER-CLASS METRICS:\n")
        f.write(f"  Moving:  {results['moving']['correct']}/{results['moving']['total']}")
        if 'f1_moving' in results:
            moving_acc = (results['moving']['correct'] / results['moving']['total'] * 100) if results['moving']['total'] > 0 else 0
            f.write(f" (Acc: {moving_acc:.2f}%, F1: {results['f1_moving']:.2f}%)")
        f.write("\n")

        f.write(f"  Stopped: {results['stopped']['correct']}/{results['stopped']['total']}")
        if 'f1_stopped' in results:
            stopped_acc = (results['stopped']['correct'] / results['stopped']['total'] * 100) if results['stopped']['total'] > 0 else 0
            f.write(f" (Acc: {stopped_acc:.2f}%, F1: {results['f1_stopped']:.2f}%)")
        f.write("\n\n")

        # Per-texture breakdown
        if 'per_texture' in results and len(results['per_texture']) > 0:
            f.write("PER-TEXTURE BREAKDOWN:\n")
            for texture in sorted(results['per_texture'].keys()):
                tex_data = results['per_texture'][texture]
                f.write(f"  {texture}:\n")
                f.write(f"    Total: {tex_data['total']} videos\n")
                f.write(f"    Accuracy: {tex_data.get('accuracy', 0):.2f}% ({tex_data['correct']}/{tex_data['total']})\n")
                if 'f1_score' in tex_data:
                    f.write(f"    F1 Score: {tex_data['f1_score']:.2f}%\n")
                    f.write(f"    Precision: {tex_data['precision']:.2f}%\n")
                    f.write(f"    Recall: {tex_data['recall']:.2f}%\n")
                f.write(f"    Moving: {tex_data['moving']['correct']}/{tex_data['moving']['total']}")
                if 'f1_moving' in tex_data:
                    f.write(f" (F1: {tex_data['f1_moving']:.2f}%)")
                f.write(f"\n    Stopped: {tex_data['stopped']['correct']}/{tex_data['stopped']['total']}")
                if 'f1_stopped' in tex_data:
                    f.write(f" (F1: {tex_data['f1_stopped']:.2f}%)")
                f.write("\n\n")

        # Per-angle breakdown
        if 'per_angle' in results and len(results['per_angle']) > 0:
            f.write("PER-ANGLE BREAKDOWN:\n")
            for angle in sorted(results['per_angle'].keys()):
                ang_data = results['per_angle'][angle]
                f.write(f"  {angle}:\n")
                f.write(f"    Total: {ang_data['total']} videos\n")
                f.write(f"    Accuracy: {ang_data.get('accuracy', 0):.2f}% ({ang_data['correct']}/{ang_data['total']})\n")
                if 'f1_score' in ang_data:
                    f.write(f"    F1 Score: {ang_data['f1_score']:.2f}%\n")
                    f.write(f"    Precision: {ang_data['precision']:.2f}%\n")
                    f.write(f"    Recall: {ang_data['recall']:.2f}%\n")
                f.write(f"    Moving: {ang_data['moving']['correct']}/{ang_data['moving']['total']}")
                if 'f1_moving' in ang_data:
                    f.write(f" (F1: {ang_data['f1_moving']:.2f}%)")
                f.write(f"\n    Stopped: {ang_data['stopped']['correct']}/{ang_data['stopped']['total']}")
                if 'f1_stopped' in ang_data:
                    f.write(f" (F1: {ang_data['f1_stopped']:.2f}%)")
                f.write("\n\n")

    def run(self):
        try:
            # Save evaluation metadata first
            self.save_evaluation_metadata()

            data = self.load_test_data()

            # Evaluate Base Model
            logger.info("--- Evaluating Base Model ---")
            base_model, processor = self.load_model(self.args.model_name_or_path)
            base_results = self.evaluate(base_model, processor, data, "Base Model", self.args.model_name_or_path)
            self.save_per_video_csv(base_results, "base")
            del base_model
            torch.cuda.empty_cache()

            lora_results = None
            if self.args.adapter_name_or_path:
                logger.info("--- Evaluating LoRA Model ---")
                lora_model, _ = self.load_model(self.args.model_name_or_path, self.args.adapter_name_or_path)
                lora_results = self.evaluate(lora_model, processor, data, "Fine-Tuned Model (LoRA)", self.args.model_name_or_path, self.args.adapter_name_or_path)
                self.save_per_video_csv(lora_results, "finetuned")

            self.generate_report(base_results, lora_results)

            logger.info("\n" + "="*80)
            logger.info(f"EVALUATION COMPLETE!")
            logger.info(f"All outputs saved to: {self.output_dir}")
            logger.info("="*80 + "\n")

        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            sys.exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Simple Evaluation Pipeline')
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--adapter_name_or_path', type=str, default=None)
    parser.add_argument('--test_dataset', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--eval_method', type=str, default='yesno',
                       choices=['yesno', 'moving_stopped'],
                       help='Evaluation method: "yesno" or "moving_stopped" (default: yesno)')
    # Ignored arguments for compatibility
    parser.add_argument('--template', type=str, default='qwen2_vl')
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--video_fps', type=float, default=2.0)
    parser.add_argument('--video_maxlen', type=int, default=128)
    parser.add_argument('--image_max_pixels', type=int)
    parser.add_argument('--image_min_pixels', type=int)
    parser.add_argument('--gpu_memory_utilization', type=float)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    evaluator = SimpleEvaluator(args)
    evaluator.run()
