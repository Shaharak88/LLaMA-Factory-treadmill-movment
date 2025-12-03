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
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load dataset info
        self.dataset_info_path = self.project_root / args.dataset_dir / "dataset_info.json"
        if not self.dataset_info_path.exists():
            raise FileNotFoundError(f"dataset_info.json not found at {self.dataset_info_path}")

        with open(self.dataset_info_path, 'r') as f:
            self.dataset_info = json.load(f)

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
        match = re.search(r'treadmill_\d+_(.+?)_(left|right|up|down)_speed[\d.]+_angle(\d+)', filename)

        if match:
            texture = match.group(1)
            angle = f"angle{match.group(3)}"
            return {'texture': texture, 'angle': angle}
        else:
            logger.warning(f"Could not parse metadata from filename: {filename}")
            return {'texture': 'unknown', 'angle': 'unknown'}

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

    def evaluate(self, model, processor, data: List[Dict], model_name: str) -> Dict:
        """Run evaluation loop."""
        logger.info(f"Evaluating {model_name} on {len(data)} samples...")

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
            })
        }
        
        for i, item in enumerate(data):
            # Extract info
            video_rel_path = item['videos'][0]
            video_path = self.project_root / video_rel_path
            
            # Get GT label from assistant message
            assistant_msg = next(m for m in item['messages'] if m['role'] == 'assistant')
            gt_text = assistant_msg['content']
            is_moving_gt = self._is_moving(gt_text)
            
            # Prepare input
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
                        {"type": "text", "text": "Is there movement in the video? Answer only with yes or no."}
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

            # Parse metadata
            metadata = self.parse_video_metadata(video_rel_path)
            texture = metadata['texture']
            angle = metadata['angle']

            # VERBOSE LOGGING: Print EVERY video prediction
            video_filename = Path(video_rel_path).name
            gt_label = "MOVING (yes)" if is_moving_gt else "STOPPED (no)"
            pred_label = "MOVING (yes)" if is_moving_pred else "STOPPED (no)"
            status_icon = "✓" if is_correct else "✗"
            logger.info(f"  [{i+1}/{len(data)}] {status_icon} {video_filename}")
            logger.info(f"      Ground Truth: {gt_label}")
            logger.info(f"      Model Output: '{output_text}'")
            logger.info(f"      Predicted:    {pred_label}")
            logger.info(f"      Texture: {texture}, Angle: {angle}")
            logger.info(f"")

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
                'angle': angle
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

        return results

    def _is_moving(self, text: str) -> bool:
        """
        Determine if text indicates moving.
        Only checks for yes/no answers to match the training format.
        """
        text = text.lower().strip()

        # Check for "yes" indicating movement (primary check)
        if text.startswith('yes') or text == 'yes' or text == 'yes.':
            return True

        # Check for "no" indicating stopped (primary check)
        if text.startswith('no') or text == 'no' or text == 'no.':
            return False

        # Check for yes/no anywhere in the response (more robust)
        if 'yes' in text and 'no' not in text:
            return True

        if 'no' in text and 'yes' not in text:
            return False

        # Default to stopped if completely unclear
        logger.warning(f"Ambiguous answer (defaulting to 'no'): '{text}'")
        return False

    def generate_report(self, base_results, lora_results=None):
        """Generate text report."""
        report_path = self.output_dir / f"evaluation_report_{self.timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("TREADMILL MOTION DETECTION - SIMPLE EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Dataset: {self.args.test_dataset}\n\n")
            
            self._write_results(f, "BASE MODEL", base_results)
            
            if lora_results:
                self._write_results(f, "FINE-TUNED MODEL", lora_results)
                
                f.write("="*70 + "\n")
                f.write("COMPARISON\n")
                f.write("="*70 + "\n")
                imp = lora_results['accuracy'] - base_results['accuracy']
                f.write(f"Improvement: {imp:+.2f}%\n")
                
        logger.info(f"Report saved to {report_path}")
        with open(report_path, 'r') as f:
            print(f.read())

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
            data = self.load_test_data()
            
            # Evaluate Base Model
            logger.info("--- Evaluating Base Model ---")
            base_model, processor = self.load_model(self.args.model_name_or_path)
            base_results = self.evaluate(base_model, processor, data, "Base Model")
            del base_model
            torch.cuda.empty_cache()
            
            lora_results = None
            if self.args.adapter_name_or_path:
                logger.info("--- Evaluating LoRA Model ---")
                lora_model, _ = self.load_model(self.args.model_name_or_path, self.args.adapter_name_or_path)
                lora_results = self.evaluate(lora_model, processor, data, "LoRA Model")
                
            self.generate_report(base_results, lora_results)
            
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
