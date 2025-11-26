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
            'details': []
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
                        {"type": "text", "text": "Analyze this video. Is the treadmill belt moving or stopped?"}
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
            
            # Update stats
            results['total'] += 1
            if is_correct: results['correct'] += 1
            else: results['incorrect'] += 1
            
            cat = 'moving' if is_moving_gt else 'stopped'
            results[cat]['total'] += 1
            if is_correct: results[cat]['correct'] += 1
            
            results['details'].append({
                'index': i,
                'video': video_rel_path,
                'gt': 'moving' if is_moving_gt else 'stopped',
                'pred': 'moving' if is_moving_pred else 'stopped',
                'correct': is_correct,
                'output': output_text
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(data)} samples")

        # Calculate accuracy
        results['accuracy'] = (results['correct'] / results['total'] * 100) if results['total'] > 0 else 0.0
        return results

    def _is_moving(self, text: str) -> bool:
        """Determine if text indicates moving."""
        text = text.lower()
        if 'moving' in text and 'not moving' not in text:
            return True
        if 'stopped' in text or 'stationary' in text or 'not moving' in text:
            return False
        return False # Default to stopped if unclear

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
        f.write("-" * len(title) + "\n")
        f.write(f"Accuracy: {results['accuracy']:.2f}% ({results['correct']}/{results['total']})\n")
        f.write(f"Moving:   {results['moving']['correct']}/{results['moving']['total']}\n")
        f.write(f"Stopped:  {results['stopped']['correct']}/{results['stopped']['total']}\n\n")

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
