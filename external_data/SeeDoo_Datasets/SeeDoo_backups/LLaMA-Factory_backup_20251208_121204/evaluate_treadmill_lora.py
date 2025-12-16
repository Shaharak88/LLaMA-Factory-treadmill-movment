import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration as QwenModel
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from pathlib import Path
from datetime import datetime
import sys

# Class to capture all output
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"evaluation_results_{timestamp}.txt"
logger = Logger(output_file)
sys.stdout = logger

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

# ==================== CONFIGURATION ====================
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA_PATH = r"C:\Users\shaha\Desktop\Qwen2.5\LLaMA-Factory\saves\qwen2vl-treadmill-lora"

MOVING_DIR = r"C:\Users\shaha\Desktop\Qwen2.5\LLaMA-Factory\data\treadmill_videos\Treadmill_moving_test"
STOPPED_DIR = r"C:\Users\shaha\Desktop\Qwen2.5\LLaMA-Factory\data\treadmill_videos\Treadmill_stopped_test"

PROMPT = "Analyze this video. Is the treadmill belt moving or stopped?"

# ==================== LOAD BASE MODEL ====================
print("="*80)
print("LOADING BASE MODEL")
print("="*80)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Loading base model with 4-bit quantization...")
base_model = QwenModel.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(f"✓ Base model loaded - VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB\n")

# ==================== INFERENCE FUNCTION ====================
def test_video(model, processor, video_path, video_num, total_videos, expected_answer):
    """Run inference on a single video and return the response"""
    video_name = Path(video_path).name
    print(f"\n{'─'*80}")
    print(f"VIDEO {video_num}/{total_videos}: {video_name}")
    print(f"Expected answer: {expected_answer}")
    print(f"{'─'*80}")
    
    torch.cuda.empty_cache()
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video", 
                    "video": str(video_path),
                    "fps": 4.0,
                    "min_pixels": 224 * 224, 
                    "max_pixels": 384 * 384
                },
                {"type": "text", "text": PROMPT}
            ]
        }
    ]
    
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    print(f"\n🎥 Raw Video Data:")
    if video_inputs is not None and len(video_inputs) > 0:
        print(f"  Number of videos: {len(video_inputs)}")
        for i, video in enumerate(video_inputs):
            if hasattr(video, 'shape'):
                print(f"  Video {i} shape: {video.shape}")
                print(f"  → Frames (dimension 0): {video.shape[0]}")
            elif isinstance(video, list):
                print(f"  Video {i} is a list with {len(video)} frames")
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
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    print(f"Model answer: {output_text}")
    
    # Check if correct
    response_lower = output_text.lower()
    if expected_answer == "MOVING":
        is_correct = "moving" in response_lower and "stopped" not in response_lower
    else:  # STOPPED
        is_correct = "stopped" in response_lower or "stationary" in response_lower or "not moving" in response_lower
    
    if is_correct:
        print(f"Result: ✓ CORRECT")
    else:
        print(f"Result: ✗ INCORRECT")
    
    # Cleanup
    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()
    
    return output_text, is_correct


def evaluate_model(model, processor, model_name):
    """Evaluate model on all test videos"""
    print(f"\n{'#'*80}")
    print(f"# EVALUATING: {model_name}")
    print(f"{'#'*80}\n")
    
    results = {
        "moving": {"correct": 0, "total": 0, "details": []},
        "stopped": {"correct": 0, "total": 0, "details": []}
    }
    
    # Test moving videos
    print(f"\n{'='*80}")
    print(f"TESTING MOVING VIDEOS")
    print(f"Directory: {MOVING_DIR}")
    print(f"{'='*80}")
    
    moving_videos = sorted(list(Path(MOVING_DIR).glob("*.MOV")))
    results["moving"]["total"] = len(moving_videos)
    print(f"Found {len(moving_videos)} moving videos")
    
    for i, video_path in enumerate(moving_videos, 1):
        response, is_correct = test_video(
            model, processor, video_path, 
            i, len(moving_videos), 
            "MOVING"
        )
        results["moving"]["details"].append((video_path.name, response, is_correct))
        if is_correct:
            results["moving"]["correct"] += 1
    
    # Test stopped videos
    print(f"\n\n{'='*80}")
    print(f"TESTING STOPPED VIDEOS")
    print(f"Directory: {STOPPED_DIR}")
    print(f"{'='*80}")
    
    stopped_videos = sorted(list(Path(STOPPED_DIR).glob("*.MOV")))
    results["stopped"]["total"] = len(stopped_videos)
    print(f"Found {len(stopped_videos)} stopped videos")
    
    for i, video_path in enumerate(stopped_videos, 1):
        response, is_correct = test_video(
            model, processor, video_path, 
            i, len(stopped_videos), 
            "STOPPED"
        )
        results["stopped"]["details"].append((video_path.name, response, is_correct))
        if is_correct:
            results["stopped"]["correct"] += 1
    
    return results


def print_summary(results, model_name):
    """Print detailed summary of results"""
    print(f"\n\n{'='*80}")
    print(f"SUMMARY FOR: {model_name}")
    print(f"{'='*80}\n")
    
    total_correct = results["moving"]["correct"] + results["stopped"]["correct"]
    total_videos = results["moving"]["total"] + results["stopped"]["total"]
    accuracy = (total_correct / total_videos * 100) if total_videos > 0 else 0
    
    print(f"OVERALL ACCURACY: {accuracy:.1f}% ({total_correct}/{total_videos})")
    print()
    
    # Moving videos summary
    moving_accuracy = (results["moving"]["correct"] / results["moving"]["total"] * 100) if results["moving"]["total"] > 0 else 0
    print(f"MOVING Videos: {moving_accuracy:.1f}% ({results['moving']['correct']}/{results['moving']['total']})")
    for name, response, is_correct in results["moving"]["details"]:
        status = "✓" if is_correct else "✗"
        print(f"  {status} {name}: {response}")
    
    print()
    
    # Stopped videos summary
    stopped_accuracy = (results["stopped"]["correct"] / results["stopped"]["total"] * 100) if results["stopped"]["total"] > 0 else 0
    print(f"STOPPED Videos: {stopped_accuracy:.1f}% ({results['stopped']['correct']}/{results['stopped']['total']})")
    for name, response, is_correct in results["stopped"]["details"]:
        status = "✓" if is_correct else "✗"
        print(f"  {status} {name}: {response}")
    
    print(f"\n{'='*80}\n")
    
    return accuracy, total_correct, total_videos


# ==================== MAIN EVALUATION ====================
if __name__ == "__main__":
    print(f"\n{'#'*80}")
    print(f"# TREADMILL MOTION DETECTION EVALUATION")
    print(f"# Timestamp: {timestamp}")
    print(f"# Output file: {output_file}")
    print(f"{'#'*80}\n")
    
    all_results = {}
    
    try:
        # ==================== TEST 1: BASELINE MODEL ====================
        print(f"\n\n{'█'*80}")
        print(f"█ TEST 1: BASELINE MODEL (No LoRA)")
        print(f"{'█'*80}\n")
        
        baseline_results = evaluate_model(base_model, processor, "BASELINE MODEL")
        baseline_acc, baseline_correct, baseline_total = print_summary(baseline_results, "BASELINE MODEL")
        all_results["baseline"] = baseline_results
        
        # ==================== TEST 2: FINE-TUNED MODEL ====================
        print(f"\n\n{'█'*80}")
        print(f"█ TEST 2: FINE-TUNED MODEL (With LoRA)")
        print(f"{'█'*80}\n")
        
        print(f"Loading LoRA adapter from: {LORA_PATH}")
        lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)
        lora_model.eval()
        print(f"✓ LoRA adapter loaded successfully\n")
        
        lora_results = evaluate_model(lora_model, processor, "FINE-TUNED MODEL (LoRA)")
        lora_acc, lora_correct, lora_total = print_summary(lora_results, "FINE-TUNED MODEL (LoRA)")
        all_results["lora"] = lora_results
        
        # ==================== FINAL COMPARISON ====================
        print(f"\n\n{'='*80}")
        print(f"FINAL COMPARISON")
        print(f"{'='*80}\n")
        
        print(f"BASELINE MODEL:     {baseline_acc:.1f}% ({baseline_correct}/{baseline_total} correct)")
        print(f"FINE-TUNED MODEL:   {lora_acc:.1f}% ({lora_correct}/{lora_total} correct)")
        
        improvement = lora_acc - baseline_acc
        print(f"\nDifference:         {improvement:+.1f}%")
        
        if improvement > 0:
            print(f"Status:             ✓ Fine-tuning IMPROVED performance")
        elif improvement < 0:
            print(f"Status:             ✗ Fine-tuning DEGRADED performance")
        else:
            print(f"Status:             = No change in performance")
        
        print(f"\n{'='*80}")
        print(f"✓✓✓ EVALUATION COMPLETED SUCCESSFULLY")
        print(f"✓✓✓ All results saved to: {output_file}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n{'!'*80}")
        print(f"ERROR OCCURRED: {e}")
        print(f"Peak memory: {torch.cuda.max_memory_allocated(0)/1024**3:.2f} GB")
        print(f"{'!'*80}\n")
        import traceback
        traceback.print_exc()
    
    finally:
        logger.close()
        sys.stdout = logger.terminal