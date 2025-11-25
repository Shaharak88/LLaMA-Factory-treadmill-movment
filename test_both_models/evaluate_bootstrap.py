import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration as QwenModel
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from pathlib import Path
from datetime import datetime
import sys
import numpy as np
from scipy import stats

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

    def isatty(self):
        return self.terminal.isatty()

    def close(self):
        self.log.close()

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"bootstrap_evaluation_{timestamp}.txt"
logger = Logger(output_file)
sys.stdout = logger

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

# ==================== CONFIGURATION ====================
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
# Docker absolute paths (works from any directory in container)
LORA_PATH = "/app/saves/qwen2vl-treadmill-lora"

MOVING_DIR = "/app/data/treadmill_videos/Treadmill_moving_test"
STOPPED_DIR = "/app/data/treadmill_videos/Treadmill_stopped_test"

PROMPT = "Analyze this video. Is the treadmill belt moving or stopped?"

# Number of bootstrap runs
NUM_RUNS = 2

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
def test_video(model, processor, video_path, video_num, total_videos, expected_answer, verbose=False):
    """Run inference on a single video and return the response"""
    video_name = Path(video_path).name

    if verbose:
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

    if verbose:
        print(f"Model answer: {output_text}")

    # Check if correct
    response_lower = output_text.lower()
    if expected_answer == "MOVING":
        is_correct = "moving" in response_lower and "stopped" not in response_lower
    else:  # STOPPED
        is_correct = "stopped" in response_lower or "stationary" in response_lower or "not moving" in response_lower

    if verbose:
        if is_correct:
            print(f"Result: ✓ CORRECT")
        else:
            print(f"Result: ✗ INCORRECT")

    # Cleanup
    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()

    return output_text, is_correct


def evaluate_model_single_run(model, processor, model_name, run_number, verbose=False):
    """Evaluate model on all test videos for a single run"""
    if verbose:
        print(f"\n{'#'*80}")
        print(f"# EVALUATING: {model_name} - RUN {run_number}/{NUM_RUNS}")
        print(f"{'#'*80}\n")
    else:
        print(f"  Run {run_number}/{NUM_RUNS}...", end=" ", flush=True)

    results = {
        "moving": {"correct": 0, "total": 0},
        "stopped": {"correct": 0, "total": 0}
    }

    # Test moving videos
    moving_videos = sorted(list(Path(MOVING_DIR).glob("*.MOV")))
    results["moving"]["total"] = len(moving_videos)

    for i, video_path in enumerate(moving_videos, 1):
        response, is_correct = test_video(
            model, processor, video_path,
            i, len(moving_videos),
            "MOVING",
            verbose=verbose
        )
        if is_correct:
            results["moving"]["correct"] += 1

    # Test stopped videos
    stopped_videos = sorted(list(Path(STOPPED_DIR).glob("*.MOV")))
    results["stopped"]["total"] = len(stopped_videos)

    for i, video_path in enumerate(stopped_videos, 1):
        response, is_correct = test_video(
            model, processor, video_path,
            i, len(stopped_videos),
            "STOPPED",
            verbose=verbose
        )
        if is_correct:
            results["stopped"]["correct"] += 1

    # Calculate accuracy
    total_correct = results["moving"]["correct"] + results["stopped"]["correct"]
    total_videos = results["moving"]["total"] + results["stopped"]["total"]
    accuracy = (total_correct / total_videos * 100) if total_videos > 0 else 0

    if not verbose:
        print(f"{accuracy:.1f}%")

    return accuracy, results


def calculate_statistics(accuracies, model_name):
    """Calculate and display statistical metrics"""
    accuracies = np.array(accuracies)

    print(f"\n{'='*80}")
    print(f"STATISTICAL SUMMARY FOR: {model_name}")
    print(f"{'='*80}\n")

    print(f"Number of runs:        {len(accuracies)}")
    print(f"Individual accuracies: {', '.join([f'{acc:.1f}%' for acc in accuracies])}")
    print()

    # Basic statistics
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)  # Sample standard deviation
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)

    print(f"Mean accuracy:         {mean_acc:.2f}%")
    print(f"Standard deviation:    {std_acc:.2f}%")
    print(f"Min accuracy:          {min_acc:.1f}%")
    print(f"Max accuracy:          {max_acc:.1f}%")

    # 95% Confidence interval
    # 95% Confidence interval
    if len(accuracies) > 1:
        sem = stats.sem(accuracies)
        if sem > 0:
            ci = stats.t.interval(
                confidence=0.95,
                df=len(accuracies)-1,
                loc=mean_acc,
                scale=sem
            )
            print(f"95% Confidence Int:    [{ci[0]:.2f}%, {ci[1]:.2f}%]")
        else:
            print(f"95% Confidence Int:    [{mean_acc:.2f}%, {mean_acc:.2f}%] (no variance)")

    print(f"{'='*80}\n")

    return mean_acc, std_acc, min_acc, max_acc


def compare_models(baseline_accuracies, lora_accuracies):
    """Compare two models statistically"""
    print(f"\n{'='*80}")
    print(f"STATISTICAL COMPARISON")
    print(f"{'='*80}\n")

    baseline_arr = np.array(baseline_accuracies)
    lora_arr = np.array(lora_accuracies)

    baseline_mean = np.mean(baseline_arr)
    lora_mean = np.mean(lora_arr)

    improvement = lora_mean - baseline_mean

    print(f"Baseline mean:         {baseline_mean:.2f}%")
    print(f"Fine-tuned mean:       {lora_mean:.2f}%")
    print(f"Mean improvement:      {improvement:+.2f}%")
    print()

    # Paired t-test (since we're comparing same videos across runs)
    if len(baseline_arr) > 1:
        t_statistic, p_value = stats.ttest_rel(lora_arr, baseline_arr)

        print(f"Paired t-test results:")
        print(f"  t-statistic:         {t_statistic:.4f}")
        print(f"  p-value:             {p_value:.4f}")
        print()

        # Explain p-value
        print("What does the p-value mean?")
        if p_value < 0.001:
            print(f"  p < 0.001: *** HIGHLY SIGNIFICANT ***")
            print(f"  → The improvement is almost certainly real, not due to chance")
        elif p_value < 0.01:
            print(f"  p < 0.01: ** VERY SIGNIFICANT **")
            print(f"  → Strong evidence the improvement is real")
        elif p_value < 0.05:
            print(f"  p < 0.05: * SIGNIFICANT *")
            print(f"  → Good evidence the improvement is real")
        elif p_value < 0.10:
            print(f"  p < 0.10: Marginally significant")
            print(f"  → Weak evidence of improvement")
        else:
            print(f"  p ≥ 0.10: NOT SIGNIFICANT")
            print(f"  → Difference could be due to random chance")

        print()
        print("In other words:")
        print(f"  There is a {p_value*100:.2f}% chance that this difference")
        print(f"  is just random variation (not a real improvement).")

    else:
        print("Note: Need more than 1 run for statistical significance testing")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(baseline_arr, ddof=1) + np.var(lora_arr, ddof=1)) / 2)
    if pooled_std > 0:
        cohens_d = (lora_mean - baseline_mean) / pooled_std
        print()
        print(f"Effect size (Cohen's d): {cohens_d:.3f}")
        if abs(cohens_d) < 0.2:
            print(f"  → Small effect size")
        elif abs(cohens_d) < 0.5:
            print(f"  → Medium effect size")
        else:
            print(f"  → Large effect size")
    else:
        print()
        print(f"Effect size (Cohen's d): Cannot calculate (zero variance)")
        print(f"  → All measurements are identical")

        print(f"\n{'='*80}\n")

    return improvement, p_value if len(baseline_arr) > 1 else None


# ==================== MAIN EVALUATION ====================
if __name__ == "__main__":
    print(f"\n{'#'*80}")
    print(f"# BOOTSTRAP EVALUATION - MULTIPLE RUNS")
    print(f"# Timestamp: {timestamp}")
    print(f"# Output file: {output_file}")
    print(f"# Number of runs per model: {NUM_RUNS}")
    print(f"{'#'*80}\n")

    baseline_accuracies = []
    lora_accuracies = []

    try:
        # ==================== BASELINE MODEL RUNS ====================
        print(f"\n{'█'*80}")
        print(f"█ BASELINE MODEL EVALUATION ({NUM_RUNS} runs)")
        print(f"{'█'*80}\n")

        for run in range(1, NUM_RUNS + 1):
            accuracy, results = evaluate_model_single_run(
                base_model,
                processor,
                "BASELINE MODEL",
                run,
                verbose=False
            )
            baseline_accuracies.append(accuracy)

        baseline_mean, baseline_std, baseline_min, baseline_max = calculate_statistics(
            baseline_accuracies,
            "BASELINE MODEL"
        )

        # ==================== FINE-TUNED MODEL RUNS ====================
        print(f"\n{'█'*80}")
        print(f"█ FINE-TUNED MODEL EVALUATION ({NUM_RUNS} runs)")
        print(f"{'█'*80}\n")

        print(f"Loading LoRA adapter from: {LORA_PATH}")
        lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)
        lora_model.eval()
        print(f"✓ LoRA adapter loaded successfully\n")

        for run in range(1, NUM_RUNS + 1):
            accuracy, results = evaluate_model_single_run(
                lora_model,
                processor,
                "FINE-TUNED MODEL",
                run,
                verbose=False
            )
            lora_accuracies.append(accuracy)

        lora_mean, lora_std, lora_min, lora_max = calculate_statistics(
            lora_accuracies,
            "FINE-TUNED MODEL (LoRA)"
        )

        # ==================== STATISTICAL COMPARISON ====================
        improvement, p_value = compare_models(baseline_accuracies, lora_accuracies)

        # ==================== FINAL SUMMARY ====================
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY")
        print(f"{'='*80}\n")

        print(f"BASELINE MODEL:")
        print(f"  Mean: {baseline_mean:.2f}% ± {baseline_std:.2f}%")
        print(f"  Range: [{baseline_min:.1f}%, {baseline_max:.1f}%]")
        print()

        print(f"FINE-TUNED MODEL:")
        print(f"  Mean: {lora_mean:.2f}% ± {lora_std:.2f}%")
        print(f"  Range: [{lora_min:.1f}%, {lora_max:.1f}%]")
        print()

        print(f"IMPROVEMENT: {improvement:+.2f}%")

        if p_value is not None:
            if p_value < 0.05:
                print(f"STATUS: ✓✓✓ STATISTICALLY SIGNIFICANT (p={p_value:.4f})")
            else:
                print(f"STATUS: ⚠ Not statistically significant (p={p_value:.4f})")

        print(f"\n{'='*80}")
        print(f"✓✓✓ BOOTSTRAP EVALUATION COMPLETED")
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
