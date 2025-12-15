# Evaluation Pipeline

## Purpose
The evaluation pipeline assesses model performance on treadmill motion detection by:
1. Loading test dataset videos
2. Running inference with base model (Qwen2.5-VL) and fine-tuned model (LoRA adapter)
3. Parsing model outputs to classify motion (moving vs stopped)
4. Calculating comprehensive metrics (accuracy, F1, precision, recall)
5. Generating detailed reports with per-video, per-texture, and per-angle breakdowns

## Location
`/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/evaluate_pipeline_simple.py`

---

## Architecture

```
SimpleEvaluator (main class)
    │
    ├──> __init__(args)
    │    ├──> Load dataset_info.json
    │    ├──> Create output directories
    │    └──> Initialize paths
    │
    ├──> run()
    │    │
    │    ├──> save_evaluation_metadata() → Complete configuration record
    │    │
    │    ├──> load_test_data() → Load test dataset JSON
    │    │
    │    ├──> Evaluate Base Model:
    │    │    ├──> load_model() → Load Qwen2.5-VL (4-bit quantization)
    │    │    ├──> evaluate() → Inference loop
    │    │    │    ├──> For each video:
    │    │    │    │    ├──> Load video and preprocess
    │    │    │    │    ├──> Generate model response
    │    │    │    │    ├──> Parse response (_is_moving)
    │    │    │    │    ├──> Compare with ground truth
    │    │    │    │    ├──> Update metrics
    │    │    │    │    └──> Log results (live log file)
    │    │    │    ├──> Calculate aggregate metrics
    │    │    │    └──> Return results dictionary
    │    │    └──> save_per_video_csv() → CSV with predictions
    │    │
    │    ├──> Evaluate Fine-Tuned Model:
    │    │    ├──> load_model() with adapter → Load base + LoRA
    │    │    ├──> evaluate() → Same inference loop
    │    │    └──> save_per_video_csv()
    │    │
    │    └──> generate_report() → Text report with all metrics
    │
    └──> Output Files:
         ├── evaluation_metadata_{timestamp}.txt
         ├── live_evaluation_log_base_{timestamp}.txt
         ├── live_evaluation_log_finetuned_{timestamp}.txt
         ├── per_video_predictions_base_{timestamp}.csv
         ├── per_video_predictions_finetuned_{timestamp}.csv
         └── evaluation_report_{timestamp}.txt
```

---

## Key Components

### 1. Initialization

**Function: `__init__(args)`**

Sets up evaluation environment and paths.

**Loading dataset info:**
```python
self.dataset_info_path = self.project_root / args.dataset_dir / "dataset_info.json"
with open(self.dataset_info_path, 'r') as f:
    self.dataset_info = json.load(f)
```

**Creating output directories:**
```python
# Primary location: inside test dataset folder
test_file = self.dataset_info[args.test_dataset]["file_name"]
test_folder = test_file.replace('.json', '')  # e.g., "_exp_20251207_143033_test"
test_dataset_dir = self.project_root / args.dataset_dir / test_folder

# Create timestamped evaluation folder
eval_folder_name = f"eval_{self.timestamp}"
self.output_dir = test_dataset_dir / eval_folder_name
self.output_dir.mkdir(parents=True, exist_ok=True)
```

**Example paths:**
- Dataset: `data/_exp_20251207_143033_test/`
- Evaluation: `data/_exp_20251207_143033_test/eval_20251207_150322/`

**Why store evaluation results with test data?**
- Co-locates videos with their evaluation results
- Easy to find evaluation for a specific dataset
- Supports multiple evaluations of same dataset (different timestamps)

**Legacy compatibility:**
```python
# ALSO create legacy output_dir for experiment tracker compatibility
self.legacy_output_dir = Path(args.output_dir)  # evaluation_results/
self.legacy_output_dir.mkdir(parents=True, exist_ok=True)
```

**Why maintain legacy location?**
- Experiment tracker expects results in `evaluation_results/`
- Backward compatibility with existing scripts
- Reports are duplicated to both locations

### 2. Model Loading

**Function: `load_model(model_path, adapter_path=None)`**

Loads model with 4-bit quantization for memory efficiency.

**Quantization configuration:**
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                        # Use 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,     # Compute in float16
    bnb_4bit_use_double_quant=True,           # Double quantization (better accuracy)
    bnb_4bit_quant_type="nf4"                 # NormalFloat4 quantization
)
```

**Loading base model:**
```python
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,                              # "Qwen/Qwen2.5-VL-7B-Instruct"
    quantization_config=quantization_config,
    device_map="auto",                       # Automatic device placement
    low_cpu_mem_usage=True,                  # Minimize CPU memory
    trust_remote_code=True                   # Allow custom code from model
)
```

**Loading LoRA adapter (for fine-tuned model):**
```python
if adapter_path:
    logger.info(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()  # Set to evaluation mode
```

**Loading processor:**
```python
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)
```

**Processor's role:**
- Tokenizes text prompts
- Processes video frames (resizing, normalization)
- Formats inputs for model

### 3. Test Data Loading

**Function: `load_test_data()`**

Loads test dataset JSON file.

```python
filename = self.dataset_info[self.args.test_dataset]["file_name"]
file_path = self.project_root / self.args.dataset_dir / filename

with open(file_path, 'r') as f:
    data = json.load(f)

return data  # List of samples
```

**Sample structure:**
```json
{
    "messages": [
        {
            "role": "user",
            "content": "Is there movement in the video? Answer only with yes or no."
        },
        {
            "role": "assistant",
            "content": "yes"
        }
    ],
    "videos": [
        "data/_exp_20251207_143033_test/treadmill_0042_subtle_gray_stripes_right_speed3.0_angle52_dist3.0_bright0.00_contr1.00_640x480_seed87.mp4"
    ]
}
```

### 4. Evaluation Prompts

**Function: `get_evaluation_prompt()`**

Returns prompt based on evaluation method.

```python
if self.args.eval_method == 'moving_stopped':
    return "Is the treadmill belt moving or stopped?"
else:  # default: yesno
    return "Is there movement in the video? Answer only with yes or no."
```

**Evaluation method comparison:**

| Method | Prompt | Expected Responses | Use Case |
|--------|--------|-------------------|----------|
| **yesno** | "Is there movement in the video? Answer only with yes or no." | "yes", "no" | Simpler, binary classification |
| **moving_stopped** | "Is the treadmill belt moving or stopped?" | "The treadmill belt is moving.", "The treadmill belt is stopped." | More explicit, domain-specific |

**Why two evaluation methods?**
- Tests model's ability to follow different instruction formats
- "yesno" is more general (works for any motion detection)
- "moving_stopped" is more specific (explicitly asks about treadmill)

### 5. Inference Loop

**Function: `evaluate(model, processor, data, model_name, base_model_path, adapter_path=None)`**

Main evaluation loop that processes all test videos.

**Step-by-step process for each video:**

**Step 1: Load video and ground truth**
```python
video_rel_path = item['videos'][0]
video_path = self.project_root / video_rel_path

# Get ground truth from assistant message
assistant_msg = next(m for m in item['messages'] if m['role'] == 'assistant')
gt_text = assistant_msg['content']  # "yes" or "no"
is_moving_gt = self._is_moving(gt_text)  # True or False
```

**Step 2: Prepare input**
```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": str(video_path),
                "fps": 4.0,                    # Sample 4 frames per second
                "min_pixels": 224 * 224,       # Minimum resolution
                "max_pixels": 384 * 384        # Maximum resolution
            },
            {"type": "text", "text": evaluation_prompt}
        ]
    }
]
```

**Video processing parameters:**
- `fps=4.0`: Sample 4 frames per second from video
  - Example: 3-second video at 30 fps → 12 frames selected
- `min_pixels=224*224`: Minimum resolution (50,176 pixels)
- `max_pixels=384*384`: Maximum resolution (147,456 pixels)
- Videos are resized to fit within these bounds while preserving aspect ratio

**Why fps=4.0?**
- Balance between information and memory
- 4 fps captures enough motion information
- Higher fps = more frames = more memory

**Step 3: Tokenize input**
```python
# Apply chat template
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  # Adds "<|assistant|>" prompt
)

# Process video frames
image_inputs, video_inputs = process_vision_info(messages)

# Tokenize everything
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(model.device)
```

**What `process_vision_info` does:**
1. Loads video file
2. Extracts frames at specified fps
3. Resizes frames to fit pixel constraints
4. Normalizes pixel values
5. Stacks frames into tensor

**Step 4: Generate prediction**
```python
with torch.no_grad():  # Disable gradient computation
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,  # Max response length
        do_sample=False      # Deterministic (greedy decoding)
    )
```

**Decoding strategy:**
- `do_sample=False`: Greedy decoding (always pick most likely token)
- Deterministic: Same input → same output
- Alternative: `do_sample=True` with temperature for randomness

**Step 5: Decode output**
```python
# Remove input tokens from output (only keep generated tokens)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# Decode to text
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]
```

**Example outputs:**
- "yes"
- "no"
- "Yes."
- "The treadmill belt is moving."
- "The belt is stopped."

**Step 6: Parse output**
```python
is_moving_pred = self._is_moving(output_text)
is_correct = (is_moving_gt == is_moving_pred)
```

See "Response Parsing" section below for details on `_is_moving()`.

**Step 7: Extract metadata**
```python
metadata = self.parse_video_metadata(video_rel_path)
texture = metadata['texture']     # e.g., "subtle_gray_stripes"
angle = metadata['angle']         # e.g., "angle52"
speed = metadata['speed']         # e.g., "3.0"
```

**Metadata parsing regex:**
```python
# Pattern: treadmill_XXXX_<texture>_<direction>_speed<X.X>_angle<X>_...
match = re.search(r'treadmill_\d+_(.+?)_(left|right|up|down)_speed([\d.]+)_angle(\d+)', filename)
texture = match.group(1)
speed = match.group(3)
angle = f"angle{match.group(4)}"
```

**Step 8: Live logging**
```python
log_lines = [
    f"[{i+1}/{len(data)}] {'✓ CORRECT' if is_correct else '✗ INCORRECT'}",
    f"Video Name: {video_filename}",
    f"Video Path (Relative): {video_rel_path}",
    f"Speed: {speed}",
    f"Label (Ground Truth): {'moving' if is_moving_gt else 'stopped'}",
    f"Model Answer (Raw): {output_text}",
    f"Parsed Classification: {'moving' if is_moving_pred else 'stopped'}",
    "-" * 80,
    ""
]

log_text = "\n".join(log_lines)
print(log_text)  # Print to console
live_log_file.write(log_text + "\n")  # Write to file
live_log_file.flush()  # Immediate disk write
```

**Why live logging?**
- Real-time monitoring of evaluation progress
- Immediate feedback on model performance
- Debugging tool (see where model makes mistakes)
- Permanent record (log file saved to disk)

**Step 9: Update metrics**
```python
# Overall metrics
results['total'] += 1
results['y_true'].append(1 if is_moving_gt else 0)
results['y_pred'].append(1 if is_moving_pred else 0)
if is_correct:
    results['correct'] += 1
else:
    results['incorrect'] += 1

# Per-class metrics
category = 'moving' if is_moving_gt else 'stopped'
results[category]['total'] += 1
if is_correct:
    results[category]['correct'] += 1

# Per-texture metrics
results['per_texture'][texture]['total'] += 1
results['per_texture'][texture]['y_true'].append(1 if is_moving_gt else 0)
results['per_texture'][texture]['y_pred'].append(1 if is_moving_pred else 0)
if is_correct:
    results['per_texture'][texture]['correct'] += 1
# ... (similar for per-angle)
```

**Step 10: Store per-video data**
```python
results['per_video_data'].append({
    'video_path': video_rel_path,
    'prediction': 'moving' if is_moving_pred else 'stopped',
    'label': 'moving' if is_moving_gt else 'stopped',
    'speed': speed,
    'correct': is_correct,
    'model_output': output_text
})
```

**After all videos processed:**

**Step 11: Calculate aggregate metrics**
```python
# Accuracy
results['accuracy'] = (results['correct'] / results['total'] * 100)

# F1 score (macro average)
results['f1_score'] = f1_score(
    results['y_true'],
    results['y_pred'],
    average='macro',      # Average F1 of both classes
    zero_division=0
) * 100

# Precision and Recall (macro average)
results['precision'] = precision_score(results['y_true'], results['y_pred'], average='macro', zero_division=0) * 100
results['recall'] = recall_score(results['y_true'], results['y_pred'], average='macro', zero_division=0) * 100

# Per-class F1 scores
f1_per_class = f1_score(results['y_true'], results['y_pred'], average=None, zero_division=0)
results['f1_stopped'] = f1_per_class[0] * 100  # class 0 = stopped
results['f1_moving'] = f1_per_class[1] * 100   # class 1 = moving
```

**Metrics explained:**

**Accuracy:**
```
Accuracy = (Correct Predictions) / (Total Predictions) × 100
```
- Simple, intuitive metric
- Can be misleading with imbalanced datasets

**F1 Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Harmonic mean of precision and recall
- Balances false positives and false negatives
- Better for imbalanced datasets

**Precision:**
```
Precision = True Positives / (True Positives + False Positives)
```
- Of all videos predicted as "moving", how many were actually moving?

**Recall:**
```
Recall = True Positives / (True Positives + False Negatives)
```
- Of all actually moving videos, how many did we correctly identify?

**Macro Average:**
- Calculate metric for each class separately
- Average the results
- Treats both classes equally (not weighted by sample count)

**Step 12: Calculate per-texture and per-angle metrics**
```python
for texture, tex_data in results['per_texture'].items():
    if len(tex_data['y_true']) > 0:
        tex_data['accuracy'] = (tex_data['correct'] / tex_data['total'] * 100)
        tex_data['f1_score'] = f1_score(tex_data['y_true'], tex_data['y_pred'], average='macro', zero_division=0) * 100
        # ... (similar for precision, recall, per-class F1)

# Same for per-angle metrics
```

**Why per-texture and per-angle breakdowns?**
- Identifies which textures/angles are hardest
- Guides data augmentation decisions
- Validates generalization across conditions

### 6. Response Parsing

**Function: `_is_moving(text)`**

Critical function that determines if model's response indicates movement.

**Parsing strategy (priority order):**

**1. Check for explicit negative indicators:**
```python
negative_indicators = [
    'not moving',
    'isn\'t moving',
    'is not moving',
    'stopped',
    'stationary',
    'not in motion',
    'no movement'
]
for indicator in text.lower():
    if indicator in text:
        return False  # Stopped
```

**2. Check for "no" response:**
```python
if text.startswith('no') or text == 'no' or text == 'no.':
    return False
```

**3. Check for positive indicators:**
```python
if 'moving' in text:
    return True

if text.startswith('yes') or text == 'yes' or text == 'yes.':
    return True
```

**4. Check for motion keywords:**
```python
motion_keywords = ['in motion', 'is moving', 'belt is moving', 'movement']
for keyword in motion_keywords:
    if keyword in text:
        return True
```

**5. Default to stopped:**
```python
logger.warning(f"Ambiguous response, defaulting to 'stopped': '{text}'")
return False
```

**Why this parsing order?**
- Negative indicators take priority (safer default)
- Handles complex responses like "The treadmill belt is not moving"
- Robust to model verbosity

**Example parsing:**
| Model Output | Parsed Result | Reasoning |
|--------------|---------------|-----------|
| "yes" | Moving | Explicit "yes" |
| "no" | Stopped | Explicit "no" |
| "The treadmill belt is moving." | Moving | Contains "moving" |
| "The treadmill belt is stopped." | Stopped | Contains "stopped" |
| "The belt isn't moving" | Stopped | Contains "isn't moving" (negative indicator) |
| "I see movement" | Moving | Contains "movement" |
| "The belt appears stationary" | Stopped | Contains "stationary" |
| "It's unclear" | Stopped | Ambiguous, default to stopped |

**Why default to stopped?**
- Conservative choice (fewer false positives)
- Ambiguous answers likely indicate model uncertainty
- Stopped is easier to verify (absence of motion)

### 7. Metrics Calculation

**Overall metrics structure:**
```python
results = {
    'total': 100,
    'correct': 85,
    'incorrect': 15,
    'accuracy': 85.0,
    'f1_score': 84.2,
    'precision': 83.5,
    'recall': 85.0,
    'f1_moving': 86.0,
    'f1_stopped': 82.4,
    'moving': {'total': 50, 'correct': 44},
    'stopped': {'total': 50, 'correct': 41},
    'per_texture': {
        'subtle_gray_stripes': {
            'total': 100,
            'correct': 85,
            'accuracy': 85.0,
            'f1_score': 84.2,
            'moving': {'total': 50, 'correct': 44},
            'stopped': {'total': 50, 'correct': 41},
            # ...
        }
    },
    'per_angle': {
        'angle52': {
            'total': 100,
            'correct': 85,
            'accuracy': 85.0,
            'f1_score': 84.2,
            # ...
        }
    }
}
```

### 8. Output Files

**1. Evaluation Metadata**
```
data/_exp_20251207_143033_test/eval_20251207_150322/evaluation_metadata_20251207_150322.txt
```

Contains complete configuration:
- Model paths
- Dataset configuration
- Evaluation parameters
- Video processing settings
- Quantization configuration
- GPU configuration
- Git information
- System information
- Command-line arguments

**Purpose:** Reproducibility and debugging

**2. Live Evaluation Logs**
```
data/_exp_20251207_143033_test/eval_20251207_150322/live_evaluation_log_base_20251207_150322.txt
data/_exp_20251207_143033_test/eval_20251207_150322/live_evaluation_log_finetuned_20251207_150322.txt
```

Contains per-video evaluation details:
- Video name and path
- Ground truth label
- Model raw output
- Parsed classification
- Correctness indicator (✓/✗)

**Purpose:** Debugging and analysis of individual predictions

**3. Per-Video Predictions CSV**
```
data/_exp_20251207_143033_test/eval_20251207_150322/per_video_predictions_base_20251207_150322.csv
data/_exp_20251207_143033_test/eval_20251207_150322/per_video_predictions_finetuned_20251207_150322.csv
```

CSV structure:
```csv
video_path,prediction,label,speed,correct,model_output
data/_exp_..._test/treadmill_0042_...,moving,moving,3.0,True,yes
data/_exp_..._test/treadmill_0043_...,stopped,stopped,0.0,True,no
```

**Purpose:** Spreadsheet analysis, plotting, error analysis

**4. Evaluation Report**
```
data/_exp_20251207_143033_test/eval_20251207_150322/evaluation_report_20251207_150322.txt
evaluation_results/evaluation_report_20251207_150322.txt (legacy copy)
```

Text report with all metrics:
- Overall metrics (accuracy, F1, precision, recall)
- Per-class metrics (moving, stopped)
- Per-texture breakdown
- Per-angle breakdown
- Comparison between base and fine-tuned models

**Purpose:** Human-readable summary, experiment tracking

### 9. Report Generation

**Function: `generate_report(base_results, lora_results=None)`**

Creates comprehensive text report.

**Report structure:**
```
==================================================
TREADMILL MOTION DETECTION - EVALUATION REPORT
==================================================

Date: 2025-12-07 15:03:22
Dataset: _exp_20251207_143033_test
Fine-Tuned Model Adapter Path: saves/_exp_20251207_143033_train_20251207_143500

BASE MODEL
==========

OVERALL METRICS:
  Accuracy:  65.00% (13/20)
  F1 Score:  63.33%
  Precision: 65.00%
  Recall:    65.00%

PER-CLASS METRICS:
  Moving:  7/10 (Acc: 70.00%, F1: 70.00%)
  Stopped: 6/10 (Acc: 60.00%, F1: 56.67%)

PER-TEXTURE BREAKDOWN:
  subtle_gray_stripes:
    Total: 20 videos
    Accuracy: 65.00% (13/20)
    F1 Score: 63.33%
    Moving: 7/10 (F1: 70.00%)
    Stopped: 6/10 (F1: 56.67%)

PER-ANGLE BREAKDOWN:
  angle52:
    Total: 20 videos
    Accuracy: 65.00% (13/20)
    F1 Score: 63.33%
    Moving: 7/10 (F1: 70.00%)
    Stopped: 6/10 (F1: 56.67%)

FINE-TUNED MODEL
================

OVERALL METRICS:
  Accuracy:  95.00% (19/20)
  F1 Score:  95.00%
  Precision: 95.00%
  Recall:    95.00%

PER-CLASS METRICS:
  Moving:  10/10 (Acc: 100.00%, F1: 100.00%)
  Stopped: 9/10 (Acc: 90.00%, F1: 90.00%)

... (similar breakdowns)

==================================================
COMPARISON
==================================================
Improvement: +30.00%
```

## Common Evaluation Issues

### 1. Inconsistent Predictions
**Symptoms:** Same video gets different predictions on different runs

**Causes:**
- `do_sample=True` (randomness enabled)
- Non-deterministic GPU operations

**Solutions:**
- Use `do_sample=False` (greedy decoding)
- Set random seed
- Use deterministic algorithms

### 2. Parsing Failures
**Symptoms:** Many warnings about ambiguous responses

**Causes:**
- Model generates unexpected formats
- Evaluation method mismatch (trained on "yesno", evaluated with "moving_stopped")

**Solutions:**
- Check eval_method matches training
- Review live log to see actual model outputs
- Enhance `_is_moving()` parsing logic

### 3. Low Base Model Performance
**Symptoms:** Base model accuracy < 60%

**Causes:**
- Task is challenging for general model
- Videos are low quality or ambiguous
- Prompt is unclear

**Solutions:**
- This is expected! Fine-tuning should improve it
- Check if fine-tuned model performs better
- Review specific failure cases in live log

### 4. No Improvement After Fine-Tuning
**Symptoms:** Fine-tuned model ≈ base model performance

**Causes:**
- Training didn't converge
- Evaluation method differs from training
- Model overfitted to training data

**Solutions:**
- Check training loss (should decrease)
- Verify eval_method matches training
- Check per-angle breakdown (overfitting shows up here)

## Next Steps
- Read `05_REPORTING.md` for HTML report generation
- Read `06_EXPERIMENT_TRACKING.md` for how results are logged
