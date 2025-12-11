# Training Pipeline

## Purpose
The training pipeline fine-tunes the Qwen2.5-VL vision-language model using LoRA (Low-Rank Adaptation) to detect treadmill motion in videos. It handles training orchestration, parameter configuration, experiment tracking, and evaluation.

## Architecture

```
run_full_pipeline.py (main orchestrator)
    │
    ├──> Validates arguments and paths
    ├──> Initializes experiment tracker
    ├──> Creates dynamic YAML config
    │
    ├──> Calls: llamafactory-cli train (LLaMA-Factory)
    │    │
    │    ├──> Loads base model (Qwen2.5-VL-7B-Instruct)
    │    ├──> Applies LoRA/DoRA adapters
    │    ├──> Trains on video dataset
    │    └──> Saves checkpoints to saves/
    │
    ├──> Calls: evaluate_pipeline_simple.py
    │    │
    │    ├──> Evaluates base model
    │    ├──> Evaluates fine-tuned model
    │    └──> Generates metrics and reports
    │
    ├──> Updates experiment tracker with results
    └──> Generates dataset comparison report (optional)
```

---

## Part 1: run_full_pipeline.py

### Location
`/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/run_full_pipeline.py`

### Purpose
Main orchestrator that coordinates training, evaluation, and experiment tracking.

### Key Components

#### 1. Argument Parsing

**Function: `parse_arguments()`**

Defines all training and evaluation parameters:

**Dataset Configuration:**
```python
parser.add_argument('--dataset_name', type=str, required=True,
                   help='Name of dataset (e.g., "_exp_20251207_143033")')
parser.add_argument('--dataset_dir', type=str, default='data',
                   help='Directory containing datasets')
```

**Model Configuration:**
```python
parser.add_argument('--model_name_or_path', type=str,
                   default='Qwen/Qwen2.5-VL-7B-Instruct',
                   help='Hugging Face model ID or local path')
parser.add_argument('--template', type=str, default='qwen2_vl',
                   help='Chat template for model')
```

**Why Qwen2.5-VL-7B-Instruct?**
- 7B parameters = good balance of performance and efficiency
- Instruction-tuned = follows instructions well
- Supports video input natively
- Pre-trained on diverse visual tasks

**LoRA Configuration:**
```python
parser.add_argument('--lora_rank', type=int, default=32,
                   help='LoRA rank (controls adapter size)')
parser.add_argument('--lora_alpha', type=int, default=64,
                   help='LoRA alpha (controls scaling)')
parser.add_argument('--lora_dropout', type=float, default=0.05,
                   help='LoRA dropout rate')
parser.add_argument('--lora_target', type=str, default='all',
                   help='LoRA target modules (all, attention, mlp)')
parser.add_argument('--adapter_type', type=str, default='lora',
                   choices=['lora', 'lora+', 'dora', 'rslora', 'pissa', 'oft'],
                   help='Adapter type to use')
parser.add_argument('--no_quantization', action='store_true',
                   help='Disable 4-bit quantization (full precision)')
```

**Adapter Types (NEW - 2025-12-08):**

| Adapter | YAML Config | Description | When to Use |
|---------|-------------|-------------|-------------|
| `lora` | (default) | Base LoRA adapter | Default choice, well-tested |
| `lora+` | `loraplus_lr_ratio: 16.0` | LoRA with learning rate ratio | When you want different LR for B matrix |
| `dora` | `use_dora: true` | Weight-Decomposed LoRA | Better generalization, slightly slower |
| `rslora` | `use_rslora: true` | Rank Stabilization LoRA | More stable training at high ranks |
| `oft` | `finetuning_type: oft` | Orthogonal Fine-Tuning | Different approach, preserves orthogonality |

> **Note:** `pissa` (PiSSA SVD initialization) is **NOT SUPPORTED** by our pipeline. It requires running `scripts/pissa_init.py` before training when using quantization, which is incompatible with our automated workflow.

---

### Multi-Adapter Comparison Script (NEW - 2025-12-08)

To compare all adapters on the same dataset, use `run_all_adapters.sh`:

**Location:** `/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/run_all_adapters.sh`

**Supported Adapters:** lora, lora+, dora, rslora, oft

**How It Works:**
1. First adapter builds the dataset (or uses existing if `--dataset-name` provided)
2. Subsequent adapters reuse the same dataset (`--skip-datasets` automatically added)
3. Each adapter gets its own experiment entry in CSV and HTML report
4. All training parameters (epochs, batch-size, learning-rate) are shared across adapters

**Basic Usage:**
```bash
# Run all 5 adapters with new dataset
./run_all_adapters.sh --epochs 7 --batch-size 14 --no-quantization -y

# Run all adapters on existing dataset
./run_all_adapters.sh --dataset-name "_exp_20251207_172213" \
    --epochs 7 --no-quantization -y

# Run specific adapters only
./run_all_adapters.sh --adapters "lora,dora,oft" \
    --epochs 7 --batch-size 14 -y

# Run with custom model name base (creates: mymodel_lora, mymodel_dora, etc.)
./run_all_adapters.sh --model-name "mymodel" --epochs 5 -y
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--adapters` | Comma-separated list of adapters | `lora,lora+,dora,rslora,oft` |
| `--dataset-name` | Use existing dataset instead of generating new | (generate new) |
| `--model-name` | Base name for models (adapter suffix added) | (auto-generated) |
| All `run_experiment.sh` args | Passed through to each experiment | - |

**Output:**
- Creates one experiment per adapter in `experiments_log.csv`
- Generates separate HTML report for each adapter
- Prints summary at end showing success/failure for each adapter

**Command examples:**
```bash
# Default LoRA
./run_experiment.sh --epochs 5

# DoRA (Weight-Decomposed LoRA)
./run_experiment.sh --adapter-type dora --epochs 5

# LoRA+ with learning rate ratio
./run_experiment.sh --adapter-type lora+ --epochs 5

# rsLoRA for stable training at high ranks
./run_experiment.sh --adapter-type rslora --lora-rank 64 --epochs 5

# PiSSA initialization
./run_experiment.sh --adapter-type pissa --epochs 5

# OFT (Orthogonal Fine-Tuning)
./run_experiment.sh --adapter-type oft --epochs 5

# Any adapter without quantization (full precision)
./run_experiment.sh --adapter-type dora --no-quantization --epochs 5
```

---

### Balanced Batch Sampling (NEW - 2025-12-09)

For binary classification tasks (moving vs stopped treadmill detection), you can enable balanced batch sampling to ensure each training batch contains exactly 50% moving and 50% stopped samples.

**What It Does:**
- Parses video filenames to extract speed parameter (`speed0.0` = stopped, `speed > 0` = moving)
- Groups samples into "moving" and "stopped" categories
- Creates batches with exactly half from each category
- Improves gradient consistency during training

**YAML Configuration:**
```yaml
# In your training config (e.g., examples/train_qlora/*.yaml)
balanced_sampling: true
per_device_train_batch_size: 4  # Must be even!
```

**Requirements:**
- Batch size must be **even** (since each batch is split 50/50)
- Video filenames must contain speed parameter (e.g., `treadmill_0000_stripes_speed3.0_angle0.mp4`)
- **Not compatible with streaming mode** (`streaming: false` required)

**How It Works:**
1. At training start, all samples are categorized by parsing video filenames
2. Two index lists are maintained: `moving_indices` and `stopped_indices`
3. During each epoch, both lists are shuffled independently
4. Batches are created by taking `batch_size/2` samples from each list
5. Samples are interleaved (moving, stopped, moving, stopped...) within each batch

**Code Location:**
- Sampler: `src/llamafactory/data/sampler.py`
- Configuration: `src/llamafactory/hparams/finetuning_args.py` (line 513)
- Trainer integration: `src/llamafactory/train/sft/trainer.py`

**When to Use:**
- Training on treadmill motion detection (binary classification)
- When you want consistent class representation in each batch
- When gradient variance from class imbalance is a concern

**Adapter parameters explained:**
- **Rank**: Controls capacity (higher = more parameters, better fit, slower)
- **Alpha**: Controls how much adapter affects base model (alpha/rank = scaling factor)
- **Target modules**: Which layers to adapt (all = entire model, attention = only attention layers)
- **Note:** OFT doesn't use lora_rank/alpha/dropout parameters

**Training Hyperparameters:**
```python
parser.add_argument('--num_train_epochs', type=int, default=5,
                   help='Number of training epochs')
parser.add_argument('--per_device_train_batch_size', type=int, default=1,
                   help='Batch size per GPU')
parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                   help='Gradient accumulation steps')
parser.add_argument('--learning_rate', type=float, default=5e-5,
                   help='Learning rate')
parser.add_argument('--lr_scheduler_type', type=str, default='cosine',
                   help='Learning rate scheduler')
parser.add_argument('--warmup_ratio', type=float, default=0.1,
                   help='Warmup ratio (portion of training for LR warmup)')
```

**Why these defaults?**
- `num_train_epochs=5`: Enough for convergence on small datasets
- `per_device_train_batch_size=1`: Videos use lots of GPU memory
- `gradient_accumulation_steps=8`: Effective batch size = 1 × 8 = 8
- `learning_rate=5e-5`: Conservative LR for fine-tuning
- `cosine` scheduler: Smooth LR decay
- `warmup_ratio=0.1`: 10% of training for warmup (prevents instability)

**Quantization:**
```python
parser.add_argument('--quantization_bit', type=int, default=4,
                   choices=[4, 8], help='Quantization bits (4 or 8)')
parser.add_argument('--bf16', action='store_true',
                   help='Use bfloat16 precision')
parser.add_argument('--fp16', action='store_true',
                   help='Use float16 precision')
```

**4-bit quantization explained:**
- Reduces model memory by ~4x
- Enables training 7B model on consumer GPUs
- Uses QLoRA (Quantized LoRA) technique
- Minimal accuracy loss (~1-2%)
- **NEW (2025-12-08):** Use `--no-quantization` flag to disable quantization for full precision training (requires more VRAM)

**Evaluation Configuration:**
```python
parser.add_argument('--eval_method', type=str, default='yesno',
                   choices=['yesno', 'moving_stopped'],
                   help='Evaluation prompt format')
parser.add_argument('--eval_batch_size', type=int, default=1,
                   help='Batch size for evaluation')
parser.add_argument('--eval_video_fps', type=float, default=4.0,
                   help='FPS for evaluation videos')
parser.add_argument('--eval_video_maxlen', type=int, default=128,
                   help='Max frames per video for evaluation')
```

**Evaluation methods:**
- **yesno**: Prompt = "Is there movement in the video? Answer only with yes or no."
  - Simpler task, clearer binary classification
  - Model outputs: "yes" (moving) or "no" (stopped)
- **moving_stopped**: Prompt = "Is the treadmill belt moving or stopped?"
  - More explicit, domain-specific
  - Model outputs: "The treadmill belt is moving." or "The treadmill belt is stopped."

**Pipeline Control:**
```python
parser.add_argument('--skip_dataset', action='store_true',
                   help='Skip dataset generation')
parser.add_argument('--skip_training', action='store_true',
                   help='Skip training')
parser.add_argument('--skip_evaluation', action='store_true',
                   help='Skip evaluation')
```

#### 2. Dynamic YAML Generation

**Function: `generate_training_config(args, model_output_dir)`**

Creates LLaMA-Factory YAML config file dynamically based on command-line arguments.

**YAML structure:**
```yaml
### model
model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
template: qwen2_vl

### method
stage: sft                    # Supervised Fine-Tuning
do_train: true
finetuning_type: lora         # Use LoRA
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target: all
use_dora: true                # Enable DoRA

### dataset
dataset: _exp_20251207_143033_train
cutoff_len: 4096              # Max sequence length
preprocessing_num_workers: 8

### output
output_dir: saves/_exp_20251207_143033_train_20251207_143500
logging_steps: 5
save_steps: 100
plot_loss: true
overwrite_output_dir: false

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-05
num_train_epochs: 5.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

### eval
val_size: 0.1                 # 10% validation split
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 100

### quantization
quantization_bit: 4
quantization_method: bitsandbytes

### vision
video_fps: 4.0
video_maxlen: 128
```

**Key design decisions:**

**1. Sequence length (`cutoff_len=4096`):**
- Videos are tokenized into long sequences
- 4096 tokens = enough for ~128 frames at 4 fps
- Longer = more context, but more memory

**2. Validation split (`val_size=0.1`):**
- Splits training set: 90% train, 10% validation
- Used for early stopping and monitoring overfitting
- Not the same as test set (test set is separate)

**3. Logging frequency:**
- `logging_steps=5`: Log metrics every 5 steps
- `save_steps=100`: Save checkpoint every 100 steps
- Frequent logging helps track training progress

**4. Video processing:**
- `video_fps=4.0`: Sample 4 frames per second from video
- `video_maxlen=128`: Max 128 frames per video
- Example: 3-second video at 30 fps → downsampled to 12 frames

**Why downsample videos?**
- Reduces memory usage dramatically
- 4 fps captures enough motion information (consistent with inference)
- Model doesn't need 30 fps for this task

**Note (2025-12-11):** Default video_fps changed from 2.0 to 4.0 to match:
- Synthetic video generation (TRAIN_FPS=4 in run_experiment.sh)
- Inference evaluation scripts (fps=4.0)
This ensures training and inference use the same frame sampling rate.

**Dynamic config generation code:**
```python
config = {
    'model_name_or_path': args.model_name_or_path,
    'template': args.template,
    'stage': 'sft',
    'do_train': True,
    'finetuning_type': 'lora',
    'lora_rank': args.lora_rank,
    'lora_alpha': args.lora_alpha,
    # ... (all other parameters)
}

# Save to YAML file
yaml_path = f'examples/train_qlora/qwen25vl_lora_pipeline_{timestamp}.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

return yaml_path
```

**Why generate config dynamically?**
- Single source of truth (command-line args)
- Easy to modify parameters without editing YAML
- Preserves configs for reproducibility (timestamped filenames)

#### 3. Training Execution

**Function: `run_training(yaml_config_path)`**

Calls LLaMA-Factory's CLI to execute training.

```python
def run_training(yaml_config_path):
    logger.info("Starting training with LLaMA-Factory...")

    # Build command
    cmd = [
        "llamafactory-cli", "train",
        str(yaml_config_path)
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    # Execute training
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=False,  # Stream output to console
        text=True,
        check=True
    )

    logger.info("Training completed successfully")
```

**What happens during training?**

1. **Model Loading:**
   - Downloads Qwen2.5-VL-7B-Instruct from Hugging Face (if not cached)
   - Loads model in 4-bit quantization
   - Initializes LoRA adapters

2. **Data Loading:**
   - Reads dataset JSON file
   - Loads videos and tokenizes inputs
   - Creates batches with DataLoader

3. **Training Loop:**
   ```
   For each epoch:
       For each batch:
           1. Forward pass (model prediction)
           2. Calculate loss (cross-entropy on answer)
           3. Backward pass (gradients)
           4. Accumulate gradients (8 steps)
           5. Update LoRA weights
           6. Log metrics (every 5 steps)
           7. Evaluate on validation (every 100 steps)
           8. Save checkpoint (every 100 steps)
   ```

4. **Outputs:**
   - Checkpoints saved to `saves/{model_name}/checkpoint-{step}/`
   - Training logs: loss, learning rate, perplexity
   - Plots: loss curves (if `plot_loss: true`)

**Training time:**
- Depends on: dataset size, epochs, GPU, quantization
- Typical: 100 videos × 5 epochs ≈ 30-60 minutes on A100

#### 4. Evaluation Execution

**Function: `run_evaluation(args, model_output_dir)`**

Runs evaluation on both base and fine-tuned models.

```python
def run_evaluation(args, model_output_dir):
    logger.info("Starting evaluation...")

    # Build evaluation command
    cmd = [
        "python", "evaluate_pipeline_simple.py",
        "--model_name_or_path", args.model_name_or_path,
        "--adapter_name_or_path", model_output_dir,  # LoRA adapter
        "--test_dataset", f"{args.dataset_name}_test",
        "--dataset_dir", args.dataset_dir,
        "--output_dir", "evaluation_results",
        "--eval_method", args.eval_method,
        "--eval_batch_size", str(args.eval_batch_size),
        "--video_fps", str(args.eval_video_fps),
        "--video_maxlen", str(args.eval_video_maxlen),
    ]

    # Add GPU memory utilization if specified
    if args.gpu_memory_utilization:
        cmd.extend(["--gpu_memory_utilization", str(args.gpu_memory_utilization)])

    logger.info(f"Running: {' '.join(cmd)}")

    # Execute evaluation
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=False,
        text=True,
        check=True
    )

    logger.info("Evaluation completed successfully")
```

**What happens during evaluation?**
See `04_EVALUATION.md` for detailed explanation.

**Key outputs:**
- `evaluation_results/evaluation_report_{timestamp}.txt` - Metrics and breakdowns
- `evaluation_results/per_video_predictions_base_{timestamp}.csv` - Base model predictions
- `evaluation_results/per_video_predictions_finetuned_{timestamp}.csv` - Fine-tuned predictions
- `evaluation_results/live_evaluation_log_*.txt` - Detailed logs per video

#### 5. Experiment Tracking

**Integration with ExperimentTracker:**

```python
from experiment_tracker import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker('experiments_log.csv')

# Start experiment (logs all parameters)
experiment_id = tracker.start_experiment(args, actual_model_path=model_output_dir)

# Update status during pipeline
tracker.update_status('dataset', 'skipped' if args.skip_dataset else 'completed')
tracker.update_status('training', 'in_progress')
# ... training happens ...
tracker.update_status('training', 'completed')

# Parse and log evaluation results
base_results, finetuned_results = parse_evaluation_results('evaluation_results')
tracker.update_evaluation_results(base_results, finetuned_results)

# Finalize
tracker.finalize_experiment()
```

**What gets logged to CSV?**
See `06_EXPERIMENT_TRACKING.md` for complete details.

**Key fields:**
- Experiment metadata (ID, timestamp)
- All dataset generation parameters (train + test)
- All model configuration parameters
- All LoRA hyperparameters
- All training hyperparameters
- Evaluation results (accuracy, F1, per-texture, per-angle)
- Model paths

#### 6. Main Execution Flow

**Function: `main()`**

```python
def main():
    args = parse_arguments()

    # Initialize tracker
    tracker = ExperimentTracker('experiments_log.csv')

    # Calculate model output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_output_dir = f"saves/{args.dataset_name}_{timestamp}"

    # Start experiment tracking
    experiment_id = tracker.start_experiment(args, actual_model_path=model_output_dir)
    logger.info(f"Started experiment {experiment_id}")

    # Stage 1: Dataset (usually skipped, handled by building_dataset.py)
    if not args.skip_dataset:
        logger.info("Dataset generation not implemented in this script")
        tracker.update_status('dataset', 'skipped')

    # Stage 2: Training
    if not args.skip_training:
        logger.info("=== TRAINING STAGE ===")
        tracker.update_status('training', 'in_progress')

        # Generate dynamic YAML config
        yaml_path = generate_training_config(args, model_output_dir)

        # Run training
        run_training(yaml_path)

        tracker.update_status('training', 'completed')
    else:
        logger.info("Skipping training")
        tracker.update_status('training', 'skipped')

    # Stage 3: Evaluation
    if not args.skip_evaluation:
        logger.info("=== EVALUATION STAGE ===")
        tracker.update_status('evaluation', 'in_progress')

        # Run evaluation
        run_evaluation(args, model_output_dir)

        # Parse results and update tracker
        base_results, finetuned_results = parse_evaluation_results('evaluation_results')
        tracker.update_evaluation_results(base_results, finetuned_results)

        tracker.update_status('evaluation', 'completed')
    else:
        logger.info("Skipping evaluation")
        tracker.update_status('evaluation', 'skipped')

    # Finalize
    tracker.finalize_experiment()
    logger.info(f"Experiment {experiment_id} completed successfully")
```

---

## Part 2: Training with LLaMA-Factory

### What is LLaMA-Factory?

LLaMA-Factory is a unified framework for fine-tuning large language models (LLMs) and vision-language models (VLMs).

**Key features:**
- Supports 100+ models (Qwen, LLaMA, Mistral, etc.)
- Multiple fine-tuning methods (LoRA, QLoRA, Full fine-tuning)
- Easy configuration via YAML files
- Built-in evaluation and logging
- Optimized for efficiency (gradient checkpointing, flash attention)

**Why use LLaMA-Factory?**
- Abstracts away complex training logic
- Handles data loading, batching, optimization
- Supports video inputs out-of-the-box for Qwen2.5-VL
- Active development and community support

### Training Process

**1. Model Initialization:**
```python
# LLaMA-Factory loads model with BitsAndBytes quantization
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    quantization_config=quantization_config,
    device_map="auto"
)
```

**2. LoRA Adapter Initialization:**
```python
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(
    r=32,                        # LoRA rank
    lora_alpha=64,               # LoRA alpha
    target_modules="all",        # Apply to all linear layers
    lora_dropout=0.05,           # Dropout
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# DoRA modification (if use_dora=true)
if use_dora:
    model = convert_lora_to_dora(model)
```

**Trainable parameters:**
- Base model: 7B parameters (frozen, quantized)
- LoRA adapters: ~10-20M parameters (trainable)
- Total GPU memory: ~12-16 GB (with 4-bit quantization)

**3. Data Processing:**
```python
# LLaMA-Factory loads JSON dataset
with open('data/_exp_20251207_143033_train.json', 'r') as f:
    dataset = json.load(f)

# For each sample:
for sample in dataset:
    messages = sample['messages']  # User and assistant messages
    video_path = sample['videos'][0]  # Path to video file

    # Load and process video
    video_frames = load_video(video_path, fps=4.0, max_frames=128)

    # Tokenize conversation
    input_ids = tokenizer.apply_chat_template(
        messages[:-1],  # Exclude assistant answer
        add_generation_prompt=True
    )

    # Tokenize answer (for loss calculation)
    answer_ids = tokenizer.encode(messages[-1]['content'])

    # Combine: [video tokens] + [prompt tokens] + [answer tokens]
    full_input_ids = [*video_frame_tokens, *input_ids, *answer_ids]
```

**4. Training Loop:**
```python
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']  # Answer tokens for loss
        )

        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Logging
        if step % logging_steps == 0:
            logger.info(f"Step {step}: loss={loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

        # Evaluation
        if step % eval_steps == 0:
            eval_loss = evaluate(model, eval_dataloader)
            logger.info(f"Eval loss: {eval_loss:.4f}")

        # Checkpointing
        if step % save_steps == 0:
            model.save_pretrained(f"saves/{model_name}/checkpoint-{step}")
```

**5. Loss Calculation:**
```python
# Cross-entropy loss on assistant answer tokens only
loss = cross_entropy(
    logits=model_outputs,  # Model predictions
    targets=answer_token_ids,  # Ground truth ("yes" or "no")
    ignore_index=IGNORE_INDEX  # Ignore padding and prompt tokens
)
```

### Optimization Techniques

**1. Gradient Checkpointing:**
- Trades compute for memory
- Recomputes activations during backward pass
- Enables larger models/batch sizes

**2. Flash Attention:**
- Faster attention computation
- Reduces memory usage
- Enabled automatically for supported models

**3. Mixed Precision Training:**
- Uses bfloat16 for computations
- Keeps master weights in float32
- Speeds up training without accuracy loss

**4. Gradient Accumulation:**
- Simulates larger batch sizes
- Accumulates gradients over multiple steps
- Updates weights after N steps

### Output Structure

After training, checkpoint directory contains:

```
saves/_exp_20251207_143033_train_20251207_143500/
├── adapter_config.json       # LoRA configuration
├── adapter_model.bin          # LoRA weights (only ~50-100 MB!)
├── trainer_state.json         # Training state (loss history, etc.)
├── training_args.bin          # Training arguments
├── checkpoint-100/            # Intermediate checkpoint
│   ├── adapter_model.bin
│   └── ...
├── checkpoint-200/
└── ...
```

**Key files:**
- `adapter_config.json`: LoRA hyperparameters (rank, alpha, target modules)
- `adapter_model.bin`: Trained LoRA weights (small, 50-100 MB)
- Base model is NOT saved (reuse original from Hugging Face)

**Loading trained model:**
```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "saves/_exp_20251207_143033_train_20251207_143500")
```

## Common Training Issues

### 1. Out of Memory (OOM)
**Symptoms:** CUDA OOM error during training

**Solutions:**
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` (keeps effective batch size)
- Use lower quantization (4-bit instead of 8-bit)
- Reduce `cutoff_len` (shorter sequences)
- Enable gradient checkpointing (if not already)

### 2. Training is Too Slow
**Symptoms:** Very long training time

**Solutions:**
- Use mixed precision (`bf16: true`)
- Enable flash attention (automatic for Qwen2.5-VL)
- Reduce `video_fps` (fewer frames per video)
- Reduce `video_maxlen` (shorter videos)
- Use multiple GPUs (DDP)

### 3. Model Not Learning
**Symptoms:** Loss not decreasing, accuracy stuck

**Solutions:**
- Check learning rate (try 1e-4 or 1e-5)
- Increase LoRA rank (more capacity)
- Train for more epochs
- Check data quality (are labels correct?)
- Reduce `lora_alpha` (less aggressive updates)

### 4. Overfitting
**Symptoms:** Training loss decreases, validation loss increases

**Solutions:**
- Reduce `num_train_epochs`
- Increase `lora_dropout`
- Add more training data
- Use early stopping (monitor validation loss)

## Next Steps
- Read `04_EVALUATION.md` for evaluation pipeline details
- Read `06_EXPERIMENT_TRACKING.md` for experiment logging
- Read `07_CONFIGURATION_FILES.md` for YAML config details
