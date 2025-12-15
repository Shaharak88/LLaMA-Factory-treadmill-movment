# Data Flow: Dataset → Sampler → Training Loop

## Complete Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          1. DATASET LOADING                                  │
│                    (src/llamafactory/data/loader.py)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Load Dataset from Source    │
                    │   • HuggingFace Hub          │
                    │   • Local JSON/JSONL files   │
                    │   • ModelScope/OpenMind      │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   align_dataset()            │
                    │   • Parse video paths        │
                    │   • Extract labels           │
                    │   • Format conversations     │
                    └───────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        2. DATASET PROCESSING                                 │
│              (src/llamafactory/data/processor/*.py)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  SupervisedDatasetProcessor   │
                    │  • Apply chat template        │
                    │  • Tokenize text             │
                    │  • Process videos/images     │
                    │  • Create input_ids, labels  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │    Processed Dataset         │
                    │    {                         │
                    │      "input_ids": [...],     │
                    │      "labels": [...],        │
                    │      "videos": [path],       │
                    │      "attention_mask": [...] │
                    │    }                         │
                    └───────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      3. SAMPLER INITIALIZATION                               │
│              (src/llamafactory/train/sft/trainer.py:125)                    │
│                 get_train_dataloader() method                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
            ┌───────────────────────┴───────────────────────┐
            │                                               │
            ▼                                               ▼
    ┌─────────────────┐                           ┌─────────────────────┐
    │  HF Samplers    │                           │  Custom Samplers    │
    └─────────────────┘                           └─────────────────────┘
            │                                               │
            ├─ hf_shuffle                                   ├─ random
            │  (RandomSampler)                              │  (RandomBatchSampler)
            │                                               │
            └─ hf_sequential                                ├─ random_no_fix
               (SequentialSampler)                          │  (RandomBatchSamplerNoIterFix)
                                                            │
                                                            ├─ balanced
                                                            │  (BalancedBatchSampler)
                                                            │
                                                            └─ feature_balanced
                                                               (FeatureBalancedBatchSampler)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      4. BATCH SAMPLER OPERATION                              │
│                   (src/llamafactory/data/sampler.py)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │  Example: FeatureBalancedBatchSampler.__iter__()     │
        │                                                       │
        │  1. Extract features from video filenames:           │
        │     • motion (moving/stopped from speed)             │
        │     • texture, stripe_gray, bg_gray                  │
        │     • direction, angle, distance                     │
        │     • brightness, contrast, resolution               │
        │                                                       │
        │  2. Calculate oversample weights:                    │
        │     • Rare features → higher weight                  │
        │     • Example: angle30 (2 videos) → weight=3x        │
        │                                                       │
        │  3. Create oversampled pools:                        │
        │     moving_pool = [idx1, idx2, idx2, idx2, ...]     │
        │     stopped_pool = [idx3, idx4, idx4, ...]           │
        │                                                       │
        │  4. Select diverse batch:                            │
        │     • Track feature counts across ENTIRE batch       │
        │     • Enforce quota: ceil(batch_size / num_values)   │
        │     • Pick samples with most NEW feature values      │
        │     • Maintain 50/50 moving/stopped balance          │
        │                                                       │
        │  5. Yield batch indices:                             │
        │     [idx1, idx3, idx2, idx4, ...]  (14 samples)     │
        └───────────────────────────────────────────────────────┘
                                    │
                                    ▼
                      [0, 5, 12, 3, 8, 1, 9, 4, 11, 2, 7, 6, 10, 13]
                           (Batch indices for this step)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         5. DATA COLLATION                                    │
│          (src/llamafactory/train/sft/trainer.py:293-307)                    │
│               LoggingCollateWrapper / data_collator                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │  1. Log batch info (optional):                       │
        │     • Video filenames                                │
        │     • Feature distribution                           │
        │     • Class balance (moving/stopped)                 │
        │                                                       │
        │  2. Collate samples into batch tensors:              │
        │     • Stack/pad input_ids                            │
        │     • Stack/pad attention_mask                       │
        │     • Stack/pad labels                               │
        │     • Process videos (load + encode)                 │
        │                                                       │
        │  3. Output collated batch:                           │
        │     {                                                │
        │       "input_ids": Tensor[14, max_len],             │
        │       "attention_mask": Tensor[14, max_len],        │
        │       "labels": Tensor[14, max_len],                │
        │       "pixel_values": Tensor[14, frames, ...]       │
        │     }                                                │
        └───────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         6. PYTORCH DATALOADER                                │
│                      (torch.utils.data.DataLoader)                          │
│              Created in get_train_dataloader() line 311                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │  DataLoader Configuration:                           │
        │  • dataset: Processed dataset                        │
        │  • batch_sampler: Custom sampler (or None)           │
        │  • collate_fn: Logging wrapper → data_collator       │
        │  • num_workers: Multi-process loading                │
        │  • pin_memory: GPU optimization                      │
        │                                                       │
        │  Iterates over batches each training step            │
        └───────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         7. TRAINING LOOP                                     │
│                  (HuggingFace Trainer / CustomSeq2SeqTrainer)               │
│                  (src/llamafactory/train/sft/trainer.py)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │  For each epoch:                                     │
        │                                                       │
        │    1. Call dataloader.__iter__()                     │
        │       → Calls batch_sampler.__iter__()               │
        │       → Generates batch indices                      │
        │                                                       │
        │    2. For each batch:                                │
        │       a. Get batch from dataloader                   │
        │       b. Move tensors to device (GPU)                │
        │       c. Forward pass: model(batch)                  │
        │       d. Compute loss                                │
        │       e. Backward pass: loss.backward()              │
        │       f. Update weights: optimizer.step()            │
        │       g. Log metrics                                 │
        │                                                       │
        │    3. End of epoch:                                  │
        │       • Sampler writes statistics                    │
        │       • Save checkpoint                              │
        │       • Run evaluation                               │
        └───────────────────────────────────────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │   Trained Model       │
                        │   + Statistics        │
                        └───────────────────────┘
```

## Key Components Explained

### 1. Dataset Loading (loader.py)
- **Location**: `src/llamafactory/data/loader.py:51`
- **Function**: `_load_single_dataset()`
- **Purpose**: Load raw data from various sources (HF Hub, local files, etc.)
- **Output**: Raw HuggingFace Dataset with video paths, conversations, etc.

### 2. Dataset Processing (processor/)
- **Location**: `src/llamafactory/data/processor/supervised.py`
- **Class**: `SupervisedDatasetProcessor`
- **Purpose**:
  - Apply chat template to format conversations
  - Tokenize text into input_ids
  - Process videos/images
  - Create training labels
- **Output**: Processed Dataset ready for batching

### 3. Sampler Creation (trainer.py)
- **Location**: `src/llamafactory/train/sft/trainer.py:125`
- **Method**: `get_train_dataloader()`
- **Purpose**: Choose which sampler to use based on `sampler_type`
- **Options**:
  - **HF Samplers**: Built-in RandomSampler, SequentialSampler
  - **Custom Samplers**: Random, Balanced, FeatureBalanced

### 4. Batch Sampling (sampler.py)
- **Location**: `src/llamafactory/data/sampler.py`
- **Key Classes**:
  - `BalancedBatchSampler` (line 74): 50/50 class balance
  - `RandomBatchSampler` (line 306): Random shuffling
  - `FeatureBalancedBatchSampler` (line 721): ALL features balanced
- **Purpose**:
  - Decide which samples go into each batch
  - Enforce diversity/balance constraints
  - Yield lists of indices
- **Output**: Batch indices `[0, 5, 12, 3, ...]`

### 5. Data Collation (trainer.py + collate_fn)
- **Location**: `src/llamafactory/train/sft/trainer.py:293-307`
- **Classes**:
  - `LoggingCollateWrapper` (sampler.py:1810)
  - `FeatureBalancedLoggingCollateWrapper` (sampler.py:1690)
- **Purpose**:
  - Log batch information (optional)
  - Convert list of samples → batch tensors
  - Pad sequences to same length
  - Process/encode videos
- **Output**: Collated batch tensors

### 6. DataLoader (PyTorch)
- **Location**: `src/llamafactory/train/sft/trainer.py:311`
- **Purpose**:
  - Orchestrate sampling + collation
  - Handle multi-worker loading
  - Provide iterator interface for training
- **Key Parameters**:
  - `dataset`: The processed dataset
  - `batch_sampler`: Custom sampler (or None)
  - `collate_fn`: Wraps data_collator with logging

### 7. Training Loop (Trainer)
- **Location**: `transformers.Seq2SeqTrainer` (extended by CustomSeq2SeqTrainer)
- **Purpose**:
  - Iterate over dataloader batches
  - Forward pass through model
  - Compute loss
  - Backward pass + optimizer step
  - Log metrics, save checkpoints

## Data Flow Example

```
Dataset Entry:
{
  "messages": [{"role": "user", "content": "<video>\nIs the treadmill moving?"},
               {"role": "assistant", "content": "Yes"}],
  "videos": ["data/train/treadmill_0042_stripes_right_speed3.0_angle0_dist2.5.mp4"]
}
                    ↓
After Processing:
{
  "input_ids": [1, 2, 3, ...],           # Tokenized question
  "labels": [-100, -100, 4, 5],          # Tokenized answer (with -100 for prompt)
  "videos": ["data/train/treadmill_..."], # Video path
  "attention_mask": [1, 1, 1, ...]
}
                    ↓
Sampler yields indices: [42, 15, 8, 91, ...] (14 samples with max diversity)
                    ↓
Collator stacks into batch:
{
  "input_ids": Tensor[14, 128],
  "labels": Tensor[14, 128],
  "pixel_values": Tensor[14, 8, 3, 224, 224],  # 14 videos, 8 frames each
  "attention_mask": Tensor[14, 128]
}
                    ↓
Trainer passes batch to model:
  outputs = model(input_ids, pixel_values, labels=labels)
  loss = outputs.loss
  loss.backward()
  optimizer.step()
```

## Critical Files

1. **Dataset Loading**: `src/llamafactory/data/loader.py`
2. **Dataset Processing**: `src/llamafactory/data/processor/supervised.py`
3. **Sampler Implementation**: `src/llamafactory/data/sampler.py`
4. **Trainer (orchestrates everything)**: `src/llamafactory/train/sft/trainer.py`

## Flow Control Points

1. **Sampler Selection**: `trainer.py:125` (get_train_dataloader)
2. **Batch Index Generation**: `sampler.py` (__iter__ method of chosen sampler)
3. **Batch Collation**: Wraps data_collator with logging
4. **Training Step**: HuggingFace Trainer.training_step()
