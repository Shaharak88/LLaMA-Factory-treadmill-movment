# PyTorch DataLoader: Internal Mechanics

## How DataLoader Actually Works

The PyTorch DataLoader is the glue that connects your dataset, sampler, and collate function. Here's exactly how it operates.

## 1. DataLoader Initialization

```python
# From trainer.py:311
dataloader = torch.utils.data.DataLoader(
    dataset=self.train_dataset,        # Your processed dataset
    batch_sampler=batch_sampler,        # Your custom sampler (or None)
    collate_fn=logging_collate_fn,      # Function to collate samples
    num_workers=4,                      # Number of worker processes
    pin_memory=True,                    # GPU optimization
    persistent_workers=True             # Keep workers alive between epochs
)
```

### What DataLoader Stores:

```
DataLoader Object:
├── dataset: Reference to your dataset
├── batch_sampler: Reference to your sampler
├── collate_fn: Function to stack samples into batches
├── num_workers: Number of subprocesses for data loading
├── worker_init_fn: Optional function to initialize workers
├── pin_memory: Whether to copy tensors to CUDA pinned memory
└── persistent_workers: Keep workers alive across epochs
```

## 2. DataLoader Iteration: The Full Cycle

### High-Level Overview

```
Training Loop                    DataLoader                    Sampler                Dataset
     │                               │                            │                      │
     │  for batch in dataloader:    │                            │                      │
     ├──────────────────────────────>│                            │                      │
     │                               │  iter(batch_sampler)       │                      │
     │                               ├───────────────────────────>│                      │
     │                               │                            │                      │
     │                               │  <──────[0,5,12,3,...]────┤  __iter__()          │
     │                               │      (batch indices)       │                      │
     │                               │                            │                      │
     │                               │  dataset[0]                │                      │
     │                               ├───────────────────────────────────────────────────>│
     │                               │  <───── sample_0 ──────────────────────────────────┤
     │                               │                            │                      │
     │                               │  dataset[5]                │                      │
     │                               ├───────────────────────────────────────────────────>│
     │                               │  <───── sample_5 ──────────────────────────────────┤
     │                               │                            │                      │
     │                               │  ... (repeat for all indices)                     │
     │                               │                            │                      │
     │                               │  collate_fn([samples])     │                      │
     │                               │  (stack into batch)        │                      │
     │                               │                            │                      │
     │  <────────── batch ───────────┤                            │                      │
     │  (collated tensors)           │                            │                      │
     │                               │                            │                      │
     │  model(batch)                 │                            │                      │
     │  loss.backward()              │                            │                      │
     │  optimizer.step()             │                            │                      │
     │                               │                            │                      │
     │  next batch...                │                            │                      │
     └───────────────────────────────>│                            │                      │
```

### Detailed Step-by-Step Execution

```python
# What happens when you do: for batch in dataloader
# This is pseudo-code showing DataLoader's internal logic

class DataLoader:
    def __iter__(self):
        """Called when training loop starts iterating"""

        # STEP 1: Create sampler iterator
        if self.batch_sampler is not None:
            # Get batch indices iterator
            self.sampler_iter = iter(self.batch_sampler)
            # This calls: batch_sampler.__iter__()
            # Returns: Iterator that yields lists of indices
        else:
            # Fallback to single-sample sampler
            self.sampler_iter = iter(self.sampler)

        # STEP 2: Initialize workers (if multi-process loading)
        if self.num_workers > 0:
            self._init_workers()

        return self  # DataLoader itself is the iterator

    def __next__(self):
        """Called to get the next batch"""

        # STEP 3: Get next batch indices from sampler
        try:
            indices = next(self.sampler_iter)
            # Example: indices = [0, 5, 12, 3, 8, 1, 9, 4, 11, 2, 7, 6, 10, 13]
        except StopIteration:
            # Sampler exhausted - end of epoch
            if self.persistent_workers:
                # Keep workers alive for next epoch
                pass
            else:
                # Shutdown workers
                self._shutdown_workers()
            raise StopIteration

        # STEP 4: Fetch samples from dataset
        if self.num_workers == 0:
            # Single-process: fetch in main thread
            batch = self._fetch_batch(indices)
        else:
            # Multi-process: workers fetch in parallel
            batch = self._get_batch_from_workers(indices)

        # STEP 5: Collate samples into batch
        batch = self.collate_fn(batch)
        # collate_fn receives: [sample_0, sample_5, sample_12, ...]
        # collate_fn returns: {input_ids: Tensor[14, 128], labels: ...}

        # STEP 6: Pin memory for faster GPU transfer (optional)
        if self.pin_memory:
            batch = self._pin_memory(batch)

        return batch

    def _fetch_batch(self, indices):
        """Fetch samples for given indices"""
        batch = []
        for idx in indices:
            # CRITICAL: This calls dataset.__getitem__(idx)
            sample = self.dataset[idx]
            batch.append(sample)
        return batch
```

## 3. The Sampler's Role

### How Sampler Interacts with DataLoader

```python
# Your FeatureBalancedBatchSampler
class FeatureBalancedBatchSampler:
    def __iter__(self):
        """Called by DataLoader to start iteration"""

        # Epoch 1: seed=42+0=42
        # Epoch 2: seed=42+1=43 (different shuffle)
        g = torch.Generator()
        g.manual_seed(self.seed + self._iter_count)
        self._iter_count += 1

        # Calculate oversample weights
        oversample_weights = self._calculate_sample_oversample_weights()

        # Create oversampled pools
        moving_pool = self._create_oversampled_pool(self.moving_indices, oversample_weights)
        stopped_pool = self._create_oversampled_pool(self.stopped_indices, oversample_weights)

        # Shuffle pools
        moving_pool = shuffle(moving_pool, generator=g)
        stopped_pool = shuffle(stopped_pool, generator=g)

        # Generate batches
        half_batch = self.batch_size // 2
        while len(moving_pool) >= half_batch and len(stopped_pool) >= half_batch:
            # Select diverse batch with quota enforcement
            batch, moving_pool, stopped_pool = self._create_diverse_batch(
                moving_pool, stopped_pool, half_batch, g
            )

            # CRITICAL: Yield batch indices to DataLoader
            yield batch  # Example: [0, 5, 12, 3, 8, 1, 9, 4, 11, 2, 7, 6, 10, 13]

            # DataLoader receives this list and fetches dataset[0], dataset[5], etc.
```

### Sampler Output → DataLoader Input

```
Sampler.__iter__():
  yield [0, 5, 12, 3, 8, 1, 9, 4, 11, 2, 7, 6, 10, 13]  ──┐
  yield [15, 20, 18, 22, 31, 28, ...]                      │
  yield [42, 35, 39, 44, ...]                              │
  ...                                                       │
                                                            │
DataLoader.__next__():                                      │
  indices = next(self.sampler_iter)  <────────────────────┘
  # indices = [0, 5, 12, 3, 8, 1, 9, 4, 11, 2, 7, 6, 10, 13]

  batch = []
  for idx in indices:
      sample = self.dataset[idx]  # Fetch each sample
      batch.append(sample)

  return self.collate_fn(batch)  # Stack into tensors
```

## 4. Dataset Access Pattern

### Single-Process (num_workers=0)

```
DataLoader (Main Thread)
    │
    ├─> Get indices from sampler: [0, 5, 12, 3, 8, 1, 9, 4, 11, 2, 7, 6, 10, 13]
    │
    ├─> for idx in [0, 5, 12, ...]:
    │       sample = dataset[idx]  # Sequential access in main thread
    │       batch.append(sample)
    │
    └─> collate_fn(batch)  # Stack into tensors
```

### Multi-Process (num_workers > 0)

```
DataLoader (Main Thread)
    │
    ├─> Get indices from sampler: [0, 5, 12, 3, 8, 9, 4, 11, 2, 7, 6, 10, 13, 14]
    │                                                                        (14 samples)
    │
    ├─> Split indices among 4 workers:
    │   Worker 0: [0, 5, 12, 3]    ─────┐
    │   Worker 1: [8, 1, 9, 4]     ─────┤
    │   Worker 2: [11, 2, 7, 6]    ─────┼──> All fetch in parallel
    │   Worker 3: [10, 13, 14]     ─────┘
    │
    │   Each worker:
    │   for idx in my_indices:
    │       sample = dataset[idx]  # Parallel I/O (load videos)
    │       my_batch.append(sample)
    │   return my_batch
    │
    ├─> Main thread collects results from all workers:
    │   all_samples = worker0_batch + worker1_batch + worker2_batch + worker3_batch
    │
    └─> collate_fn(all_samples)  # Stack into tensors in main thread
```

## 5. The Collate Function

### What Collate Does

```python
# Simplified collate function logic
def collate_fn(batch):
    """
    Input: List of samples
    [
        {"input_ids": [1,2,3], "labels": [4,5], "videos": ["path1.mp4"]},
        {"input_ids": [6,7], "labels": [8,9,10], "videos": ["path2.mp4"]},
        ...
    ]

    Output: Batch tensors
    {
        "input_ids": Tensor[14, max_len],  # Padded to same length
        "labels": Tensor[14, max_len],
        "pixel_values": Tensor[14, 8, 3, 224, 224],  # Videos loaded & encoded
        "attention_mask": Tensor[14, max_len]
    }
    """

    # STEP 1: Find max sequence length
    max_len = max(len(sample["input_ids"]) for sample in batch)

    # STEP 2: Pad all sequences to max_len
    input_ids = []
    labels = []
    for sample in batch:
        # Pad input_ids
        padded_ids = sample["input_ids"] + [pad_token_id] * (max_len - len(sample["input_ids"]))
        input_ids.append(padded_ids)

        # Pad labels
        padded_labels = sample["labels"] + [-100] * (max_len - len(sample["labels"]))
        labels.append(padded_labels)

    # STEP 3: Convert to tensors
    batch_tensors = {
        "input_ids": torch.tensor(input_ids),      # [14, max_len]
        "labels": torch.tensor(labels),            # [14, max_len]
        "attention_mask": torch.ones_like(input_ids)  # [14, max_len]
    }

    # STEP 4: Load and encode videos
    pixel_values = []
    for sample in batch:
        video_path = sample["videos"][0]
        # Load video file, extract frames, encode to tensors
        frames = load_video(video_path)  # Shape: [8, 3, 224, 224]
        pixel_values.append(frames)

    batch_tensors["pixel_values"] = torch.stack(pixel_values)  # [14, 8, 3, 224, 224]

    return batch_tensors
```

### Collate Function in Your System

```python
# From trainer.py:293-307
if sampler_type == "feature_balanced":
    logging_collate_fn = FeatureBalancedLoggingCollateWrapper(
        collate_fn=self.data_collator,  # The actual collate logic
        output_dir=self.args.output_dir,
        sampler=batch_sampler,
        log_filename=log_filename,
    )

# When DataLoader calls: logging_collate_fn(batch)
# It does:
class FeatureBalancedLoggingCollateWrapper:
    def __call__(self, batch):
        # 1. Log batch info (features, video names, etc.)
        self._log_batch(batch)

        # 2. Call actual collate function
        return self.collate_fn(batch)  # self.data_collator does the real work
```

## 6. Complete Iteration Example

### One Training Step

```
EPOCH 1, BATCH 1
════════════════════════════════════════════════════════════════

Training Loop:
    for batch in dataloader:  # Start iteration
        ↓

DataLoader.__iter__():
    self.sampler_iter = iter(self.batch_sampler)
    ↓

FeatureBalancedBatchSampler.__iter__():
    # Calculate which samples to include in first batch
    # Use round-robin + quota enforcement for diversity
    yield [0, 5, 12, 3, 8, 1, 9, 4, 11, 2, 7, 6, 10, 13]
    ↓

DataLoader.__next__():
    indices = next(self.sampler_iter)
    # indices = [0, 5, 12, 3, 8, 1, 9, 4, 11, 2, 7, 6, 10, 13]

    # Fetch samples (parallel if num_workers > 0)
    batch = []
    for idx in indices:
        sample = dataset[idx]
        # sample = {
        #     "input_ids": [1, 2, 3, ...],
        #     "labels": [4, 5, ...],
        #     "videos": ["data/train/treadmill_0000_...mp4"]
        # }
        batch.append(sample)

    # Collate into tensors
    batch = logging_collate_fn(batch)
    # 1. Log batch info
    # 2. Stack samples into tensors
    # 3. Load videos from disk
    # 4. Return: {
    #      "input_ids": Tensor[14, 128],
    #      "labels": Tensor[14, 128],
    #      "pixel_values": Tensor[14, 8, 3, 224, 224]
    #    }

    return batch
    ↓

Training Loop receives batch:
    batch = {
        "input_ids": Tensor[14, 128],
        "labels": Tensor[14, 128],
        "pixel_values": Tensor[14, 8, 3, 224, 224],
        "attention_mask": Tensor[14, 128]
    }

    # Move to GPU
    batch = {k: v.to("cuda") for k, v in batch.items()}

    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()
    optimizer.zero_grad()

════════════════════════════════════════════════════════════════
BATCH 1 COMPLETE - MOVE TO BATCH 2
════════════════════════════════════════════════════════════════

Training Loop:
    # Continue for loop - get next batch
    ↓

DataLoader.__next__():
    indices = next(self.sampler_iter)
    # Sampler yields next batch: [15, 20, 18, 22, ...]
    ↓

    # Repeat: fetch → collate → return
    ...
```

## 7. Multi-Worker Details

### Worker Process Architecture

```
Main Process (Training Loop)
│
├─> DataLoader
    │
    ├─> Sampler (runs in main process)
    │   └─> Yields: [0, 5, 12, 3, 8, 1, 9, 4, 11, 2, 7, 6, 10, 13]
    │
    ├─> Split indices among workers:
    │
    │   ┌─────────────────────────────────────────────────────────┐
    │   │ Worker 0 (subprocess)                                   │
    │   │   Dataset copy (read-only)                              │
    │   │   Indices: [0, 5, 12, 3]                                │
    │   │   for idx in [0, 5, 12, 3]:                            │
    │   │       sample = dataset[idx]  # Load video from disk     │
    │   │   Send samples back to main ──────────────┐             │
    │   └───────────────────────────────────────────│─────────────┘
    │                                               │
    │   ┌───────────────────────────────────────────│─────────────┐
    │   │ Worker 1 (subprocess)                     │             │
    │   │   Indices: [8, 1, 9, 4]                   │             │
    │   │   Fetch samples... ───────────────────────┼──────┐      │
    │   └───────────────────────────────────────────│──────│──────┘
    │                                               │      │
    │   ┌───────────────────────────────────────────│──────│──────┐
    │   │ Worker 2 (subprocess)                     │      │      │
    │   │   Indices: [11, 2, 7, 6]                  │      │      │
    │   │   Fetch samples... ───────────────────────┼──────┼──┐   │
    │   └───────────────────────────────────────────│──────│──│───┘
    │                                               │      │  │
    │   ┌───────────────────────────────────────────│──────│──│───┐
    │   │ Worker 3 (subprocess)                     │      │  │   │
    │   │   Indices: [10, 13]                       │      │  │   │
    │   │   Fetch samples... ───────────────────────┼──────┼──┼─┐ │
    │   └───────────────────────────────────────────│──────│──│─│─┘
    │                                               │      │  │ │
    │   Main Process Collects Results: ◄────────────┴──────┴──┴─┘
    │   all_samples = [worker0_samples + worker1_samples + ...]
    │
    └─> Collate (runs in main process)
        batch = collate_fn(all_samples)
        └─> Return batch to training loop
```

### Why Multi-Worker Helps

**Without Workers (num_workers=0)**:
```
Time per batch = Load 14 videos + Collate
               ≈ 14 × 50ms + 5ms = 705ms per batch
               └─ All sequential in main thread
```

**With 4 Workers (num_workers=4)**:
```
Time per batch = Load 14 videos in parallel + Collate
               ≈ max(14/4 × 50ms) + 5ms = 180ms per batch
               └─ 4 videos loaded in parallel per worker

Speedup: 705ms / 180ms ≈ 4x faster
```

## 8. Memory and State Management

### Persistent Workers

```python
# persistent_workers=True (recommended for multi-epoch training)
DataLoader(
    persistent_workers=True,  # Keep workers alive across epochs
    num_workers=4
)

EPOCH 1:
    __iter__() → Create 4 worker processes
    __next__() → Workers fetch batches
    __next__() → Workers fetch batches
    ...
    StopIteration → Epoch done, but workers stay alive

EPOCH 2:
    __iter__() → Reuse existing 4 workers (no spawn overhead)
    __next__() → Workers fetch batches
    ...

Benefit: No process spawn overhead between epochs (saves ~1-2 seconds per epoch)
```

### Cached DataLoader

```python
# From trainer.py:98
self._cached_train_dataloader = None

def get_train_dataloader(self):
    # Return cached dataloader if it exists
    if self._cached_train_dataloader is not None:
        return self._cached_train_dataloader

    # Create dataloader once
    self._cached_train_dataloader = DataLoader(...)
    return self._cached_train_dataloader

Why? Preserve sampler's internal state (_iter_count) across epochs
Without caching: New sampler each epoch → epoch counter resets → same shuffle
With caching: Same sampler → _iter_count increments → different shuffle each epoch
```

## 9. Key Insights

### DataLoader is Just Glue

```
DataLoader doesn't "do" much - it coordinates:
├── Sampler: Decides which samples (yields indices)
├── Dataset: Provides samples (dataset[idx])
└── Collate: Stacks samples into batch tensors

All the "smart" logic is in:
- Sampler: Feature balancing, diversity, round-robin
- Collate: Video loading, padding, tensor creation
```

### The Critical Path

```
1. Sampler yields: [0, 5, 12, 3, ...]  ← Your custom logic here
2. DataLoader fetches: dataset[0], dataset[5], ...
3. Collate stacks: List → Tensors
4. Training loop: Forward → Backward → Update

Slowest step? Usually step 2 (dataset[idx]) if loading videos from disk
Solution? Multi-worker loading (num_workers > 0)
```

### Common Confusion

**Q: Does DataLoader know about features/balancing?**
A: No! DataLoader just takes indices from sampler and fetches samples.

**Q: Where does feature balancing happen?**
A: In `FeatureBalancedBatchSampler.__iter__()` - it yields carefully chosen indices.

**Q: When are videos loaded from disk?**
A: In `dataset[idx]` call (inside DataLoader) or in `collate_fn` (depends on implementation).

**Q: Can sampler access dataset?**
A: Yes! Sampler receives dataset reference and can call `dataset[idx]` to inspect samples.
