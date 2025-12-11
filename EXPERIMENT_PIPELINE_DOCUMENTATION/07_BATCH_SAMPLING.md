# Batch Sampling Documentation

**Branch:** `sampler`
**Created:** December 9, 2025
**Purpose:** Custom batch samplers for training with class balancing and consistent logging

---

## Overview

This feature introduces custom batch samplers that replace HuggingFace's default `RandomSampler`. The samplers provide:

1. **BalancedBatchSampler** - Ensures 50/50 class balance per batch (moving vs stopped)
2. **RandomBatchSampler** - Standard random shuffling with consistent logging
3. **FeatureBalancedBatchSampler** - **(NEW 2025-12-11)** Balances ALL features across epochs with fixed batch sizes
4. **LoggingCollateWrapper** / **FeatureBalancedLoggingCollateWrapper** - Per-batch logging of video names and class/feature distribution

### Why Custom Samplers?

- **Class Imbalance Problem:** Datasets may have unequal numbers of moving/stopped videos
- **Batch-Level Balance:** Even with balanced datasets, random sampling can create batches with skewed class distributions
- **Feature Representation:** Different features (angles, distances, textures) may have varying counts - need fair representation
- **Consistent Logging:** Track exactly which samples are in each batch for debugging and reproducibility
- **Distributed Support:** Works with multi-GPU training

---

## Sampler Types

### 1. BalancedBatchSampler

**Purpose:** Ensures each training batch contains exactly 50% moving and 50% stopped samples.

**How it works:**
1. Parses video filenames to extract `speed` parameter
2. Categorizes samples: `speed > 0.0` = moving, `speed == 0.0` = stopped
3. Maintains separate lists for each class
4. Yields batches with half samples from each class (interleaved)

**Requirements:**
- Batch size must be **even**
- Video filenames must contain `speed` parameter (e.g., `speed3.0`, `speed0.0`)
- Not compatible with streaming mode

**When to use:**
- Training binary classifiers where class balance matters
- When dataset has class imbalance
- When you want guaranteed per-batch balance

### 2. RandomBatchSampler

**Purpose:** Standard random shuffling (like HuggingFace default) but with consistent interface and logging.

**How it works:**
1. Shuffles all dataset indices randomly
2. Yields batches of consecutive indices from shuffled list
3. No class balancing - batches may have any class distribution

**Requirements:**
- Any batch size allowed (doesn't need to be even)
- Works with any dataset

**When to use:**
- When class balance per batch is not critical
- When you want logging but not balancing
- Default behavior when `balanced_sampling: false`

### 3. FeatureBalancedBatchSampler (NEW - 2025-12-11)

**Purpose:** Ensures fair representation of ALL features across the entire training run, with fixed batch sizes.

**How it works:**
1. **Feature Extraction:** Parses ALL features from video filenames at initialization
2. **Cumulative Tracking:** Maintains counts of how many times each feature value has been seen
3. **Priority Scoring:** Each sample gets a priority score = sum of cumulative counts for its features
4. **Priority Selection:** Samples with lower scores (under-represented features) are prioritized
5. **Within-Priority Shuffle:** Samples with similar priority scores are shuffled for randomness
6. **Fixed Batch Size:** All batches are exactly the specified size (pads if needed)

**Features Tracked:**
| Feature | Extracted From | Example Values |
|---------|----------------|----------------|
| `texture` | `treadmill_0000_<texture>_stripe` | `subtle_gray_stripes` |
| `stripe_gray` | `stripe<N>` | `125`, `226`, `227` |
| `bg_gray` | `bg<N>` | `120`, `140` |
| `direction` | `_<dir>_` | `left`, `right`, `up`, `down` |
| `motion` | Derived from `speed<N>` | `moving`, `stopped` |
| `speed` | `speed<N>` | `0.0`, `3.0`, `14.0` |
| `angle` | `angle<N>` | `0`, `30`, `45` |
| `distance` | `dist<N>` (rounded to 1 decimal) | `1.0`, `1.1`, `2.0` |
| `dist_randomized` | Presence of `_distrand_` | `yes`, `no` |
| `center_randomized` | Presence of `_centerrand_` | `yes` |
| `brightness` | `bright<N>` | `0.00`, `0.10` |
| `contrast` | `contr<N>` | `1.00`, `1.10` |
| `resolution` | `<W>x<H>` | `640x480` |

**Requirements:**
- Fixed batch size (all batches will be exactly this size)
- Video filenames must follow the naming convention
- Not compatible with streaming mode

**When to use:**
- When you want ALL features to be fairly represented across training
- When dataset has imbalanced feature distributions (e.g., more angle=0 than angle=30)
- When training across multiple epochs and want cumulative balancing
- When you need guaranteed fixed batch sizes

**Key Difference from BalancedBatchSampler:**
- `BalancedBatchSampler`: Balances ONLY moving/stopped per batch
- `FeatureBalancedBatchSampler`: Balances ALL features across ALL epochs

### 4. Distributed Variants

All three samplers have distributed versions for multi-GPU training:
- `DistributedBalancedBatchSampler`
- `DistributedRandomBatchSampler`
- `DistributedFeatureBalancedBatchSampler`

These automatically split batches across processes and handle padding.

### 5. LoggingCollateWrapper / FeatureBalancedLoggingCollateWrapper

**Purpose:** Wraps the collate function to log batch contents before collation.

**LoggingCollateWrapper (for balanced/random samplers):**
- Batch number
- Class distribution (count and percentage of moving/stopped)
- Video filenames with speed and class label

**FeatureBalancedLoggingCollateWrapper (for feature_balanced sampler):**
- Batch number and size
- Per-batch feature distribution (all features)
- Cumulative feature distribution (key features: motion, direction, angle)
- Video filenames with motion, direction, angle, and speed

**Output files:**
- `balanced_sampling_log.txt` - when `sampler_type: balanced`
- `random_sampling_log.txt` - when `sampler_type: random` or `random_no_fix`
- `feature_balanced_sampling_log.txt` - when `sampler_type: feature_balanced`

---

## Behavior Summary

| Config | Sampler Used | Logging File |
|--------|--------------|--------------|
| `sampler_type: hf_shuffle` | HF RandomSampler | None (HF default) |
| `sampler_type: hf_sequential` | HF SequentialSampler | None (HF default) |
| `sampler_type: random_no_fix` | RandomBatchSamplerNoIterFix | `random_sampling_log.txt` |
| `sampler_type: random` (default) | RandomBatchSampler | `random_sampling_log.txt` |
| `sampler_type: balanced` | BalancedBatchSampler | `balanced_sampling_log.txt` |
| `sampler_type: feature_balanced` | FeatureBalancedBatchSampler | `feature_balanced_sampling_log.txt` |

**Legacy Config (Deprecated):**
| Config | Maps To |
|--------|---------|
| `balanced_sampling: true` | `sampler_type: balanced` |
| `disable_shuffling: true` | `sampler_type: hf_sequential` |

---

## Files Modified

### Core Source Files

| File | Change Type | Description |
|------|-------------|-------------|
| `src/llamafactory/data/sampler.py` | **CREATED** | All sampler classes: `BalancedBatchSampler`, `DistributedBalancedBatchSampler`, `RandomBatchSampler`, `DistributedRandomBatchSampler`, `FeatureBalancedBatchSampler`, `DistributedFeatureBalancedBatchSampler`, `LoggingCollateWrapper`, `FeatureBalancedLoggingCollateWrapper`, `extract_features_from_path()` helper |
| `src/llamafactory/data/__init__.py` | Modified | Export new sampler classes and helper functions |
| `src/llamafactory/train/sft/trainer.py` | Modified | Override `get_train_dataloader()` to use custom samplers including feature_balanced |
| `src/llamafactory/hparams/finetuning_args.py` | Modified | Add `sampler_type` config field with 6 options (default: "random") |

### Pipeline/CLI Files

| File | Change Type | Description |
|------|-------------|-------------|
| `run_experiment.sh` | Modified | Add `--sampler-type` CLI flag with 6 options |
| `run_full_pipeline.py` | Modified | Add `--sampler_type` argument, include in YAML config |
| `experiment_tracker.py` | Modified | Add `sampler_type` column to CSV tracking |

### Documentation Files

| File | Change Type | Description |
|------|-------------|-------------|
| `EXPERIMENT_PIPELINE_DOCUMENTATION/01_ORCHESTRATION.md` | Modified | Document `--balanced-sampling` in training params |
| `EXPERIMENT_PIPELINE_DOCUMENTATION/03_TRAINING.md` | Modified | Add "Balanced Batch Sampling" section |
| `EXPERIMENT_PIPELINE_DOCUMENTATION/QUICK_REFERENCE.md` | Modified | Add CLI usage examples and task reference |
| `EXPERIMENT_PIPELINE_DOCUMENTATION/07_BATCH_SAMPLING.md` | **CREATED** | This file |

---

## Usage

### CLI Usage

```bash
# Use feature_balanced sampler (balance ALL features across epochs)
./run_experiment.sh --sampler-type feature_balanced --batch-size 14 --epochs 5

# Use balanced sampler (50/50 moving/stopped per batch)
./run_experiment.sh --sampler-type balanced --batch-size 14 --epochs 5

# Use random sampler with per-epoch shuffling (DEFAULT)
./run_experiment.sh --sampler-type random --batch-size 14 --epochs 5

# Use HuggingFace default shuffle
./run_experiment.sh --sampler-type hf_shuffle --batch-size 14 --epochs 5

# Use sequential (no shuffle)
./run_experiment.sh --sampler-type hf_sequential --batch-size 14 --epochs 5
```

### YAML Config

```yaml
# Feature-balanced sampling (balance ALL features across epochs)
sampler_type: feature_balanced
per_device_train_batch_size: 14

# Balanced sampling (50/50 per batch)
sampler_type: balanced
per_device_train_batch_size: 14  # Must be even!

# Random sampling with per-epoch shuffling (DEFAULT)
sampler_type: random
per_device_train_batch_size: 14  # Any size allowed

# HuggingFace default
sampler_type: hf_shuffle
per_device_train_batch_size: 14
```

---

## Log Output Examples

### Balanced/Random Sampler Log

```
================================================================================
BATCH 1
================================================================================
Class Distribution: 7 MOVING (50.0%) | 7 STOPPED (50.0%)
Total samples: 14
--------------------------------------------------------------------------------
Videos in this batch:
    1. [MOVING ] speed=3.0    | treadmill_0000_stripes_right_speed3.0_angle0.mp4
    2. [STOPPED] speed=0.0    | treadmill_0001_stripes_right_speed0.0_angle0.mp4
    3. [MOVING ] speed=6.0    | treadmill_0002_stripes_right_speed6.0_angle30.mp4
    4. [STOPPED] speed=0.0    | treadmill_0003_stripes_right_speed0.0_angle30.mp4
    ...
--------------------------------------------------------------------------------
```

### Feature-Balanced Sampler Log

```
================================================================================
BATCH 1 (size=14)
================================================================================
Batch Feature Distribution:
  angle: 0:7, 30:7
  direction: left:14
  distance: 1.1:4, 1.2:5, 1.0:5
  motion: moving:7, stopped:7
  speed: 0.0:7, 14.0:7
  texture: subtle_gray_stripes:14
--------------------------------------------------------------------------------
Cumulative Distribution (key features):
  motion: moving:7(50%), stopped:7(50%)
  direction: left:14(100%)
  angle: 0:7(50%), 30:7(50%)
--------------------------------------------------------------------------------
Videos in this batch:
    1. [moving ] dir=left  angle=0   speed=14.0 | treadmill_0000_subtle_gray_stripes_stripe226_bg120_left_speed14.0_angle0_dist1.11_distrand.mp4
    2. [stopped] dir=left  angle=30  speed=0.0  | treadmill_0001_subtle_gray_stripes_stripe226_bg120_left_speed0.0_angle30_dist1.04_distrand.mp4
    3. [moving ] dir=left  angle=30  speed=14.0 | treadmill_0002_subtle_gray_stripes_stripe226_bg120_left_speed14.0_angle30_dist1.13_distrand.mp4
    ...
--------------------------------------------------------------------------------
```

---

## Implementation Details

### Speed Extraction

```python
def extract_speed_from_path(video_path: str) -> Optional[float]:
    """Extract speed value from filename using regex: speed([\d.]+)"""
    match = re.search(r"speed([\d.]+)", video_path)
    if match:
        return float(match.group(1))
    return None
```

### Class Classification

```python
def is_video_moving(video_path: str) -> Optional[bool]:
    """speed > 0.0 = moving, speed == 0.0 = stopped"""
    speed = extract_speed_from_path(video_path)
    if speed is not None:
        return speed > 0.0
    return None
```

### Feature Extraction (for FeatureBalancedBatchSampler)

```python
def extract_features_from_path(video_path: str) -> dict[str, str]:
    """Extract ALL features from video filename for feature-balanced sampling."""
    filename = os.path.basename(video_path)
    features = {}

    # Texture type (between second underscore and stripe/bg)
    texture_match = re.search(r"treadmill_\d+_([a-z_]+)_stripe", filename)
    if texture_match:
        features["texture"] = texture_match.group(1)

    # Stripe gray level
    stripe_match = re.search(r"stripe(\d+)", filename)
    if stripe_match:
        features["stripe_gray"] = stripe_match.group(1)

    # Background gray level
    bg_match = re.search(r"bg(\d+)", filename)
    if bg_match:
        features["bg_gray"] = bg_match.group(1)

    # Direction (left, right, up, down)
    dir_match = re.search(r"_(left|right|up|down)_", filename)
    if dir_match:
        features["direction"] = dir_match.group(1)

    # Speed and motion
    speed_match = re.search(r"speed([\d.]+)", filename)
    if speed_match:
        speed_val = float(speed_match.group(1))
        features["motion"] = "moving" if speed_val > 0 else "stopped"
        features["speed"] = speed_match.group(1)

    # View angle, distance, brightness, contrast, resolution...
    # (additional feature extraction)
    return features
```

### Priority Scoring (for FeatureBalancedBatchSampler)

```python
def _calculate_sample_priority(self, idx: int) -> float:
    """
    Lower score = higher priority (more under-represented).

    Score = sum of cumulative counts for all features of this sample.
    Samples with under-represented features will have lower cumulative counts,
    and thus lower scores, making them higher priority.
    """
    features = self.sample_features.get(idx, {})
    score = 0.0
    for feat_name, feat_value in features.items():
        if feat_name in self.cumulative_feature_counts:
            count = self.cumulative_feature_counts[feat_name].get(feat_value, 0)
            score += count
    return score
```

### Trainer Integration

The `CustomSeq2SeqTrainer.get_train_dataloader()` method is overridden to:

1. Check `sampler_type` configuration
2. For HF samplers (`hf_shuffle`, `hf_sequential`) → use HuggingFace default
3. Check if streaming mode → fall back to HuggingFace default (incompatible)
4. If `sampler_type: random` → use `RandomBatchSampler`
5. If `sampler_type: random_no_fix` → use `RandomBatchSamplerNoIterFix`
6. If `sampler_type: balanced` → use `BalancedBatchSampler`
7. If `sampler_type: feature_balanced` → use `FeatureBalancedBatchSampler`
8. Wrap collate function with appropriate logging wrapper

---

## Commits

| Commit | Description |
|--------|-------------|
| `d657bd8` | Initial balanced batch sampling implementation |
| `fd8cb88` | Add `--balanced-sampling` CLI flag |
| `abf922b` | Add per-batch logging with `LoggingCollateWrapper` |
| `2a74178` | Add `RandomBatchSampler` as default when balanced_sampling is OFF |
| `ddce795` | Add `--sampler-type` flag with 5 sampler options |
| `63706e3` | Add `feature_balanced` sampler for cross-epoch feature balancing |

---

## Notes

- **Batch size constraint:** Only `BalancedBatchSampler` requires even batch size; `FeatureBalancedBatchSampler` produces fixed-size batches
- **Streaming incompatibility:** Custom samplers require full dataset access (not streaming)
- **Filename dependency:** Samplers parse filenames for `speed` parameter (balanced/random) or ALL features (feature_balanced)
- **Experiment tracking:** `sampler_type` is tracked in `experiments_log.csv`
- **Feature extraction:** `FeatureBalancedBatchSampler` extracts texture, stripe_gray, bg_gray, direction, motion, speed, angle, distance, brightness, contrast, resolution

---

## Bug Fixes (December 10, 2025)

### Fixed: Epoch Shuffling Bug

**Problem:** HuggingFace Trainer calls `set_epoch()` on `dataloader.sampler`, but NOT on `dataloader.batch_sampler`. Since our samplers use `batch_sampler`, `set_epoch()` was never called, causing:
1. **Same shuffle order every epoch** - `self.epoch` stayed at 0 forever
2. **Same samples permanently excluded** - With `drop_last=True`, the same samples were dropped every epoch

**Solution:** Auto-increment `self.epoch` at the end of each `__iter__()` call in both samplers.

### Fixed: Data Loss Per Epoch

**Problem:** With `drop_last=True` and 64 samples, batch_size=14: only 56 samples (4 batches × 14) were used per epoch, dropping 8 samples (12.5%).

**Solution:** Changed `drop_last=False` so remaining samples are yielded as a partial final batch.

**Before (bug):**
```
Epoch: [Batch1: 14] [Batch2: 14] [Batch3: 14] [Batch4: 14] [DROPPED: 8]
       └─────────────── 56 samples seen (same ones dropped every epoch!) ───────────────┘
```

**After (fixed):**
```
Epoch: [Batch1: 14] [Batch2: 14] [Batch3: 14] [Batch4: 14] [Batch5: 8]
       └─────────────── 64 samples seen (100%), different order each epoch ───────────────┘
```

### Commit

| Commit | Description |
|--------|-------------|
| `664626c` | Fix sampler epoch bug and ensure all samples are trained each epoch |

---

**Last Updated:** December 11, 2025 (Added FeatureBalancedBatchSampler for cross-epoch feature balancing)
