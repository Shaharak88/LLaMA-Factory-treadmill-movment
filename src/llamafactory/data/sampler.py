# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Balanced batch sampler for binary classification tasks.

This module provides samplers that ensure each training batch contains
exactly 50% of each class (e.g., 50% moving, 50% stopped videos).

The sampler extracts class labels from video filenames by parsing the
speed parameter: speed > 0.0 = moving, speed == 0.0 = stopped.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional

import torch
from torch.utils.data import Dataset, Sampler

from ..extras import logging


logger = logging.get_logger(__name__)


def extract_speed_from_path(video_path: str) -> Optional[float]:
    """
    Extract speed value from video filename.

    Args:
        video_path: Path to video file containing speed in filename.
            Example: "data/train/treadmill_0000_stripes_right_speed6.0_angle0.mp4"

    Returns:
        Speed value as float, or None if not found.
    """
    match = re.search(r"speed([\d.]+)", video_path)
    if match:
        return float(match.group(1))
    return None


def is_video_moving(video_path: str) -> Optional[bool]:
    """
    Determine if video shows moving treadmill based on filename.

    Args:
        video_path: Path to video file.

    Returns:
        True if moving (speed > 0), False if stopped (speed == 0), None if unknown.
    """
    speed = extract_speed_from_path(video_path)
    if speed is not None:
        return speed > 0.0
    return None


class BalancedBatchSampler(Sampler[List[int]]):
    """
    Sampler that yields batches with exactly 50% of each class.

    For video classification (moving vs stopped treadmill), this ensures
    each training batch has balanced representation of both classes.

    The sampler:
    1. Categorizes all samples by parsing video filenames for speed
    2. Maintains separate lists for moving and stopped samples
    3. Yields batches with half samples from each class

    Args:
        dataset: The dataset to sample from (must have 'videos' column).
        batch_size: Total batch size (must be even).
        drop_last: Whether to drop the last incomplete batch.
        shuffle: Whether to shuffle indices within each class.
        seed: Random seed for reproducibility.

    Raises:
        ValueError: If batch_size is not even.

    Example:
        >>> sampler = BalancedBatchSampler(dataset, batch_size=4)
        >>> for batch_indices in sampler:
        ...     # batch_indices contains 2 moving + 2 stopped samples
        ...     pass
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        if batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even for balanced sampling, got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.half_batch = batch_size // 2
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0  # Can be set via set_epoch() for compatibility
        self._iter_count = 0  # Tracks __iter__ calls for reliable per-epoch shuffling

        # Categorize samples by class
        self.moving_indices: List[int] = []
        self.stopped_indices: List[int] = []
        self._categorize_samples()

        logger.info_rank0(
            f"BalancedBatchSampler initialized: {len(self.moving_indices)} moving, "
            f"{len(self.stopped_indices)} stopped samples"
        )

    def _categorize_samples(self) -> None:
        """Categorize all samples into moving/stopped based on video filename."""
        unknown_count = 0

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            videos = sample.get("videos", None)

            if videos and len(videos) > 0:
                # Use first video path for classification
                video_path = videos[0] if isinstance(videos, list) else videos
                is_moving = is_video_moving(video_path)

                if is_moving is True:
                    self.moving_indices.append(idx)
                elif is_moving is False:
                    self.stopped_indices.append(idx)
                else:
                    # Cannot determine class from filename
                    unknown_count += 1
            else:
                unknown_count += 1

        if unknown_count > 0:
            logger.warning_rank0(
                f"BalancedBatchSampler: {unknown_count} samples could not be categorized "
                f"(missing videos or speed not in filename)"
            )

    def set_epoch(self, epoch: int) -> None:
        """
        Set epoch for deterministic shuffling.

        This ensures different orderings across epochs while maintaining
        reproducibility when using the same seed.

        Args:
            epoch: Current epoch number.
        """
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        """Yield balanced batches of indices."""
        g = torch.Generator()
        # Use _iter_count for seeding, then increment IMMEDIATELY (before any yields)
        # This ensures each __iter__ call gets a different seed, regardless of generator exhaustion
        g.manual_seed(self.seed + self._iter_count)
        self._iter_count += 1

        # Shuffle indices within each class if enabled
        if self.shuffle:
            moving_perm = torch.randperm(len(self.moving_indices), generator=g).tolist()
            stopped_perm = torch.randperm(len(self.stopped_indices), generator=g).tolist()
        else:
            moving_perm = list(range(len(self.moving_indices)))
            stopped_perm = list(range(len(self.stopped_indices)))

        moving_shuffled = [self.moving_indices[i] for i in moving_perm]
        stopped_shuffled = [self.stopped_indices[i] for i in stopped_perm]

        # Determine number of complete batches (limited by smaller class)
        num_moving_batches = len(moving_shuffled) // self.half_batch
        num_stopped_batches = len(stopped_shuffled) // self.half_batch
        num_batches = min(num_moving_batches, num_stopped_batches)

        # Generate balanced batches
        for batch_idx in range(num_batches):
            start = batch_idx * self.half_batch
            end = start + self.half_batch

            batch_moving = moving_shuffled[start:end]
            batch_stopped = stopped_shuffled[start:end]

            # Interleave samples (helps with gradient variance)
            batch = []
            for m, s in zip(batch_moving, batch_stopped):
                batch.extend([m, s])

            yield batch

        # Handle remaining samples if not dropping last
        if not self.drop_last:
            remaining_moving = moving_shuffled[num_batches * self.half_batch :]
            remaining_stopped = stopped_shuffled[num_batches * self.half_batch :]

            if remaining_moving or remaining_stopped:
                # Yield partial batch (won't be perfectly balanced)
                yield remaining_moving + remaining_stopped

        # Note: epoch increment moved to START of __iter__ using _iter_count
        # This ensures reliable per-epoch shuffling regardless of generator exhaustion

    def __len__(self) -> int:
        """Return number of batches."""
        num_moving_batches = len(self.moving_indices) // self.half_batch
        num_stopped_batches = len(self.stopped_indices) // self.half_batch
        num_batches = min(num_moving_batches, num_stopped_batches)

        if not self.drop_last:
            remaining = len(self.moving_indices) % self.half_batch + len(self.stopped_indices) % self.half_batch
            if remaining > 0:
                num_batches += 1

        return num_batches


class DistributedBalancedBatchSampler(BalancedBatchSampler):
    """
    Balanced batch sampler with distributed training support.

    Each process gets a subset of batches, ensuring all processes
    have the same number of batches (padding if necessary).

    Args:
        dataset: The dataset to sample from.
        batch_size: Total batch size per process (must be even).
        num_replicas: Number of distributed processes.
        rank: Current process rank.
        drop_last: Whether to drop incomplete batches.
        shuffle: Whether to shuffle within each class.
        seed: Random seed.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__(dataset, batch_size, drop_last, shuffle, seed)

        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Distributed package not available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Distributed package not available")
            rank = torch.distributed.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches assigned to this rank."""
        all_batches = list(super().__iter__())

        # Pad to make divisible by num_replicas
        remainder = len(all_batches) % self.num_replicas
        if remainder != 0:
            # Pad with repeated batches from the beginning
            padding = self.num_replicas - remainder
            all_batches.extend(all_batches[:padding])

        # Select batches for this rank
        batches_per_replica = len(all_batches) // self.num_replicas
        start_idx = self.rank * batches_per_replica
        end_idx = start_idx + batches_per_replica

        for batch in all_batches[start_idx:end_idx]:
            yield batch

    def __len__(self) -> int:
        """Return number of batches for this rank."""
        total_batches = super().__len__()
        # Ceiling division to account for padding
        return (total_batches + self.num_replicas - 1) // self.num_replicas


class RandomBatchSampler(Sampler[List[int]]):
    """
    Simple random batch sampler that shuffles all samples without class balancing.

    Unlike BalancedBatchSampler, this does NOT enforce class balance.
    Each batch contains randomly selected samples from the entire dataset.

    Used when balanced_sampling is OFF but we still want consistent
    behavior and logging compared to BalancedBatchSampler.

    Args:
        dataset: The dataset to sample from.
        batch_size: Total batch size (any size allowed, unlike BalancedBatchSampler).
        drop_last: Whether to drop the last incomplete batch.
        shuffle: Whether to shuffle indices.
        seed: Random seed for reproducibility.

    Example:
        >>> sampler = RandomBatchSampler(dataset, batch_size=4)
        >>> for batch_indices in sampler:
        ...     # batch_indices contains 4 randomly selected samples
        ...     pass
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0  # Can be set via set_epoch() for compatibility
        self._iter_count = 0  # Tracks __iter__ calls for reliable per-epoch shuffling
        self.all_indices = list(range(len(dataset)))

        logger.info_rank0(
            f"RandomBatchSampler initialized: {len(self.all_indices)} samples, "
            f"batch_size={batch_size}, shuffle={shuffle}"
        )

    def set_epoch(self, epoch: int) -> None:
        """
        Set epoch for deterministic shuffling.

        This ensures different orderings across epochs while maintaining
        reproducibility when using the same seed.

        Args:
            epoch: Current epoch number.
        """
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches of indices."""
        g = torch.Generator()
        # Use _iter_count for seeding, then increment IMMEDIATELY (before any yields)
        # This ensures each __iter__ call gets a different seed, regardless of generator exhaustion
        g.manual_seed(self.seed + self._iter_count)
        self._iter_count += 1

        if self.shuffle:
            perm = torch.randperm(len(self.all_indices), generator=g).tolist()
            indices = [self.all_indices[i] for i in perm]
        else:
            indices = self.all_indices.copy()

        # Yield batches
        num_batches = len(indices) // self.batch_size
        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            yield indices[start:end]

        # Handle remaining if not drop_last
        if not self.drop_last:
            remaining = indices[num_batches * self.batch_size:]
            if remaining:
                yield remaining

        # Note: epoch increment moved to START of __iter__ using _iter_count
        # This ensures reliable per-epoch shuffling regardless of generator exhaustion

    def __len__(self) -> int:
        """Return number of batches."""
        if self.drop_last:
            return len(self.all_indices) // self.batch_size
        return (len(self.all_indices) + self.batch_size - 1) // self.batch_size


class RandomBatchSamplerNoIterFix(Sampler[List[int]]):
    """
    Random batch sampler WITHOUT the per-epoch shuffling fix.

    This sampler uses only self.epoch for seeding (which stays at 0 since
    HuggingFace doesn't call set_epoch() on batch samplers). This means
    the SAME shuffle order is used every epoch.

    Useful for:
    - Debugging/comparing behavior with vs without the iter fix
    - Reproducing the old behavior where batch 1 of epoch 1 == batch 1 of epoch 2
    - Testing deterministic training scenarios

    Unlike RandomBatchSampler (which uses _iter_count for different per-epoch shuffling),
    this sampler produces identical batch ordering across all epochs.

    Args:
        dataset: The dataset to sample from.
        batch_size: Total batch size (any size allowed).
        drop_last: Whether to drop the last incomplete batch.
        shuffle: Whether to shuffle indices.
        seed: Random seed for reproducibility.

    Example:
        >>> sampler = RandomBatchSamplerNoIterFix(dataset, batch_size=4)
        >>> # Epoch 1 and Epoch 2 will have identical batch order
        >>> for batch_indices in sampler:
        ...     pass
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0  # NOT auto-incremented - same order every epoch
        self.all_indices = list(range(len(dataset)))

        logger.info_rank0(
            f"RandomBatchSamplerNoIterFix initialized: {len(self.all_indices)} samples, "
            f"batch_size={batch_size}, shuffle={shuffle} (SAME ORDER EVERY EPOCH)"
        )

    def set_epoch(self, epoch: int) -> None:
        """
        Set epoch for deterministic shuffling.

        Note: HuggingFace Trainer does NOT call this on batch_sampler,
        so self.epoch will stay at 0 unless explicitly set externally.
        This results in the same shuffle order every epoch.

        Args:
            epoch: Current epoch number.
        """
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches of indices (same order every epoch)."""
        g = torch.Generator()
        # Use only self.epoch for seeding (stays at 0 = same order every epoch)
        # This is intentional - NO _iter_count increment here
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            perm = torch.randperm(len(self.all_indices), generator=g).tolist()
            indices = [self.all_indices[i] for i in perm]
        else:
            indices = self.all_indices.copy()

        # Yield batches
        num_batches = len(indices) // self.batch_size
        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            yield indices[start:end]

        # Handle remaining if not drop_last
        if not self.drop_last:
            remaining = indices[num_batches * self.batch_size:]
            if remaining:
                yield remaining

        # Note: NO epoch increment here - intentionally same order every epoch

    def __len__(self) -> int:
        """Return number of batches."""
        if self.drop_last:
            return len(self.all_indices) // self.batch_size
        return (len(self.all_indices) + self.batch_size - 1) // self.batch_size


class DistributedRandomBatchSamplerNoIterFix(RandomBatchSamplerNoIterFix):
    """
    Random batch sampler WITHOUT iter fix, with distributed training support.

    Each process gets a subset of batches, ensuring all processes
    have the same number of batches (padding if necessary).
    Same shuffle order every epoch (no _iter_count fix).

    Args:
        dataset: The dataset to sample from.
        batch_size: Total batch size per process.
        num_replicas: Number of distributed processes.
        rank: Current process rank.
        drop_last: Whether to drop incomplete batches.
        shuffle: Whether to shuffle indices.
        seed: Random seed.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__(dataset, batch_size, drop_last, shuffle, seed)

        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Distributed package not available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Distributed package not available")
            rank = torch.distributed.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches assigned to this rank."""
        all_batches = list(super().__iter__())

        # Pad to make divisible by num_replicas
        remainder = len(all_batches) % self.num_replicas
        if remainder != 0:
            # Pad with repeated batches from the beginning
            padding = self.num_replicas - remainder
            all_batches.extend(all_batches[:padding])

        # Select batches for this rank
        batches_per_replica = len(all_batches) // self.num_replicas
        start_idx = self.rank * batches_per_replica
        end_idx = start_idx + batches_per_replica

        for batch in all_batches[start_idx:end_idx]:
            yield batch

    def __len__(self) -> int:
        """Return number of batches for this rank."""
        total_batches = super().__len__()
        # Ceiling division to account for padding
        return (total_batches + self.num_replicas - 1) // self.num_replicas


class DistributedRandomBatchSampler(RandomBatchSampler):
    """
    Random batch sampler with distributed training support.

    Each process gets a subset of batches, ensuring all processes
    have the same number of batches (padding if necessary).

    Args:
        dataset: The dataset to sample from.
        batch_size: Total batch size per process.
        num_replicas: Number of distributed processes.
        rank: Current process rank.
        drop_last: Whether to drop incomplete batches.
        shuffle: Whether to shuffle indices.
        seed: Random seed.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__(dataset, batch_size, drop_last, shuffle, seed)

        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Distributed package not available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Distributed package not available")
            rank = torch.distributed.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches assigned to this rank."""
        all_batches = list(super().__iter__())

        # Pad to make divisible by num_replicas
        remainder = len(all_batches) % self.num_replicas
        if remainder != 0:
            # Pad with repeated batches from the beginning
            padding = self.num_replicas - remainder
            all_batches.extend(all_batches[:padding])

        # Select batches for this rank
        batches_per_replica = len(all_batches) // self.num_replicas
        start_idx = self.rank * batches_per_replica
        end_idx = start_idx + batches_per_replica

        for batch in all_batches[start_idx:end_idx]:
            yield batch

    def __len__(self) -> int:
        """Return number of batches for this rank."""
        total_batches = super().__len__()
        # Ceiling division to account for padding
        return (total_batches + self.num_replicas - 1) // self.num_replicas


def extract_features_from_path(video_path: str) -> dict[str, str]:
    """
    Extract all features from video filename for feature-balanced sampling.

    Parses features from filenames like:
    treadmill_0000_subtle_gray_stripes_stripe226_bg120_left_speed0.0_angle0_dist1.13_distrand_bright0.00_contr1.00_640x480_seed42.mp4

    Args:
        video_path: Path to video file.

    Returns:
        Dictionary of feature_name -> feature_value (as strings for grouping).
    """
    import re
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

    # Speed (categorize as moving/stopped for balance)
    speed_match = re.search(r"speed([\d.]+)", filename)
    if speed_match:
        speed_val = float(speed_match.group(1))
        features["motion"] = "moving" if speed_val > 0 else "stopped"
        # Also store actual speed for granular balancing
        features["speed"] = speed_match.group(1)

    # View angle
    angle_match = re.search(r"angle(\d+)", filename)
    if angle_match:
        features["angle"] = angle_match.group(1)

    # Distance
    dist_match = re.search(r"dist([\d.]+)", filename)
    if dist_match:
        # Round to 1 decimal for grouping
        dist_val = round(float(dist_match.group(1)), 1)
        features["distance"] = str(dist_val)

    # Distance randomization flag
    if "_distrand_" in filename:
        features["dist_randomized"] = "yes"
    elif "_dist" in filename and "_distrand_" not in filename:
        features["dist_randomized"] = "no"

    # Center randomization
    if "_center_randomized_" in filename or "_centerrand_" in filename:
        features["center_randomized"] = "yes"

    # Brightness
    bright_match = re.search(r"bright([\d.-]+)", filename)
    if bright_match:
        features["brightness"] = bright_match.group(1)

    # Contrast
    contr_match = re.search(r"contr([\d.]+)", filename)
    if contr_match:
        features["contrast"] = contr_match.group(1)

    # Resolution
    res_match = re.search(r"(\d+x\d+)", filename)
    if res_match:
        features["resolution"] = res_match.group(1)

    return features


class FeatureBalancedBatchSampler(Sampler[List[int]]):
    """
    Sampler that balances ALL features with per-batch diversity and per-epoch fairness.

    This sampler ensures:
    1. Per-batch diversity: Each batch contains diverse feature combinations
    2. 50/50 moving/stopped balance per batch
    3. Per-epoch fairness: All features represented equally within each epoch
    4. Round-robin cycling: Different feature ordering across epochs
    5. Detailed statistics tracking: Logs how many times each video is seen

    Features extracted from video filenames:
    - texture, stripe_gray, bg_gray, direction, motion, speed, angle,
      distance, brightness, contrast, resolution

    Key behaviors:
    - Fixed batch size (always exactly batch_size)
    - Per-batch: Maximizes feature diversity + 50/50 moving/stopped
    - Per-epoch: Ensures all features represented equally
    - Cumulative tracking: Tracks video-level counts across all epochs
    - Statistics file: Writes detailed stats after each epoch and at end

    Args:
        dataset: The dataset to sample from (must have 'videos' column).
        batch_size: Fixed batch size (all batches exactly this size).
        drop_last: Whether to drop samples that don't fit in complete batches.
        shuffle: Whether to shuffle within feature groups.
        seed: Random seed for reproducibility.
        output_dir: Output directory for statistics file (optional).

    Example:
        >>> sampler = FeatureBalancedBatchSampler(dataset, batch_size=14)
        >>> for batch_indices in sampler:
        ...     # batch has 14 samples with diverse features + 50/50 balance
        ...     pass
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        output_dir: Optional[str] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._iter_count = 0
        self.output_dir = Path(output_dir) if output_dir else None

        # Feature tracking
        self.sample_features: dict[int, dict[str, str]] = {}  # idx -> features dict
        self.feature_values: dict[str, set[str]] = {}  # feature_name -> set of values
        self.cumulative_feature_counts: dict[str, dict[str, int]] = {}  # feature -> value -> count

        # Video-level tracking (how many times each video seen)
        self.video_seen_counts: dict[int, int] = {}  # idx -> count
        self.epoch_stats: list[dict] = []  # Per-epoch statistics

        # Categorize by motion class
        self.moving_indices: list[int] = []
        self.stopped_indices: list[int] = []

        # Extract features from all samples
        self._extract_all_features()

        # Initialize cumulative counts
        self._init_cumulative_counts()

        # Initialize video counts
        for idx in range(len(self.dataset)):
            self.video_seen_counts[idx] = 0

        # Distance-specific tracking for priority-weighted sampling
        self.distance_priority_scores: dict[str, float] = {}  # distance -> priority score
        self.distance_target_count: int = 0  # target samples per distance
        self.moving_distance_groups: dict[str, dict[tuple, list[int]]] = {}  # distance -> (angle, dir) -> indices
        self.stopped_distance_groups: dict[str, dict[tuple, list[int]]] = {}  # distance -> (angle, dir) -> indices

        # Initialize distance groups and priorities
        self._init_distance_groups()
        self._update_distance_priorities()

        logger.info_rank0(
            f"FeatureBalancedBatchSampler initialized: {len(self.dataset)} samples, "
            f"batch_size={batch_size}, features tracked: {list(self.feature_values.keys())}"
        )
        logger.info_rank0(f"  Moving: {len(self.moving_indices)}, Stopped: {len(self.stopped_indices)}")
        for feat, values in self.feature_values.items():
            logger.info_rank0(f"  {feat}: {sorted(values)}")

        # Log per-batch quota limits (critical for understanding diversity enforcement)
        half_batch = batch_size // 2
        logger.info_rank0(f"\n  PER-BATCH QUOTA LIMITS (max samples per feature value per half-batch of {half_batch}):")
        for feat_name, values in self.feature_values.items():
            num_values = len(values)
            if num_values > 0:
                quota = (half_batch + num_values - 1) // num_values  # ceil division
                logger.info_rank0(f"    {feat_name}: max {quota} per value (ceil({half_batch}/{num_values}))")

    def _extract_all_features(self) -> None:
        """Extract features from all samples and categorize by motion class."""
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            videos = sample.get("videos", None)

            if videos and len(videos) > 0:
                video_path = videos[0] if isinstance(videos, list) else videos
                features = extract_features_from_path(video_path)
                self.sample_features[idx] = features

                # Categorize by motion class
                if features.get("motion") == "moving":
                    self.moving_indices.append(idx)
                elif features.get("motion") == "stopped":
                    self.stopped_indices.append(idx)

                # Track all unique values for each feature
                for feat_name, feat_value in features.items():
                    if feat_name not in self.feature_values:
                        self.feature_values[feat_name] = set()
                    self.feature_values[feat_name].add(feat_value)
            else:
                self.sample_features[idx] = {}

    def _init_cumulative_counts(self) -> None:
        """Initialize cumulative feature counts to zero."""
        for feat_name, values in self.feature_values.items():
            self.cumulative_feature_counts[feat_name] = {v: 0 for v in values}

    def _calculate_sample_oversample_weights(self) -> dict[int, int]:
        """
        Calculate oversample weights for each sample based on feature rarity.

        For each feature, samples with rare values should appear more often.
        The weight is the maximum oversample factor across all features.

        Example: If angle0 has 6 samples and angle30 has 2 samples,
        angle30 videos get weight=3 (6/2=3x more appearances).

        Note: motion is excluded - 50/50 balance handled separately.

        Returns:
            Dict mapping sample index -> oversample weight (how many times to include)
        """
        # Calculate counts per feature value (excluding motion - handled separately)
        feature_value_counts: dict[str, dict[str, int]] = {}
        for feat_name in self.feature_values.keys():
            if feat_name == "motion":  # Skip motion - handled via 50/50 balance
                continue
            feature_value_counts[feat_name] = {}
            for idx, features in self.sample_features.items():
                value = features.get(feat_name, "unknown")
                feature_value_counts[feat_name][value] = feature_value_counts[feat_name].get(value, 0) + 1

        # Calculate max count per feature (target for balancing)
        feature_max_counts: dict[str, int] = {}
        for feat_name, value_counts in feature_value_counts.items():
            if value_counts:
                feature_max_counts[feat_name] = max(value_counts.values())

        # Calculate oversample weight for each sample
        sample_weights: dict[int, int] = {}
        for idx, features in self.sample_features.items():
            max_weight = 1
            for feat_name, value in features.items():
                if feat_name == "motion":
                    continue
                if feat_name in feature_value_counts and feat_name in feature_max_counts:
                    value_count = feature_value_counts[feat_name].get(value, 1)
                    max_count = feature_max_counts[feat_name]
                    # Weight = how many times more this sample should appear
                    weight = max(1, round(max_count / value_count))
                    max_weight = max(max_weight, weight)
            sample_weights[idx] = max_weight

        # Log oversample statistics
        weights_distribution = {}
        for weight in sample_weights.values():
            weights_distribution[weight] = weights_distribution.get(weight, 0) + 1
        logger.info_rank0(f"Oversample weights distribution: {dict(sorted(weights_distribution.items()))}")

        return sample_weights

    def _create_oversampled_pool(self, indices: list[int], weights: dict[int, int]) -> list[int]:
        """
        Create an oversampled pool where rare-feature samples appear multiple times.

        Args:
            indices: Original list of sample indices
            weights: Dict mapping index -> oversample weight

        Returns:
            Oversampled pool with rare samples repeated
        """
        pool = []
        for idx in indices:
            weight = weights.get(idx, 1)
            pool.extend([idx] * weight)
        return pool

    def _select_diverse_samples(
        self,
        pool: list[int],
        target_count: int,
        generator: torch.Generator,
    ) -> list[int]:
        """
        Select samples ensuring per-batch diversity across ALL features using QUOTA-BASED selection.

        QUOTA-BASED SELECTION (the key fix):
        For each feature, enforces max ceil(target_count / num_unique_values) samples
        per feature value per batch. This ensures:
        - Distance 1.0 (even with 64 videos) can only appear once per half-batch if there are 10+ distances
        - Rare distances (with few videos) get fair representation via oversampling pool
        - No single feature value dominates a batch

        Example: batch_size=8, half_batch=4, 10 distances, 2 angles
        - max_per_distance = ceil(4 / 10) = 1 → each distance at most ONCE per half-batch
        - max_per_angle = ceil(4 / 2) = 2 → each angle at most TWICE per half-batch
        - Result: 4 different distances selected, roughly balanced angles

        Args:
            pool: Available sample indices (may contain duplicates from oversampling)
            target_count: Number of samples to select
            generator: Random generator

        Returns:
            List of selected indices with maximum feature diversity enforced by quotas
        """
        if len(pool) == 0:
            return []

        selected = []

        # Calculate max quota per feature value: ceil(target_count / num_unique_values)
        # This ensures no feature value appears more than its fair share in a batch
        feature_max_quota: dict[str, int] = {}
        for feat_name, values in self.feature_values.items():
            num_values = len(values)
            if num_values > 0:
                # ceil(target_count / num_values) using integer arithmetic
                feature_max_quota[feat_name] = (target_count + num_values - 1) // num_values

        # Track how many times each feature value has been selected in THIS batch
        feature_value_counts: dict[str, dict[str, int]] = {}
        for feat_name, values in self.feature_values.items():
            feature_value_counts[feat_name] = {v: 0 for v in values}

        available = pool.copy()

        # Shuffle available pool
        perm = torch.randperm(len(available), generator=generator).tolist()
        available = [available[i] for i in perm]

        while len(selected) < target_count and available:
            best_idx = None
            best_score = -1
            best_pos = -1

            # Find sample that:
            # 1. Doesn't exceed quota for ANY feature
            # 2. Adds most diversity (prioritize NEW feature values with count=0)
            for pos, idx in enumerate(available):
                features = self.sample_features.get(idx, {})

                # Check if this sample would exceed quota for any feature
                exceeds_quota = False
                for feat_name, feat_value in features.items():
                    max_q = feature_max_quota.get(feat_name, target_count)  # Default to target_count if unknown
                    current_count = feature_value_counts.get(feat_name, {}).get(feat_value, 0)
                    if current_count >= max_q:
                        exceeds_quota = True
                        break

                if exceeds_quota:
                    continue  # Skip this sample, would exceed quota

                # Score = number of NEW feature values (values with count=0 in this batch)
                score = 0
                for feat_name, feat_value in features.items():
                    if feature_value_counts.get(feat_name, {}).get(feat_value, 0) == 0:
                        score += 1

                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_pos = pos

            if best_idx is None:
                # No samples available that don't exceed quota
                # Fall back: take any sample (edge case when pool is exhausted of valid options)
                if available:
                    best_idx = available[0]
                    best_pos = 0
                else:
                    break

            # Add selected sample
            selected.append(best_idx)

            # Update feature value counts for this batch
            features = self.sample_features.get(best_idx, {})
            for feat_name, feat_value in features.items():
                if feat_name in feature_value_counts:
                    if feat_value not in feature_value_counts[feat_name]:
                        feature_value_counts[feat_name][feat_value] = 0
                    feature_value_counts[feat_name][feat_value] += 1

            # Remove from available (only this occurrence if duplicated)
            available.pop(best_pos)

        return selected

    def _init_distance_groups(self) -> None:
        """
        Pre-group samples by distance (primary) and (angle, direction) (secondary).

        Creates hierarchical grouping structure:
            moving_distance_groups[distance][(angle, direction)] = [sample_indices]
            stopped_distance_groups[distance][(angle, direction)] = [sample_indices]

        This enables distance-prioritized sampling while maintaining angle/direction diversity
        within each distance group.
        """
        # Group moving samples
        for idx in self.moving_indices:
            features = self.sample_features.get(idx, {})
            distance = features.get("distance", "unknown")
            angle = features.get("angle", "unknown")
            direction = features.get("direction", "unknown")

            if distance not in self.moving_distance_groups:
                self.moving_distance_groups[distance] = {}

            key = (angle, direction)
            if key not in self.moving_distance_groups[distance]:
                self.moving_distance_groups[distance][key] = []

            self.moving_distance_groups[distance][key].append(idx)

        # Group stopped samples
        for idx in self.stopped_indices:
            features = self.sample_features.get(idx, {})
            distance = features.get("distance", "unknown")
            angle = features.get("angle", "unknown")
            direction = features.get("direction", "unknown")

            if distance not in self.stopped_distance_groups:
                self.stopped_distance_groups[distance] = {}

            key = (angle, direction)
            if key not in self.stopped_distance_groups[distance]:
                self.stopped_distance_groups[distance][key] = []

            self.stopped_distance_groups[distance][key].append(idx)

        # Log distance grouping statistics
        moving_distances = sorted(self.moving_distance_groups.keys())
        stopped_distances = sorted(self.stopped_distance_groups.keys())
        logger.info_rank0(f"Distance-based hierarchical grouping complete:")
        logger.info_rank0(f"  Moving distances: {moving_distances}")
        logger.info_rank0(f"  Stopped distances: {stopped_distances}")

    def _update_distance_priorities(self) -> None:
        """
        Calculate priority scores for each distance value based on representation gap.

        Priority calculation:
            priority = 1.0 + (target_count - current_count) / target_count

        Higher priority (> 1.0) = under-represented, will get more samples
        Lower priority (< 1.0) = over-represented, will get fewer samples
        Equal priority (= 1.0) = perfectly balanced
        """
        # Get all unique distances
        all_distances = set()
        if "distance" in self.cumulative_feature_counts:
            all_distances = set(self.cumulative_feature_counts["distance"].keys())

        if not all_distances or len(all_distances) == 0:
            return

        # Calculate target count per distance
        total_samples = sum(self.cumulative_feature_counts["distance"].values())

        if total_samples == 0:
            # First epoch - equal priority for all distances
            self.distance_target_count = len(self.dataset) // len(all_distances)
            for dist in all_distances:
                self.distance_priority_scores[dist] = 1.0
            logger.info_rank0(f"Distance priorities initialized: all distances have equal priority (1.0)")
        else:
            # Calculate target and priorities based on cumulative counts
            self.distance_target_count = total_samples // len(all_distances)

            # Calculate priority for each distance
            for dist in all_distances:
                current_count = self.cumulative_feature_counts["distance"].get(dist, 0)
                gap = self.distance_target_count - current_count
                # Normalize to 0-2 range (2 = very under-represented, 0 = over-represented)
                priority = max(0.0, 1.0 + (gap / max(1, self.distance_target_count)))
                self.distance_priority_scores[dist] = priority

            # Log top under-represented distances
            sorted_priorities = sorted(
                self.distance_priority_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            logger.info_rank0(
                f"Distance priorities updated - Top 5 under-represented: "
                f"{[(d, f'{p:.3f}') for d, p in sorted_priorities]}"
            )

    def _group_by_features(self, indices: list[int], feature_keys: list[str]) -> dict[tuple, list[int]]:
        """
        Group indices by their feature values.

        Args:
            indices: List of sample indices to group.
            feature_keys: Which features to use for grouping.

        Returns:
            Dict mapping feature_tuple -> list of indices with those features.
        """
        groups: dict[tuple, list[int]] = {}
        for idx in indices:
            features = self.sample_features.get(idx, {})
            # Create tuple of feature values for grouping
            key = tuple(features.get(f, "unknown") for f in feature_keys)
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)
        return groups

    def _regroup_pool_by_distance(self, pool: list[int], motion_class: str) -> dict[str, list[int]]:
        """
        Re-group pool indices by their distance values.

        Args:
            pool: List of sample indices currently available.
            motion_class: "moving" or "stopped" (for logging).

        Returns:
            Dict mapping distance -> list of indices with that distance.

        Note:
            This is needed because the pool changes dynamically as batches are created,
            so we need to re-group the remaining samples by distance for each batch.
        """
        groups: dict[str, list[int]] = {}
        for idx in pool:
            features = self.sample_features.get(idx, {})
            distance = features.get("distance", "unknown")

            if distance not in groups:
                groups[distance] = []
            groups[distance].append(idx)

        return groups

    def _select_distance_prioritized_samples(
        self,
        distance_groups: dict[str, list[int]],
        target_count: int,
        generator: torch.Generator,
        motion_class: str,
    ) -> list[int]:
        """
        Select samples prioritizing under-represented distances.

        Algorithm:
        1. Create priority-sorted list of distances
        2. Round-robin through distances, weighted by priority
        3. Higher priority distances get more samples per cycle (1-3)
        4. Lower priority distances get fewer samples (1)
        5. Continue until target_count samples selected

        Args:
            distance_groups: Dict of distance -> [indices].
            target_count: Number of samples to select.
            generator: Random generator for shuffling.
            motion_class: "moving" or "stopped" (for logging).

        Returns:
            List of selected indices with distance-prioritized diversity.
        """
        selected = []

        # Create list of (distance, priority, available_indices)
        distance_queue = []
        for distance, indices in distance_groups.items():
            if len(indices) > 0:
                priority = self.distance_priority_scores.get(distance, 1.0)
                distance_queue.append(
                    {"distance": distance, "priority": priority, "indices": indices.copy()}
                )

        # Sort by priority (descending)
        distance_queue.sort(key=lambda x: x["priority"], reverse=True)

        # Round-robin with priority weighting
        cycle_count = 0
        max_cycles = target_count * 2  # Safety limit to avoid infinite loops

        while len(selected) < target_count and len(distance_queue) > 0 and cycle_count < max_cycles:
            cycle_count += 1

            # Go through each distance in priority order
            for dist_info in distance_queue:
                if len(selected) >= target_count:
                    break

                if len(dist_info["indices"]) == 0:
                    continue

                # Higher priority = more samples per cycle (1-3)
                # priority >= 1.5 -> 3 samples, priority >= 1.2 -> 2 samples, else 1 sample
                if dist_info["priority"] >= 1.5:
                    samples_to_take = 3
                elif dist_info["priority"] >= 1.2:
                    samples_to_take = 2
                else:
                    samples_to_take = 1

                samples_to_take = min(
                    samples_to_take, target_count - len(selected), len(dist_info["indices"])
                )

                # Randomly select from this distance group
                # Get shuffled indices, then sort in descending order so we can pop safely
                # (popping from higher indices first keeps lower indices valid)
                perm = torch.randperm(len(dist_info["indices"]), generator=generator).tolist()
                indices_to_pop = sorted(perm[:samples_to_take], reverse=True)
                for pop_idx in indices_to_pop:
                    selected.append(dist_info["indices"].pop(pop_idx))

            # Remove empty distance groups
            distance_queue = [d for d in distance_queue if len(d["indices"]) > 0]

        # If still need more samples, take any available
        if len(selected) < target_count:
            all_remaining = []
            for dist_info in distance_queue:
                all_remaining.extend(dist_info["indices"])

            needed = target_count - len(selected)
            if len(all_remaining) >= needed:
                perm = torch.randperm(len(all_remaining), generator=generator).tolist()
                selected.extend([all_remaining[i] for i in perm[:needed]])
            elif len(all_remaining) > 0:
                # Take what's available
                selected.extend(all_remaining)

        return selected

    def _create_diverse_batch(
        self,
        moving_pool: list[int],
        stopped_pool: list[int],
        half_batch: int,
        generator: torch.Generator,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Create one batch with ALL-FEATURE diversity + 50/50 moving/stopped balance.

        This method ensures:
        1. Maximum diversity across ALL features (distance, angle, direction, etc.)
        2. At most 1 sample per feature value when unique_values > half_batch
        3. Maintaining 50/50 moving/stopped balance per batch
        4. Rare feature values get selected (from oversampled pool)

        Args:
            moving_pool: Available moving indices (may have duplicates from oversampling).
            stopped_pool: Available stopped indices (may have duplicates from oversampling).
            half_batch: Number of samples per class (batch_size // 2).
            generator: Random generator for shuffling.

        Returns:
            (batch_indices, remaining_moving, remaining_stopped)
        """
        # Select moving samples with ALL-feature diversity
        moving_batch = self._select_diverse_samples(moving_pool, half_batch, generator)

        # Select stopped samples with ALL-feature diversity
        stopped_batch = self._select_diverse_samples(stopped_pool, half_batch, generator)

        # Interleave moving and stopped for batch (preserves 50/50 balance)
        batch = []
        for m, s in zip(moving_batch, stopped_batch):
            batch.extend([m, s])
        # Add any extras
        batch.extend(moving_batch[len(stopped_batch):])
        batch.extend(stopped_batch[len(moving_batch):])

        # Remove selected from pools (remove only ONE occurrence per selected item)
        remaining_moving = moving_pool.copy()
        for idx in moving_batch:
            if idx in remaining_moving:
                remaining_moving.remove(idx)  # Removes first occurrence only

        remaining_stopped = stopped_pool.copy()
        for idx in stopped_batch:
            if idx in remaining_stopped:
                remaining_stopped.remove(idx)  # Removes first occurrence only

        return batch, remaining_moving, remaining_stopped

    def _update_counts(self, indices: List[int]) -> None:
        """Update cumulative feature counts and video seen counts."""
        for idx in indices:
            # Update video-level tracking
            self.video_seen_counts[idx] += 1

            # Update feature counts
            features = self.sample_features.get(idx, {})
            for feat_name, feat_value in features.items():
                if feat_name in self.cumulative_feature_counts:
                    if feat_value in self.cumulative_feature_counts[feat_name]:
                        self.cumulative_feature_counts[feat_name][feat_value] += 1

    def _write_statistics(self, epoch_num: int, is_final: bool = False) -> None:
        """Write detailed statistics to file."""
        if not self.output_dir:
            return

        stats_file = self.output_dir / "feature_balanced_statistics.txt"
        mode = "a" if stats_file.exists() and not is_final else "w"

        with open(stats_file, mode, encoding="utf-8") as f:
            if is_final:
                f.write("=" * 80 + "\n")
                f.write("FINAL FEATURE-BALANCED SAMPLER STATISTICS\n")
                f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
            else:
                f.write(f"\n{'='*80}\n")
                f.write(f"EPOCH {epoch_num} STATISTICS\n")
                f.write(f"{'='*80}\n")

            # Video-level statistics
            f.write(f"\nVIDEO-LEVEL STATISTICS:\n")
            f.write(f"Total videos: {len(self.dataset)}\n")

            seen_counts = {}
            for count in self.video_seen_counts.values():
                seen_counts[count] = seen_counts.get(count, 0) + 1

            f.write(f"\nDistribution of how many times videos were seen:\n")
            for count in sorted(seen_counts.keys()):
                num_videos = seen_counts[count]
                pct = (num_videos / len(self.dataset) * 100)
                f.write(f"  Seen {count}x: {num_videos} videos ({pct:.1f}%)\n")

            # Feature value statistics
            f.write(f"\nCUMULATIVE FEATURE REPRESENTATION:\n")
            for feat_name in sorted(self.cumulative_feature_counts.keys()):
                counts = self.cumulative_feature_counts[feat_name]
                total = sum(counts.values())
                f.write(f"\n{feat_name} (total: {total}):\n")
                for value in sorted(counts.keys()):
                    count = counts[value]
                    pct = (count / total * 100) if total > 0 else 0
                    f.write(f"  {value}: {count} ({pct:.1f}%)\n")

            # Distance-specific balance analysis (new section)
            if "distance" in self.cumulative_feature_counts:
                f.write(f"\n{'='*80}\n")
                f.write(f"DISTANCE BALANCE ANALYSIS:\n")
                f.write(f"{'='*80}\n")

                distance_counts = self.cumulative_feature_counts["distance"]
                total_dist = sum(distance_counts.values())
                num_distances = len(distance_counts)
                target_per_dist = total_dist // num_distances if num_distances > 0 else 0

                f.write(f"\nSummary:\n")
                f.write(f"  Total samples: {total_dist}\n")
                f.write(f"  Unique distances: {num_distances}\n")
                f.write(f"  Target per distance: {target_per_dist}\n")
                f.write(f"  Ideal representation: {100/num_distances:.2f}%\n\n")

                # Calculate balance metrics
                deviations = []
                f.write(f"Per-Distance Statistics:\n")
                f.write(f"  {'Status':<6} {'Distance':<10} {'Count':<8} {'%':<8} {'Deviation':<15} {'Priority'}\n")
                f.write(f"  {'-'*70}\n")

                for distance in sorted(distance_counts.keys(), key=lambda x: float(x) if x != "unknown" else 999.0):
                    count = distance_counts[distance]
                    pct = (count / total_dist * 100) if total_dist > 0 else 0
                    deviation = count - target_per_dist
                    deviation_pct = (deviation / target_per_dist * 100) if target_per_dist > 0 else 0
                    deviations.append(abs(deviation_pct))

                    # Get priority score for next epoch
                    priority = self.distance_priority_scores.get(distance, 1.0)

                    # Status indicator
                    if abs(deviation_pct) < 10:
                        status = "✓"  # Excellent
                    elif abs(deviation_pct) < 20:
                        status = "⚠"  # Acceptable
                    else:
                        status = "✗"  # Needs improvement

                    f.write(
                        f"  {status:<6} {distance:<10} {count:<8} {pct:>6.2f}% "
                        f"{deviation:>+5d} ({deviation_pct:>+6.2f}%) {priority:>7.3f}\n"
                    )

                # Overall balance metrics
                avg_deviation = sum(deviations) / len(deviations) if deviations else 0
                max_deviation = max(deviations) if deviations else 0

                f.write(f"\nBalance Metrics:\n")
                f.write(f"  Average deviation: {avg_deviation:.2f}%\n")
                f.write(f"  Maximum deviation: {max_deviation:.2f}%\n")

                if max_deviation < 10:
                    status_msg = "✓ EXCELLENT BALANCE"
                elif max_deviation < 20:
                    status_msg = "⚠ GOOD BALANCE"
                else:
                    status_msg = "✗ NEEDS IMPROVEMENT"

                f.write(f"  Status: {status_msg}\n")
                f.write(f"{'='*80}\n\n")

            # Per-video details
            f.write(f"\nPER-VIDEO DETAILS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Index':<8} {'Count':<8} {'Features'}\n")
            f.write("-" * 80 + "\n")

            for idx in sorted(self.video_seen_counts.keys()):
                count = self.video_seen_counts[idx]
                features = self.sample_features.get(idx, {})
                feat_str = ", ".join([f"{k}={v}" for k, v in sorted(features.items())])
                f.write(f"{idx:<8} {count:<8} {feat_str}\n")

            f.write("=" * 80 + "\n\n")

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        """Yield diverse, balanced batches with ALL-feature balancing and oversampling."""
        epoch_start_time = datetime.now()
        g = torch.Generator()
        g.manual_seed(self.seed + self._iter_count)

        # Calculate oversample weights based on feature rarity
        oversample_weights = self._calculate_sample_oversample_weights()

        # Update distance priorities at start of each epoch
        self._update_distance_priorities()

        # Round-robin offset based on epoch for cycling
        epoch_offset = self._iter_count % max(1, len(self.moving_indices) // self.batch_size)

        self._iter_count += 1

        # Apply round-robin rotation to starting positions
        moving_base = self.moving_indices[epoch_offset:] + self.moving_indices[:epoch_offset]
        stopped_base = self.stopped_indices[epoch_offset:] + self.stopped_indices[:epoch_offset]

        # Create OVERSAMPLED pools - rare features appear multiple times
        moving_pool = self._create_oversampled_pool(moving_base, oversample_weights)
        stopped_pool = self._create_oversampled_pool(stopped_base, oversample_weights)

        logger.info_rank0(
            f"Epoch {self._iter_count}: Oversampled pools - "
            f"moving: {len(moving_base)} -> {len(moving_pool)}, "
            f"stopped: {len(stopped_base)} -> {len(stopped_pool)}"
        )

        # Shuffle pools
        if self.shuffle:
            perm_m = torch.randperm(len(moving_pool), generator=g).tolist()
            moving_pool = [moving_pool[i] for i in perm_m]
            perm_s = torch.randperm(len(stopped_pool), generator=g).tolist()
            stopped_pool = [stopped_pool[i] for i in perm_s]

        half_batch = self.batch_size // 2
        batches_yielded = 0

        # Generate batches with diversity
        while len(moving_pool) >= half_batch and len(stopped_pool) >= half_batch:
            batch, moving_pool, stopped_pool = self._create_diverse_batch(
                moving_pool, stopped_pool, half_batch, g
            )

            # Update counts
            self._update_counts(batch)

            batches_yielded += 1
            yield batch

        # Handle remaining samples if not drop_last
        if not self.drop_last and (moving_pool or stopped_pool):
            remaining = moving_pool + stopped_pool
            # Pad to full batch size
            if len(remaining) < self.batch_size:
                # Repeat under-represented samples
                all_indices = self.moving_indices + self.stopped_indices
                needed = self.batch_size - len(remaining)
                # Add samples not yet in remaining
                candidates = [idx for idx in all_indices if idx not in remaining]
                if len(candidates) >= needed:
                    perm = torch.randperm(len(candidates), generator=g).tolist()
                    remaining.extend([candidates[i] for i in perm[:needed]])
                else:
                    # Need to repeat - prioritize least-seen
                    sorted_by_seen = sorted(all_indices, key=lambda idx: self.video_seen_counts[idx])
                    remaining.extend(sorted_by_seen[:needed])

            self._update_counts(remaining[:self.batch_size])
            batches_yielded += 1
            yield remaining[:self.batch_size]

        # Write epoch statistics
        self._write_statistics(self._iter_count, is_final=False)

        logger.info_rank0(
            f"Epoch {self._iter_count} complete: {batches_yielded} batches yielded"
        )

    def __len__(self) -> int:
        """Return number of batches."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def get_feature_distribution_report(self) -> str:
        """Generate a report of cumulative feature distribution."""
        lines = ["=" * 80, "CUMULATIVE FEATURE DISTRIBUTION", "=" * 80]

        for feat_name in sorted(self.cumulative_feature_counts.keys()):
            counts = self.cumulative_feature_counts[feat_name]
            total = sum(counts.values())
            lines.append(f"\n{feat_name}:")
            for value in sorted(counts.keys()):
                count = counts[value]
                pct = (count / total * 100) if total > 0 else 0
                lines.append(f"  {value}: {count} ({pct:.1f}%)")

        return "\n".join(lines)

    def finalize(self) -> None:
        """Write final statistics at end of training."""
        self._write_statistics(self._iter_count, is_final=True)
        self._print_final_summary()
        logger.info_rank0(f"Final statistics written to: {self.output_dir / 'feature_balanced_statistics.txt'}")

    def _print_final_summary(self) -> None:
        """Print clear summary of video and feature statistics to console and file."""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("FINAL TRAINING STATISTICS SUMMARY")
        lines.append("=" * 80)

        # Per-video statistics
        lines.append("\n" + "-" * 40)
        lines.append("PER-VIDEO SEEN COUNTS:")
        lines.append("-" * 40)
        sorted_videos = sorted(self.video_seen_counts.items(), key=lambda x: x[1], reverse=True)
        for idx, count in sorted_videos:
            features = self.sample_features.get(idx, {})
            motion = features.get("motion", "?")
            distance = features.get("distance", "?")
            angle = features.get("angle", "?")
            lines.append(f"  Video {idx}: {count} times (motion={motion}, dist={distance}, angle={angle})")

        # Summary of seen counts
        lines.append("\n" + "-" * 40)
        lines.append("VIDEO SEEN DISTRIBUTION:")
        lines.append("-" * 40)
        seen_counts = {}
        for count in self.video_seen_counts.values():
            seen_counts[count] = seen_counts.get(count, 0) + 1
        for count in sorted(seen_counts.keys()):
            num_videos = seen_counts[count]
            lines.append(f"  Seen {count}x: {num_videos} videos")

        # Per-feature statistics
        lines.append("\n" + "-" * 40)
        lines.append("PER-FEATURE VALUE COUNTS (how many times model saw each value):")
        lines.append("-" * 40)
        for feat_name in sorted(self.cumulative_feature_counts.keys()):
            counts = self.cumulative_feature_counts[feat_name]
            total = sum(counts.values())
            lines.append(f"\n  {feat_name.upper()}:")
            for value in sorted(counts.keys(), key=lambda x: counts[x], reverse=True):
                count = counts[value]
                pct = (count / total * 100) if total > 0 else 0
                lines.append(f"    {value}: {count} ({pct:.1f}%)")

        lines.append("\n" + "=" * 80)

        # Print to console
        summary = "\n".join(lines)
        logger.info_rank0(summary)

        # Also write to file
        if self.output_dir:
            summary_file = self.output_dir / "training_summary.txt"
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(summary)


class DistributedFeatureBalancedBatchSampler(FeatureBalancedBatchSampler):
    """
    Feature-balanced batch sampler with distributed training support.

    Each process gets a subset of batches, ensuring all processes
    have the same number of batches (padding if necessary).

    Args:
        dataset: The dataset to sample from.
        batch_size: Fixed batch size per process.
        num_replicas: Number of distributed processes.
        rank: Current process rank.
        drop_last: Whether to drop incomplete batches.
        shuffle: Whether to shuffle within priority groups.
        seed: Random seed.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__(dataset, batch_size, drop_last, shuffle, seed)

        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Distributed package not available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Distributed package not available")
            rank = torch.distributed.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches assigned to this rank."""
        all_batches = list(super().__iter__())

        # Pad to make divisible by num_replicas
        remainder = len(all_batches) % self.num_replicas
        if remainder != 0:
            padding = self.num_replicas - remainder
            all_batches.extend(all_batches[:padding])

        # Select batches for this rank
        batches_per_replica = len(all_batches) // self.num_replicas
        start_idx = self.rank * batches_per_replica
        end_idx = start_idx + batches_per_replica

        for batch in all_batches[start_idx:end_idx]:
            yield batch

    def __len__(self) -> int:
        """Return number of batches for this rank."""
        total_batches = super().__len__()
        return (total_batches + self.num_replicas - 1) // self.num_replicas


class FeatureBalancedLoggingCollateWrapper:
    """
    Wrapper for feature-balanced sampler that logs detailed feature info per batch.

    Logs feature distribution for each batch to both console and a log file.

    Args:
        collate_fn: Original collate function to wrap.
        output_dir: Model output directory.
        sampler: The FeatureBalancedBatchSampler (to access feature tracking).
        log_filename: Name of log file.
    """

    def __init__(
        self,
        collate_fn: Callable,
        output_dir: str,
        sampler: FeatureBalancedBatchSampler,
        log_filename: str = "feature_balanced_sampling_log.txt",
    ):
        self.collate_fn = collate_fn
        self.output_dir = Path(output_dir)
        self.sampler = sampler
        self.log_file = self.output_dir / log_filename
        self.batch_count = 0

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._init_log_file()

    def _init_log_file(self) -> None:
        """Initialize log file with header."""
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("FEATURE-BALANCED SAMPLING BATCH LOG\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write(f"Batch size: {self.sampler.batch_size}\n")
            f.write(f"Features tracked: {list(self.sampler.feature_values.keys())}\n")
            f.write("=" * 80 + "\n\n")

    def _log_batch(self, batch: List[dict]) -> None:
        """Log detailed feature distribution for this batch."""
        self.batch_count += 1

        # Count features in this batch
        batch_feature_counts: dict[str, dict[str, int]] = {}
        video_details = []

        for sample in batch:
            videos = sample.get("videos", [])
            if videos:
                video_path = videos[0] if isinstance(videos, list) else videos
                video_name = os.path.basename(video_path)
                features = extract_features_from_path(video_path)

                video_details.append((video_name, features))

                for feat_name, feat_value in features.items():
                    if feat_name not in batch_feature_counts:
                        batch_feature_counts[feat_name] = {}
                    if feat_value not in batch_feature_counts[feat_name]:
                        batch_feature_counts[feat_name][feat_value] = 0
                    batch_feature_counts[feat_name][feat_value] += 1

        # Format log message
        log_lines = []
        log_lines.append(f"\n{'='*80}")
        log_lines.append(f"BATCH {self.batch_count} (size={len(batch)})")
        log_lines.append(f"{'='*80}")

        # Batch feature distribution
        log_lines.append("Batch Feature Distribution:")
        for feat_name in sorted(batch_feature_counts.keys()):
            counts = batch_feature_counts[feat_name]
            parts = [f"{v}:{c}" for v, c in sorted(counts.items())]
            log_lines.append(f"  {feat_name}: {', '.join(parts)}")

        log_lines.append("-" * 80)

        # Cumulative distribution (key features only)
        log_lines.append("Cumulative Distribution (key features):")
        for feat_name in ["motion", "direction", "angle"]:
            if feat_name in self.sampler.cumulative_feature_counts:
                counts = self.sampler.cumulative_feature_counts[feat_name]
                total = sum(counts.values())
                parts = []
                for v in sorted(counts.keys()):
                    c = counts[v]
                    pct = (c / total * 100) if total > 0 else 0
                    parts.append(f"{v}:{c}({pct:.0f}%)")
                log_lines.append(f"  {feat_name}: {', '.join(parts)}")

        log_lines.append("-" * 80)

        # Video list
        log_lines.append("Videos in this batch:")
        for i, (name, features) in enumerate(video_details, 1):
            motion = features.get("motion", "?")
            direction = features.get("direction", "?")
            angle = features.get("angle", "?")
            speed = features.get("speed", "?")
            log_lines.append(f"  {i:3d}. [{motion:7s}] dir={direction:5s} angle={angle:3s} speed={speed:4s} | {name}")

        log_lines.append("-" * 80)

        log_message = "\n".join(log_lines)

        # Print to console
        logger.info_rank0(log_message)

        # Write to file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")

    def __call__(self, batch: List[dict]) -> Any:
        """Log batch info and call original collate function."""
        self._log_batch(batch)
        return self.collate_fn(batch)


class LoggingCollateWrapper:
    """
    Wrapper around collate function that logs batch information.

    Logs video names and class distribution (moving/stopped) for each batch
    to both console and a log file in the model output directory.

    Args:
        collate_fn: Original collate function to wrap.
        output_dir: Model output directory (saves/{model_name}).
        log_filename: Name of log file (default: balanced_sampling_log.txt).
    """

    def __init__(
        self,
        collate_fn: Callable,
        output_dir: str,
        log_filename: str = "balanced_sampling_log.txt",
    ):
        self.collate_fn = collate_fn
        self.output_dir = Path(output_dir)
        self.log_file = self.output_dir / log_filename
        self.batch_count = 0

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize log file with header
        self._init_log_file()

    def _init_log_file(self) -> None:
        """Initialize log file with header."""
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("BALANCED SAMPLING BATCH LOG\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write("=" * 80 + "\n\n")

    def _log_batch(self, batch: List[dict]) -> None:
        """
        Log batch information to console and file.

        Args:
            batch: List of samples in the batch.
        """
        self.batch_count += 1

        # Extract video info from batch
        video_info = []
        moving_count = 0
        stopped_count = 0

        for sample in batch:
            videos = sample.get("videos", [])
            if videos:
                video_path = videos[0] if isinstance(videos, list) else videos
                # Get just the filename for cleaner logging
                video_name = os.path.basename(video_path)
                speed = extract_speed_from_path(video_path)

                if speed is not None:
                    is_moving = speed > 0.0
                    class_label = "MOVING" if is_moving else "STOPPED"
                    if is_moving:
                        moving_count += 1
                    else:
                        stopped_count += 1
                else:
                    class_label = "UNKNOWN"

                video_info.append((video_name, speed, class_label))

        # Format log message
        total = moving_count + stopped_count
        moving_pct = (moving_count / total * 100) if total > 0 else 0
        stopped_pct = (stopped_count / total * 100) if total > 0 else 0

        log_lines = []
        log_lines.append(f"\n{'='*80}")
        log_lines.append(f"BATCH {self.batch_count}")
        log_lines.append(f"{'='*80}")
        log_lines.append(f"Class Distribution: {moving_count} MOVING ({moving_pct:.1f}%) | {stopped_count} STOPPED ({stopped_pct:.1f}%)")
        log_lines.append(f"Total samples: {len(batch)}")
        log_lines.append("-" * 80)
        log_lines.append("Videos in this batch:")

        for i, (name, speed, label) in enumerate(video_info, 1):
            speed_str = f"speed={speed:.1f}" if speed is not None else "speed=?"
            log_lines.append(f"  {i:3d}. [{label:7s}] {speed_str:12s} | {name}")

        log_lines.append("-" * 80)

        # Join all lines
        log_message = "\n".join(log_lines)

        # Print to console
        logger.info_rank0(log_message)

        # Write to file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")

    def __call__(self, batch: List[dict]) -> Any:
        """
        Log batch info and call original collate function.

        Args:
            batch: List of samples to collate.

        Returns:
            Collated batch from original collate function.
        """
        # Log batch information before collating
        self._log_batch(batch)

        # Call original collate function
        return self.collate_fn(batch)
