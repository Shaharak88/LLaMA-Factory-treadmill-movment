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

        logger.info_rank0(
            f"FeatureBalancedBatchSampler initialized: {len(self.dataset)} samples, "
            f"batch_size={batch_size}, features tracked: {list(self.feature_values.keys())}"
        )
        logger.info_rank0(f"  Moving: {len(self.moving_indices)}, Stopped: {len(self.stopped_indices)}")
        for feat, values in self.feature_values.items():
            logger.info_rank0(f"  {feat}: {sorted(values)}")

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

    def _create_diverse_batch(
        self,
        moving_pool: list[int],
        stopped_pool: list[int],
        half_batch: int,
        generator: torch.Generator,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Create one batch with maximum diversity + 50/50 balance.

        Args:
            moving_pool: Available moving indices.
            stopped_pool: Available stopped indices.
            half_batch: Number of samples per class (batch_size // 2).
            generator: Random generator for shuffling.

        Returns:
            (batch_indices, remaining_moving, remaining_stopped)
        """
        # Group moving samples by key features (angle, distance, direction)
        moving_groups = self._group_by_features(moving_pool, ["angle", "distance", "direction"])
        stopped_groups = self._group_by_features(stopped_pool, ["angle", "distance", "direction"])

        # Select diverse samples from moving group
        moving_batch = []
        moving_group_keys = list(moving_groups.keys())
        if self.shuffle:
            perm = torch.randperm(len(moving_group_keys), generator=generator).tolist()
            moving_group_keys = [moving_group_keys[i] for i in perm]

        # Round-robin through groups to get diverse samples
        for _ in range(half_batch):
            if not moving_group_keys:
                break
            # Cycle through groups
            for group_key in moving_group_keys[:]:
                if len(moving_batch) >= half_batch:
                    break
                if moving_groups[group_key]:
                    # Pop one sample from this group
                    idx = moving_groups[group_key].pop(0)
                    moving_batch.append(idx)
                    # Remove group if empty
                    if not moving_groups[group_key]:
                        moving_group_keys.remove(group_key)

        # If we need more moving samples, take any available
        while len(moving_batch) < half_batch and moving_pool:
            for group in moving_groups.values():
                if group and len(moving_batch) < half_batch:
                    moving_batch.append(group.pop(0))

        # Select diverse samples from stopped group (same logic)
        stopped_batch = []
        stopped_group_keys = list(stopped_groups.keys())
        if self.shuffle:
            perm = torch.randperm(len(stopped_group_keys), generator=generator).tolist()
            stopped_group_keys = [stopped_group_keys[i] for i in perm]

        for _ in range(half_batch):
            if not stopped_group_keys:
                break
            for group_key in stopped_group_keys[:]:
                if len(stopped_batch) >= half_batch:
                    break
                if stopped_groups[group_key]:
                    idx = stopped_groups[group_key].pop(0)
                    stopped_batch.append(idx)
                    if not stopped_groups[group_key]:
                        stopped_group_keys.remove(group_key)

        while len(stopped_batch) < half_batch and stopped_pool:
            for group in stopped_groups.values():
                if group and len(stopped_batch) < half_batch:
                    stopped_batch.append(group.pop(0))

        # Interleave moving and stopped for batch
        batch = []
        for m, s in zip(moving_batch, stopped_batch):
            batch.extend([m, s])
        # Add any extras
        batch.extend(moving_batch[len(stopped_batch):])
        batch.extend(stopped_batch[len(moving_batch):])

        # Remove selected from pools
        remaining_moving = [idx for idx in moving_pool if idx not in moving_batch]
        remaining_stopped = [idx for idx in stopped_pool if idx not in stopped_batch]

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
        """Yield diverse, balanced batches with round-robin cycling."""
        epoch_start_time = datetime.now()
        g = torch.Generator()
        g.manual_seed(self.seed + self._iter_count)

        # Round-robin offset based on epoch for cycling
        epoch_offset = self._iter_count % max(1, len(self.moving_indices) // self.batch_size)

        self._iter_count += 1

        # Apply round-robin rotation to starting positions
        moving_pool = self.moving_indices[epoch_offset:] + self.moving_indices[:epoch_offset]
        stopped_pool = self.stopped_indices[epoch_offset:] + self.stopped_indices[:epoch_offset]

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
        logger.info_rank0(f"Final statistics written to: {self.output_dir / 'feature_balanced_statistics.txt'}")


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
