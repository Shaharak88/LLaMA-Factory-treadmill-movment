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
        self.epoch = 0

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
        g.manual_seed(self.seed + self.epoch)

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

        # Auto-increment epoch for next iteration (since HuggingFace doesn't call set_epoch on batch_sampler)
        self.epoch += 1

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
        self.epoch = 0
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

        # Auto-increment epoch for next iteration (since HuggingFace doesn't call set_epoch on batch_sampler)
        self.epoch += 1

    def __len__(self) -> int:
        """Return number of batches."""
        if self.drop_last:
            return len(self.all_indices) // self.batch_size
        return (len(self.all_indices) + self.batch_size - 1) // self.batch_size


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
