#!/usr/bin/env python3
"""
Building Dataset - Synthetic Treadmill Video Dataset Generator & Formatter

This script automates the complete pipeline for creating synthetic treadmill video
datasets formatted for LLaMA-Factory training:
  1. Generates synthetic videos using synthetic_data_generation.py
  2. Creates ShareGPT-formatted JSON dataset
  3. Automatically updates dataset_info.json
  4. Generates summary reports

Features:
  - 50/50 split of moving vs stopped videos
  - Simple, compact prompts and responses
  - Automatic parameter variation for diversity
  - Reproducible with seed control
  - Full validation and error handling

Author: AI-Generated
Date: 2025-11-26
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import itertools


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Orchestrates synthetic video generation and dataset formatting for LLaMA-Factory.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize dataset builder with configuration.

        Args:
            args: Command-line arguments
        """
        self.args = args
        self.dataset_name = args.dataset_name
        self.output_dir = Path(args.output_dir)
        self.num_videos = args.num_videos
        self.train_split = args.train_split
        self.moving_ratio = args.moving_ratio
        self.seed = args.seed
        self.rng = random.Random(self.seed)

        # Paths
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.dataset_json_path = self.data_dir / f"{self.dataset_name}.json"
        self.dataset_info_path = self.data_dir / "dataset_info.json"
        self.synthetic_script = self.data_dir / "synthetic_treadmill" / "synthetic_data_generation.py"

        # Statistics tracking
        self.stats = {
            'total_videos': 0,
            'moving_videos': 0,
            'stopped_videos': 0,
            'texture_distribution': {},
            'direction_distribution': {},
            'speed_range': {'min': float('inf'), 'max': float('-inf')},
            'total_size_bytes': 0
        }

        # Log file
        self.log_file = self.output_dir / f"{self.dataset_name}_build_log.txt"

    def setup_logging(self) -> None:
        """Setup file logging in addition to console logging."""
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
        logger.info(f"Dataset Builder started - {self.dataset_name}")
        logger.info(f"Configuration: {vars(self.args)}")

    def create_directories(self) -> None:
        """Create necessary directories for dataset generation."""
        logger.info("Creating directories...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Output directory: {self.output_dir.absolute()}")

    def validate_prerequisites(self) -> None:
        """Validate that all required files and dependencies exist."""
        logger.info("Validating prerequisites...")

        # Check if synthetic_data_generation.py exists
        if not self.synthetic_script.exists():
            raise FileNotFoundError(
                f"Synthetic generation script not found: {self.synthetic_script}"
            )
        logger.info(f"  [OK] Found synthetic generation script")

        # Check if dataset_info.json exists
        if not self.dataset_info_path.exists():
            raise FileNotFoundError(
                f"dataset_info.json not found: {self.dataset_info_path}"
            )
        logger.info(f"  [OK] Found dataset_info.json")

        # Check Python availability
        try:
            result = subprocess.run(
                ['python3', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            logger.info(f"  [OK] Python3 available: {result.stdout.strip()}")
        except Exception as e:
            raise RuntimeError(f"Python3 not available: {e}")

    def _parse_parameter_values(self, param_str: str, param_type=str):
        """
        Parse comma-separated parameter values.

        Args:
            param_str: Comma-separated values (e.g., "4,6,8" or "stripes,noise")
            param_type: Type to convert values to (int, float, str)

        Returns:
            List of parsed values
        """
        if ',' in str(param_str):
            values = [param_type(v.strip()) for v in str(param_str).split(',')]
            return values
        else:
            return [param_type(param_str)]

    def _generate_all_combinations(self) -> List[Dict]:
        """
        Generate all parameter combinations based on comma-separated arguments.

        Returns:
            List[Dict]: All parameter combinations
        """
        # Parse all parameters that might have multiple values
        texture_types = self._parse_parameter_values(self.args.texture_type, str)
        directions = self._parse_parameter_values(self.args.direction, str)
        fps_values = self._parse_parameter_values(self.args.fps, int)
        durations = self._parse_parameter_values(self.args.duration, float)
        resolutions = self._parse_parameter_values(self.args.resolution, str)
        view_angles = self._parse_parameter_values(self.args.view_angle, float)
        brightness_values = self._parse_parameter_values(self.args.brightness, float)
        contrast_values = self._parse_parameter_values(self.args.contrast, float)
        lighting_variations = self._parse_parameter_values(self.args.lighting_variation, str)
        lighting_intensities = self._parse_parameter_values(self.args.lighting_intensity, float)
        motion_blurs = self._parse_parameter_values(self.args.motion_blur, int)
        camera_noises = self._parse_parameter_values(self.args.camera_noise, float)
        edge_widths = self._parse_parameter_values(self.args.edge_width, float)

        # Parse subtle_gray_stripes parameters
        stripe_widths = self._parse_parameter_values(self.args.stripe_width, int) if hasattr(self.args, 'stripe_width') else [10]
        stripe_spacings = self._parse_parameter_values(self.args.stripe_spacing, int) if hasattr(self.args, 'stripe_spacing') else [60]
        stripe_grays = self._parse_parameter_values(self.args.stripe_gray, int) if hasattr(self.args, 'stripe_gray') else [125]
        background_grays = self._parse_parameter_values(self.args.background_gray, int) if hasattr(self.args, 'background_gray') else [140]
        stripe_distance_variances = self._parse_parameter_values(self.args.stripe_distance_variance, float) if hasattr(self.args, 'stripe_distance_variance') else [0.0]

        # Parse speed range
        if hasattr(self.args, 'speed_range') and self.args.speed_range:
            speed_parts = self.args.speed_range.split(',')
            if len(speed_parts) == 2:
                speed_min, speed_max = map(float, speed_parts)
                speeds = [speed_min, speed_max]
            else:
                speeds = [float(s.strip()) for s in speed_parts]
        else:
            speeds = [1.0, 8.0]

        # Generate all combinations
        all_combinations = list(itertools.product(
            texture_types,
            directions,
            speeds,
            fps_values,
            durations,
            resolutions,
            view_angles,
            brightness_values,
            contrast_values,
            lighting_variations,
            lighting_intensities,
            motion_blurs,
            camera_noises,
            edge_widths,
            stripe_widths,
            stripe_spacings,
            stripe_grays,
            background_grays,
            stripe_distance_variances
        ))

        logger.info(f"  Generated {len(all_combinations)} parameter combinations")

        # Create config dictionaries
        configs = []
        for idx, combo in enumerate(all_combinations):
            (texture, direction, speed, fps, duration, resolution, view_angle,
             brightness, contrast, lighting_var, lighting_int, motion_blur,
             camera_noise, edge_width, stripe_width, stripe_spacing, stripe_gray,
             background_gray, stripe_distance_variance) = combo

            config = {
                'index': idx,
                'seed': self.seed + idx,
                'is_moving': speed > 0.0,
                'texture_type': texture,
                'direction': direction,
                'speed': speed,
                'fps': fps,
                'duration': duration,
                'resolution': resolution,
                'view_angle': view_angle,
                'brightness': brightness,
                'contrast': contrast,
                'lighting_variation': lighting_var,
                'lighting_intensity': lighting_int,
                'motion_blur': motion_blur,
                'camera_noise': camera_noise,
                'edge_width': edge_width,
                'stripe_width': stripe_width,
                'stripe_spacing': stripe_spacing,
                'stripe_gray': stripe_gray,
                'background_gray': background_gray,
                'stripe_distance_variance': stripe_distance_variance
            }
            configs.append(config)

        return configs

    def generate_video_configs(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate video configurations for moving and stopped videos.

        Returns:
            Tuple[List[Dict], List[Dict]]: (moving_configs, stopped_configs)
        """
        logger.info("Generating video configurations...")

        # Check if we're in combination mode (any parameter has comma)
        if (',' in str(self.args.texture_type) or ',' in str(self.args.view_angle) or
            ',' in str(self.args.background_gray) or ',' in str(self.args.stripe_gray)):
            all_configs = self._generate_all_combinations()
            # Separate moving and stopped
            moving_configs = [c for c in all_configs if c['is_moving']]
            stopped_configs = [c for c in all_configs if not c['is_moving']]

            logger.info(f"  Generated: {len(moving_configs)} moving, {len(stopped_configs)} stopped videos")

            return moving_configs, stopped_configs
        else:
            # Original behavior
            # Calculate number of moving vs stopped videos (always 50/50)
            num_moving = self.num_videos // 2
            num_stopped = self.num_videos - num_moving

            logger.info(f"  Target: {num_moving} moving, {num_stopped} stopped videos")

            moving_configs = self._generate_configs(num_moving, is_moving=True)
            stopped_configs = self._generate_configs(num_stopped, is_moving=False)

            return moving_configs, stopped_configs

    def _generate_configs(self, count: int, is_moving: bool) -> List[Dict]:
        """
        Generate video configurations with varied parameters.

        Args:
            count: Number of configurations to generate
            is_moving: True for moving videos, False for stopped

        Returns:
            List[Dict]: List of configuration dictionaries
        """
        configs = []

        # Available parameter options
        textures = ['stripes', 'noise', 'rubber', 'grid', 'diamond_plate', 'factory_dark', 'factory_dark_stripes']
        directions = ['left', 'right', 'up', 'down']
        lighting_types = ['none', 'vignette', 'gradient_lr', 'gradient_tb', 'spotlight']

        # Speed ranges
        if is_moving:
            if hasattr(self.args, 'speed_range') and self.args.speed_range:
                speed_min, speed_max = map(float, self.args.speed_range.split(','))
            else:
                speed_min, speed_max = 1.0, 8.0
        else:
            speed_min, speed_max = 0.0, 0.0

        for i in range(count):
            config = {
                'index': len(configs),
                'seed': self.seed + len(configs),
                'is_moving': is_moving
            }

            # Fixed parameters from args or defaults
            config['resolution'] = getattr(self.args, 'resolution', '640x480')
            config['fps'] = getattr(self.args, 'fps', 30)
            config['duration'] = getattr(self.args, 'duration', 5.0)

            # Vary parameters if requested
            if self.args.vary_parameters:
                config['texture_type'] = self.rng.choice(textures)
                config['direction'] = self.rng.choice(directions)
                config['speed'] = self.rng.uniform(speed_min, speed_max) if is_moving else 0.0
                config['view_angle'] = self.rng.uniform(-30, 30)
                config['brightness'] = self.rng.uniform(-0.2, 0.2)
                config['contrast'] = self.rng.uniform(0.7, 1.3)
                config['lighting_variation'] = self.rng.choice(lighting_types)
                config['lighting_intensity'] = self.rng.uniform(0.3, 0.8)
                config['motion_blur'] = self.rng.randint(1, 3) if config['speed'] > 5.0 else 0
                config['camera_noise'] = self.rng.uniform(0.0, 0.3)
                config['edge_width'] = self.rng.uniform(0.05, 0.15)
            else:
                # Use provided parameters
                config['texture_type'] = getattr(self.args, 'texture_type', 'stripes')
                config['direction'] = getattr(self.args, 'direction', 'right')
                config['speed'] = self.rng.uniform(speed_min, speed_max) if is_moving else 0.0
                config['view_angle'] = getattr(self.args, 'view_angle', 0.0)
                config['brightness'] = getattr(self.args, 'brightness', 0.0)
                config['contrast'] = getattr(self.args, 'contrast', 1.0)
                config['lighting_variation'] = getattr(self.args, 'lighting_variation', 'none')
                config['lighting_intensity'] = getattr(self.args, 'lighting_intensity', 0.5)
                config['motion_blur'] = getattr(self.args, 'motion_blur', 0)
                config['camera_noise'] = getattr(self.args, 'camera_noise', 0.0)
                config['edge_width'] = getattr(self.args, 'edge_width', 0.1)
                config['stripe_width'] = getattr(self.args, 'stripe_width', 10)
                config['stripe_spacing'] = getattr(self.args, 'stripe_spacing', 60)
                config['stripe_gray'] = getattr(self.args, 'stripe_gray', 125)
                config['background_gray'] = getattr(self.args, 'background_gray', 140)
                config['stripe_distance_variance'] = getattr(self.args, 'stripe_distance_variance', 0.0)

            configs.append(config)

        return configs

    def generate_videos(self, configs: List[Dict]) -> List[str]:
        """
        Generate synthetic videos using the synthetic_data_generation.py script.

        Args:
            configs: List of video configurations

        Returns:
            List[str]: List of generated video file paths
        """
        logger.info(f"Generating {len(configs)} videos...")
        generated_files = []

        for i, config in enumerate(configs):
            logger.info(f"  [{i+1}/{len(configs)}] Generating video...")
            logger.info(f"    Texture: {config['texture_type']}, Direction: {config['direction']}, "
                       f"Speed: {config['speed']:.2f}")

            # Build command
            cmd = [
                'python3',
                str(self.synthetic_script),
                '--num_videos', '1',
                '--output_dir', str(self.output_dir),
                '--seed', str(config['seed']),
                '--texture_type', config['texture_type'],
                '--direction', config['direction'],
                '--speed', str(config['speed']),
                '--view_angle', str(config['view_angle']),
                '--brightness', str(config['brightness']),
                '--contrast', str(config['contrast']),
                '--lighting_variation', config['lighting_variation'],
                '--lighting_intensity', str(config['lighting_intensity']),
                '--motion_blur', str(config['motion_blur']),
                '--camera_noise', str(config['camera_noise']),
                '--edge_width', str(config['edge_width']),
                '--resolution', config['resolution'],
                '--fps', str(config['fps']),
                '--duration', str(config['duration'])
            ]

            # Add subtle_gray_stripes specific parameters if present in config
            if 'stripe_width' in config:
                cmd.extend(['--stripe_width', str(config['stripe_width'])])
            if 'stripe_spacing' in config:
                cmd.extend(['--stripe_spacing', str(config['stripe_spacing'])])
            if 'stripe_gray' in config:
                cmd.extend(['--stripe_gray', str(config['stripe_gray'])])
            if 'background_gray' in config:
                cmd.extend(['--background_gray', str(config['background_gray'])])
            if 'stripe_distance_variance' in config:
                cmd.extend(['--stripe_distance_variance', str(config['stripe_distance_variance'])])

            try:
                # Run video generation
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout per video
                    cwd=str(self.project_root)
                )

                if result.returncode != 0:
                    logger.error(f"    [FAIL] Video generation failed:")
                    logger.error(f"    {result.stderr}")
                    continue

                # Find generated video file
                # Parse output to find filename
                video_file = self._find_generated_video(config)

                if video_file and video_file.exists():
                    generated_files.append(str(video_file))
                    file_size = video_file.stat().st_size
                    self.stats['total_size_bytes'] += file_size
                    logger.info(f"    [OK] Generated: {video_file.name} ({file_size / 1024:.1f} KB)")

                    # Update statistics
                    self._update_stats(config)
                else:
                    logger.error(f"    [FAIL] Generated video not found")

            except subprocess.TimeoutExpired:
                logger.error(f"    [FAIL] Video generation timed out")
            except Exception as e:
                logger.error(f"    [FAIL] Error: {e}")

        logger.info(f"Successfully generated {len(generated_files)}/{len(configs)} videos")
        return generated_files

    def _find_generated_video(self, config: Dict) -> Optional[Path]:
        """
        Find the most recently generated video file matching config parameters.

        Args:
            config: Video configuration

        Returns:
            Optional[Path]: Path to generated video or None
        """
        # Look for files matching the pattern
        pattern = f"treadmill_*_{config['texture_type']}_*_speed{config['speed']:.1f}_*.mp4"
        matching_files = list(self.output_dir.glob(pattern))

        if matching_files:
            # Return most recent file
            return max(matching_files, key=lambda p: p.stat().st_mtime)

        # Fallback: look for any recent .mp4 files
        all_videos = list(self.output_dir.glob("*.mp4"))
        if all_videos:
            return max(all_videos, key=lambda p: p.stat().st_mtime)

        return None

    def _update_stats(self, config: Dict) -> None:
        """Update statistics with video configuration."""
        self.stats['total_videos'] += 1

        if config['is_moving']:
            self.stats['moving_videos'] += 1
            speed = config['speed']
            self.stats['speed_range']['min'] = min(self.stats['speed_range']['min'], speed)
            self.stats['speed_range']['max'] = max(self.stats['speed_range']['max'], speed)
        else:
            self.stats['stopped_videos'] += 1

        texture = config['texture_type']
        self.stats['texture_distribution'][texture] = \
            self.stats['texture_distribution'].get(texture, 0) + 1

        direction = config['direction']
        self.stats['direction_distribution'][direction] = \
            self.stats['direction_distribution'].get(direction, 0) + 1

    def create_dataset_json(self, video_files: List[str]) -> None:
        """
        Create ShareGPT-formatted dataset JSON file.

        Args:
            video_files: List of video file paths
        """
        logger.info(f"Creating dataset JSON: {self.dataset_json_path}")

        dataset_entries = []

        for video_path in video_files:
            # Extract speed from filename to determine if moving or stopped
            is_moving = self._is_video_moving(video_path)

            # Create relative path from LLaMA-Factory root
            video_rel_path = f"data/{self.dataset_name}/{Path(video_path).name}"

            # Create dataset entry
            entry = self._create_dataset_entry(video_rel_path, is_moving)
            dataset_entries.append(entry)

        # Shuffle for better training
        self.rng.shuffle(dataset_entries)

        # Write JSON file
        with open(self.dataset_json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_entries, f, indent=2, ensure_ascii=False)

        logger.info(f"  [OK] Created dataset with {len(dataset_entries)} entries")

    def _is_video_moving(self, video_path: str) -> bool:
        """
        Determine if video is moving based on filename.

        Args:
            video_path: Path to video file

        Returns:
            bool: True if moving, False if stopped
        """
        filename = Path(video_path).name

        # Extract speed from filename (e.g., speed2.5 or speed0.0)
        match = re.search(r'speed([\d.]+)', filename)
        if match:
            speed = float(match.group(1))
            return speed > 0.0

        # Fallback: assume stopped if can't determine
        logger.warning(f"  Could not determine speed from filename: {filename}, assuming stopped")
        return False

    def _create_dataset_entry(self, video_path: str, is_moving: bool) -> Dict:
        """
        Create a single dataset entry in ShareGPT format.

        Args:
            video_path: Relative path to video file
            is_moving: Whether video shows moving belt

        Returns:
            Dict: Dataset entry
        """
        # Simple, compact prompts and responses
        user_prompt = "<video>Analyze this video. Is the treadmill belt moving or stopped?"

        if is_moving:
            assistant_response = "The treadmill belt is moving."
        else:
            assistant_response = "The treadmill belt is stopped."

        return {
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt
                },
                {
                    "role": "assistant",
                    "content": assistant_response
                }
            ],
            "videos": [video_path]
        }

    def update_dataset_info(self) -> None:
        """Update dataset_info.json with new dataset entry."""
        logger.info(f"Updating dataset_info.json...")

        # Backup original file
        backup_path = self.dataset_info_path.with_suffix('.json.backup')
        shutil.copy2(self.dataset_info_path, backup_path)
        logger.info(f"  Created backup: {backup_path.name}")

        # Load existing dataset_info.json
        with open(self.dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)

        # Add new dataset entry
        dataset_info[self.dataset_name] = {
            "file_name": f"{self.dataset_name}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "videos": "videos"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }

        # Write updated dataset_info.json
        with open(self.dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)

        logger.info(f"  [OK] Added '{self.dataset_name}' to dataset_info.json")

    def generate_summary_report(self) -> None:
        """Generate and save summary report of dataset generation."""
        logger.info("Generating summary report...")

        report_path = self.output_dir / f"{self.dataset_name}_summary.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"Dataset Generation Summary: {self.dataset_name}\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Seed: {self.seed}\n\n")

            f.write("Dataset Statistics:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Total Videos: {self.stats['total_videos']}\n")
            f.write(f"  Moving Videos: {self.stats['moving_videos']} "
                   f"({self.stats['moving_videos']/self.stats['total_videos']*100:.1f}%)\n")
            f.write(f"  Stopped Videos: {self.stats['stopped_videos']} "
                   f"({self.stats['stopped_videos']/self.stats['total_videos']*100:.1f}%)\n\n")

            if self.stats['moving_videos'] > 0:
                f.write(f"  Speed Range: {self.stats['speed_range']['min']:.2f} - "
                       f"{self.stats['speed_range']['max']:.2f} px/frame\n\n")

            f.write("Texture Distribution:\n")
            for texture, count in sorted(self.stats['texture_distribution'].items()):
                pct = count / self.stats['total_videos'] * 100
                f.write(f"  {texture:15s}: {count:3d} ({pct:.1f}%)\n")
            f.write("\n")

            f.write("Direction Distribution:\n")
            for direction, count in sorted(self.stats['direction_distribution'].items()):
                pct = count / self.stats['total_videos'] * 100
                f.write(f"  {direction:15s}: {count:3d} ({pct:.1f}%)\n")
            f.write("\n")

            total_size_mb = self.stats['total_size_bytes'] / (1024 * 1024)
            avg_size_mb = total_size_mb / self.stats['total_videos'] if self.stats['total_videos'] > 0 else 0
            f.write(f"File Sizes:\n")
            f.write(f"  Total: {total_size_mb:.2f} MB\n")
            f.write(f"  Average: {avg_size_mb:.2f} MB per video\n\n")

            f.write("Output Files:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Videos: {self.output_dir}/\n")
            f.write(f"  Dataset JSON: {self.dataset_json_path}\n")
            f.write(f"  Dataset Info: {self.dataset_info_path}\n")
            f.write(f"  Log File: {self.log_file}\n\n")

            f.write("=" * 70 + "\n")
            f.write("Dataset ready for training!\n")
            f.write("=" * 70 + "\n")

        # Print summary to console
        with open(report_path, 'r') as f:
            logger.info("\n" + f.read())

    def build(self) -> None:
        """Execute complete dataset building pipeline."""
        try:
            logger.info("=" * 70)
            logger.info(f"Building Dataset: {self.dataset_name}")
            logger.info("=" * 70)

            # Setup - create directories first, then setup logging
            self.create_directories()
            self.setup_logging()
            self.validate_prerequisites()

            # Generate video configurations
            moving_configs, stopped_configs = self.generate_video_configs()
            all_configs = moving_configs + stopped_configs

            # Shuffle for varied generation order
            self.rng.shuffle(all_configs)

            # Generate videos
            generated_files = self.generate_videos(all_configs)

            if len(generated_files) == 0:
                raise RuntimeError("No videos were successfully generated")

            # Create dataset JSON
            self.create_dataset_json(generated_files)

            # Update dataset_info.json
            self.update_dataset_info()

            # Generate summary report
            self.generate_summary_report()

            logger.info("=" * 70)
            logger.info("[OK] Dataset building complete!")
            logger.info("=" * 70)

        except KeyboardInterrupt:
            logger.warning("\n\nDataset building interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"\n\nError during dataset building: {e}", exc_info=True)
            sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Automated synthetic treadmill dataset builder for LLaMA-Factory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Generate 100 videos with varied parameters
  python building_dataset.py \\
    --dataset_name synthetic_treadmill_varied \\
    --num_videos 100 \\
    --vary_parameters \\
    --seed 42

  # Generate specific texture type dataset
  python building_dataset.py \\
    --dataset_name synthetic_stripes_motion \\
    --num_videos 50 \\
    --texture_type stripes \\
    --vary_parameters \\
    --speed_range 1.0,8.0

  # Generate all combinations of parameters (NEW!)
  # Example: 4 textures x 2 speeds = 8 videos
  python building_dataset.py \\
    --dataset_name combo_dataset \\
    --texture_type stripes,noise,rubber,grid \\
    --speed_range 2.0,5.0 \\
    --fps 30 \\
    --duration 5.0

  # Multiple parameters with combinations
  # Example: 2 fps x 2 durations x 2 textures = 8 videos
  python building_dataset.py \\
    --dataset_name multi_combo \\
    --fps 15,30 \\
    --duration 3.0,5.0 \\
    --texture_type stripes,noise

  # Generate small test dataset
  python building_dataset.py \\
    --dataset_name test_dataset \\
    --num_videos 10 \\
    --seed 123

Docker Usage:

  # Run inside Docker container
  docker exec llamafactory python3 /app/building_dataset.py \\
    --dataset_name my_dataset \\
    --num_videos 50 \\
    --vary_parameters

  # Combination mode in Docker
  docker exec llamafactory python3 /app/building_dataset.py \\
    --dataset_name combo_test \\
    --texture_type stripes,noise,rubber,grid \\
    --speed_range 3.0

Notes:
  - Moving vs stopped split is always 50/50 (when not using combination mode)
  - COMBINATION MODE: Use comma-separated values for any parameter to generate
    all combinations (e.g., --fps 15,30 --texture_type stripes,noise)
  - Use --vary_parameters for diverse training data with random variation
  - Set seed for reproducibility
  - Videos saved to data/<dataset_name>/
  - Dataset JSON created at data/<dataset_name>.json
  - dataset_info.json automatically updated
        """
    )

    # Required arguments
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Name for the dataset (e.g., "synthetic_treadmill_stripes")')

    # Basic parameters
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for videos (default: data/<dataset_name>)')
    parser.add_argument('--num_videos', type=int, default=10,
                       help='Total number of videos to generate (default: 10)')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Training split ratio (default: 0.8) - for future use')
    parser.add_argument('--moving_ratio', type=float, default=0.5,
                       help='Ratio of moving videos (default: 0.5 = 50%% moving, 50%% stopped)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    # Video generation parameters
    parser.add_argument('--vary_parameters', action='store_true',
                       help='Automatically vary parameters for diversity')
    parser.add_argument('--texture_type', type=str, default='stripes',
                       help='Texture type (default: stripes). Accepts comma-separated values for combinations (e.g., stripes,noise,rubber). Valid types: stripes, noise, rubber, grid, diamond_plate, factory_dark, factory_dark_stripes, subtle_gray_stripes')
    parser.add_argument('--direction', type=str, default='right',
                       help='Motion direction (default: right). Accepts comma-separated values (e.g., left,right). Valid directions: left, right, up, down')
    parser.add_argument('--speed_range', type=str, default='1.0,8.0',
                       help='Speed values as comma-separated list (default: 1.0,8.0 for min,max). Can specify multiple speeds (e.g., 2.0,4.0,6.0)')
    parser.add_argument('--resolution', type=str, default='640x480',
                       help='Video resolution as WxH (default: 640x480). Accepts comma-separated values (e.g., 640x480,800x600)')
    parser.add_argument('--fps', type=str, default='30',
                       help='Frames per second (default: 30). Accepts comma-separated values (e.g., 15,30)')
    parser.add_argument('--duration', type=str, default='5.0',
                       help='Video duration in seconds (default: 5.0). Accepts comma-separated values (e.g., 3.0,5.0)')

    # Camera/lighting parameters
    parser.add_argument('--view_angle', type=str, default='0.0',
                       help='Camera viewing angle in degrees (default: 0.0). Accepts comma-separated values (e.g., -15,0,15)')
    parser.add_argument('--brightness', type=str, default='0.0',
                       help='Brightness adjustment (default: 0.0). Accepts comma-separated values (e.g., -0.2,0.0,0.2)')
    parser.add_argument('--contrast', type=str, default='1.0',
                       help='Contrast adjustment (default: 1.0). Accepts comma-separated values (e.g., 0.8,1.0,1.2)')
    parser.add_argument('--lighting_variation', type=str, default='none',
                       help='Lighting variation type (default: none). Accepts comma-separated values (e.g., none,vignette,spotlight). Valid types: none, vignette, gradient_lr, gradient_tb, spotlight')
    parser.add_argument('--lighting_intensity', type=str, default='0.5',
                       help='Lighting intensity (default: 0.5). Accepts comma-separated values (e.g., 0.3,0.5,0.7)')
    parser.add_argument('--motion_blur', type=str, default='0',
                       help='Motion blur amount (default: 0). Accepts comma-separated values (e.g., 0,1,2)')
    parser.add_argument('--camera_noise', type=str, default='0.0',
                       help='Camera noise level (default: 0.0). Accepts comma-separated values (e.g., 0.0,0.1,0.2)')
    parser.add_argument('--edge_width', type=str, default='0.1',
                       help='Belt enclosure edge width as percentage (default: 0.1). Accepts comma-separated values (e.g., 0.05,0.1,0.15)')

    # Subtle gray stripes parameters (for subtle_gray_stripes texture type)
    parser.add_argument('--stripe_width', type=str, default='10',
                       help='Stripe width in pixels for subtle_gray_stripes (default: 10). Accepts comma-separated values (e.g., 5,10,15)')
    parser.add_argument('--stripe_spacing', type=str, default='60',
                       help='Stripe spacing in pixels for subtle_gray_stripes (default: 60). Accepts comma-separated values (e.g., 40,60,80)')
    parser.add_argument('--stripe_gray', type=str, default='125',
                       help='Stripe gray level (0-255) for subtle_gray_stripes (default: 125). Accepts comma-separated values (e.g., 110,115,120,125,130,135)')
    parser.add_argument('--background_gray', type=str, default='140',
                       help='Background gray level (0-255) for subtle_gray_stripes (default: 140). Accepts comma-separated values (e.g., 135,140,145)')

    args = parser.parse_args()

    # Set default output_dir if not provided
    if args.output_dir is None:
        args.output_dir = f"data/{args.dataset_name}"

    return args


def main():
    """Main entry point."""
    args = parse_arguments()
    builder = DatasetBuilder(args)
    builder.build()


if __name__ == '__main__':
    main()
