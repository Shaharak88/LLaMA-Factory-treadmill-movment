#!/usr/bin/env python3
"""
Synthetic Treadmill Video Dataset Generator

This script generates synthetic videos of moving treadmill/conveyor-belt surfaces
with no objects on them. All visual parameters are controllable for creating
diverse, reproducible datasets for computer vision tasks.

Author: AI-Generated
Date: 2025-11-26
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import cv2
from datetime import datetime


class TreadmillTextureGenerator:
    """
    Generates various treadmill/conveyor belt textures with controllable parameters.

    Supports multiple texture types including stripes, noise patterns, rubber,
    grid patterns, and more. Each texture is generated programmatically using
    NumPy operations for full reproducibility.
    """

    def __init__(self, width: int, height: int, seed: int):
        """
        Initialize texture generator with fixed dimensions and random seed.

        Args:
            width: Width of texture in pixels
            height: Height of texture in pixels
            seed: Random seed for reproducible texture generation
        """
        self.width = width
        self.height = height
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_stripes(self, stripe_width: int = 20, orientation: str = 'horizontal',
                        base_color: Tuple[int, int, int] = (80, 80, 80),
                        stripe_color: Tuple[int, int, int] = (120, 120, 120)) -> np.ndarray:
        """
        Generate stripe pattern texture (simulates conveyor belt segments).

        Args:
            stripe_width: Width of each stripe in pixels
            orientation: 'horizontal' or 'vertical'
            base_color: RGB color of base stripes
            stripe_color: RGB color of alternating stripes

        Returns:
            numpy.ndarray: RGB image of shape (height, width, 3)
        """
        texture = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if orientation == 'horizontal':
            for y in range(self.height):
                if (y // stripe_width) % 2 == 0:
                    texture[y, :] = base_color
                else:
                    texture[y, :] = stripe_color
        else:  # vertical
            for x in range(self.width):
                if (x // stripe_width) % 2 == 0:
                    texture[:, x] = base_color
                else:
                    texture[:, x] = stripe_color

        # Add subtle noise for realism
        noise = self.rng.randint(-5, 5, texture.shape, dtype=np.int16)
        texture = np.clip(texture.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return texture

    def generate_noise_pattern(self, noise_scale: float = 0.3,
                              base_color: Tuple[int, int, int] = (100, 100, 100)) -> np.ndarray:
        """
        Generate random noise texture (simulates rough rubber surface).

        Args:
            noise_scale: Scale factor for noise intensity (0.0 to 1.0)
            base_color: RGB base color

        Returns:
            numpy.ndarray: RGB image of shape (height, width, 3)
        """
        texture = np.ones((self.height, self.width, 3), dtype=np.float32)
        texture[:, :, 0] = base_color[0]
        texture[:, :, 1] = base_color[1]
        texture[:, :, 2] = base_color[2]

        # Generate Perlin-like noise using multiple octaves
        for octave in range(3):
            scale = 2 ** octave
            noise = self.rng.randn(self.height // scale, self.width // scale)
            noise = cv2.resize(noise, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            texture += noise[:, :, np.newaxis] * (noise_scale * 255 / (scale + 1))

        texture = np.clip(texture, 0, 255).astype(np.uint8)
        return texture

    def generate_rubber_pattern(self, bump_density: float = 0.02,
                               base_color: Tuple[int, int, int] = (60, 60, 60)) -> np.ndarray:
        """
        Generate rubber texture with raised bumps/dimples.

        Args:
            bump_density: Density of bumps (0.0 to 1.0)
            base_color: RGB base color of rubber

        Returns:
            numpy.ndarray: RGB image of shape (height, width, 3)
        """
        texture = np.ones((self.height, self.width, 3), dtype=np.uint8)
        texture[:, :] = base_color

        # Add circular bumps at random locations
        num_bumps = int(self.width * self.height * bump_density)
        for _ in range(num_bumps):
            cx = self.rng.randint(0, self.width)
            cy = self.rng.randint(0, self.height)
            radius = self.rng.randint(3, 8)

            # Create gradient around bump center
            y_grid, x_grid = np.ogrid[:self.height, :self.width]
            distance = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
            mask = distance <= radius

            # Apply shading to create 3D effect
            shading = np.clip(255 - (distance * 30), 0, 50)
            texture[mask] = np.clip(texture[mask].astype(np.int16) +
                                   shading[mask, np.newaxis].astype(np.int16), 0, 255).astype(np.uint8)

        return texture

    def generate_grid_pattern(self, grid_size: int = 30,
                            line_width: int = 2,
                            base_color: Tuple[int, int, int] = (90, 90, 90),
                            line_color: Tuple[int, int, int] = (140, 140, 140)) -> np.ndarray:
        """
        Generate grid pattern texture (simulates tiled surface).

        Args:
            grid_size: Size of each grid cell in pixels
            line_width: Width of grid lines
            base_color: RGB base color
            line_color: RGB color of grid lines

        Returns:
            numpy.ndarray: RGB image of shape (height, width, 3)
        """
        texture = np.ones((self.height, self.width, 3), dtype=np.uint8)
        texture[:, :] = base_color

        # Draw vertical lines
        for x in range(0, self.width, grid_size):
            texture[:, x:x+line_width] = line_color

        # Draw horizontal lines
        for y in range(0, self.height, grid_size):
            texture[y:y+line_width, :] = line_color

        # Add subtle texture variation
        noise = self.rng.randint(-3, 3, texture.shape, dtype=np.int16)
        texture = np.clip(texture.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return texture

    def generate_diamond_plate(self, diamond_size: int = 40,
                              base_color: Tuple[int, int, int] = (70, 70, 70)) -> np.ndarray:
        """
        Generate diamond plate pattern (metal tread plate texture).

        Args:
            diamond_size: Size of diamond pattern in pixels
            base_color: RGB base color

        Returns:
            numpy.ndarray: RGB image of shape (height, width, 3)
        """
        texture = np.ones((self.height, self.width, 3), dtype=np.uint8)
        texture[:, :] = base_color

        # Create diamond pattern using rotated squares
        half_size = diamond_size // 2
        for y in range(-diamond_size, self.height + diamond_size, diamond_size):
            for x in range(-diamond_size, self.width + diamond_size, diamond_size):
                # Offset every other row
                offset_x = half_size if (y // diamond_size) % 2 == 0 else 0
                cx = x + offset_x
                cy = y

                # Draw diamond (rotated square)
                pts = np.array([
                    [cx, cy - half_size],
                    [cx + half_size, cy],
                    [cx, cy + half_size],
                    [cx - half_size, cy]
                ], dtype=np.int32)

                cv2.polylines(texture, [pts], True,
                            (base_color[0] + 30, base_color[1] + 30, base_color[2] + 30), 2)

        return texture

    def generate_texture(self, texture_type: str, **kwargs) -> np.ndarray:
        """
        Main interface for generating textures of specified type.

        Args:
            texture_type: One of 'stripes', 'noise', 'rubber', 'grid', 'diamond_plate'
            **kwargs: Additional parameters passed to specific texture generators

        Returns:
            numpy.ndarray: RGB texture image
        """
        if texture_type == 'stripes':
            return self.generate_stripes(**kwargs)
        elif texture_type == 'noise':
            return self.generate_noise_pattern(**kwargs)
        elif texture_type == 'rubber':
            return self.generate_rubber_pattern(**kwargs)
        elif texture_type == 'grid':
            return self.generate_grid_pattern(**kwargs)
        elif texture_type == 'diamond_plate':
            return self.generate_diamond_plate(**kwargs)
        else:
            raise ValueError(f"Unknown texture type: {texture_type}")


class TreadmillMotionSimulator:
    """
    Simulates treadmill/conveyor belt motion by translating textures across frames.

    Implements various motion patterns (left, right, up, down) with controllable
    speed. Uses seamless tiling to create infinite scrolling effect.
    """

    def __init__(self, width: int, height: int, direction: str, speed: float):
        """
        Initialize motion simulator.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            direction: Motion direction ('left', 'right', 'up', 'down')
            speed: Motion speed in pixels per frame
        """
        self.width = width
        self.height = height
        self.direction = direction
        self.speed = speed
        self.offset = 0.0

    def create_seamless_texture(self, base_texture: np.ndarray) -> np.ndarray:
        """
        Create larger texture that tiles seamlessly for infinite scrolling.

        The texture is extended by 2x in the motion direction to allow
        wrapping without visible seams.

        Args:
            base_texture: Original texture to make seamless

        Returns:
            numpy.ndarray: Extended seamless texture
        """
        if self.direction in ['left', 'right']:
            # Tile horizontally
            return np.hstack([base_texture, base_texture])
        else:  # up or down
            # Tile vertically
            return np.vstack([base_texture, base_texture])

    def apply_motion(self, texture: np.ndarray, frame_index: int) -> np.ndarray:
        """
        Apply motion to texture for current frame.

        Motion is simulated by translating the texture and wrapping at edges.
        The offset accumulates over frames to create smooth continuous motion.

        Args:
            texture: Seamless texture to animate
            frame_index: Current frame number (for calculating offset)

        Returns:
            numpy.ndarray: Frame with applied motion
        """
        # Calculate cumulative offset
        self.offset = (frame_index * self.speed)

        if self.direction == 'right':
            # Positive X translation
            offset_x = int(self.offset) % self.width
            frame = texture[:self.height, offset_x:offset_x + self.width]

        elif self.direction == 'left':
            # Negative X translation
            offset_x = int(self.offset) % self.width
            offset_x = self.width - offset_x
            frame = texture[:self.height, offset_x:offset_x + self.width]

        elif self.direction == 'down':
            # Positive Y translation
            offset_y = int(self.offset) % self.height
            frame = texture[offset_y:offset_y + self.height, :self.width]

        elif self.direction == 'up':
            # Negative Y translation
            offset_y = int(self.offset) % self.height
            offset_y = self.height - offset_y
            frame = texture[offset_y:offset_y + self.height, :self.width]

        else:
            raise ValueError(f"Unknown direction: {self.direction}")

        return frame.copy()


class CameraEffectsProcessor:
    """
    Applies camera and lighting effects to simulate realistic viewing conditions.

    Includes perspective transforms for view angle simulation, brightness/contrast
    adjustments, lighting gradients, vignetting, motion blur, and noise.
    """

    def __init__(self, width: int, height: int, seed: int):
        """
        Initialize effects processor.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            seed: Random seed for stochastic effects
        """
        self.width = width
        self.height = height
        self.rng = np.random.RandomState(seed)

    def apply_view_angle(self, frame: np.ndarray, angle_degrees: float) -> np.ndarray:
        """
        Apply perspective transform to simulate camera viewing angle.

        Simulates tilting the camera by applying a perspective warp.
        0° = frontal view (no distortion)
        Positive angles = top-tilted view
        Negative angles = bottom-tilted view

        Args:
            frame: Input frame
            angle_degrees: Viewing angle in degrees (-45 to 45 recommended)

        Returns:
            numpy.ndarray: Perspective-transformed frame
        """
        if abs(angle_degrees) < 0.1:
            return frame

        # Calculate perspective transform based on viewing angle
        # Positive angle = compress top, expand bottom (viewing from above)
        # Negative angle = expand top, compress bottom (viewing from below)

        angle_rad = np.radians(angle_degrees)
        scale_factor = 1.0 - abs(np.sin(angle_rad)) * 0.3

        if angle_degrees > 0:
            # Viewing from above
            src_points = np.float32([
                [0, 0],
                [self.width, 0],
                [self.width, self.height],
                [0, self.height]
            ])
            dst_points = np.float32([
                [self.width * 0.1, 0],
                [self.width * 0.9, 0],
                [self.width, self.height],
                [0, self.height]
            ])
        else:
            # Viewing from below
            src_points = np.float32([
                [0, 0],
                [self.width, 0],
                [self.width, self.height],
                [0, self.height]
            ])
            dst_points = np.float32([
                [0, 0],
                [self.width, 0],
                [self.width * 0.9, self.height],
                [self.width * 0.1, self.height]
            ])

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(frame, matrix, (self.width, self.height))

        return warped

    def apply_brightness_contrast(self, frame: np.ndarray,
                                  brightness: float, contrast: float) -> np.ndarray:
        """
        Adjust brightness and contrast of frame.

        Brightness: additive adjustment (range: -1.0 to 1.0)
        Contrast: multiplicative adjustment (range: 0.5 to 2.0)

        Args:
            frame: Input frame
            brightness: Brightness factor (-1.0 to 1.0, 0.0 = no change)
            contrast: Contrast factor (0.5 to 2.0, 1.0 = no change)

        Returns:
            numpy.ndarray: Adjusted frame
        """
        # Apply contrast (multiplicative)
        adjusted = frame.astype(np.float32) * contrast

        # Apply brightness (additive)
        adjusted = adjusted + (brightness * 255)

        # Clip to valid range
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        return adjusted

    def apply_lighting_gradient(self, frame: np.ndarray,
                               variation_type: str,
                               intensity: float) -> np.ndarray:
        """
        Apply lighting gradients to simulate non-uniform illumination.

        Simulates realistic lighting conditions like spotlights, shadows,
        or directional lighting.

        Args:
            frame: Input frame
            variation_type: Type of lighting ('none', 'vignette', 'gradient_lr',
                          'gradient_tb', 'spotlight')
            intensity: Strength of lighting effect (0.0 to 1.0)

        Returns:
            numpy.ndarray: Frame with lighting applied
        """
        if variation_type == 'none' or intensity < 0.01:
            return frame

        h, w = frame.shape[:2]
        lighting_mask = np.ones((h, w), dtype=np.float32)

        if variation_type == 'vignette':
            # Darkening at edges
            y_grid, x_grid = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            distance = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            lighting_mask = 1.0 - (distance / max_distance) * intensity * 0.5

        elif variation_type == 'gradient_lr':
            # Left-to-right gradient
            x_grid = np.linspace(0, 1, w)
            lighting_mask = 1.0 - (x_grid * intensity * 0.3)
            lighting_mask = np.tile(lighting_mask, (h, 1))

        elif variation_type == 'gradient_tb':
            # Top-to-bottom gradient
            y_grid = np.linspace(0, 1, h)
            lighting_mask = 1.0 - (y_grid * intensity * 0.3)
            lighting_mask = np.tile(lighting_mask[:, np.newaxis], (1, w))

        elif variation_type == 'spotlight':
            # Circular spotlight in center
            y_grid, x_grid = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            distance = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            lighting_mask = np.exp(-(distance**2) / (2 * (max_distance * 0.5)**2))
            lighting_mask = 0.5 + lighting_mask * intensity * 0.5

        # Apply lighting mask to frame
        adjusted = frame.astype(np.float32)
        adjusted *= lighting_mask[:, :, np.newaxis]
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        return adjusted

    def apply_motion_blur(self, frame: np.ndarray, direction: str,
                         blur_amount: int) -> np.ndarray:
        """
        Apply directional motion blur to simulate camera motion or fast belt speed.

        Args:
            frame: Input frame
            direction: Motion direction ('left', 'right', 'up', 'down')
            blur_amount: Blur kernel size (0 = no blur)

        Returns:
            numpy.ndarray: Blurred frame
        """
        if blur_amount < 1:
            return frame

        # Create directional blur kernel
        kernel_size = blur_amount * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

        if direction in ['left', 'right']:
            # Horizontal blur
            kernel[blur_amount, :] = 1.0
        else:  # up or down
            # Vertical blur
            kernel[:, blur_amount] = 1.0

        kernel /= kernel.sum()

        # Apply blur
        blurred = cv2.filter2D(frame, -1, kernel)

        return blurred

    def apply_camera_noise(self, frame: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Add camera sensor noise to simulate realistic image capture.

        Args:
            frame: Input frame
            noise_level: Noise intensity (0.0 to 1.0)

        Returns:
            numpy.ndarray: Noisy frame
        """
        if noise_level < 0.01:
            return frame

        noise = self.rng.randn(frame.shape[0], frame.shape[1], frame.shape[2])
        noise = noise * noise_level * 20

        noisy = frame.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        return noisy


class SyntheticVideoGenerator:
    """
    Main orchestrator for synthetic treadmill video generation.

    Combines texture generation, motion simulation, and camera effects
    to produce complete synthetic video datasets with full parameter control.
    """

    def __init__(self, config: dict):
        """
        Initialize video generator with configuration.

        Args:
            config: Dictionary containing all generation parameters
        """
        self.config = config
        self.width, self.height = config['resolution']
        self.fps = config['fps']
        self.duration = config['duration']
        self.num_frames = int(self.fps * self.duration)

        # Initialize components
        self.texture_gen = TreadmillTextureGenerator(
            self.width, self.height, config['seed']
        )
        self.motion_sim = TreadmillMotionSimulator(
            self.width, self.height, config['direction'], config['speed']
        )
        self.effects = CameraEffectsProcessor(
            self.width, self.height, config['seed']
        )

    def generate_video(self, output_path: str) -> None:
        """
        Generate complete synthetic video with all effects applied.

        Args:
            output_path: Path where video file will be saved
        """
        print(f"Generating video: {output_path}")
        print(f"  Parameters: {self.config['texture_type']}, {self.config['direction']}, "
              f"{self.config['speed']} px/frame, {self.num_frames} frames")

        # Generate base texture
        print("  Step 1/4: Generating texture...")
        base_texture = self.texture_gen.generate_texture(
            self.config['texture_type']
        )

        # Create seamless scrolling texture
        print("  Step 2/4: Creating seamless texture...")
        seamless_texture = self.motion_sim.create_seamless_texture(base_texture)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps,
                             (self.width, self.height))

        if not out.isOpened():
            raise RuntimeError(f"Failed to create video writer for {output_path}")

        # Generate frames
        print("  Step 3/4: Generating frames...")
        for frame_idx in range(self.num_frames):
            if frame_idx % 30 == 0:
                print(f"    Frame {frame_idx}/{self.num_frames}")

            # Apply motion
            frame = self.motion_sim.apply_motion(seamless_texture, frame_idx)

            # Apply camera effects
            frame = self.effects.apply_view_angle(frame, self.config['view_angle'])
            frame = self.effects.apply_brightness_contrast(
                frame, self.config['brightness'], self.config['contrast']
            )
            frame = self.effects.apply_lighting_gradient(
                frame, self.config['lighting_variation'],
                self.config['lighting_intensity']
            )

            # Apply optional motion blur
            if self.config.get('motion_blur', 0) > 0:
                frame = self.effects.apply_motion_blur(
                    frame, self.config['direction'], self.config['motion_blur']
                )

            # Apply camera noise
            frame = self.effects.apply_camera_noise(frame, self.config['camera_noise'])

            # Write frame (convert RGB to BGR for OpenCV)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        # Cleanup
        print("  Step 4/4: Finalizing video...")
        out.release()
        print(f"  ✓ Video saved: {output_path}")

    @staticmethod
    def generate_filename(config: dict, video_idx: int) -> str:
        """
        Generate descriptive filename from parameters.

        Args:
            config: Configuration dictionary
            video_idx: Video index number

        Returns:
            str: Filename encoding key parameters
        """
        filename = (
            f"treadmill_{video_idx:04d}_"
            f"{config['texture_type']}_"
            f"{config['direction']}_"
            f"speed{config['speed']:.1f}_"
            f"angle{config['view_angle']:.0f}_"
            f"bright{config['brightness']:.2f}_"
            f"contr{config['contrast']:.2f}_"
            f"{config['resolution'][0]}x{config['resolution'][1]}_"
            f"seed{config['seed']}.mp4"
        )
        return filename


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for video generation.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Generate synthetic treadmill/conveyor belt videos for computer vision',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 5 videos with default settings
  python synthetic_data_generation.py --num_videos 5

  # Generate with specific texture and motion
  python synthetic_data_generation.py --texture_type rubber --direction left --speed 3.0

  # Generate with camera angle and lighting
  python synthetic_data_generation.py --view_angle 30 --lighting_variation vignette --lighting_intensity 0.8

  # Generate high-speed with motion blur
  python synthetic_data_generation.py --speed 10 --motion_blur 3

  # Generate full dataset with varied parameters
  python synthetic_data_generation.py --num_videos 20 --seed 42 --output_dir ./dataset
        """
    )

    # Basic parameters
    parser.add_argument('--num_videos', type=int, default=1,
                       help='Number of videos to generate (default: 1)')
    parser.add_argument('--output_dir', type=str, default='./synthetic_videos',
                       help='Output directory for generated videos (default: ./synthetic_videos)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    # Video parameters
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Video duration in seconds (default: 5.0)')
    parser.add_argument('--resolution', type=str, default='640x480',
                       help='Video resolution as WxH (default: 640x480)')

    # Motion parameters
    parser.add_argument('--direction', type=str, default='right',
                       choices=['left', 'right', 'up', 'down'],
                       help='Belt motion direction (default: right)')
    parser.add_argument('--speed', type=float, default=2.0,
                       help='Motion speed in pixels per frame (default: 2.0)')

    # Texture parameters
    parser.add_argument('--texture_type', type=str, default='stripes',
                       choices=['stripes', 'noise', 'rubber', 'grid', 'diamond_plate'],
                       help='Type of belt texture (default: stripes)')
    parser.add_argument('--background_color', type=str, default='80,80,80',
                       help='Background color as R,G,B (default: 80,80,80)')

    # Camera/lighting parameters
    parser.add_argument('--view_angle', type=float, default=0.0,
                       help='Camera viewing angle in degrees, -45 to 45 (default: 0.0)')
    parser.add_argument('--brightness', type=float, default=0.0,
                       help='Brightness adjustment, -1.0 to 1.0 (default: 0.0)')
    parser.add_argument('--contrast', type=float, default=1.0,
                       help='Contrast adjustment, 0.5 to 2.0 (default: 1.0)')
    parser.add_argument('--lighting_variation', type=str, default='none',
                       choices=['none', 'vignette', 'gradient_lr', 'gradient_tb', 'spotlight'],
                       help='Type of lighting variation (default: none)')
    parser.add_argument('--lighting_intensity', type=float, default=0.5,
                       help='Intensity of lighting variation, 0.0 to 1.0 (default: 0.5)')

    # Additional effects
    parser.add_argument('--motion_blur', type=int, default=0,
                       help='Motion blur amount in pixels, 0 to 10 (default: 0)')
    parser.add_argument('--camera_noise', type=float, default=0.0,
                       help='Camera sensor noise level, 0.0 to 1.0 (default: 0.0)')

    # Variation mode
    parser.add_argument('--vary_parameters', action='store_true',
                       help='Automatically vary parameters across multiple videos')

    return parser.parse_args()


def parse_resolution(resolution_str: str) -> Tuple[int, int]:
    """
    Parse resolution string into width and height.

    Args:
        resolution_str: String in format 'WxH' or 'W,H'

    Returns:
        Tuple[int, int]: (width, height)
    """
    if 'x' in resolution_str:
        w, h = resolution_str.split('x')
    elif ',' in resolution_str:
        w, h = resolution_str.split(',')
    else:
        raise ValueError(f"Invalid resolution format: {resolution_str}")

    return (int(w), int(h))


def parse_color(color_str: str) -> Tuple[int, int, int]:
    """
    Parse color string into RGB tuple.

    Args:
        color_str: String in format 'R,G,B'

    Returns:
        Tuple[int, int, int]: (R, G, B)
    """
    parts = color_str.split(',')
    if len(parts) != 3:
        raise ValueError(f"Invalid color format: {color_str}")

    return tuple(int(x) for x in parts)


def generate_varied_configs(base_config: dict, num_videos: int) -> list:
    """
    Generate varied configurations for diverse dataset.

    Creates parameter variations across texture types, motion directions,
    speeds, and camera/lighting conditions.

    Args:
        base_config: Base configuration dictionary
        num_videos: Number of varied configs to generate

    Returns:
        list: List of configuration dictionaries
    """
    configs = []
    rng = np.random.RandomState(base_config['seed'])

    textures = ['stripes', 'noise', 'rubber', 'grid', 'diamond_plate']
    directions = ['left', 'right', 'up', 'down']
    lighting_types = ['none', 'vignette', 'gradient_lr', 'gradient_tb', 'spotlight']

    for i in range(num_videos):
        config = base_config.copy()
        config['seed'] = base_config['seed'] + i

        # Vary texture
        config['texture_type'] = textures[i % len(textures)]

        # Vary direction
        config['direction'] = directions[i % len(directions)]

        # Vary speed (1.0 to 8.0)
        config['speed'] = rng.uniform(1.0, 8.0)

        # Vary view angle (-30 to 30)
        config['view_angle'] = rng.uniform(-30, 30)

        # Vary brightness (-0.2 to 0.2)
        config['brightness'] = rng.uniform(-0.2, 0.2)

        # Vary contrast (0.7 to 1.3)
        config['contrast'] = rng.uniform(0.7, 1.3)

        # Vary lighting
        config['lighting_variation'] = lighting_types[i % len(lighting_types)]
        config['lighting_intensity'] = rng.uniform(0.3, 0.8)

        # Occasionally add motion blur for high-speed videos
        if config['speed'] > 5.0:
            config['motion_blur'] = rng.randint(1, 4)

        # Add some camera noise
        config['camera_noise'] = rng.uniform(0.0, 0.3)

        configs.append(config)

    return configs


def main():
    """
    Main entry point for synthetic video generation.
    """
    print("=" * 70)
    print("Synthetic Treadmill Video Dataset Generator")
    print("=" * 70)
    print()

    # Parse arguments
    args = parse_arguments()

    # Parse resolution and color
    width, height = parse_resolution(args.resolution)
    bg_color = parse_color(args.background_color)

    # Create base configuration
    base_config = {
        'resolution': (width, height),
        'fps': args.fps,
        'duration': args.duration,
        'direction': args.direction,
        'speed': args.speed,
        'texture_type': args.texture_type,
        'view_angle': args.view_angle,
        'brightness': args.brightness,
        'contrast': args.contrast,
        'lighting_variation': args.lighting_variation,
        'lighting_intensity': args.lighting_intensity,
        'motion_blur': args.motion_blur,
        'camera_noise': args.camera_noise,
        'seed': args.seed,
        'background_color': bg_color,
    }

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    print()

    # Generate video configurations
    if args.vary_parameters and args.num_videos > 1:
        print("Generating varied parameter configurations...")
        configs = generate_varied_configs(base_config, args.num_videos)
    else:
        configs = [base_config.copy() for _ in range(args.num_videos)]
        for i, config in enumerate(configs):
            config['seed'] = args.seed + i

    print(f"Generating {len(configs)} video(s)...")
    print()

    # Generate videos
    start_time = datetime.now()

    for idx, config in enumerate(configs):
        print(f"[Video {idx + 1}/{len(configs)}]")

        # Generate filename
        filename = SyntheticVideoGenerator.generate_filename(config, idx)
        output_path = str(output_dir / filename)

        # Create generator and generate video
        generator = SyntheticVideoGenerator(config)
        generator.generate_video(output_path)
        print()

    # Summary
    elapsed = datetime.now() - start_time
    print("=" * 70)
    print(f"✓ Generation complete!")
    print(f"  Generated: {len(configs)} video(s)")
    print(f"  Output: {output_dir.absolute()}")
    print(f"  Time elapsed: {elapsed.total_seconds():.1f} seconds")
    print("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        sys.exit(1)
