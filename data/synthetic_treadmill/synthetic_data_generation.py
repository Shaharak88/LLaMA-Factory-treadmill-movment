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
                        stripe_color: Tuple[int, int, int] = (120, 120, 122),
                        motion_direction: str = None) -> np.ndarray:
        """
        Generate stripe pattern texture (simulates conveyor belt segments).

        IMPORTANT: Stripes are automatically oriented perpendicular to motion direction
        so that movement is visible. If motion_direction is provided, it overrides
        the orientation parameter.

        Args:
            stripe_width: Width of each stripe in pixels
            orientation: 'horizontal' or 'vertical' (ignored if motion_direction is set)
            base_color: RGB color of base stripes
            stripe_color: RGB color of alternating stripes
            motion_direction: 'left', 'right', 'up', 'down' - auto-sets perpendicular stripes

        Returns:
            numpy.ndarray: RGB image of shape (height, width, 3)
        """
        # Auto-orient stripes perpendicular to motion for visibility
        if motion_direction:
            if motion_direction in ['left', 'right']:
                # Horizontal motion needs vertical stripes
                orientation = 'vertical'
            else:  # up or down
                # Vertical motion needs horizontal stripes
                orientation = 'horizontal'
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

    def generate_factory_dark(self, base_color: Tuple[int, int, int] = (25, 25, 25),
                             texture_intensity: float = 0.15) -> np.ndarray:
        """
        Generate dark/black factory conveyor belt texture (no stripes).

        Simulates the appearance of real industrial conveyor belts with dark rubber
        surface and subtle texture variations typical of factory equipment.

        Args:
            base_color: RGB base color (very dark, typically 20-35 for each channel)
            texture_intensity: Intensity of subtle texture variations (0.0 to 1.0)

        Returns:
            numpy.ndarray: RGB image of shape (height, width, 3)
        """
        texture = np.ones((self.height, self.width, 3), dtype=np.float32)
        texture[:, :, 0] = base_color[0]
        texture[:, :, 1] = base_color[1]
        texture[:, :, 2] = base_color[2]

        # Add subtle noise for realistic worn rubber appearance
        # Use multiple noise octaves for natural variation
        for octave in range(4):
            scale = 2 ** octave
            noise = self.rng.randn(self.height // scale, self.width // scale)
            noise = cv2.resize(noise, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            texture += noise[:, :, np.newaxis] * (texture_intensity * 255 / (scale + 2))

        # Add very subtle directional wear marks (like real conveyor belts)
        num_marks = int(self.width * self.height * 0.0001)  # Very sparse
        for _ in range(num_marks):
            x = self.rng.randint(0, self.width)
            y = self.rng.randint(0, self.height)
            length = self.rng.randint(10, 30)
            angle = self.rng.uniform(0, np.pi)

            # Draw subtle wear mark
            end_x = int(x + length * np.cos(angle))
            end_y = int(y + length * np.sin(angle))
            end_x = np.clip(end_x, 0, self.width - 1)
            end_y = np.clip(end_y, 0, self.height - 1)

            # Very subtle brightness variation for wear marks
            brightness_shift = self.rng.uniform(3, 8)
            cv2.line(texture, (x, y), (end_x, end_y),
                    (base_color[0] + brightness_shift,
                     base_color[1] + brightness_shift,
                     base_color[2] + brightness_shift), 1)

        texture = np.clip(texture, 0, 255).astype(np.uint8)
        return texture

    def generate_factory_dark_stripes(self, stripe_width: int = 50, orientation: str = 'horizontal',
                                     base_color: Tuple[int, int, int] = (20, 20, 20),
                                     stripe_color: Tuple[int, int, int] = (35, 35, 35),
                                     motion_direction: str = None) -> np.ndarray:
        """
        Generate dark factory conveyor with widely-spaced stripes.

        Similar to generate_stripes but with darker colors suitable for factory/industrial
        environments and wider default spacing between stripes.

        Args:
            stripe_width: Width of each stripe in pixels (default 50 - wider than standard)
            orientation: 'horizontal' or 'vertical' (ignored if motion_direction is set)
            base_color: RGB color of base stripes (very dark)
            stripe_color: RGB color of alternating stripes (slightly lighter dark)
            motion_direction: 'left', 'right', 'up', 'down' - auto-sets perpendicular stripes

        Returns:
            numpy.ndarray: RGB image of shape (height, width, 3)
        """
        # Auto-orient stripes perpendicular to motion for visibility
        if motion_direction:
            if motion_direction in ['left', 'right']:
                orientation = 'vertical'
            else:  # up or down
                orientation = 'horizontal'

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

        # Add subtle noise for realism (less than standard to maintain dark look)
        noise = self.rng.randint(-3, 3, texture.shape, dtype=np.int16)
        texture = np.clip(texture.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return texture

    def generate_subtle_gray_stripes(self, stripe_width: int = 10, stripe_spacing: int = 60,
                                     orientation: str = 'horizontal',
                                     stripe_gray: int = 125, background_gray: int = 140,
                                     motion_direction: str = None,
                                     stripe_distance_variance: float = 0.0) -> np.ndarray:
        """
        Generate subtle low-contrast gray stripe pattern for motion detection challenges.

        This texture creates narrow gray stripes on a gray background with minimal contrast,
        designed to test the threshold at which vision models can detect belt movement.
        Both the stripes and background are shades of gray with low perceptual difference.

        Args:
            stripe_width: Width of each stripe in pixels (default: 10 - thinner than standard)
            stripe_spacing: Spacing between stripe centers in pixels (default: 60 - wider apart)
            orientation: 'horizontal' or 'vertical' (ignored if motion_direction is set)
            stripe_gray: Gray level for stripes, 0-255 (default: 125)
            background_gray: Gray level for background, 0-255 (default: 140)
            motion_direction: 'left', 'right', 'up', 'down' - auto-sets perpendicular stripes
            stripe_distance_variance: Variance (std dev) for randomized stripe spacing.
                                    0 = fixed spacing. >0 = randomized spacing.

        Returns:
            numpy.ndarray: RGB image of shape (height, width, 3)
        """
        # Auto-orient stripes perpendicular to motion for visibility
        if motion_direction:
            if motion_direction in ['left', 'right']:
                orientation = 'vertical'
            else:  # up or down
                orientation = 'horizontal'

        # Create base texture with background gray
        texture = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        texture[:, :] = (background_gray, background_gray, background_gray)

        # Add stripes
        if orientation == 'horizontal':
            y = 0
            while y < self.height:
                # Draw stripe of specified width
                y_end = min(y + stripe_width, self.height)
                texture[y:y_end, :] = (stripe_gray, stripe_gray, stripe_gray)
                
                # Calculate next position
                if stripe_distance_variance > 0:
                    # Randomize spacing
                    # Use normal distribution centered on stripe_spacing
                    spacing = self.rng.normal(stripe_spacing, stripe_distance_variance)
                    # Ensure minimum spacing of stripe_width + 1 pixel
                    spacing = max(stripe_width + 1, spacing)
                    y += int(spacing)
                else:
                    y += stripe_spacing
        else:  # vertical
            x = 0
            while x < self.width:
                # Draw stripe of specified width
                x_end = min(x + stripe_width, self.width)
                texture[:, x:x_end] = (stripe_gray, stripe_gray, stripe_gray)
                
                # Calculate next position
                if stripe_distance_variance > 0:
                    # Randomize spacing
                    # Use normal distribution centered on stripe_spacing
                    spacing = self.rng.normal(stripe_spacing, stripe_distance_variance)
                    # Ensure minimum spacing of stripe_width + 1 pixel
                    spacing = max(stripe_width + 1, spacing)
                    x += int(spacing)
                else:
                    x += stripe_spacing

        # Add subtle noise for realism (as requested by user)
        noise = self.rng.randint(-3, 3, texture.shape, dtype=np.int16)
        texture = np.clip(texture.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return texture

    def generate_texture(self, texture_type: str, **kwargs) -> np.ndarray:
        """
        Main interface for generating textures of specified type.

        Args:
            texture_type: One of 'stripes', 'noise', 'rubber', 'grid', 'diamond_plate',
                         'factory_dark', 'factory_dark_stripes', 'subtle_gray_stripes'
            **kwargs: Additional parameters passed to specific texture generators

        Returns:
            numpy.ndarray: RGB texture image
        """
        if texture_type == 'stripes':
            # Filter kwargs to only include parameters that generate_stripes accepts
            stripes_params = {'stripe_width', 'orientation', 'base_color', 'stripe_color', 'motion_direction'}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in stripes_params}
            return self.generate_stripes(**filtered_kwargs)
        elif texture_type == 'factory_dark_stripes':
            # Filter kwargs to only include parameters that generate_factory_dark_stripes accepts
            factory_dark_stripes_params = {'stripe_width', 'orientation', 'base_color', 'stripe_color', 'motion_direction'}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in factory_dark_stripes_params}
            return self.generate_factory_dark_stripes(**filtered_kwargs)
        elif texture_type == 'subtle_gray_stripes':
            return self.generate_subtle_gray_stripes(**kwargs)
        else:
            # Remove motion_direction for non-stripe textures (they don't use it)
            kwargs.pop('motion_direction', None)

            if texture_type == 'noise':
                # Filter kwargs to only include parameters that generate_noise_pattern accepts
                noise_params = {'noise_scale', 'base_color'}
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in noise_params}
                return self.generate_noise_pattern(**filtered_kwargs)
            elif texture_type == 'rubber':
                # Filter kwargs to only include parameters that generate_rubber_pattern accepts
                rubber_params = {'bump_density', 'base_color'}
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in rubber_params}
                return self.generate_rubber_pattern(**filtered_kwargs)
            elif texture_type == 'grid':
                # Filter kwargs to only include parameters that generate_grid_pattern accepts
                grid_params = {'grid_size', 'line_width', 'base_color', 'line_color'}
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in grid_params}
                return self.generate_grid_pattern(**filtered_kwargs)
            elif texture_type == 'diamond_plate':
                # Filter kwargs to only include parameters that generate_diamond_plate accepts
                diamond_params = {'diamond_size', 'base_color'}
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in diamond_params}
                return self.generate_diamond_plate(**filtered_kwargs)
            elif texture_type == 'factory_dark':
                # Filter kwargs to only include parameters that generate_factory_dark accepts
                factory_dark_params = {'base_color', 'texture_intensity'}
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in factory_dark_params}
                return self.generate_factory_dark(**filtered_kwargs)
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


class ObjectPlacementGenerator:
    """
    Generates and places objects on the treadmill belt surface with motion tracking.

    Supports various object types (boxes, circles) with different colors, sizes,
    and positions. Objects move with the belt motion like real items on a conveyor.
    """

    def __init__(self, width: int, height: int, seed: int, direction: str, speed: float):
        """
        Initialize object placement generator with motion tracking.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            seed: Random seed for reproducible placement
            direction: Belt motion direction ('left', 'right', 'up', 'down')
            speed: Belt speed in pixels per frame
        """
        self.width = width
        self.height = height
        self.direction = direction
        self.speed = speed
        self.rng = np.random.RandomState(seed)

        # Define available colors (BGR format for OpenCV)
        self.colors = {
            'red': (0, 0, 200),
            'blue': (200, 0, 0),
            'green': (0, 180, 0),
            'yellow': (0, 200, 200),
            'orange': (0, 100, 255),
            'purple': (200, 0, 200),
            'cyan': (255, 200, 0),
            'white': (220, 220, 220)
        }

        # Define size mappings (as percentage of image dimensions)
        self.size_mappings = {
            'small': 0.08,
            'medium': 0.15,
            'large': 0.25
        }

        # Object tracking list: each object is {x, y, type, size, color}
        self.objects = []

    def _get_position_coordinates(self, position: str, object_size: int,
                                   edge_width_percent: float) -> tuple:
        """
        Calculate object center coordinates based on position descriptor.

        Args:
            position: Position descriptor ('center', 'left', 'right', 'random')
            object_size: Size of the object (for boundary calculations)
            edge_width_percent: Belt enclosure edge width percentage

        Returns:
            tuple: (x, y) coordinates for object center
        """
        # Calculate usable belt area (excluding enclosure edges)
        edge_width = int(self.width * edge_width_percent)
        edge_height = int(self.height * edge_width_percent)

        usable_left = edge_width + object_size
        usable_right = self.width - edge_width - object_size
        usable_top = edge_height + object_size
        usable_bottom = self.height - edge_height - object_size

        # Ensure we have valid bounds
        if usable_left >= usable_right:
            usable_left = self.width // 4
            usable_right = 3 * self.width // 4
        if usable_top >= usable_bottom:
            usable_top = self.height // 4
            usable_bottom = 3 * self.height // 4

        center_x = self.width // 2
        center_y = self.height // 2

        if position == 'center':
            return (center_x, center_y)
        elif position == 'left':
            x = (usable_left + center_x) // 2
            y = center_y
            return (x, y)
        elif position == 'right':
            x = (center_x + usable_right) // 2
            y = center_y
            return (x, y)
        elif position == 'random':
            # Ensure valid bounds for randint (low must be < high)
            if usable_right - usable_left < 2:
                x = center_x
            else:
                x = self.rng.randint(usable_left, usable_right)

            if usable_bottom - usable_top < 2:
                y = center_y
            else:
                y = self.rng.randint(usable_top, usable_bottom)
            return (x, y)
        else:
            return (center_x, center_y)

    def _parse_size(self, size: str) -> int:
        """
        Parse size parameter into pixel dimensions.

        Args:
            size: Size descriptor ('small', 'medium', 'large') or numeric value

        Returns:
            int: Object size in pixels
        """
        if size in self.size_mappings:
            # Use percentage of smaller dimension to ensure object fits
            base_size = min(self.width, self.height)
            return int(base_size * self.size_mappings[size])
        else:
            try:
                # Try to parse as numeric value (0.0-1.0 as percentage, >1 as pixels)
                numeric_size = float(size)
                if 0.0 < numeric_size <= 1.0:
                    base_size = min(self.width, self.height)
                    return int(base_size * numeric_size)
                else:
                    return int(numeric_size)
            except ValueError:
                # Default to medium if parsing fails
                base_size = min(self.width, self.height)
                return int(base_size * self.size_mappings['medium'])

    def place_box(self, frame: np.ndarray, position: tuple, size: int,
                  color: tuple) -> np.ndarray:
        """
        Place a box (rectangle) object on the frame.

        Args:
            frame: Input frame (RGB)
            position: (x, y) coordinates for box center
            size: Size of the box in pixels
            color: RGB color tuple

        Returns:
            numpy.ndarray: Frame with box placed
        """
        result = frame.copy()
        x, y = position
        half_size = size // 2

        # Calculate box corners
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(self.width, x + half_size)
        y2 = min(self.height, y + half_size)

        # Convert RGB to BGR for OpenCV
        bgr_color = (color[2], color[1], color[0])

        # Draw filled rectangle
        cv2.rectangle(result, (x1, y1), (x2, y2), bgr_color, -1)

        # Add subtle shading for 3D effect
        shadow_color = tuple(max(0, c - 40) for c in bgr_color)
        cv2.rectangle(result, (x1, y1), (x2, y2), shadow_color, 2)

        return result

    def place_circle(self, frame: np.ndarray, position: tuple, size: int,
                     color: tuple) -> np.ndarray:
        """
        Place a circular object on the frame.

        Args:
            frame: Input frame (RGB)
            position: (x, y) coordinates for circle center
            size: Diameter of the circle in pixels
            color: RGB color tuple

        Returns:
            numpy.ndarray: Frame with circle placed
        """
        result = frame.copy()
        x, y = position
        radius = size // 2

        # Convert RGB to BGR for OpenCV
        bgr_color = (color[2], color[1], color[0])

        # Draw filled circle
        cv2.circle(result, (x, y), radius, bgr_color, -1)

        # Add subtle shading for 3D effect
        shadow_color = tuple(max(0, c - 40) for c in bgr_color)
        cv2.circle(result, (x, y), radius, shadow_color, 2)

        # Add highlight for sphere effect
        highlight_color = tuple(min(255, c + 60) for c in bgr_color)
        highlight_offset = radius // 3
        cv2.circle(result, (x - highlight_offset, y - highlight_offset),
                  radius // 4, highlight_color, -1)

        return result

    def initialize_objects(self, num_objects: int, object_type: str,
                          object_size: str, position: str,
                          edge_width_percent: float) -> None:
        """
        Initialize objects at starting positions based on belt direction.

        Objects start at the entry edge of the belt and will move with belt motion.

        Args:
            num_objects: Number of objects to initialize
            object_type: Type of object ('box', 'circle', 'random')
            object_size: Size descriptor ('small', 'medium', 'large', or numeric)
            position: Position descriptor (affects cross-belt positioning)
            edge_width_percent: Belt enclosure edge width percentage
        """
        self.objects = []
        size_pixels = self._parse_size(object_size)
        color_names = list(self.colors.keys())

        # Calculate usable belt area
        edge_width = int(self.width * edge_width_percent)
        edge_height = int(self.height * edge_width_percent)

        for i in range(num_objects):
            # Determine object type
            if object_type == 'random':
                current_type = self.rng.choice(['box', 'circle'])
            else:
                current_type = object_type

            # Select random color
            color_name = self.rng.choice(color_names)
            color_bgr = self.colors[color_name]
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

            # Determine starting position based on visual belt motion
            # Objects start at the entry edge and move with the visual belt flow
            if self.direction == 'right':
                # Visual motion is LEFT, so start from RIGHT edge
                x = self.width - edge_width - size_pixels
                # Ensure valid bounds for randint
                y_low = edge_height + size_pixels
                y_high = self.height - edge_height - size_pixels
                if y_high - y_low < 2:
                    y = self.height // 2
                else:
                    y = self.rng.randint(y_low, y_high)
            elif self.direction == 'left':
                # Visual motion is RIGHT, so start from LEFT edge
                x = edge_width + size_pixels
                # Ensure valid bounds for randint
                y_low = edge_height + size_pixels
                y_high = self.height - edge_height - size_pixels
                if y_high - y_low < 2:
                    y = self.height // 2
                else:
                    y = self.rng.randint(y_low, y_high)
            elif self.direction == 'down':
                # Visual motion is UP, so start from BOTTOM edge
                # Ensure valid bounds for randint
                x_low = edge_width + size_pixels
                x_high = self.width - edge_width - size_pixels
                if x_high - x_low < 2:
                    x = self.width // 2
                else:
                    x = self.rng.randint(x_low, x_high)
                y = self.height - edge_height - size_pixels
            elif self.direction == 'up':
                # Visual motion is DOWN, so start from TOP edge
                # Ensure valid bounds for randint
                x_low = edge_width + size_pixels
                x_high = self.width - edge_width - size_pixels
                if x_high - x_low < 2:
                    x = self.width // 2
                else:
                    x = self.rng.randint(x_low, x_high)
                y = edge_height + size_pixels

            # Space objects out along visual belt motion direction
            if self.direction in ['left', 'right']:
                offset = i * (self.width // (num_objects + 1))
                if self.direction == 'right':
                    # Visual motion left, so space leftward (decrease X)
                    x -= offset
                else:
                    # Visual motion right, so space rightward (increase X)
                    x += offset
            else:  # up or down
                offset = i * (self.height // (num_objects + 1))
                if self.direction == 'down':
                    # Visual motion up, so space upward (decrease Y)
                    y -= offset
                else:
                    # Visual motion down, so space downward (increase Y)
                    y += offset

            # Store object info
            obj = {
                'x': x,
                'y': y,
                'type': current_type,
                'size': size_pixels,
                'color': color_rgb
            }
            self.objects.append(obj)

    def update_object_positions(self) -> None:
        """
        Update object positions based on belt motion.
        Removes objects that have exited the viewable area.

        Note: Objects move WITH the belt texture. The texture scrolling creates
        visual motion in the opposite direction to the offset change.
        """
        objects_to_keep = []

        for obj in self.objects:
            # Update position based on direction and speed
            # Objects move WITH the belt visual motion (opposite to texture offset direction)
            # When texture offset increases right, visual effect is belt moving LEFT
            # So objects should move LEFT (decrease X) to match visual belt motion
            if self.direction == 'right':
                # Belt appears to move left, so objects move left
                obj['x'] -= self.speed
            elif self.direction == 'left':
                # Belt appears to move right, so objects move right
                obj['x'] += self.speed
            elif self.direction == 'down':
                # Belt appears to move up, so objects move up
                obj['y'] -= self.speed
            elif self.direction == 'up':
                # Belt appears to move down, so objects move down
                obj['y'] += self.speed

            # Check if object is still in viewable area (with generous margin for smooth exit)
            margin = obj['size'] * 2
            if (obj['x'] > -margin and obj['x'] < self.width + margin and
                obj['y'] > -margin and obj['y'] < self.height + margin):
                objects_to_keep.append(obj)

        self.objects = objects_to_keep

    def render_objects(self, frame: np.ndarray) -> np.ndarray:
        """
        Render all tracked objects onto the frame.

        Args:
            frame: Input frame (RGB)

        Returns:
            numpy.ndarray: Frame with objects rendered
        """
        result = frame.copy()

        for obj in self.objects:
            pos = (int(obj['x']), int(obj['y']))

            # Place object based on type
            if obj['type'] == 'box':
                result = self.place_box(result, pos, obj['size'], obj['color'])
            elif obj['type'] == 'circle':
                result = self.place_circle(result, pos, obj['size'], obj['color'])

        return result


class CameraEffectsProcessor:
    """
    Applies camera and lighting effects to simulate realistic viewing conditions.

    Includes perspective transforms for view angle simulation, brightness/contrast
    adjustments, lighting gradients, vignetting, motion blur, gaussian blur, and noise.
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

    def apply_gaussian_blur(self, frame: np.ndarray, blur_intensity: float) -> np.ndarray:
        """
        Apply gaussian blur to simulate out-of-focus camera effect.

        Args:
            frame: Input frame
            blur_intensity: Blur intensity (0.0 to 1.0)
                          0.0 = no blur
                          light (0.1-0.3) = slight out of focus
                          medium (0.4-0.6) = moderate blur
                          heavy (0.7-1.0) = strong blur

        Returns:
            numpy.ndarray: Blurred frame
        """
        if blur_intensity < 0.01:
            return frame

        # Map intensity to kernel size (must be odd)
        # Intensity 0.1-1.0 maps to kernel size 3-31
        kernel_size = int(3 + (blur_intensity * 28))
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Apply gaussian blur
        blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

        return blurred

    def apply_blur(self, frame: np.ndarray, blur_type: str, blur_intensity: float,
                   direction: str = None) -> np.ndarray:
        """
        Apply blur effect based on specified type and intensity.

        Args:
            frame: Input frame
            blur_type: Type of blur ('motion', 'gaussian', 'random', 'none')
            blur_intensity: Blur intensity (0.0 to 1.0) or string ('light', 'medium', 'heavy')
            direction: Motion direction for motion blur (required if blur_type='motion')

        Returns:
            numpy.ndarray: Blurred frame
        """
        if blur_type == 'none':
            return frame

        # Parse intensity if it's a string
        intensity_value = blur_intensity
        if isinstance(blur_intensity, str):
            intensity_map = {
                'light': 0.2,
                'medium': 0.5,
                'heavy': 0.8
            }
            intensity_value = intensity_map.get(blur_intensity.lower(), 0.5)

        # Select blur type
        if blur_type == 'random':
            blur_type = self.rng.choice(['motion', 'gaussian'])

        # Apply selected blur
        if blur_type == 'motion':
            if direction is None:
                direction = 'right'  # Default direction
            # Convert intensity to blur amount (1-10 range)
            blur_amount = int(1 + (intensity_value * 9))
            return self.apply_motion_blur(frame, direction, blur_amount)
        elif blur_type == 'gaussian':
            return self.apply_gaussian_blur(frame, intensity_value)
        else:
            return frame

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

    def apply_belt_enclosure(self, frame: np.ndarray,
                            edge_width_percent: float = 0.1,
                            enclosure_color: Tuple[int, int, int] = (40, 40, 40),
                            edge_color: Tuple[int, int, int] = (60, 60, 60)) -> np.ndarray:
        """
        Add realistic belt enclosure and edges to simulate actual treadmill/conveyor appearance.

        Creates a frame around the belt to show:
        - Side rails/edges of the belt
        - Metal/plastic enclosure typical of treadmills
        - Visible belt boundaries

        Args:
            frame: Input frame showing belt texture
            edge_width_percent: Width of edge/enclosure as percentage of frame (0.05 to 0.2)
            enclosure_color: RGB color of enclosure/frame (dark gray/black)
            edge_color: RGB color of belt edges (slightly lighter)

        Returns:
            numpy.ndarray: Frame with belt enclosure overlay
        """
        h, w = frame.shape[:2]
        result = frame.copy()

        # Calculate edge dimensions
        edge_width = int(w * edge_width_percent)
        edge_height = int(h * edge_width_percent)

        # Draw left and right edges (side rails)
        cv2.rectangle(result, (0, 0), (edge_width, h), enclosure_color, -1)
        cv2.rectangle(result, (w - edge_width, 0), (w, h), enclosure_color, -1)

        # Draw top and bottom edges
        cv2.rectangle(result, (0, 0), (w, edge_height), enclosure_color, -1)
        cv2.rectangle(result, (0, h - edge_height), (w, h), enclosure_color, -1)

        # Draw belt edge lines (inner border showing belt edge)
        # Left belt edge
        cv2.line(result, (edge_width, edge_height), (edge_width, h - edge_height),
                edge_color, 3)
        # Right belt edge
        cv2.line(result, (w - edge_width, edge_height), (w - edge_width, h - edge_height),
                edge_color, 3)
        # Top belt edge
        cv2.line(result, (edge_width, edge_height), (w - edge_width, edge_height),
                edge_color, 3)
        # Bottom belt edge
        cv2.line(result, (edge_width, h - edge_height), (w - edge_width, h - edge_height),
                edge_color, 3)

        # Add subtle shading/gradient to enclosure for 3D effect
        # Left side gradient
        for i in range(edge_width):
            alpha = i / edge_width
            shade = int(enclosure_color[0] * (1 - alpha * 0.3))
            cv2.line(result, (i, edge_height), (i, h - edge_height),
                    (shade, shade, shade), 1)

        # Right side gradient
        for i in range(edge_width):
            alpha = i / edge_width
            shade = int(enclosure_color[0] * (1 - alpha * 0.3))
            cv2.line(result, (w - edge_width + i, edge_height),
                    (w - edge_width + i, h - edge_height),
                    (shade, shade, shade), 1)

        # Add corner details (screws/bolts for realism)
        bolt_radius = max(3, int(edge_width * 0.15))
        bolt_color = (80, 80, 80)

        # Corner positions (inset from edges)
        corner_offset = edge_width // 2
        corners = [
            (corner_offset, corner_offset),  # Top-left
            (w - corner_offset, corner_offset),  # Top-right
            (corner_offset, h - corner_offset),  # Bottom-left
            (w - corner_offset, h - corner_offset)  # Bottom-right
        ]

        for corner in corners:
            cv2.circle(result, corner, bolt_radius, bolt_color, -1)
            cv2.circle(result, corner, bolt_radius // 2, (100, 100, 100), -1)

        return result


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
        self.object_gen = ObjectPlacementGenerator(
            self.width, self.height, config['seed'],
            config['direction'], config['speed']
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
        texture_kwargs = {'motion_direction': self.config['direction']}

        # Add subtle_gray_stripes specific parameters if present
        if 'stripe_width' in self.config:
            texture_kwargs['stripe_width'] = self.config['stripe_width']
        if 'stripe_spacing' in self.config:
            texture_kwargs['stripe_spacing'] = self.config['stripe_spacing']
        if 'stripe_gray' in self.config:
            texture_kwargs['stripe_gray'] = self.config['stripe_gray']
        if 'background_gray' in self.config:
            texture_kwargs['background_gray'] = self.config['background_gray']
        if 'stripe_distance_variance' in self.config:
            texture_kwargs['stripe_distance_variance'] = self.config['stripe_distance_variance']

        base_texture = self.texture_gen.generate_texture(
            self.config['texture_type'],
            **texture_kwargs
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

        # Initialize objects if enabled (before frame loop)
        if self.config.get('add_object', False):
            self.object_gen.initialize_objects(
                num_objects=self.config.get('num_objects', 1),
                object_type=self.config.get('object_type', 'box'),
                object_size=self.config.get('object_size', 'medium'),
                position=self.config.get('object_position', 'center'),
                edge_width_percent=self.config.get('edge_width', 0.1)
            )

        # Generate frames
        print("  Step 3/4: Generating frames...")
        for frame_idx in range(self.num_frames):
            if frame_idx % 30 == 0:
                print(f"    Frame {frame_idx}/{self.num_frames}")

            # Apply motion
            frame = self.motion_sim.apply_motion(seamless_texture, frame_idx)

            # Render moving objects BEFORE belt enclosure (so they disappear under edges)
            if self.config.get('add_object', False):
                frame = self.object_gen.render_objects(frame)
                # Update object positions for next frame
                self.object_gen.update_object_positions()

            # Apply belt enclosure AFTER objects (creates fixed frame that objects go under)
            frame = self.effects.apply_belt_enclosure(
                frame,
                edge_width_percent=self.config.get('edge_width', 0.1)
            )

            # Apply camera effects
            frame = self.effects.apply_view_angle(frame, self.config['view_angle'])
            frame = self.effects.apply_brightness_contrast(
                frame, self.config['brightness'], self.config['contrast']
            )
            frame = self.effects.apply_lighting_gradient(
                frame, self.config['lighting_variation'],
                self.config['lighting_intensity']
            )

            # Apply blur effects (new unified blur system)
            if self.config.get('add_blur', False):
                # Use new blur system with random variation
                blur_type = self.config.get('blur_type', 'gaussian')
                blur_intensity = self.config.get('blur_intensity', 0.3)

                # If random blur variation is enabled, randomize per frame
                if self.config.get('random_blur_variation', False):
                    # Randomly vary intensity slightly per frame
                    if isinstance(blur_intensity, (int, float)):
                        intensity_variation = self.effects.rng.uniform(-0.1, 0.1)
                        blur_intensity = max(0.0, min(1.0, blur_intensity + intensity_variation))

                frame = self.effects.apply_blur(
                    frame,
                    blur_type=blur_type,
                    blur_intensity=blur_intensity,
                    direction=self.config['direction']
                )
            # Legacy motion blur support (for backward compatibility)
            elif self.config.get('motion_blur', 0) > 0:
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
        print(f"  [OK] Video saved: {output_path}")

    @staticmethod
    def generate_filename(config: dict, video_idx: int) -> str:
        """
        Generate descriptive filename from parameters.

        Args:
            config: Configuration dictionary
            video_idx: Video index number

        Returns:
            str: Filename encoding key parameters including object and blur info
        """
        # Build base filename
        parts = [
            f"treadmill_{video_idx:04d}",
            config['texture_type'],
        ]

        # Add stripe gray parameters for subtle_gray_stripes texture
        if config['texture_type'] == 'subtle_gray_stripes':
            stripe_gray = config.get('stripe_gray', 125)
            background_gray = config.get('background_gray', 140)
            parts.append(f"stripe{stripe_gray}")
            parts.append(f"bg{background_gray}")

        # Continue with other parameters
        parts.extend([
            config['direction'],
            f"speed{config['speed']:.1f}",
            f"angle{config['view_angle']:.0f}",
            f"bright{config['brightness']:.2f}",
            f"contr{config['contrast']:.2f}",
        ])

        # Add object information if objects are enabled
        if config.get('add_object', False):
            obj_type = config.get('object_type', 'box')
            obj_num = config.get('num_objects', 1)
            obj_size = config.get('object_size', 'medium')
            obj_pos = config.get('object_position', 'center')
            parts.append(f"obj_{obj_type}x{obj_num}_{obj_size}_{obj_pos}")

        # Add blur information if blur is enabled
        if config.get('add_blur', False):
            blur_type = config.get('blur_type', 'gaussian')
            blur_intensity = config.get('blur_intensity', 0.3)

            # Format intensity: use descriptor if string, numeric if float
            if isinstance(blur_intensity, str):
                intensity_str = blur_intensity
            else:
                intensity_str = f"{blur_intensity:.1f}"

            blur_str = f"blur_{blur_type}_{intensity_str}"

            # Add variation flag if enabled
            if config.get('random_blur_variation', False):
                blur_str += "_var"

            parts.append(blur_str)

        # Add resolution and seed
        parts.append(f"{config['resolution'][0]}x{config['resolution'][1]}")
        parts.append(f"seed{config['seed']}")

        filename = "_".join(parts) + ".mp4"
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
                       choices=['stripes', 'noise', 'rubber', 'grid', 'diamond_plate',
                               'factory_dark', 'factory_dark_stripes', 'subtle_gray_stripes'],
                       help='Type of belt texture (default: stripes)')
    parser.add_argument('--background_color', type=str, default='80,80,80',
                       help='Background color as R,G,B (default: 80,80,80)')

    # Subtle gray stripes parameters (for subtle_gray_stripes texture type)
    parser.add_argument('--stripe_width', type=str, default='10',
                       help='Stripe width in pixels for subtle_gray_stripes (default: 10). Supports comma-separated values.')
    parser.add_argument('--stripe_spacing', type=str, default='60',
                       help='Stripe spacing in pixels for subtle_gray_stripes (default: 60). Supports comma-separated values.')
    parser.add_argument('--stripe_gray', type=str, default='125',
                       help='Stripe gray level (0-255) for subtle_gray_stripes (default: 125). Supports comma-separated values.')
    parser.add_argument('--background_gray', type=str, default='140',
                       help='Background gray level (0-255) for subtle_gray_stripes (default: 140). Supports comma-separated values.')
    parser.add_argument('--stripe_distance_variance', type=str, default='0.0',
                       help='Variance (std dev) for stripe spacing in subtle_gray_stripes (default: 0.0). Supports comma-separated values.')

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
    parser.add_argument('--edge_width', type=float, default=0.1,
                       help='Belt enclosure edge width as percentage, 0.05 to 0.2 (default: 0.1)')

    # Object placement parameters
    parser.add_argument('--add-object', action='store_true',
                       help='Enable object placement on treadmill belt')
    parser.add_argument('--object-type', type=str, default='box',
                       choices=['box', 'circle', 'random'],
                       help='Type of object to place (default: box)')
    parser.add_argument('--object-position', type=str, default='center',
                       choices=['center', 'left', 'right', 'random'],
                       help='Position of object on belt (default: center)')
    parser.add_argument('--object-size', type=str, default='medium',
                       help='Size of object: small/medium/large or numeric value (default: medium)')
    parser.add_argument('--num-objects', type=int, default=1,
                       help='Number of objects to place (default: 1)')

    # Camera blur parameters
    parser.add_argument('--add-blur', action='store_true',
                       help='Enable camera blur effects')
    parser.add_argument('--blur-type', type=str, default='gaussian',
                       choices=['motion', 'gaussian', 'random'],
                       help='Type of blur effect (default: gaussian)')
    parser.add_argument('--blur-intensity', type=str, default='0.3',
                       help='Blur intensity: light/medium/heavy or 0.0-1.0 (default: 0.3)')
    parser.add_argument('--random-blur-variation', action='store_true',
                       help='Add random blur intensity variation across frames')

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

    textures = ['stripes', 'noise', 'rubber', 'grid', 'diamond_plate',
                'factory_dark', 'factory_dark_stripes']
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

    # Parse subtle_gray_stripes parameters (can be single or comma-separated values)
    stripe_width = int(args.stripe_width.split(',')[0]) if ',' not in args.stripe_width else args.stripe_width
    stripe_spacing = int(args.stripe_spacing.split(',')[0]) if ',' not in args.stripe_spacing else args.stripe_spacing
    stripe_gray = int(args.stripe_gray.split(',')[0]) if ',' not in args.stripe_gray else args.stripe_gray
    background_gray = int(args.background_gray.split(',')[0]) if ',' not in args.background_gray else args.background_gray
    stripe_distance_variance = float(args.stripe_distance_variance.split(',')[0]) if ',' not in args.stripe_distance_variance else args.stripe_distance_variance

    # Parse blur intensity (can be string like 'light' or numeric like '0.5')
    blur_intensity = args.blur_intensity
    try:
        blur_intensity = float(blur_intensity)
    except ValueError:
        # Keep as string if it's a descriptor like 'light', 'medium', 'heavy'
        pass

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
        'edge_width': args.edge_width,
        'seed': args.seed,
        'background_color': bg_color,
        'stripe_width': stripe_width,
        'stripe_spacing': stripe_spacing,
        'stripe_gray': stripe_gray,
        'background_gray': background_gray,
        'stripe_distance_variance': stripe_distance_variance,
        # Object placement parameters
        'add_object': args.add_object,
        'object_type': args.object_type,
        'object_position': args.object_position,
        'object_size': args.object_size,
        'num_objects': args.num_objects,
        # Camera blur parameters
        'add_blur': args.add_blur,
        'blur_type': args.blur_type,
        'blur_intensity': blur_intensity,
        'random_blur_variation': args.random_blur_variation,
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
    print(f"[OK] Generation complete!")
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
