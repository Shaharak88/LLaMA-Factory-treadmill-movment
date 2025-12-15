#!/usr/bin/env python3
"""
Script to extract video metadata from server dataset and save to CSV.

Connects to remote server, lists videos in a dataset directory,
parses filenames to extract parameters, and saves metadata to local CSV.

Uses dynamic regex extraction to automatically detect ALL numeric parameters
in filenames (e.g., speed, angle, dist, stripe, bg, etc.) without requiring
hardcoded field names.
"""

import subprocess
import csv
import re
import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Union


# Features to skip in dynamic extraction (not experimental parameters)
SKIP_FEATURES = {'seed', 'treadmill', 'x', 'mp'}


def extract_all_features(filename: str) -> Dict[str, Union[int, float]]:
    """
    Dynamically extract ALL experimental features from a video filename using regex.

    Automatically detects patterns like: name123, name12.34
    Uses the exact feature names from the filename (e.g., 'dist' not 'distance').

    Args:
        filename: Video filename containing encoded feature values

    Returns:
        Dictionary with feature names as keys (exactly as in filename)
        and their numeric values.

    Example:
        >>> extract_all_features("treadmill_stripe226_bg120_left_speed14.0_angle30_dist5.0.mp4")
        {'stripe': 226, 'bg': 120, 'speed': 14.0, 'angle': 30, 'dist': 5.0}

        >>> extract_all_features("video_blur0.5_object1_contr1.2.mp4")
        {'blur': 0.5, 'object': 1, 'contr': 1.2}
    """
    features = {}

    # Pattern: word characters followed by number (int or float)
    # Matches: dist5.0, angle30, stripe226, blur0.5, object1, contr1.00
    # The (?<![a-zA-Z]) ensures we don't match partial words
    pattern = r'(?<![a-zA-Z])([a-zA-Z]+)(\d+\.?\d*)'

    for match in re.finditer(pattern, filename):
        key = match.group(1).lower()
        value_str = match.group(2)

        # Skip non-experimental features
        if key in SKIP_FEATURES:
            continue

        # Skip if value string is empty
        if not value_str:
            continue

        # Convert to float or int
        if '.' in value_str:
            value = float(value_str)
        else:
            value = int(value_str)

        features[key] = value

    return features


def parse_video_filename(filename: str) -> Dict[str, str]:
    """
    Parse video filename to extract all parameters using DYNAMIC REGEX EXTRACTION.

    This function combines manual parsing for complex fields (texture, direction, etc.)
    with automatic regex-based extraction of ALL numeric parameters (speed, angle,
    dist, stripe, bg, bright, contr, etc.).

    Expected format:
    treadmill_{index}_{texture}_{direction}_speed{speed}_angle{angle}_dist{dist}_bright{brightness}_contr{contrast}
    [_obj_{object_info}][_blur_{blur_info}]_{width}x{height}_seed{seed}.mp4

    Args:
        filename: Video filename to parse

    Returns:
        Dictionary containing extracted parameters (all numeric values + complex fields)
    """
    # Initialize with video name
    metadata = {
        'video_name': filename
    }

    # STEP 1: Use dynamic regex extraction to get ALL numeric parameters automatically
    # This captures: speed, angle, dist, stripe, bg, bright, contr, seed, etc.
    dynamic_features = extract_all_features(filename)

    # Add all dynamically extracted features to metadata (as strings for CSV compatibility)
    for key, value in dynamic_features.items():
        # Special handling: 'seed' should be kept as an integer string
        if key == 'seed':
            metadata['seed'] = str(int(value))
        else:
            # Keep numeric precision for other values
            if isinstance(value, float):
                metadata[key] = str(value)
            else:
                metadata[key] = str(value)

    # STEP 2: Manual parsing for complex fields that aren't simple key-value pairs
    # Remove .mp4 extension
    name = filename.replace('.mp4', '')
    parts = name.split('_')

    try:
        # Extract index (treadmill_0000)
        if parts[0] == 'treadmill':
            metadata['index'] = parts[1]
            remaining_idx = 2
        else:
            metadata['index'] = ''
            remaining_idx = 0

        # Extract texture (can be multi-part like "subtle_gray_stripes")
        texture_parts = []
        while remaining_idx < len(parts):
            if parts[remaining_idx] in ['up', 'down', 'left', 'right', 'stationary']:
                break
            # Stop if we hit parameter values (digits after parameter names)
            if re.match(r'^(stripe|bg|speed|angle|dist|bright|contr)\d+', parts[remaining_idx]):
                break
            texture_parts.append(parts[remaining_idx])
            remaining_idx += 1
        metadata['texture'] = '_'.join(texture_parts) if texture_parts else ''

        # Extract direction
        if remaining_idx < len(parts) and parts[remaining_idx] in ['up', 'down', 'left', 'right', 'stationary']:
            metadata['direction'] = parts[remaining_idx]
            remaining_idx += 1
        else:
            metadata['direction'] = ''

        # Parse remaining parts for complex fields
        i = remaining_idx
        while i < len(parts):
            part = parts[i]

            # Object (format: obj_boxx1_30_center)
            if part == 'obj':
                metadata['object_enabled'] = 'yes'
                # Next parts are: type, num, size, position (e.g., boxx1_30_center)
                if i + 1 < len(parts):
                    obj_info = parts[i + 1]
                    # Parse object info: e.g., "boxx1" or "conex2"
                    obj_match = re.match(r'([a-z]+)x(\d+)', obj_info)
                    if obj_match:
                        metadata['object_type'] = obj_match.group(1)
                        metadata['num_objects'] = obj_match.group(2)

                    # Size
                    if i + 2 < len(parts) and parts[i + 2] not in ['blur', 'gaussian', 'motion'] \
                       and not re.match(r'\d+x\d+', parts[i + 2]):
                        metadata['object_size'] = parts[i + 2]

                        # Position
                        if i + 3 < len(parts) and not re.match(r'\d+x\d+', parts[i + 3]) \
                           and parts[i + 3] not in ['blur', 'gaussian', 'motion']:
                            metadata['object_position'] = parts[i + 3]
                            i += 3
                        else:
                            i += 2
                    else:
                        i += 1

            # Blur (format: blur_gaussian_0.3 or blur_gaussian_0.3_var)
            elif part == 'blur':
                metadata['blur_enabled'] = 'yes'
                if i + 1 < len(parts):
                    metadata['blur_type'] = parts[i + 1]
                    i += 1
                if i + 1 < len(parts) and not re.match(r'\d+x\d+', parts[i + 1]):
                    intensity_or_var = parts[i + 1]
                    if intensity_or_var == 'var':
                        metadata['blur_variation'] = 'yes'
                    else:
                        metadata['blur_intensity'] = intensity_or_var
                        # Check for variation flag
                        if i + 2 < len(parts) and parts[i + 2] == 'var':
                            metadata['blur_variation'] = 'yes'
                            i += 1
                    i += 1

            # Resolution (640x480)
            elif re.match(r'\d+x\d+', part):
                metadata['resolution'] = part

            # Check for special flags that might not have been captured
            elif part == 'distrand':
                metadata['distance_randomized'] = 'yes'

            i += 1

        # Set defaults for complex fields if not found
        if 'object_enabled' not in metadata:
            metadata['object_enabled'] = 'no'
        if 'object_type' not in metadata:
            metadata['object_type'] = ''
        if 'num_objects' not in metadata:
            metadata['num_objects'] = ''
        if 'object_size' not in metadata:
            metadata['object_size'] = ''
        if 'object_position' not in metadata:
            metadata['object_position'] = ''
        if 'blur_enabled' not in metadata:
            metadata['blur_enabled'] = 'no'
        if 'blur_type' not in metadata:
            metadata['blur_type'] = ''
        if 'blur_intensity' not in metadata:
            metadata['blur_intensity'] = ''
        if 'blur_variation' not in metadata:
            metadata['blur_variation'] = 'no'
        if 'resolution' not in metadata:
            metadata['resolution'] = ''
        if 'distance_randomized' not in metadata:
            metadata['distance_randomized'] = 'no'

    except Exception as e:
        print(f"Warning: Error parsing filename '{filename}': {e}")

    return metadata


def load_video_order_from_json(json_path: str) -> tuple[str, List[str]]:
    """
    Load video paths from dataset JSON file in order.

    Args:
        json_path: Path to dataset JSON file

    Returns:
        Tuple of (dataset_name, list of video filenames in JSON order)
    """
    print(f"Reading JSON file: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON to be a list, got {type(data)}")

    # Extract video paths in order
    video_filenames = []
    for idx, entry in enumerate(data):
        if 'videos' not in entry:
            print(f"Warning: Entry {idx} missing 'videos' field, skipping")
            continue

        videos = entry['videos']
        if isinstance(videos, list):
            if len(videos) == 0:
                print(f"Warning: Entry {idx} has empty videos list, skipping")
                continue
            video_path = videos[0]
        else:
            video_path = videos

        # Extract just the filename (not full path)
        filename = os.path.basename(video_path)
        video_filenames.append(filename)

    # Extract dataset name from JSON filename
    # Format: data/_exp_20251209_dist_augment_train.json -> _exp_20251209_dist_augment_train
    json_filename = os.path.basename(json_path)
    dataset_name = json_filename.replace('.json', '')

    print(f"Dataset: {dataset_name}")
    print(f"Found {len(video_filenames)} videos in JSON (in order)")

    return dataset_name, video_filenames


def save_metadata_to_csv(metadata_list: List[Dict[str, str]], output_file: str):
    """
    Save video metadata to CSV file using DYNAMIC column detection.

    Automatically detects all columns present in the metadata and orders them
    intelligently (fixed fields first, then dynamically detected numeric parameters).

    Args:
        metadata_list: List of metadata dictionaries
        output_file: Path to output CSV file
    """
    if not metadata_list:
        print("No metadata to save")
        return

    # Collect all unique keys across all videos
    all_keys = set()
    for metadata in metadata_list:
        all_keys.update(metadata.keys())

    # Define priority order for known fixed fields (always first)
    priority_fields = [
        'video_name',
        'index',
        'texture',
        'direction',
    ]

    # Common numeric parameters (ordered logically)
    common_numeric = [
        'speed',
        'angle',
        'dist',  # IMPORTANT: This is now included!
        'bright',
        'contr',
        'stripe',
        'bg',
    ]

    # Secondary fields
    secondary_fields = [
        'object_enabled',
        'object_type',
        'num_objects',
        'object_size',
        'object_position',
        'blur_enabled',
        'blur_type',
        'blur_intensity',
        'blur_variation',
        'distance_randomized',
        'resolution',
        'seed',
    ]

    # Build final fieldnames list
    fieldnames = []

    # Add priority fields that exist
    for field in priority_fields:
        if field in all_keys:
            fieldnames.append(field)
            all_keys.remove(field)

    # Add common numeric parameters that exist
    for field in common_numeric:
        if field in all_keys:
            fieldnames.append(field)
            all_keys.remove(field)

    # Add secondary fields that exist
    for field in secondary_fields:
        if field in all_keys:
            fieldnames.append(field)
            all_keys.remove(field)

    # Add any remaining fields (dynamically detected parameters not in our lists)
    # Sort them alphabetically for consistency
    remaining = sorted(list(all_keys))
    fieldnames.extend(remaining)

    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(metadata_list)

        print(f"\nMetadata saved to: {output_file}")
        print(f"Total videos: {len(metadata_list)}")
        print(f"Columns detected: {len(fieldnames)}")
        print(f"Column names: {', '.join(fieldnames)}")

    except Exception as e:
        print(f"Error saving CSV: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract video metadata from JSON dataset file and save to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract metadata from dataset JSON (preserves video order from JSON)
  python extract_video_metadata.py data/_exp_20251209_dist_augment_train.json

  # Specify custom output directory
  python extract_video_metadata.py data/my_dataset_train.json --output-dir ./data
"""
    )

    parser.add_argument(
        'json_file',
        help='Path to dataset JSON file (e.g., data/_exp_20251209_dist_augment_train.json)'
    )

    parser.add_argument(
        '--output-dir',
        default='./data',
        help='Base directory for output (default: ./data). CSV will be saved in {output-dir}/{dataset_name}/'
    )

    args = parser.parse_args()

    # Load video order from JSON
    try:
        dataset_name, video_filenames = load_video_order_from_json(args.json_file)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    # Create output directory: ./data/{dataset_name}/
    dataset_output_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    # Use FIXED filename (no timestamp)
    output_file = os.path.join(
        dataset_output_dir,
        f"{dataset_name}_metadata.csv"
    )

    print("=" * 60)
    print("Video Metadata Extraction (JSON-Based)")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"JSON file: {args.json_file}")
    print(f"Output file: {output_file}")
    print("=" * 60)
    print()

    if not video_filenames:
        print("No videos found in JSON")
        return

    # Parse metadata from filenames (in JSON order)
    print("\nParsing video filenames...")
    metadata_list = []
    for idx, video in enumerate(video_filenames):
        metadata = parse_video_filename(video)
        # Override index with sequential position matching JSON order (not filename number)
        metadata['index'] = idx
        metadata_list.append(metadata)

    # Save to CSV (preserves JSON order)
    save_metadata_to_csv(metadata_list, output_file)

    print("\nDone!")


if __name__ == '__main__':
    main()
