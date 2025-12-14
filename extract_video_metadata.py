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


def list_videos_on_server(server: str, dataset_path: str, docker_container: Optional[str] = None) -> List[str]:
    """
    Connect to server and list all video files in the dataset directory.

    Args:
        server: SSH server address
        dataset_path: Path to dataset directory on server (or in Docker container)
        docker_container: Optional Docker container name to execute commands in

    Returns:
        List of video filenames
    """
    try:
        # Build command based on whether we're using Docker or not
        if docker_container:
            # SSH into server and execute docker command
            docker_cmd = f'docker exec {docker_container} bash -c "ls -1 {dataset_path}/*.mp4 2>/dev/null"'
            cmd = ['ssh', server, docker_cmd]
        else:
            # Direct SSH command
            cmd = ['ssh', server, f'ls -1 "{dataset_path}"/*.mp4 2>/dev/null']

        print(f"Connecting to {server}...")
        if docker_container:
            print(f"Using Docker container: {docker_container}")
        print(f"Listing videos in {dataset_path}...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Extract just the filenames (basename)
        videos = []
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if line and line.endswith('.mp4'):
                # Get just the filename, not the full path
                videos.append(os.path.basename(line))

        print(f"Found {len(videos)} videos")

        return videos

    except subprocess.CalledProcessError as e:
        print(f"Error connecting to server or listing files: {e}")
        print(f"stderr: {e.stderr}")
        return []


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
        description='Extract video metadata from server dataset and save to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract metadata for a specific dataset (uses default Docker container: llamafactory)
  python extract_video_metadata.py preformat_right_only_20251130_141204

  # Specify custom output directory
  python extract_video_metadata.py preformat_right_only_20251130_141204 --output-dir ./metadata

  # Use custom Docker container
  python extract_video_metadata.py my_dataset --docker-container my_container

  # Use custom server without Docker
  python extract_video_metadata.py my_dataset --server user@hostname --docker-container ""
"""
    )

    parser.add_argument(
        'dataset_name',
        help='Name of the dataset (will look in /app/data/{dataset_name}/ on server)'
    )

    parser.add_argument(
        '--server',
        default='seedoo@hetzner-gpu.tail9e6e7.ts.net',
        help='SSH server address (default: seedoo@hetzner-gpu.tail9e6e7.ts.net)'
    )

    parser.add_argument(
        '--dataset-base-path',
        default='/app/data',
        help='Base path for datasets on server (default: /app/data)'
    )

    parser.add_argument(
        '--docker-container',
        default='llamafactory',
        help='Docker container name to execute commands in (default: llamafactory)'
    )

    parser.add_argument(
        '--output-dir',
        default='./data',
        help='Local directory to save CSV file (default: ./data)'
    )

    args = parser.parse_args()

    # Construct full dataset path on server
    dataset_path = f"{args.dataset_base_path}/{args.dataset_name}"

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(
        args.output_dir,
        f"{args.dataset_name}_metadata_{timestamp}.csv"
    )

    print("=" * 60)
    print("Video Metadata Extraction")
    print("=" * 60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Server: {args.server}")
    print(f"Docker container: {args.docker_container}")
    print(f"Remote path: {dataset_path}")
    print(f"Output file: {output_file}")
    print("=" * 60)
    print()

    # Get list of videos from server
    videos = list_videos_on_server(args.server, dataset_path, args.docker_container)

    if not videos:
        print("No videos found or error occurred")
        return

    # Parse metadata from filenames
    print("\nParsing video filenames...")
    metadata_list = []
    for video in videos:
        metadata = parse_video_filename(video)
        metadata_list.append(metadata)

    # Save to CSV
    save_metadata_to_csv(metadata_list, output_file)

    print("\nDone!")


if __name__ == '__main__':
    main()
