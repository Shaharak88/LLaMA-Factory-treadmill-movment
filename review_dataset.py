#!/usr/bin/env python3
"""
All-in-one script to review video datasets.

Just provide the dataset name and this script will:
1. Extract metadata from server and create CSV
2. Download videos to your local PC
3. Generate interactive HTML report with working video players
"""

import subprocess
import csv
import json
import os
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter


# ============================================================================
# METADATA EXTRACTION (from extract_video_metadata.py)
# ============================================================================

def parse_video_filename(filename: str) -> Dict[str, str]:
    """Parse video filename to extract all parameters."""
    metadata = {
        'video_name': filename,
        'index': '',
        'texture': '',
        'direction': '',
        'speed': '',
        'angle': '',
        'brightness': '',
        'contrast': '',
        'stripe_contrast': 'N/A',
        'background_contrast': 'N/A',
        'object_enabled': 'no',
        'object_type': '',
        'num_objects': '',
        'object_size': '',
        'object_position': '',
        'blur_enabled': 'no',
        'blur_type': '',
        'blur_intensity': '',
        'blur_variation': 'no',
        'resolution': '',
        'seed': ''
    }

    name = filename.replace('.mp4', '')
    parts = name.split('_')

    try:
        if parts[0] == 'treadmill':
            metadata['index'] = parts[1]
            remaining_idx = 2
        else:
            remaining_idx = 0

        texture_parts = []
        while remaining_idx < len(parts):
            if parts[remaining_idx] in ['up', 'down', 'left', 'right', 'stationary']:
                break
            # Stop if we hit stripe/bg parameter values (digits after stripe/bg)
            if re.match(r'^stripe\d+$', parts[remaining_idx]) or re.match(r'^bg\d+$', parts[remaining_idx]):
                break
            texture_parts.append(parts[remaining_idx])
            remaining_idx += 1
        metadata['texture'] = '_'.join(texture_parts)

        # Extract stripe_gray and background_gray if present (for subtle_gray_stripes)
        if remaining_idx < len(parts) and re.match(r'^stripe\d+$', parts[remaining_idx]):
            metadata['stripe_contrast'] = parts[remaining_idx].replace('stripe', '')
            remaining_idx += 1
        if remaining_idx < len(parts) and re.match(r'^bg\d+$', parts[remaining_idx]):
            metadata['background_contrast'] = parts[remaining_idx].replace('bg', '')
            remaining_idx += 1

        if remaining_idx < len(parts):
            metadata['direction'] = parts[remaining_idx]
            remaining_idx += 1

        i = remaining_idx
        while i < len(parts):
            part = parts[i]

            if part.startswith('speed'):
                metadata['speed'] = part.replace('speed', '')
            elif part.startswith('angle'):
                metadata['angle'] = part.replace('angle', '')
            elif part.startswith('bright'):
                metadata['brightness'] = part.replace('bright', '')
            elif part.startswith('contr'):
                metadata['contrast'] = part.replace('contr', '')
            elif part == 'obj':
                metadata['object_enabled'] = 'yes'
                if i + 1 < len(parts):
                    obj_info = parts[i + 1]
                    obj_match = re.match(r'([a-z]+)x(\d+)', obj_info)
                    if obj_match:
                        metadata['object_type'] = obj_match.group(1)
                        metadata['num_objects'] = obj_match.group(2)
                    if i + 2 < len(parts) and parts[i + 2] not in ['blur', 'gaussian', 'motion'] \
                       and not re.match(r'\d+x\d+', parts[i + 2]):
                        metadata['object_size'] = parts[i + 2]
                        if i + 3 < len(parts) and not re.match(r'\d+x\d+', parts[i + 3]) \
                           and parts[i + 3] not in ['blur', 'gaussian', 'motion']:
                            metadata['object_position'] = parts[i + 3]
                            i += 3
                        else:
                            i += 2
                    else:
                        i += 1
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
                        if i + 2 < len(parts) and parts[i + 2] == 'var':
                            metadata['blur_variation'] = 'yes'
                            i += 1
                    i += 1
            elif re.match(r'\d+x\d+', part):
                metadata['resolution'] = part
            elif part.startswith('seed'):
                metadata['seed'] = part.replace('seed', '')

            i += 1

    except Exception as e:
        print(f"Warning: Error parsing filename '{filename}': {e}")

    return metadata


def list_videos_on_server(server: str, dataset_path: str, docker_container: Optional[str] = None) -> List[str]:
    """Connect to server and list all video files in the dataset directory."""
    try:
        if docker_container:
            docker_cmd = f'docker exec {docker_container} bash -c "ls -1 {dataset_path}/*.mp4 2>/dev/null"'
            cmd = ['ssh', server, docker_cmd]
        else:
            cmd = ['ssh', server, f'ls -1 "{dataset_path}"/*.mp4 2>/dev/null']

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        videos = []
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if line and line.endswith('.mp4'):
                videos.append(os.path.basename(line))

        return videos

    except subprocess.CalledProcessError as e:
        print(f"Error connecting to server or listing files: {e}")
        print(f"stderr: {e.stderr}")
        return []


def extract_metadata(dataset_name: str, server: str, remote_base: str,
                    output_dir: str, docker_container: str = None) -> str:
    """Extract metadata from server and save to CSV."""
    dataset_path = f"{remote_base}/{dataset_name}"

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"{dataset_name}_metadata_{timestamp}.csv")

    print(f"📋 Extracting metadata from server...")
    print(f"   Dataset: {dataset_name}")
    print(f"   Server: {server}")
    if docker_container:
        print(f"   Docker container: {docker_container}")
    print(f"   Remote path: {dataset_path}")

    # Get list of videos from server
    videos = list_videos_on_server(server, dataset_path, docker_container)

    if not videos:
        print("❌ No videos found or error occurred")
        return None

    print(f"   Found {len(videos)} videos")

    # Parse metadata from filenames
    metadata_list = []
    for video in videos:
        metadata = parse_video_filename(video)
        metadata_list.append(metadata)

    # Save to CSV
    fieldnames = [
        'video_name', 'index', 'texture', 'direction', 'speed', 'angle',
        'brightness', 'contrast', 'stripe_contrast', 'background_contrast',
        'object_enabled', 'object_type', 'num_objects', 'object_size', 'object_position',
        'blur_enabled', 'blur_type', 'blur_intensity', 'blur_variation',
        'resolution', 'seed'
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_list)

    print(f"   ✅ Metadata saved to: {output_file}")

    return output_file


# ============================================================================
# VIDEO DOWNLOAD
# ============================================================================

def download_videos_from_server(dataset_name: str, server: str, remote_base: str,
                                local_dir: str, docker_container: str = None) -> str:
    """Download videos from remote server using direct file copy."""
    local_video_dir = os.path.join(local_dir, dataset_name)
    os.makedirs(local_video_dir, exist_ok=True)

    print(f"\n📥 Downloading videos from server...")
    print(f"   Remote: {server}:{remote_base}/{dataset_name}/")
    print(f"   Local: {local_video_dir}/")

    try:
        # Step 1: Get list of video files
        if docker_container:
            print(f"   (via Docker container: {docker_container})")
            list_cmd = f'docker exec {docker_container} bash -c "ls {remote_base}/{dataset_name}/*.mp4"'
            result = subprocess.run(['ssh', server, list_cmd],
                                   capture_output=True, text=True, check=True)
        else:
            result = subprocess.run(['ssh', server, f'ls {remote_base}/{dataset_name}/*.mp4'],
                                   capture_output=True, text=True, check=True)

        video_files = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]

        if not video_files:
            print("   ❌ No videos found!")
            return local_video_dir

        print(f"   Found {len(video_files)} videos to download")

        # Step 2: Copy from Docker to temp location if needed
        temp_path = None
        if docker_container:
            temp_path = f"/tmp/{dataset_name}_videos/"
            print(f"   Copying from Docker to server temp location...")
            docker_copy_cmd = f'mkdir -p {temp_path} && docker cp {docker_container}:{remote_base}/{dataset_name}/. {temp_path}'
            subprocess.run(['ssh', server, docker_copy_cmd], check=True)

        # Step 3: Download each video file
        print(f"   Downloading files:")
        for i, remote_file in enumerate(video_files, 1):
            filename = os.path.basename(remote_file)
            local_file = os.path.join(local_video_dir, filename)

            # Skip if already exists
            if os.path.exists(local_file):
                print(f"   [{i}/{len(video_files)}] ⏭️  {filename} (already exists)")
                continue

            # Download using scp
            if temp_path:
                source = f"{server}:{temp_path}{filename}"
            else:
                source = f"{server}:{remote_file}"

            print(f"   [{i}/{len(video_files)}] ⬇️  {filename}...", end='', flush=True)

            result = subprocess.run(['scp', '-q', source, local_file],
                                   capture_output=True, text=True)

            if result.returncode == 0 and os.path.exists(local_file):
                file_size = os.path.getsize(local_file)
                print(f" ✅ ({file_size:,} bytes)")
            else:
                print(f" ❌ Failed")

        # Step 4: Clean up temp directory
        if temp_path:
            subprocess.run(['ssh', server, f'rm -rf {temp_path}'], check=False)

        # Step 5: Verify downloads
        downloaded_files = list(Path(local_video_dir).glob('*.mp4'))
        print(f"\n   ✅ Downloaded {len(downloaded_files)} videos successfully")

        return local_video_dir

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error downloading videos: {e}")
        print(f"   stderr: {e.stderr}")
        return local_video_dir
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return local_video_dir


# ============================================================================
# REPORT GENERATION (from generate_video_report.py)
# ============================================================================

def load_metadata(csv_path: str) -> List[Dict[str, str]]:
    """Load metadata from CSV file."""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def analyze_distribution(metadata: List[Dict[str, str]]) -> Dict[str, Any]:
    """Analyze parameter distributions and identify under/over-represented groups."""
    stats = {
        'total_videos': len(metadata),
        'distributions': {},
        'under_represented': [],
        'over_represented': []
    }

    key_params = ['texture', 'direction', 'speed', 'angle', 'brightness',
                  'contrast', 'object_enabled', 'blur_enabled']

    for param in key_params:
        values = [row.get(param, '') for row in metadata if row.get(param, '')]
        counter = Counter(values)

        total = sum(counter.values())
        avg = total / len(counter) if counter else 0
        std_threshold = avg * 0.5

        stats['distributions'][param] = {
            'counts': dict(counter.most_common()),
            'unique_values': len(counter),
            'average_per_value': round(avg, 2)
        }

        for value, count in counter.items():
            ratio = count / avg if avg > 0 else 0

            if count < avg - std_threshold:
                stats['under_represented'].append({
                    'parameter': param,
                    'value': value,
                    'count': count,
                    'expected': round(avg, 2),
                    'ratio': round(ratio, 2)
                })
            elif count > avg + std_threshold:
                stats['over_represented'].append({
                    'parameter': param,
                    'value': value,
                    'count': count,
                    'expected': round(avg, 2),
                    'ratio': round(ratio, 2)
                })

    return stats


def group_videos(metadata: List[Dict[str, str]]) -> Dict[str, Dict[str, List[Dict]]]:
    """Group videos by different parameters for tabbed viewing."""
    groups = {
        'texture': defaultdict(list),
        'direction': defaultdict(list),
        'speed_range': defaultdict(list),
        'angle': defaultdict(list),
        'objects': defaultdict(list),
        'blur': defaultdict(list)
    }

    for video in metadata:
        groups['texture'][video.get('texture', 'unknown')].append(video)
        groups['direction'][video.get('direction', 'unknown')].append(video)

        try:
            speed = float(video.get('speed', '0'))
            if speed == 0:
                speed_range = 'stationary'
            elif speed < 5:
                speed_range = 'slow (0-5)'
            elif speed < 10:
                speed_range = 'medium (5-10)'
            else:
                speed_range = 'fast (10+)'
            groups['speed_range'][speed_range].append(video)
        except ValueError:
            groups['speed_range']['unknown'].append(video)

        groups['angle'][f"{video.get('angle', 'unknown')}°"].append(video)

        obj_enabled = video.get('object_enabled', 'no')
        obj_type = video.get('object_type', 'none')
        obj_key = f"{obj_type}" if obj_enabled == 'yes' else 'no_objects'
        groups['objects'][obj_key].append(video)

        blur_enabled = video.get('blur_enabled', 'no')
        blur_type = video.get('blur_type', 'none')
        blur_key = f"{blur_type}" if blur_enabled == 'yes' else 'no_blur'
        groups['blur'][blur_key].append(video)

    return groups


def generate_html_report(metadata: List[Dict[str, str]], stats: Dict[str, Any],
                        groups: Dict[str, Dict[str, List[Dict]]], output_path: str,
                        video_base_url: str):
    """Generate interactive HTML report."""

    # This is the same HTML generation code from generate_video_report.py
    # (keeping it compact here - it's the same as before)
    from generate_video_report import generate_html
    generate_html(metadata, stats, groups, output_path, video_base_url)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='All-in-one script to review video datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Review a dataset (does everything automatically)
  python review_dataset.py preformat_right_only_20251130_141204

  # Review with custom output location
  python review_dataset.py my_experiment_train --output reports/my_report.html

  # Custom server settings
  python review_dataset.py my_dataset --server user@myserver.com --remote-path /data
"""
    )

    parser.add_argument(
        'dataset_name',
        help='Dataset name on the server (e.g., exp_20251201_train)'
    )

    parser.add_argument(
        '--output',
        default='video_dataset_report.html',
        help='Output HTML report path (default: video_dataset_report.html)'
    )

    parser.add_argument(
        '--server',
        default='seedoo@hetzner-gpu.tail9e6e7.ts.net',
        help='SSH server address (default: seedoo@hetzner-gpu.tail9e6e7.ts.net)'
    )

    parser.add_argument(
        '--remote-path',
        default='/app/data',
        help='Remote base path for datasets (default: /app/data)'
    )

    parser.add_argument(
        '--docker-container',
        default='llamafactory',
        help='Docker container name (default: llamafactory, empty string for none)'
    )

    parser.add_argument(
        '--metadata-dir',
        default='./data',
        help='Local directory to save metadata CSV (default: ./data)'
    )

    parser.add_argument(
        '--video-dir',
        default='./data/videos',
        help='Local directory to download videos (default: ./data/videos)'
    )

    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip video download (for testing or if videos already downloaded)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("📹 VIDEO DATASET REVIEW - ALL-IN-ONE")
    print("=" * 70)
    print(f"Dataset: {args.dataset_name}")
    print(f"Server: {args.server}")
    print(f"Output: {args.output}")
    print("=" * 70)
    print()

    # Step 1: Extract metadata from server
    csv_path = extract_metadata(
        args.dataset_name,
        args.server,
        args.remote_path,
        args.metadata_dir,
        args.docker_container if args.docker_container else None
    )

    if not csv_path:
        print("\n❌ Failed to extract metadata. Exiting.")
        return

    # Step 2: Download videos from server
    if not args.skip_download:
        local_video_path = download_videos_from_server(
            args.dataset_name,
            args.server,
            args.remote_path,
            args.video_dir,
            args.docker_container if args.docker_container else None
        )
    else:
        print("\n⏭️  Skipping video download (--skip-download)")
        local_video_path = os.path.join(args.video_dir, args.dataset_name)

    # Use relative path for local HTTP server
    output_dir = os.path.dirname(os.path.abspath(args.output))
    local_video_abs = os.path.abspath(local_video_path)
    video_url = os.path.relpath(local_video_abs, output_dir)

    # Step 3: Load metadata and analyze
    print(f"\n📊 Analyzing dataset...")
    metadata = load_metadata(csv_path)
    stats = analyze_distribution(metadata)
    print(f"   Total videos: {len(metadata)}")
    print(f"   Under-represented groups: {len(stats['under_represented'])}")
    print(f"   Over-represented groups: {len(stats['over_represented'])}")

    # Step 4: Group videos
    print(f"📁 Grouping videos by parameters...")
    groups = group_videos(metadata)

    # Step 5: Generate HTML report
    print(f"🎨 Generating HTML report...")

    # Import and use the generate_html function from generate_video_report.py
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_video_report import generate_html

    generate_html(metadata, stats, groups, args.output, video_url)

    # Verify videos exist
    downloaded_files = list(Path(local_video_path).glob('*.mp4'))

    # Done!
    print("\n" + "=" * 70)
    print("✅ ALL DONE!")
    print("=" * 70)
    print(f"📄 Metadata CSV: {csv_path}")
    print(f"📂 Videos: {local_video_path} ({len(downloaded_files)} files)")
    print(f"📊 HTML Report: {args.output}")
    print("=" * 70)
    print()
    print("🎬 TO VIEW THE REPORT WITH WORKING VIDEOS:")
    print()
    print("OPTION 1 (Easiest):")
    print("   ./view_report.sh " + args.output)
    print()
    print("OPTION 2 (Manual):")
    print("   Step 1: python3 -m http.server 8000")
    print(f"   Step 2: Open http://localhost:8000/{args.output}")
    print()
    print("OPTION 3 (Quick test):")
    print(f"   python3 -m http.server 8000 &")
    print(f"   explorer.exe 'http://localhost:8000/{args.output}'")
    print()
    print("⚠️  IMPORTANT: Videos ONLY work with the web server running!")
    print("   They won't play if you just double-click the HTML file.")
    print("=" * 70)


if __name__ == '__main__':
    main()
