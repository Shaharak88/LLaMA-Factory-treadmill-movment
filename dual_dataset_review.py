#!/usr/bin/env python3
"""
Dual Dataset Video Review Tool

One-command solution to:
1. Extract metadata from server for both train and test datasets
2. Download videos to local PC for both datasets
3. Generate interactive HTML report with video playback and comparison

Just provide the base experiment name and everything happens automatically!

Usage:
    python3 dual_dataset_review.py BASE_EXPERIMENT_NAME

Example:
    python3 dual_dataset_review.py _exp_20251201_172008
"""

import subprocess
import csv
import json
import os
import re
import argparse
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter


# ============================================================================
# STEP 1: METADATA EXTRACTION FROM SERVER
# ============================================================================

def parse_video_filename(filename: str) -> Dict[str, str]:
    """
    Parse video filename to extract all parameters.

    Supports both formats:
    - Old: treadmill_0000_subtle_gray_stripes_left_speed8.0_...
    - New: treadmill_0000_subtle_gray_stripes_stripe125_bg140_left_speed8.0_...
    """
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
        # Extract index (treadmill_0000)
        if parts[0] == 'treadmill':
            metadata['index'] = parts[1]
            remaining_idx = 2
        else:
            remaining_idx = 0

        # Extract texture (can be multi-part like "subtle_gray_stripes")
        texture_parts = []
        while remaining_idx < len(parts):
            if parts[remaining_idx] in ['up', 'down', 'left', 'right', 'stationary']:
                break
            # Stop if we hit stripe/bg parameter values (new format)
            if re.match(r'^stripe\d+$', parts[remaining_idx]) or re.match(r'^bg\d+$', parts[remaining_idx]):
                break
            texture_parts.append(parts[remaining_idx])
            remaining_idx += 1
        metadata['texture'] = '_'.join(texture_parts)

        # Extract stripe_gray and background_gray if present (NEW FORMAT)
        if remaining_idx < len(parts) and re.match(r'^stripe\d+$', parts[remaining_idx]):
            metadata['stripe_contrast'] = parts[remaining_idx].replace('stripe', '')
            remaining_idx += 1
        if remaining_idx < len(parts) and re.match(r'^bg\d+$', parts[remaining_idx]):
            metadata['background_contrast'] = parts[remaining_idx].replace('bg', '')
            remaining_idx += 1

        # Extract direction
        if remaining_idx < len(parts):
            metadata['direction'] = parts[remaining_idx]
            remaining_idx += 1

        # Parse remaining parts
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
        return []


def extract_metadata(dataset_name: str, server: str, remote_base: str,
                    output_dir: str, docker_container: str = None) -> str:
    """
    Extract metadata from server and save to CSV.

    Uses extract_video_metadata.py script with dynamic feature extraction.
    This ensures consistency with standalone metadata extraction and includes
    all features (distance, etc.) automatically.
    """
    print(f"📋 Extracting metadata using extract_video_metadata.py...")
    print(f"   Dataset: {dataset_name}")
    print(f"   Server: {server}")
    if docker_container:
        print(f"   Docker: {docker_container}")

    # Build command to run extract_video_metadata.py
    script_path = Path(__file__).parent / "extract_video_metadata.py"
    cmd = [
        "python3", str(script_path),
        dataset_name,
        "--server", server,
        "--dataset-base-path", remote_base,
        "--output-dir", output_dir,
    ]

    if docker_container:
        cmd.extend(["--docker-container", docker_container])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)

        # Find the generated CSV file (extract_video_metadata.py saves to dataset folder)
        dataset_dir = os.path.join(output_dir, dataset_name)
        csv_files = list(Path(dataset_dir).glob(f"{dataset_name}_metadata_*.csv"))

        if csv_files:
            # Return the most recent one
            latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
            print(f"   ✅ Metadata CSV: {latest_csv}")
            return str(latest_csv)
        else:
            print("   ❌ No CSV file found after extraction")
            return None

    except subprocess.CalledProcessError as e:
        print(f"   ❌ Extraction failed: {e}")
        print(f"   stderr: {e.stderr}")
        return None


# ============================================================================
# STEP 2: VIDEO DOWNLOAD FROM SERVER
# ============================================================================

def download_videos(dataset_name: str, server: str, remote_base: str,
                   local_dir: str, docker_container: str = None) -> str:
    """Download videos from remote server."""
    local_video_dir = os.path.join(local_dir, dataset_name)
    os.makedirs(local_video_dir, exist_ok=True)

    print(f"\n📥 Downloading videos...")
    print(f"   Remote: {server}:{remote_base}/{dataset_name}/")
    print(f"   Local: {local_video_dir}/")

    try:
        # Get list of videos
        if docker_container:
            print(f"   Via Docker: {docker_container}")
            list_cmd = f'docker exec {docker_container} bash -c "ls {remote_base}/{dataset_name}/*.mp4"'
            result = subprocess.run(['ssh', server, list_cmd],
                                   capture_output=True, text=True, check=True)
        else:
            result = subprocess.run(['ssh', server, f'ls {remote_base}/{dataset_name}/*.mp4'],
                                   capture_output=True, text=True, check=True)

        video_files = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]

        if not video_files:
            print("   ❌ No videos found")
            return local_video_dir

        print(f"   Found {len(video_files)} videos")

        # Copy from Docker to temp if needed
        temp_path = None
        if docker_container:
            temp_path = f"/tmp/{dataset_name}_videos/"
            print(f"   Copying from Docker to temp...")
            docker_copy_cmd = f'mkdir -p {temp_path} && docker cp {docker_container}:{remote_base}/{dataset_name}/. {temp_path}'
            subprocess.run(['ssh', server, docker_copy_cmd], check=True)

        # Download each video
        print(f"   Downloading:")
        for i, remote_file in enumerate(video_files, 1):
            filename = os.path.basename(remote_file)
            local_file = os.path.join(local_video_dir, filename)

            if os.path.exists(local_file):
                print(f"   [{i}/{len(video_files)}] ⏭️  {filename} (exists)")
                continue

            source = f"{server}:{temp_path}{filename}" if temp_path else f"{server}:{remote_file}"
            print(f"   [{i}/{len(video_files)}] ⬇️  {filename}...", end='', flush=True)

            result = subprocess.run(['scp', '-q', source, local_file],
                                   capture_output=True, text=True)

            if result.returncode == 0 and os.path.exists(local_file):
                size = os.path.getsize(local_file)
                print(f" ✅ ({size:,} bytes)")
            else:
                print(f" ❌")

        # Cleanup
        if temp_path:
            subprocess.run(['ssh', server, f'rm -rf {temp_path}'], check=False)

        downloaded = list(Path(local_video_dir).glob('*.mp4'))
        print(f"\n   ✅ Downloaded {len(downloaded)} videos")

        return local_video_dir

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return local_video_dir


# ============================================================================
# STEP 2.5: CONVERT VIDEOS TO H264 CODEC
# ============================================================================

def convert_videos_to_h264(video_dir: str, parallel_jobs: int = 8) -> None:
    """Convert all videos in directory from mpeg4 to h264 codec for browser compatibility."""
    video_files = list(Path(video_dir).glob('*.mp4'))

    if not video_files:
        print("\n⚠️  No videos to convert")
        return

    print(f"\n🎬 Converting videos to H.264 codec...")
    print(f"   Directory: {video_dir}")
    print(f"   Videos: {len(video_files)}")
    print(f"   Parallel jobs: {parallel_jobs}")

    # Check if conversion is needed
    sample_file = video_files[0]
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=codec_name', '-of',
         'default=noprint_wrappers=1:nokey=1', str(sample_file)],
        capture_output=True, text=True
    )

    if result.stdout.strip() == 'h264':
        print("   ✅ Videos already in H.264 format")
        return

    print(f"   Converting from {result.stdout.strip()} to h264...")

    # Create temp directory for converted videos
    temp_dir = Path(video_dir).parent / f"{Path(video_dir).name}_h264_temp"
    temp_dir.mkdir(exist_ok=True)

    # Convert videos in parallel
    converted_count = 0
    failed_count = 0

    for i, video_file in enumerate(video_files, 1):
        output_file = temp_dir / video_file.name

        cmd = [
            'ffmpeg', '-i', str(video_file),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-c:a', 'copy',
            str(output_file), '-y', '-loglevel', 'error'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and output_file.exists():
            converted_count += 1
            print(f"   [{i}/{len(video_files)}] ✅ {video_file.name}")
        else:
            failed_count += 1
            print(f"   [{i}/{len(video_files)}] ❌ {video_file.name}")

    if converted_count > 0:
        # Replace original videos with converted ones
        print(f"\n   Replacing original videos...")
        for video_file in video_files:
            video_file.unlink()

        for converted_file in temp_dir.glob('*.mp4'):
            converted_file.rename(Path(video_dir) / converted_file.name)

        temp_dir.rmdir()
        print(f"   ✅ Converted {converted_count}/{len(video_files)} videos to H.264")

        if failed_count > 0:
            print(f"   ⚠️  Failed to convert {failed_count} videos")
    else:
        print(f"\n   ❌ Conversion failed for all videos")
        temp_dir.rmdir()


# ============================================================================
# STEP 3: GENERATE INTERACTIVE HTML REPORT
# ============================================================================

def load_metadata(csv_path: str) -> List[Dict[str, str]]:
    """Load metadata from CSV."""
    with open(csv_path, 'r') as f:
        return list(csv.DictReader(f))


def analyze_distribution(metadata: List[Dict[str, str]]) -> Dict[str, Any]:
    """Analyze parameter distributions."""
    stats = {
        'total_videos': len(metadata),
        'distributions': {},
        'under_represented': [],
        'over_represented': []
    }

    params = ['texture', 'direction', 'speed', 'angle', 'brightness',
              'contrast', 'object_enabled', 'blur_enabled']

    for param in params:
        values = [row.get(param, '') for row in metadata if row.get(param, '')]
        counter = Counter(values)
        total = sum(counter.values())
        avg = total / len(counter) if counter else 0
        threshold = avg * 0.5

        stats['distributions'][param] = {
            'counts': dict(counter.most_common()),
            'unique_values': len(counter),
            'average_per_value': round(avg, 2)
        }

        for value, count in counter.items():
            ratio = count / avg if avg > 0 else 0
            if count < avg - threshold:
                stats['under_represented'].append({
                    'parameter': param, 'value': value, 'count': count,
                    'expected': round(avg, 2), 'ratio': round(ratio, 2)
                })
            elif count > avg + threshold:
                stats['over_represented'].append({
                    'parameter': param, 'value': value, 'count': count,
                    'expected': round(avg, 2), 'ratio': round(ratio, 2)
                })

    return stats


def group_videos(metadata: List[Dict[str, str]]) -> Dict[str, Dict[str, List[Dict]]]:
    """Group videos by parameters."""
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
            speed_range = ('stationary' if speed == 0 else
                          'slow (0-5)' if speed < 5 else
                          'medium (5-10)' if speed < 10 else 'fast (10+)')
            groups['speed_range'][speed_range].append(video)
        except ValueError:
            groups['speed_range']['unknown'].append(video)

        groups['angle'][f"{video.get('angle', 'unknown')}°"].append(video)

        obj_enabled = video.get('object_enabled', 'no')
        obj_type = video.get('object_type', 'none')
        obj_key = obj_type if obj_enabled == 'yes' else 'no_objects'
        groups['objects'][obj_key].append(video)

        blur_enabled = video.get('blur_enabled', 'no')
        blur_type = video.get('blur_type', 'none')
        blur_key = blur_type if blur_enabled == 'yes' else 'no_blur'
        groups['blur'][blur_key].append(video)

    return groups


def generate_pdf_plot(train_values: List[float], test_values: List[float],
                     param_name: str, xlabel: str = None) -> str:
    """
    Generate PDF (Probability Density Function) plot for continuous variables.

    Args:
        train_values: List of numerical values from training dataset
        test_values: List of numerical values from test dataset
        param_name: Name of the parameter being plotted
        xlabel: Optional custom x-axis label

    Returns:
        Base64-encoded PNG image string with data URI prefix
    """
    # Import plotting libraries only when needed
    import numpy as np
    from scipy import stats
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Train KDE
    if len(train_values) > 1:
        train_kde = stats.gaussian_kde(train_values)
        x_min = min(train_values)
        x_max = max(train_values)
        x_train = np.linspace(x_min, x_max, 200)
        ax.plot(x_train, train_kde(x_train), label='Train', color='#667eea', linewidth=2.5, alpha=0.8)
        ax.fill_between(x_train, train_kde(x_train), alpha=0.2, color='#667eea')

    # Test KDE
    if len(test_values) > 1:
        test_kde = stats.gaussian_kde(test_values)
        x_min = min(test_values)
        x_max = max(test_values)
        x_test = np.linspace(x_min, x_max, 200)
        ax.plot(x_test, test_kde(x_test), label='Test', color='#764ba2', linewidth=2.5, alpha=0.8)
        ax.fill_between(x_test, test_kde(x_test), alpha=0.2, color='#764ba2')

    ax.set_xlabel(xlabel or param_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax.set_title(f'{param_name} Distribution Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Convert to base64
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)

    return f'data:image/png;base64,{image_base64}'


def generate_categorical_comparison(train_counter: Counter, test_counter: Counter,
                                    param_name: str) -> str:
    """
    Generate side-by-side bar chart for categorical parameter comparison.

    Args:
        train_counter: Counter object with train dataset value counts
        test_counter: Counter object with test dataset value counts
        param_name: Name of the parameter being compared

    Returns:
        Base64-encoded PNG image string with data URI prefix
    """
    # Import plotting libraries only when needed
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Get all unique categories
    all_categories = sorted(set(list(train_counter.keys()) + list(test_counter.keys())))

    if not all_categories:
        return ""

    # Prepare data
    train_counts = [train_counter.get(cat, 0) for cat in all_categories]
    test_counts = [test_counter.get(cat, 0) for cat in all_categories]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(all_categories))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, train_counts, width, label='Train',
                   color='#667eea', alpha=0.8, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, test_counts, width, label='Test',
                   color='#764ba2', alpha=0.8, edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel(param_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'{param_name} Distribution Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(all_categories, rotation=45, ha='right')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Convert to base64
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)

    return f'data:image/png;base64,{image_base64}'


def generate_html_report(metadata: List[Dict[str, str]], stats: Dict[str, Any],
                        groups: Dict[str, Dict[str, List[Dict]]], output_path: str,
                        video_url: str):
    """Generate beautiful interactive HTML report."""

    metadata_json = json.dumps(metadata, indent=2)
    stats_json = json.dumps(stats, indent=2)
    groups_json = json.dumps({k: {gk: len(gv) for gk, gv in v.items()}
                              for k, v in groups.items()}, indent=2)

    # Sample videos (max 5 per group)
    samples = {}
    for category, category_groups in groups.items():
        samples[category] = {}
        for group_name, group_videos in category_groups.items():
            samples[category][group_name] = group_videos[:5]
    samples_json = json.dumps(samples, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Dataset Review Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .tabs {{
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
            overflow-x: auto;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .tab {{
            padding: 15px 25px;
            cursor: pointer;
            border: none;
            background: transparent;
            font-size: 1em;
            font-weight: 500;
            color: #495057;
            transition: all 0.3s;
            white-space: nowrap;
        }}
        .tab:hover {{ background: rgba(102, 126, 234, 0.1); color: #667eea; }}
        .tab.active {{ background: white; color: #667eea; border-bottom: 3px solid #667eea; }}
        .tab-content {{ padding: 30px; display: none; }}
        .tab-content.active {{ display: block; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{ font-size: 1em; opacity: 0.9; margin-bottom: 10px; }}
        .stat-card .value {{ font-size: 2.5em; font-weight: bold; }}
        .distribution-chart {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .distribution-chart h3 {{ margin-bottom: 15px; color: #667eea; }}
        .bar-item {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }}
        .bar-label {{
            width: 150px;
            font-size: 0.9em;
            color: #495057;
        }}
        .bar {{
            flex: 1;
            height: 25px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 4px;
            position: relative;
            margin-right: 10px;
        }}
        .bar-value {{
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            color: white;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .alert {{
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .alert-warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            color: #856404;
        }}
        .alert-info {{
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            color: #0c5460;
        }}
        .video-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }}
        .video-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        .video-card:hover {{ transform: translateY(-5px); }}
        .video-card video {{
            width: 100%;
            border-radius: 8px;
            margin-bottom: 15px;
            background: #000;
        }}
        .video-card h4 {{
            font-size: 0.9em;
            color: #667eea;
            margin-bottom: 10px;
            word-break: break-all;
        }}
        .video-metadata {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            font-size: 0.85em;
        }}
        .metadata-item {{
            display: flex;
            justify-content: space-between;
            padding: 5px;
            background: white;
            border-radius: 4px;
        }}
        .metadata-label {{ font-weight: 600; color: #495057; }}
        .metadata-value {{ color: #667eea; }}
        .group-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0 15px 0;
            display: flex;
            justify-content: space-between;
        }}
        .search-box {{
            width: 100%;
            padding: 12px;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            font-size: 1em;
            margin-bottom: 20px;
        }}
        .search-box:focus {{ outline: none; border-color: #667eea; }}
        @media (max-width: 768px) {{
            .video-grid {{ grid-template-columns: 1fr; }}
            .stats-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📹 Video Dataset Review Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Total Videos: {len(metadata)}</p>
        </header>

        <div class="tabs">
            <button class="tab active" onclick="switchTab(event, 'overview')">Overview</button>
            <button class="tab" onclick="switchTab(event, 'defaults')">Dataset Defaults</button>
            <button class="tab" onclick="switchTab(event, 'texture')">By Texture</button>
            <button class="tab" onclick="switchTab(event, 'direction')">By Direction</button>
            <button class="tab" onclick="switchTab(event, 'speed')">By Speed</button>
            <button class="tab" onclick="switchTab(event, 'angle')">By Angle</button>
            <button class="tab" onclick="switchTab(event, 'objects')">Objects</button>
            <button class="tab" onclick="switchTab(event, 'blur')">Blur Effects</button>
            <button class="tab" onclick="switchTab(event, 'all')">All Videos</button>
        </div>

        <div id="overview" class="tab-content active">
            <h2>📊 Statistical Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Videos</h3>
                    <div class="value">{stats['total_videos']}</div>
                </div>
                <div class="stat-card">
                    <h3>Unique Textures</h3>
                    <div class="value">{stats['distributions'].get('texture', {}).get('unique_values', 0) if 'texture' in stats['distributions'] else 0}</div>
                </div>
                <div class="stat-card">
                    <h3>Directions</h3>
                    <div class="value">{stats['distributions'].get('direction', {}).get('unique_values', 0) if 'direction' in stats['distributions'] else 0}</div>
                </div>
                <div class="stat-card">
                    <h3>Speed Variations</h3>
                    <div class="value">{stats['distributions'].get('speed', {}).get('unique_values', 0) if 'speed' in stats['distributions'] else 0}</div>
                </div>
            </div>
            <div id="distribution-charts"></div>
            <div id="representation-alerts"></div>
        </div>

        <div id="defaults" class="tab-content">
            <h2>📋 Dataset Defaults & Common Values</h2>
            <p style="margin-bottom: 20px; color: #495057;">Most common value for each parameter.</p>
            <div id="defaults-table"></div>
        </div>

        <div id="texture" class="tab-content"></div>
        <div id="direction" class="tab-content"></div>
        <div id="speed" class="tab-content"></div>
        <div id="angle" class="tab-content"></div>
        <div id="objects" class="tab-content"></div>
        <div id="blur" class="tab-content"></div>

        <div id="all" class="tab-content">
            <h2>All Videos</h2>
            <input type="text" class="search-box" id="searchBox"
                   placeholder="Search by name, texture, direction, speed..."
                   onkeyup="filterVideos()">
            <div id="allVideosGrid" class="video-grid"></div>
        </div>
    </div>

    <script>
        const metadata = {metadata_json};
        const stats = {stats_json};
        const samples = {samples_json};
        const videoBaseUrl = '{video_url}';

        function switchTab(event, tabName) {{
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
            event.currentTarget.classList.add('active');
            if (tabName === 'all' && !document.getElementById('allVideosGrid').innerHTML) {{
                renderAllVideos();
            }}
        }}

        function renderVideoCard(video) {{
            const videoUrl = videoBaseUrl ? `${{videoBaseUrl}}/${{video.video_name}}` : video.video_name;
            return `
                <div class="video-card">
                    <video controls preload="metadata">
                        <source src="${{videoUrl}}" type="video/mp4">
                        Your browser does not support video.
                    </video>
                    <h4>${{video.video_name}}</h4>
                    <div class="video-metadata">
                        <div class="metadata-item">
                            <span class="metadata-label">Index:</span>
                            <span class="metadata-value">${{video.index || 'N/A'}}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Texture:</span>
                            <span class="metadata-value">${{video.texture || 'N/A'}}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Direction:</span>
                            <span class="metadata-value">${{video.direction || 'N/A'}}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Speed:</span>
                            <span class="metadata-value">${{video.speed || 'N/A'}}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Angle:</span>
                            <span class="metadata-value">${{video.angle || 'N/A'}}°</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Brightness:</span>
                            <span class="metadata-value">${{video.brightness || 'N/A'}}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Contrast:</span>
                            <span class="metadata-value">${{video.contrast || 'N/A'}}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Stripe Gray:</span>
                            <span class="metadata-value">${{video.stripe_contrast || 'N/A'}}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Background Gray:</span>
                            <span class="metadata-value">${{video.background_contrast || 'N/A'}}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Object Enabled:</span>
                            <span class="metadata-value">${{video.object_enabled || 'no'}}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Object Type:</span>
                            <span class="metadata-value">${{video.object_type || 'N/A'}}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Num Objects:</span>
                            <span class="metadata-value">${{video.num_objects || 'N/A'}}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Object Size:</span>
                            <span class="metadata-value">${{video.object_size || 'N/A'}}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Object Position:</span>
                            <span class="metadata-value">${{video.object_position || 'N/A'}}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Blur Enabled:</span>
                            <span class="metadata-value">${{video.blur_enabled || 'no'}}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Blur Type:</span>
                            <span class="metadata-value">${{video.blur_type || 'N/A'}}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Blur Intensity:</span>
                            <span class="metadata-value">${{video.blur_intensity || 'N/A'}}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Blur Variation:</span>
                            <span class="metadata-value">${{video.blur_variation || 'N/A'}}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Resolution:</span>
                            <span class="metadata-value">${{video.resolution || 'N/A'}}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Seed:</span>
                            <span class="metadata-value">${{video.seed || 'N/A'}}</span>
                        </div>
                    </div>
                </div>
            `;
        }}

        function renderDistributionCharts() {{
            const container = document.getElementById('distribution-charts');
            for (const [param, data] of Object.entries(stats.distributions)) {{
                const chartDiv = document.createElement('div');
                chartDiv.className = 'distribution-chart';
                const counts = data.counts;
                const maxCount = Math.max(...Object.values(counts));
                let barsHtml = '';
                for (const [value, count] of Object.entries(counts)) {{
                    const width = (count / maxCount) * 100;
                    barsHtml += `
                        <div class="bar-item">
                            <div class="bar-label">${{value || 'empty'}}</div>
                            <div class="bar" style="width: ${{width}}%">
                                <span class="bar-value">${{count}}</span>
                            </div>
                        </div>
                    `;
                }}
                chartDiv.innerHTML = `
                    <h3>${{param.charAt(0).toUpperCase() + param.slice(1)}} Distribution</h3>
                    <div>${{barsHtml}}</div>
                `;
                container.appendChild(chartDiv);
            }}
        }}

        function renderRepresentationAlerts() {{
            const container = document.getElementById('representation-alerts');
            if (stats.under_represented.length > 0) {{
                const div = document.createElement('div');
                div.className = 'alert alert-warning';
                div.innerHTML = `
                    <h3>⚠️ Under-represented Groups</h3>
                    <ul>
                        ${{stats.under_represented.map(item =>
                            `<li><strong>${{item.parameter}}: ${{item.value}}</strong> - ${{item.count}} videos (expected ~${{item.expected}})</li>`
                        ).join('')}}
                    </ul>
                `;
                container.appendChild(div);
            }}
            if (stats.over_represented.length > 0) {{
                const div = document.createElement('div');
                div.className = 'alert alert-info';
                div.innerHTML = `
                    <h3>📈 Over-represented Groups</h3>
                    <ul>
                        ${{stats.over_represented.map(item =>
                            `<li><strong>${{item.parameter}}: ${{item.value}}</strong> - ${{item.count}} videos (expected ~${{item.expected}})</li>`
                        ).join('')}}
                    </ul>
                `;
                container.appendChild(div);
            }}
        }}

        function renderDefaultsTable() {{
            const container = document.getElementById('defaults-table');
            const fields = [
                'index', 'texture', 'direction', 'speed', 'angle', 'brightness', 'contrast',
                'stripe_contrast', 'background_contrast', 'object_enabled', 'object_type',
                'num_objects', 'object_size', 'object_position', 'blur_enabled', 'blur_type',
                'blur_intensity', 'blur_variation', 'resolution', 'seed'
            ];
            const defaults = {{}};
            fields.forEach(field => {{
                const counter = {{}};
                metadata.forEach(video => {{
                    const value = video[field] || 'N/A';
                    counter[value] = (counter[value] || 0) + 1;
                }});
                let maxCount = 0, mostCommon = 'N/A';
                for (const [value, count] of Object.entries(counter)) {{
                    if (count > maxCount) {{
                        maxCount = count;
                        mostCommon = value;
                    }}
                }}
                defaults[field] = {{
                    value: mostCommon,
                    count: maxCount,
                    percentage: ((maxCount / metadata.length) * 100).toFixed(1)
                }};
            }});
            let html = `
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <thead>
                            <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                                <th style="padding: 15px; text-align: left;">Parameter</th>
                                <th style="padding: 15px; text-align: left;">Most Common Value</th>
                                <th style="padding: 15px; text-align: center;">Count</th>
                                <th style="padding: 15px; text-align: center;">Percentage</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
            fields.forEach((field, i) => {{
                const bg = i % 2 === 0 ? '#ffffff' : '#f8f9fa';
                const def = defaults[field];
                html += `
                    <tr style="background: ${{bg}};">
                        <td style="padding: 12px; font-weight: 600; border-bottom: 1px solid #dee2e6;">
                            ${{field.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}}
                        </td>
                        <td style="padding: 12px; color: #667eea; font-family: monospace; border-bottom: 1px solid #dee2e6;">
                            ${{def.value}}
                        </td>
                        <td style="padding: 12px; text-align: center; border-bottom: 1px solid #dee2e6;">
                            ${{def.count}}
                        </td>
                        <td style="padding: 12px; text-align: center; border-bottom: 1px solid #dee2e6;">
                            ${{def.percentage}}%
                        </td>
                    </tr>
                `;
            }});
            html += '</tbody></table></div>';
            container.innerHTML = html;
        }}

        function renderGroupTab(tabId, categoryName) {{
            const container = document.getElementById(tabId);
            const categoryData = samples[categoryName];
            let html = `<h2>${{categoryName.charAt(0).toUpperCase() + categoryName.slice(1)}} Groups</h2>`;
            for (const [groupName, videos] of Object.entries(categoryData)) {{
                html += `
                    <div class="group-header">
                        <h3>${{groupName}}</h3>
                        <span>${{videos.length}} samples</span>
                    </div>
                    <div class="video-grid">
                        ${{videos.map(video => renderVideoCard(video)).join('')}}
                    </div>
                `;
            }}
            container.innerHTML = html;
        }}

        function renderAllVideos() {{
            document.getElementById('allVideosGrid').innerHTML =
                metadata.map(video => renderVideoCard(video)).join('');
        }}

        function filterVideos() {{
            const search = document.getElementById('searchBox').value.toLowerCase();
            document.querySelectorAll('#allVideosGrid .video-card').forEach(card => {{
                card.style.display = card.textContent.toLowerCase().includes(search) ? 'block' : 'none';
            }});
        }}

        window.addEventListener('DOMContentLoaded', () => {{
            renderDistributionCharts();
            renderRepresentationAlerts();
            renderDefaultsTable();
            renderGroupTab('texture', 'texture');
            renderGroupTab('direction', 'direction');
            renderGroupTab('speed', 'speed_range');
            renderGroupTab('angle', 'angle');
            renderGroupTab('objects', 'objects');
            renderGroupTab('blur', 'blur');
        }});
    </script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)


def generate_dual_html_report(train_metadata: List[Dict[str, str]], test_metadata: List[Dict[str, str]],
                              train_stats: Dict[str, Any], test_stats: Dict[str, Any],
                              train_groups: Dict[str, Dict[str, List[Dict]]],
                              test_groups: Dict[str, Dict[str, List[Dict]]],
                              output_path: str, train_video_url: str, test_video_url: str,
                              experiment_name: str):
    """
    Generate dual HTML report with separate tabs for train/test datasets and comparison.

    Args:
        train_metadata: Training dataset metadata
        test_metadata: Test dataset metadata
        train_stats: Training dataset statistics
        test_stats: Test dataset statistics
        train_groups: Training dataset grouped videos
        test_groups: Test dataset grouped videos
        output_path: Path to save HTML file
        train_video_url: URL/path for train videos
        test_video_url: URL/path for test videos
        experiment_name: Base experiment name
    """
    print(f"\n🎨 Generating comparison charts...")

    # Generate PDF plots for continuous variables
    comparison_plots = {}

    # Speed distribution
    try:
        train_speeds = [float(v['speed']) for v in train_metadata if v.get('speed', '').replace('.', '').replace('-', '').isdigit()]
        test_speeds = [float(v['speed']) for v in test_metadata if v.get('speed', '').replace('.', '').replace('-', '').isdigit()]
        if train_speeds and test_speeds:
            comparison_plots['speed'] = generate_pdf_plot(train_speeds, test_speeds, 'Speed', 'Speed (units/sec)')
            print(f"   ✅ Speed distribution plot")
    except Exception as e:
        print(f"   ⚠️  Speed plot failed: {e}")
        comparison_plots['speed'] = ""

    # Angle distribution
    try:
        train_angles = [float(v['angle']) for v in train_metadata if v.get('angle', '').replace('.', '').replace('-', '').isdigit()]
        test_angles = [float(v['angle']) for v in test_metadata if v.get('angle', '').replace('.', '').replace('-', '').isdigit()]
        if train_angles and test_angles:
            comparison_plots['angle'] = generate_pdf_plot(train_angles, test_angles, 'View Angle', 'Angle (degrees)')
            print(f"   ✅ Angle distribution plot")
    except Exception as e:
        print(f"   ⚠️  Angle plot failed: {e}")
        comparison_plots['angle'] = ""

    # Check for subtle gray textures and generate conditional plots
    train_textures = [v.get('texture', '') for v in train_metadata]
    test_textures = [v.get('texture', '') for v in test_metadata]
    has_subtle_gray = any('subtle' in t.lower() and 'gray' in t.lower() for t in train_textures + test_textures)

    if has_subtle_gray:
        # Stripe gray distribution
        try:
            train_stripe = [float(v['stripe_contrast']) for v in train_metadata
                           if v.get('stripe_contrast', 'N/A') not in ['N/A', ''] and v.get('stripe_contrast', '').replace('.', '').isdigit()]
            test_stripe = [float(v['stripe_contrast']) for v in test_metadata
                          if v.get('stripe_contrast', 'N/A') not in ['N/A', ''] and v.get('stripe_contrast', '').replace('.', '').isdigit()]
            if train_stripe and test_stripe:
                comparison_plots['stripe_gray'] = generate_pdf_plot(train_stripe, test_stripe, 'Stripe Gray Values', 'Gray Value')
                print(f"   ✅ Stripe gray distribution plot")
        except Exception as e:
            print(f"   ⚠️  Stripe gray plot failed: {e}")
            comparison_plots['stripe_gray'] = ""

        # Background gray distribution
        try:
            train_bg = [float(v['background_contrast']) for v in train_metadata
                       if v.get('background_contrast', 'N/A') not in ['N/A', ''] and v.get('background_contrast', '').replace('.', '').isdigit()]
            test_bg = [float(v['background_contrast']) for v in test_metadata
                      if v.get('background_contrast', 'N/A') not in ['N/A', ''] and v.get('background_contrast', '').replace('.', '').isdigit()]
            if train_bg and test_bg:
                comparison_plots['background_gray'] = generate_pdf_plot(train_bg, test_bg, 'Background Gray Values', 'Gray Value')
                print(f"   ✅ Background gray distribution plot")
        except Exception as e:
            print(f"   ⚠️  Background gray plot failed: {e}")
            comparison_plots['background_gray'] = ""

    # Generate categorical comparisons
    # Texture comparison
    try:
        train_texture_counter = Counter([v.get('texture', 'unknown') for v in train_metadata])
        test_texture_counter = Counter([v.get('texture', 'unknown') for v in test_metadata])
        comparison_plots['texture'] = generate_categorical_comparison(train_texture_counter, test_texture_counter, 'Texture')
        print(f"   ✅ Texture comparison chart")
    except Exception as e:
        print(f"   ⚠️  Texture comparison failed: {e}")
        comparison_plots['texture'] = ""

    # Direction comparison
    try:
        train_direction_counter = Counter([v.get('direction', 'unknown') for v in train_metadata])
        test_direction_counter = Counter([v.get('direction', 'unknown') for v in test_metadata])
        comparison_plots['direction'] = generate_categorical_comparison(train_direction_counter, test_direction_counter, 'Direction')
        print(f"   ✅ Direction comparison chart")
    except Exception as e:
        print(f"   ⚠️  Direction comparison failed: {e}")
        comparison_plots['direction'] = ""

    # Prepare JSON data for both datasets
    train_metadata_json = json.dumps(train_metadata, indent=2)
    test_metadata_json = json.dumps(test_metadata, indent=2)
    train_stats_json = json.dumps(train_stats, indent=2)
    test_stats_json = json.dumps(test_stats, indent=2)

    # Sample videos (max 5 per group) for both datasets
    train_samples = {}
    for category, category_groups in train_groups.items():
        train_samples[category] = {}
        for group_name, group_videos in category_groups.items():
            train_samples[category][group_name] = group_videos[:5]
    train_samples_json = json.dumps(train_samples, indent=2)

    test_samples = {}
    for category, category_groups in test_groups.items():
        test_samples[category] = {}
        for group_name, group_videos in category_groups.items():
            test_samples[category][group_name] = group_videos[:5]
    test_samples_json = json.dumps(test_samples, indent=2)

    # Calculate train/test ratio
    train_count = len(train_metadata)
    test_count = len(test_metadata)
    ratio = f"{train_count/test_count:.2f}" if test_count > 0 else "N/A"

    # Build comparison tab HTML
    comparison_html = f"""
        <div class="stats-comparison-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px;">
            <div class="stat-card">
                <h3>Train Videos</h3>
                <div class="value">{train_count}</div>
            </div>
            <div class="stat-card">
                <h3>Test Videos</h3>
                <div class="value">{test_count}</div>
            </div>
            <div class="stat-card">
                <h3>Train/Test Ratio</h3>
                <div class="value">{ratio}</div>
            </div>
        </div>

        <div class="pdf-plot" style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="color: #667eea; margin-bottom: 15px;">Speed Distribution (PDF)</h3>
            <img src="{comparison_plots.get('speed', '')}" style="width: 100%; max-width: 900px; display: block; margin: 0 auto;">
        </div>

        <div class="pdf-plot" style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="color: #667eea; margin-bottom: 15px;">Angle Distribution (PDF)</h3>
            <img src="{comparison_plots.get('angle', '')}" style="width: 100%; max-width: 900px; display: block; margin: 0 auto;">
        </div>
    """

    # Add conditional gray value plots
    if has_subtle_gray and comparison_plots.get('stripe_gray'):
        comparison_html += f"""
        <div class="pdf-plot" style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="color: #667eea; margin-bottom: 15px;">Stripe Gray Values Distribution (PDF)</h3>
            <img src="{comparison_plots['stripe_gray']}" style="width: 100%; max-width: 900px; display: block; margin: 0 auto;">
        </div>
        """

    if has_subtle_gray and comparison_plots.get('background_gray'):
        comparison_html += f"""
        <div class="pdf-plot" style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="color: #667eea; margin-bottom: 15px;">Background Gray Values Distribution (PDF)</h3>
            <img src="{comparison_plots['background_gray']}" style="width: 100%; max-width: 900px; display: block; margin: 0 auto;">
        </div>
        """

    comparison_html += f"""
        <div class="categorical-comparison" style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="color: #667eea; margin-bottom: 15px;">Texture Distribution Comparison</h3>
            <img src="{comparison_plots.get('texture', '')}" style="width: 100%; max-width: 900px; display: block; margin: 0 auto;">
        </div>

        <div class="categorical-comparison" style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="color: #667eea; margin-bottom: 15px;">Direction Distribution Comparison</h3>
            <img src="{comparison_plots.get('direction', '')}" style="width: 100%; max-width: 900px; display: block; margin: 0 auto;">
        </div>
    """

    # Generate full HTML (continuing in next edit due to size...)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dual Dataset Review Report - {experiment_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        header .experiment-name {{ font-size: 1.2em; opacity: 0.9; font-family: monospace; }}
        .tabs {{
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
            overflow-x: auto;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .tab {{
            padding: 15px 20px;
            cursor: pointer;
            border: none;
            background: transparent;
            font-size: 0.9em;
            font-weight: 500;
            color: #495057;
            transition: all 0.3s;
            white-space: nowrap;
        }}
        .tab:hover {{ background: rgba(102, 126, 234, 0.1); color: #667eea; }}
        .tab.active {{ background: white; color: #667eea; border-bottom: 3px solid #667eea; }}
        .tab.train-tab {{ border-left: 3px solid #667eea; }}
        .tab.test-tab {{ border-left: 3px solid #764ba2; }}
        .tab.comparison-tab {{ background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); font-weight: 600; }}
        .tab-content {{ padding: 30px; display: none; }}
        .tab-content.active {{ display: block; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{ font-size: 1em; opacity: 0.9; margin-bottom: 10px; }}
        .stat-card .value {{ font-size: 2.5em; font-weight: bold; }}
        .distribution-chart {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .distribution-chart h3 {{ margin-bottom: 15px; color: #667eea; }}
        .bar-item {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }}
        .bar-label {{
            width: 150px;
            font-size: 0.9em;
            color: #495057;
        }}
        .bar {{
            flex: 1;
            height: 25px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 4px;
            position: relative;
            margin-right: 10px;
        }}
        .bar-value {{
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            color: white;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .alert {{
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .alert-warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            color: #856404;
        }}
        .alert-info {{
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            color: #0c5460;
        }}
        .video-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }}
        .video-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        .video-card:hover {{ transform: translateY(-5px); }}
        .video-card video {{
            width: 100%;
            border-radius: 8px;
            margin-bottom: 15px;
            background: #000;
        }}
        .video-card h4 {{
            font-size: 0.9em;
            color: #667eea;
            margin-bottom: 10px;
            word-break: break-all;
        }}
        .video-metadata {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            font-size: 0.85em;
        }}
        .metadata-item {{
            display: flex;
            justify-content: space-between;
            padding: 5px;
            background: white;
            border-radius: 4px;
        }}
        .metadata-label {{ font-weight: 600; color: #495057; }}
        .metadata-value {{ color: #667eea; }}
        .group-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0 15px 0;
            display: flex;
            justify-content: space-between;
        }}
        .search-box {{
            width: 100%;
            padding: 12px;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            font-size: 1em;
            margin-bottom: 20px;
        }}
        .search-box:focus {{ outline: none; border-color: #667eea; }}
        @media (max-width: 768px) {{
            .video-grid {{ grid-template-columns: 1fr; }}
            .stats-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📹 Dual Dataset Review Report</h1>
            <p class="experiment-name">{experiment_name}</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Train: {train_count} videos | Test: {test_count} videos</p>
        </header>

        <div class="tabs">
            <button class="tab train-tab active" onclick="switchTab(event, 'train-overview')">🔵 Train: Overview</button>
            <button class="tab train-tab" onclick="switchTab(event, 'train-defaults')">🔵 Train: Defaults</button>
            <button class="tab train-tab" onclick="switchTab(event, 'train-texture')">🔵 Train: Texture</button>
            <button class="tab train-tab" onclick="switchTab(event, 'train-direction')">🔵 Train: Direction</button>
            <button class="tab train-tab" onclick="switchTab(event, 'train-speed')">🔵 Train: Speed</button>
            <button class="tab train-tab" onclick="switchTab(event, 'train-angle')">🔵 Train: Angle</button>
            <button class="tab train-tab" onclick="switchTab(event, 'train-objects')">🔵 Train: Objects</button>
            <button class="tab train-tab" onclick="switchTab(event, 'train-blur')">🔵 Train: Blur</button>
            <button class="tab train-tab" onclick="switchTab(event, 'train-all')">🔵 Train: All Videos</button>

            <button class="tab test-tab" onclick="switchTab(event, 'test-overview')">🟣 Test: Overview</button>
            <button class="tab test-tab" onclick="switchTab(event, 'test-defaults')">🟣 Test: Defaults</button>
            <button class="tab test-tab" onclick="switchTab(event, 'test-texture')">🟣 Test: Texture</button>
            <button class="tab test-tab" onclick="switchTab(event, 'test-direction')">🟣 Test: Direction</button>
            <button class="tab test-tab" onclick="switchTab(event, 'test-speed')">🟣 Test: Speed</button>
            <button class="tab test-tab" onclick="switchTab(event, 'test-angle')">🟣 Test: Angle</button>
            <button class="tab test-tab" onclick="switchTab(event, 'test-objects')">🟣 Test: Objects</button>
            <button class="tab test-tab" onclick="switchTab(event, 'test-blur')">🟣 Test: Blur</button>
            <button class="tab test-tab" onclick="switchTab(event, 'test-all')">🟣 Test: All Videos</button>

            <button class="tab comparison-tab" onclick="switchTab(event, 'comparison')">📊 Distribution Comparison</button>
        </div>

        <!-- TRAIN TABS -->
        <div id="train-overview" class="tab-content active">
            <h2>📊 Train Dataset - Statistical Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Videos</h3>
                    <div class="value">{train_stats['total_videos']}</div>
                </div>
                <div class="stat-card">
                    <h3>Unique Textures</h3>
                    <div class="value">{train_stats['distributions'].get('texture', {}).get('unique_values', 0) if 'texture' in train_stats['distributions'] else 0}</div>
                </div>
                <div class="stat-card">
                    <h3>Directions</h3>
                    <div class="value">{train_stats['distributions'].get('direction', {}).get('unique_values', 0) if 'direction' in train_stats['distributions'] else 0}</div>
                </div>
                <div class="stat-card">
                    <h3>Speed Variations</h3>
                    <div class="value">{train_stats['distributions'].get('speed', {}).get('unique_values', 0) if 'speed' in train_stats['distributions'] else 0}</div>
                </div>
            </div>
            <div id="train-distribution-charts"></div>
            <div id="train-representation-alerts"></div>
        </div>

        <div id="train-defaults" class="tab-content">
            <h2>📋 Train Dataset - Defaults & Common Values</h2>
            <p style="margin-bottom: 20px; color: #495057;">Most common value for each parameter.</p>
            <div id="train-defaults-table"></div>
        </div>

        <div id="train-texture" class="tab-content"></div>
        <div id="train-direction" class="tab-content"></div>
        <div id="train-speed" class="tab-content"></div>
        <div id="train-angle" class="tab-content"></div>
        <div id="train-objects" class="tab-content"></div>
        <div id="train-blur" class="tab-content"></div>

        <div id="train-all" class="tab-content">
            <h2>Train: All Videos</h2>
            <input type="text" class="search-box" id="trainSearchBox"
                   placeholder="Search by name, texture, direction, speed..."
                   onkeyup="filterTrainVideos()">
            <div id="trainAllVideosGrid" class="video-grid"></div>
        </div>

        <!-- TEST TABS -->
        <div id="test-overview" class="tab-content">
            <h2>📊 Test Dataset - Statistical Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Videos</h3>
                    <div class="value">{test_stats['total_videos']}</div>
                </div>
                <div class="stat-card">
                    <h3>Unique Textures</h3>
                    <div class="value">{test_stats['distributions'].get('texture', {}).get('unique_values', 0) if 'texture' in test_stats['distributions'] else 0}</div>
                </div>
                <div class="stat-card">
                    <h3>Directions</h3>
                    <div class="value">{test_stats['distributions'].get('direction', {}).get('unique_values', 0) if 'direction' in test_stats['distributions'] else 0}</div>
                </div>
                <div class="stat-card">
                    <h3>Speed Variations</h3>
                    <div class="value">{test_stats['distributions'].get('speed', {}).get('unique_values', 0) if 'speed' in test_stats['distributions'] else 0}</div>
                </div>
            </div>
            <div id="test-distribution-charts"></div>
            <div id="test-representation-alerts"></div>
        </div>

        <div id="test-defaults" class="tab-content">
            <h2>📋 Test Dataset - Defaults & Common Values</h2>
            <p style="margin-bottom: 20px; color: #495057;">Most common value for each parameter.</p>
            <div id="test-defaults-table"></div>
        </div>

        <div id="test-texture" class="tab-content"></div>
        <div id="test-direction" class="tab-content"></div>
        <div id="test-speed" class="tab-content"></div>
        <div id="test-angle" class="tab-content"></div>
        <div id="test-objects" class="tab-content"></div>
        <div id="test-blur" class="tab-content"></div>

        <div id="test-all" class="tab-content">
            <h2>Test: All Videos</h2>
            <input type="text" class="search-box" id="testSearchBox"
                   placeholder="Search by name, texture, direction, speed..."
                   onkeyup="filterTestVideos()">
            <div id="testAllVideosGrid" class="video-grid"></div>
        </div>

        <!-- COMPARISON TAB -->
        <div id="comparison" class="tab-content">
            <h2>📊 Train vs Test Distribution Comparison</h2>
            {comparison_html}
        </div>
    </div>

    <script>
        const trainMetadata = {train_metadata_json};
        const testMetadata = {test_metadata_json};
        const trainStats = {train_stats_json};
        const testStats = {test_stats_json};
        const trainSamples = {train_samples_json};
        const testSamples = {test_samples_json};
        const trainVideoBaseUrl = '{train_video_url}';
        const testVideoBaseUrl = '{test_video_url}';

        function switchTab(event, tabName) {{
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
            event.currentTarget.classList.add('active');

            // Lazy load videos
            if (tabName === 'train-all' && !document.getElementById('trainAllVideosGrid').innerHTML) {{
                renderAllVideos('train');
            }} else if (tabName === 'test-all' && !document.getElementById('testAllVideosGrid').innerHTML) {{
                renderAllVideos('test');
            }}
        }}

        function renderVideoCard(video, baseUrl) {{
            const videoUrl = baseUrl ? `${{baseUrl}}/${{video.video_name}}` : video.video_name;
            return `
                <div class="video-card">
                    <video controls preload="metadata">
                        <source src="${{videoUrl}}" type="video/mp4">
                        Your browser does not support video.
                    </video>
                    <h4>${{video.video_name}}</h4>
                    <div class="video-metadata">
                        <div class="metadata-item"><span class="metadata-label">Index:</span><span class="metadata-value">${{video.index || 'N/A'}}</span></div>
                        <div class="metadata-item"><span class="metadata-label">Texture:</span><span class="metadata-value">${{video.texture || 'N/A'}}</span></div>
                        <div class="metadata-item"><span class="metadata-label">Direction:</span><span class="metadata-value">${{video.direction || 'N/A'}}</span></div>
                        <div class="metadata-item"><span class="metadata-label">Speed:</span><span class="metadata-value">${{video.speed || 'N/A'}}</span></div>
                        <div class="metadata-item"><span class="metadata-label">Angle:</span><span class="metadata-value">${{video.angle || 'N/A'}}°</span></div>
                        <div class="metadata-item"><span class="metadata-label">Brightness:</span><span class="metadata-value">${{video.brightness || 'N/A'}}</span></div>
                        <div class="metadata-item"><span class="metadata-label">Contrast:</span><span class="metadata-value">${{video.contrast || 'N/A'}}</span></div>
                        <div class="metadata-item"><span class="metadata-label">Stripe Gray:</span><span class="metadata-value">${{video.stripe_contrast || 'N/A'}}</span></div>
                        <div class="metadata-item"><span class="metadata-label">Background Gray:</span><span class="metadata-value">${{video.background_contrast || 'N/A'}}</span></div>
                        <div class="metadata-item"><span class="metadata-label">Resolution:</span><span class="metadata-value">${{video.resolution || 'N/A'}}</span></div>
                        <div class="metadata-item"><span class="metadata-label">Seed:</span><span class="metadata-value">${{video.seed || 'N/A'}}</span></div>
                    </div>
                </div>
            `;
        }}

        function renderDistributionCharts(stats, containerId) {{
            const container = document.getElementById(containerId);
            for (const [param, data] of Object.entries(stats.distributions)) {{
                const chartDiv = document.createElement('div');
                chartDiv.className = 'distribution-chart';
                const counts = data.counts;
                const maxCount = Math.max(...Object.values(counts));
                let barsHtml = '';
                for (const [value, count] of Object.entries(counts)) {{
                    const width = (count / maxCount) * 100;
                    barsHtml += `
                        <div class="bar-item">
                            <div class="bar-label">${{value || 'empty'}}</div>
                            <div class="bar" style="width: ${{width}}%">
                                <span class="bar-value">${{count}}</span>
                            </div>
                        </div>
                    `;
                }}
                chartDiv.innerHTML = `
                    <h3>${{param.charAt(0).toUpperCase() + param.slice(1)}} Distribution</h3>
                    <div>${{barsHtml}}</div>
                `;
                container.appendChild(chartDiv);
            }}
        }}

        function renderRepresentationAlerts(stats, containerId) {{
            const container = document.getElementById(containerId);
            if (stats.under_represented.length > 0) {{
                const div = document.createElement('div');
                div.className = 'alert alert-warning';
                div.innerHTML = `
                    <h3>⚠️ Under-represented Groups</h3>
                    <ul>
                        ${{stats.under_represented.map(item =>
                            `<li><strong>${{item.parameter}}: ${{item.value}}</strong> - ${{item.count}} videos (expected ~${{item.expected}})</li>`
                        ).join('')}}
                    </ul>
                `;
                container.appendChild(div);
            }}
            if (stats.over_represented.length > 0) {{
                const div = document.createElement('div');
                div.className = 'alert alert-info';
                div.innerHTML = `
                    <h3>📈 Over-represented Groups</h3>
                    <ul>
                        ${{stats.over_represented.map(item =>
                            `<li><strong>${{item.parameter}}: ${{item.value}}</strong> - ${{item.count}} videos (expected ~${{item.expected}})</li>`
                        ).join('')}}
                    </ul>
                `;
                container.appendChild(div);
            }}
        }}

        function renderDefaultsTable(metadata, containerId) {{
            const container = document.getElementById(containerId);
            const fields = ['index', 'texture', 'direction', 'speed', 'angle', 'brightness', 'contrast',
                'stripe_contrast', 'background_contrast', 'object_enabled', 'object_type',
                'num_objects', 'object_size', 'object_position', 'blur_enabled', 'blur_type',
                'blur_intensity', 'blur_variation', 'resolution', 'seed'];
            const defaults = {{}};
            fields.forEach(field => {{
                const counter = {{}};
                metadata.forEach(video => {{
                    const value = video[field] || 'N/A';
                    counter[value] = (counter[value] || 0) + 1;
                }});
                let maxCount = 0, mostCommon = 'N/A';
                for (const [value, count] of Object.entries(counter)) {{
                    if (count > maxCount) {{
                        maxCount = count;
                        mostCommon = value;
                    }}
                }}
                defaults[field] = {{
                    value: mostCommon,
                    count: maxCount,
                    percentage: ((maxCount / metadata.length) * 100).toFixed(1)
                }};
            }});
            let html = `
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <thead>
                            <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                                <th style="padding: 15px; text-align: left;">Parameter</th>
                                <th style="padding: 15px; text-align: left;">Most Common Value</th>
                                <th style="padding: 15px; text-align: center;">Count</th>
                                <th style="padding: 15px; text-align: center;">Percentage</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
            fields.forEach((field, i) => {{
                const bg = i % 2 === 0 ? '#ffffff' : '#f8f9fa';
                const def = defaults[field];
                html += `
                    <tr style="background: ${{bg}};">
                        <td style="padding: 12px; font-weight: 600; border-bottom: 1px solid #dee2e6;">
                            ${{field.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}}
                        </td>
                        <td style="padding: 12px; color: #667eea; font-family: monospace; border-bottom: 1px solid #dee2e6;">
                            ${{def.value}}
                        </td>
                        <td style="padding: 12px; text-align: center; border-bottom: 1px solid #dee2e6;">
                            ${{def.count}}
                        </td>
                        <td style="padding: 12px; text-align: center; border-bottom: 1px solid #dee2e6;">
                            ${{def.percentage}}%
                        </td>
                    </tr>
                `;
            }});
            html += '</tbody></table></div>';
            container.innerHTML = html;
        }}

        function renderGroupTab(tabId, categoryName, samples, videoBaseUrl) {{
            const container = document.getElementById(tabId);
            const categoryData = samples[categoryName];
            let html = `<h2>${{categoryName.charAt(0).toUpperCase() + categoryName.slice(1)}} Groups</h2>`;
            for (const [groupName, videos] of Object.entries(categoryData)) {{
                html += `
                    <div class="group-header">
                        <h3>${{groupName}}</h3>
                        <span>${{videos.length}} samples</span>
                    </div>
                    <div class="video-grid">
                        ${{videos.map(video => renderVideoCard(video, videoBaseUrl)).join('')}}
                    </div>
                `;
            }}
            container.innerHTML = html;
        }}

        function renderAllVideos(dataset) {{
            const metadata = dataset === 'train' ? trainMetadata : testMetadata;
            const baseUrl = dataset === 'train' ? trainVideoBaseUrl : testVideoBaseUrl;
            const gridId = dataset === 'train' ? 'trainAllVideosGrid' : 'testAllVideosGrid';
            document.getElementById(gridId).innerHTML = metadata.map(video => renderVideoCard(video, baseUrl)).join('');
        }}

        function filterTrainVideos() {{
            const search = document.getElementById('trainSearchBox').value.toLowerCase();
            document.querySelectorAll('#trainAllVideosGrid .video-card').forEach(card => {{
                card.style.display = card.textContent.toLowerCase().includes(search) ? 'block' : 'none';
            }});
        }}

        function filterTestVideos() {{
            const search = document.getElementById('testSearchBox').value.toLowerCase();
            document.querySelectorAll('#testAllVideosGrid .video-card').forEach(card => {{
                card.style.display = card.textContent.toLowerCase().includes(search) ? 'block' : 'none';
            }});
        }}

        window.addEventListener('DOMContentLoaded', () => {{
            // Train dataset
            renderDistributionCharts(trainStats, 'train-distribution-charts');
            renderRepresentationAlerts(trainStats, 'train-representation-alerts');
            renderDefaultsTable(trainMetadata, 'train-defaults-table');
            renderGroupTab('train-texture', 'texture', trainSamples, trainVideoBaseUrl);
            renderGroupTab('train-direction', 'direction', trainSamples, trainVideoBaseUrl);
            renderGroupTab('train-speed', 'speed_range', trainSamples, trainVideoBaseUrl);
            renderGroupTab('train-angle', 'angle', trainSamples, trainVideoBaseUrl);
            renderGroupTab('train-objects', 'objects', trainSamples, trainVideoBaseUrl);
            renderGroupTab('train-blur', 'blur', trainSamples, trainVideoBaseUrl);

            // Test dataset
            renderDistributionCharts(testStats, 'test-distribution-charts');
            renderRepresentationAlerts(testStats, 'test-representation-alerts');
            renderDefaultsTable(testMetadata, 'test-defaults-table');
            renderGroupTab('test-texture', 'texture', testSamples, testVideoBaseUrl);
            renderGroupTab('test-direction', 'direction', testSamples, testVideoBaseUrl);
            renderGroupTab('test-speed', 'speed_range', testSamples, testVideoBaseUrl);
            renderGroupTab('test-angle', 'angle', testSamples, testVideoBaseUrl);
            renderGroupTab('test-objects', 'objects', testSamples, testVideoBaseUrl);
            renderGroupTab('test-blur', 'blur', testSamples, testVideoBaseUrl);
        }});
    </script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"   ✅ HTML report saved: {output_path}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Dual Dataset Video Review Tool - Process train and test datasets together',
        epilog='Example: python3 dual_dataset_review.py _exp_20251201_172008'
    )

    parser.add_argument('experiment_name', help='Base experiment name (without _train or _test suffix)')
    parser.add_argument('--server', default='seedoo@hetzner-gpu.tail9e6e7.ts.net', help='SSH server')
    parser.add_argument('--remote-path', default='/app/data', help='Remote dataset base path')
    parser.add_argument('--docker-container', default='llamafactory', help='Docker container name')
    parser.add_argument('--skip-download', action='store_true', help='Skip video download')

    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs('analytics/reports', exist_ok=True)
    os.makedirs('analytics/data', exist_ok=True)

    print("=" * 70)
    print("📹 DUAL DATASET REVIEW - TRAIN & TEST")
    print("=" * 70)
    print(f"Experiment: {args.experiment_name}")
    print(f"Server: {args.server}")
    print("=" * 70)
    print()

    # Construct dataset names
    train_dataset = f"{args.experiment_name}_train"
    test_dataset = f"{args.experiment_name}_test"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"📋 Train Dataset: {train_dataset}")
    print(f"📋 Test Dataset: {test_dataset}")
    print()

    # ========================================================================
    # TRAIN DATASET PROCESSING
    # ========================================================================
    print("=" * 70)
    print("🔵 PROCESSING TRAIN DATASET")
    print("=" * 70)

    # Step 1: Extract train metadata
    train_csv_path = extract_metadata(
        train_dataset, args.server, args.remote_path,
        'analytics/data', args.docker_container
    )
    if not train_csv_path:
        print("\n❌ Failed to extract train metadata")
        return

    # Step 2: Download train videos
    if not args.skip_download:
        train_video_path = download_videos(
            train_dataset, args.server, args.remote_path,
            './data/videos', args.docker_container
        )
    else:
        print("\n⏭️  Skipping train download")
        train_video_path = os.path.join('./data/videos', train_dataset)

    # Step 3: Convert train videos to H.264
    convert_videos_to_h264(train_video_path)

    # ========================================================================
    # TEST DATASET PROCESSING
    # ========================================================================
    print("\n" + "=" * 70)
    print("🟣 PROCESSING TEST DATASET")
    print("=" * 70)

    # Step 1: Extract test metadata
    test_csv_path = extract_metadata(
        test_dataset, args.server, args.remote_path,
        'analytics/data', args.docker_container
    )
    if not test_csv_path:
        print("\n❌ Failed to extract test metadata")
        return

    # Step 2: Download test videos
    if not args.skip_download:
        test_video_path = download_videos(
            test_dataset, args.server, args.remote_path,
            './data/videos', args.docker_container
        )
    else:
        print("\n⏭️  Skipping test download")
        test_video_path = os.path.join('./data/videos', test_dataset)

    # Step 3: Convert test videos to H.264
    convert_videos_to_h264(test_video_path)

    # ========================================================================
    # ANALYSIS AND REPORT GENERATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 ANALYZING BOTH DATASETS")
    print("=" * 70)

    # Load and analyze train dataset
    print("\n🔵 Analyzing train dataset...")
    train_metadata = load_metadata(train_csv_path)
    train_stats = analyze_distribution(train_metadata)
    train_groups = group_videos(train_metadata)

    # Load and analyze test dataset
    print("🟣 Analyzing test dataset...")
    test_metadata = load_metadata(test_csv_path)
    test_stats = analyze_distribution(test_metadata)
    test_groups = group_videos(test_metadata)

    # Generate output paths
    html_output = f"analytics/reports/{args.experiment_name}_dual_report_{timestamp}.html"

    # Calculate video URLs relative to HTML output
    output_dir = os.path.dirname(os.path.abspath(html_output))
    train_video_abs = os.path.abspath(train_video_path)
    test_video_abs = os.path.abspath(test_video_path)
    train_video_url = os.path.relpath(train_video_abs, output_dir)
    test_video_url = os.path.relpath(test_video_abs, output_dir)

    # Generate dual HTML report
    print(f"\n📊 Generating dual HTML report...")
    generate_dual_html_report(
        train_metadata, test_metadata,
        train_stats, test_stats,
        train_groups, test_groups,
        html_output,
        train_video_url, test_video_url,
        args.experiment_name
    )

    # Count downloaded videos
    train_videos = list(Path(train_video_path).glob('*.mp4'))
    test_videos = list(Path(test_video_path).glob('*.mp4'))

    # Final summary
    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)
    print(f"📄 Train Metadata: {train_csv_path}")
    print(f"📄 Test Metadata: {test_csv_path}")
    print(f"📂 Train Videos: {train_video_path} ({len(train_videos)} files)")
    print(f"📂 Test Videos: {test_video_path} ({len(test_videos)} files)")
    print(f"📊 HTML Report: {html_output}")
    print("=" * 70)
    print()
    print("🎬 TO VIEW THE REPORT:")
    print()

    # Convert to Windows path if on WSL
    html_abs_path = os.path.abspath(html_output)
    if html_abs_path.startswith('/mnt/'):
        # Convert WSL path to Windows path
        windows_path = html_abs_path.replace('/mnt/c/', 'C:\\').replace('/mnt/d/', 'D:\\').replace('/', '\\')
        print(f"   📂 Double-click to open: {windows_path}")
        print()
        print("   Or from WSL/Linux:")

    print(f"   📂 file://{html_abs_path}")
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
