#!/usr/bin/env python3
"""
Complete Dataset Video Review Tool

One-command solution to:
1. Extract metadata from server
2. Download videos to local PC
3. Generate interactive HTML report with video playback

Just provide the dataset name and everything happens automatically!

Usage:
    python3 complete_dataset_review.py YOUR_DATASET_NAME

Example:
    python3 complete_dataset_review.py _exp_20251201_172008_train
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
    """Extract metadata from server and save to CSV."""
    dataset_path = f"{remote_base}/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"{dataset_name}_metadata_{timestamp}.csv")

    print(f"📋 Extracting metadata from server...")
    print(f"   Dataset: {dataset_name}")
    print(f"   Server: {server}")
    if docker_container:
        print(f"   Docker: {docker_container}")
    print(f"   Path: {dataset_path}")

    # Get list of videos
    videos = list_videos_on_server(server, dataset_path, docker_container)

    if not videos:
        print("❌ No videos found")
        return None

    print(f"   Found {len(videos)} videos")

    # Parse metadata
    metadata_list = [parse_video_filename(video) for video in videos]

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

    print(f"   ✅ Saved: {output_file}")
    return output_file


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


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Complete Video Dataset Review Tool',
        epilog='Example: python3 complete_dataset_review.py _exp_20251201_172008_train'
    )

    parser.add_argument('dataset_name', help='Dataset name on server')
    parser.add_argument('--output', default='dataset_report.html', help='Output HTML file')
    parser.add_argument('--server', default='seedoo@hetzner-gpu.tail9e6e7.ts.net', help='SSH server')
    parser.add_argument('--remote-path', default='/app/data', help='Remote dataset base path')
    parser.add_argument('--docker-container', default='llamafactory', help='Docker container name')
    parser.add_argument('--metadata-dir', default='./data', help='Local metadata directory')
    parser.add_argument('--video-dir', default='./data/videos', help='Local video directory')
    parser.add_argument('--skip-download', action='store_true', help='Skip video download')

    args = parser.parse_args()

    print("=" * 70)
    print("📹 COMPLETE VIDEO DATASET REVIEW")
    print("=" * 70)
    print(f"Dataset: {args.dataset_name}")
    print(f"Server: {args.server}")
    print(f"Output: {args.output}")
    print("=" * 70)
    print()

    # Step 1: Extract metadata
    csv_path = extract_metadata(
        args.dataset_name, args.server, args.remote_path,
        args.metadata_dir, args.docker_container
    )
    if not csv_path:
        print("\n❌ Failed to extract metadata")
        return

    # Step 2: Download videos
    if not args.skip_download:
        local_video_path = download_videos(
            args.dataset_name, args.server, args.remote_path,
            args.video_dir, args.docker_container
        )
    else:
        print("\n⏭️  Skipping download")
        local_video_path = os.path.join(args.video_dir, args.dataset_name)

    # Step 2.5: Convert videos to H.264 for browser compatibility
    convert_videos_to_h264(local_video_path)

    # Step 3: Call enhanced report generation
    print(f"\n📊 Generating enhanced video report...")
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from generate_enhanced_video_report import main as generate_enhanced_report

        # Find the metadata CSV
        csv_files = list(Path(args.metadata_dir).glob(f"{args.dataset_name}_metadata_*.csv"))
        if csv_files:
            # Sort by modification time and get the most recent
            csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            metadata_csv = str(csv_files[0])

            # Generate enhanced report
            sys.argv = ['generate_enhanced_video_report.py', metadata_csv]
            generate_enhanced_report()
            print(f"   ✅ Enhanced report generated")
        else:
            print(f"   ⚠️  No metadata CSV found, skipping enhanced report")
    except Exception as e:
        print(f"   ⚠️  Could not generate enhanced report: {e}")
        print(f"   Falling back to basic report")

    # Step 4: Generate basic HTML report (fallback)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    local_video_abs = os.path.abspath(local_video_path)
    video_url = os.path.relpath(local_video_abs, output_dir)

    print(f"\n📊 Analyzing dataset...")
    metadata = load_metadata(csv_path)
    stats = analyze_distribution(metadata)
    groups = group_videos(metadata)

    print(f"🎨 Generating HTML report...")
    generate_html_report(metadata, stats, groups, args.output, video_url)

    downloaded = list(Path(local_video_path).glob('*.mp4'))

    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)
    print(f"📄 Metadata: {csv_path}")
    print(f"📂 Videos: {local_video_path} ({len(downloaded)} files)")
    print(f"📊 Report: {args.output}")
    print("=" * 70)
    print()
    print("🎬 TO VIEW THE REPORT:")
    print()
    print("   python3 -m http.server 8000")
    print(f"   Then open: http://localhost:8000/{args.output}")
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
