#!/usr/bin/env python3
"""
Download Dataset Videos and Generate Local Review Report

Complete one-command solution:
1. Downloads videos from server Docker container
2. Re-encodes videos to browser-compatible H.264 format
3. Generates HTML report with absolute file:// paths for Windows

Usage:
    python3 download_and_review_dataset.py DATASET_NAME

Example:
    python3 download_and_review_dataset.py _exp_20251201_172008_train
"""

import subprocess
import csv
import json
import os
import re
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter


################################################################################
# CONFIGURATION
################################################################################

SERVER_USER = "seedoo"
SERVER_HOST = "hetzner-gpu.tail9e6e7.ts.net"
SERVER_SSH = f"{SERVER_USER}@{SERVER_HOST}"
DOCKER_CONTAINER = "llamafactory"
REMOTE_DATA_PATH = "/app/data"

LOCAL_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATA_DIR = os.path.join(LOCAL_BASE_DIR, "data")
LOCAL_VIDEO_DIR = os.path.join(LOCAL_DATA_DIR, "videos")


################################################################################
# VIDEO FILENAME PARSER
################################################################################

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
        # Extract index
        if parts[0] == 'treadmill':
            metadata['index'] = parts[1]
            remaining_idx = 2
        else:
            remaining_idx = 0

        # Extract texture
        texture_parts = []
        while remaining_idx < len(parts):
            if parts[remaining_idx] in ['up', 'down', 'left', 'right', 'stationary']:
                break
            if re.match(r'^stripe\d+$', parts[remaining_idx]) or re.match(r'^bg\d+$', parts[remaining_idx]):
                break
            texture_parts.append(parts[remaining_idx])
            remaining_idx += 1
        metadata['texture'] = '_'.join(texture_parts)

        # Extract stripe_contrast and background_contrast
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
        print(f"⚠️  Warning: Error parsing '{filename}': {e}")

    return metadata


################################################################################
# STEP 1: LIST VIDEOS ON SERVER
################################################################################

def list_videos_on_server(dataset_name: str) -> List[str]:
    """List all video files in the dataset on the server Docker container."""
    print(f"\n📋 Listing videos on server...")
    print(f"   Dataset: {dataset_name}")
    print(f"   Server: {SERVER_SSH}")
    print(f"   Container: {DOCKER_CONTAINER}")

    try:
        # Use bash -c to properly expand wildcards in docker exec
        cmd = [
            'ssh', SERVER_SSH,
            f'docker exec {DOCKER_CONTAINER} bash -c "ls -1 {REMOTE_DATA_PATH}/{dataset_name}/*.mp4"'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        videos = []
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if line and line.endswith('.mp4'):
                videos.append(os.path.basename(line))

        print(f"   ✅ Found {len(videos)} videos")
        return videos

    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        print(f"   Stderr: {e.stderr}")
        return []


################################################################################
# STEP 2: DOWNLOAD VIDEOS FROM SERVER
################################################################################

def download_videos(dataset_name: str) -> str:
    """Download videos from server Docker container to local PC."""
    local_dataset_dir = os.path.join(LOCAL_VIDEO_DIR, dataset_name)
    os.makedirs(local_dataset_dir, exist_ok=True)

    print(f"\n📥 Downloading videos...")
    print(f"   Remote: {SERVER_SSH}:{DOCKER_CONTAINER}:{REMOTE_DATA_PATH}/{dataset_name}/")
    print(f"   Local: {local_dataset_dir}/")

    try:
        # Step 1: Copy from Docker container to server temp directory
        temp_path = f"/tmp/video_download_{dataset_name}"
        print(f"   Step 1: Copying from container to server temp...")

        copy_cmd = f"rm -rf {temp_path} && mkdir -p {temp_path} && docker cp {DOCKER_CONTAINER}:{REMOTE_DATA_PATH}/{dataset_name}/. {temp_path}/"
        subprocess.run(['ssh', SERVER_SSH, copy_cmd], check=True)

        # Step 2: Rsync from server temp to local PC
        print(f"   Step 2: Downloading from server to local PC...")

        rsync_cmd = [
            'rsync', '-avz', '--progress',
            f'{SERVER_SSH}:{temp_path}/',
            f'{local_dataset_dir}/'
        ]
        subprocess.run(rsync_cmd, check=True)

        # Step 3: Cleanup server temp
        print(f"   Step 3: Cleaning up server temp...")
        subprocess.run(['ssh', SERVER_SSH, f'rm -rf {temp_path}'], check=False)

        # Count downloaded files
        downloaded = list(Path(local_dataset_dir).glob('*.mp4'))
        print(f"   ✅ Downloaded {len(downloaded)} videos")

        return local_dataset_dir

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return local_dataset_dir


################################################################################
# STEP 3: RE-ENCODE VIDEOS TO WEB-COMPATIBLE FORMAT
################################################################################

def reencode_video_for_web(input_path: str, output_path: str) -> bool:
    """
    Re-encode video to H.264 with browser-compatible settings.

    Uses ffmpeg to convert to H.264 baseline profile with YUV420p pixel format,
    which is universally supported by all modern browsers.
    """
    try:
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-i', input_path,
            '-c:v', 'libx264',  # H.264 codec
            '-profile:v', 'baseline',  # Baseline profile for maximum compatibility
            '-level', '3.0',
            '-pix_fmt', 'yuv420p',  # Standard pixel format
            '-movflags', '+faststart',  # Enable streaming/fast start
            '-loglevel', 'error'  # Only show errors
        ]

        # Add audio codec if audio exists, otherwise remove audio
        cmd.extend(['-an'])  # Remove audio (treadmill videos don't have audio)
        cmd.append(output_path)

        subprocess.run(cmd, check=True, capture_output=True)
        return True

    except subprocess.CalledProcessError as e:
        print(f"      ❌ FFmpeg error: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print(f"      ❌ FFmpeg not found. Please install ffmpeg.")
        return False


def reencode_videos_for_browser(dataset_dir: str) -> int:
    """Re-encode all videos in dataset directory to web-compatible format."""
    print(f"\n🎬 Re-encoding videos for browser compatibility...")
    print(f"   Converting to H.264 baseline profile (yuv420p)")

    videos = list(Path(dataset_dir).glob('*.mp4'))

    if not videos:
        print(f"   ⚠️  No videos found to re-encode")
        return 0

    # Create a backup directory
    backup_dir = os.path.join(dataset_dir, '_original_videos_backup')
    os.makedirs(backup_dir, exist_ok=True)

    success_count = 0

    for i, video_path in enumerate(videos, 1):
        video_name = video_path.name
        print(f"   [{i}/{len(videos)}] {video_name}...", end='', flush=True)

        # Backup original
        backup_path = os.path.join(backup_dir, video_name)
        shutil.copy2(str(video_path), backup_path)

        # Re-encode to temp file
        temp_output = str(video_path) + '.tmp.mp4'

        if reencode_video_for_web(str(video_path), temp_output):
            # Replace original with re-encoded version
            os.replace(temp_output, str(video_path))
            print(f" ✅")
            success_count += 1
        else:
            print(f" ❌ (keeping original)")
            if os.path.exists(temp_output):
                os.remove(temp_output)

    print(f"   ✅ Successfully re-encoded {success_count}/{len(videos)} videos")
    print(f"   📁 Original videos backed up to: {backup_dir}")

    return success_count


################################################################################
# STEP 4: GENERATE METADATA CSV
################################################################################

def generate_metadata_csv(dataset_name: str, dataset_dir: str) -> str:
    """Generate metadata CSV from video filenames."""
    print(f"\n📊 Generating metadata...")

    videos = sorted([v.name for v in Path(dataset_dir).glob('*.mp4')])

    if not videos:
        print(f"   ❌ No videos found")
        return None

    metadata_list = [parse_video_filename(video) for video in videos]

    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(LOCAL_DATA_DIR, f"{dataset_name}_metadata_{timestamp}.csv")

    fieldnames = [
        'video_name', 'index', 'texture', 'direction', 'speed', 'angle',
        'brightness', 'contrast', 'stripe_contrast', 'background_contrast',
        'object_enabled', 'object_type', 'num_objects', 'object_size', 'object_position',
        'blur_enabled', 'blur_type', 'blur_intensity', 'blur_variation',
        'resolution', 'seed'
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_list)

    print(f"   ✅ Saved: {csv_path}")
    return csv_path


################################################################################
# STEP 5: ANALYZE AND GENERATE HTML REPORT
################################################################################

def convert_wsl_to_windows_path(wsl_path: str) -> str:
    """Convert WSL path to Windows path."""
    if wsl_path[1:3] == ':/':
        return wsl_path.replace('\\', '/')

    if wsl_path.startswith('/mnt/'):
        parts = wsl_path.split('/')
        if len(parts) >= 3:
            drive_letter = parts[2].upper()
            rest_of_path = '/'.join(parts[3:])
            return f"{drive_letter}:/{rest_of_path}"

    return wsl_path


def load_metadata_with_paths(csv_path: str, dataset_dir: str) -> List[Dict[str, str]]:
    """Load metadata from CSV and add absolute video paths."""
    with open(csv_path, 'r') as f:
        metadata = list(csv.DictReader(f))

    for video in metadata:
        video_path = os.path.join(dataset_dir, video['video_name'])
        abs_path = os.path.abspath(video_path)
        windows_path = convert_wsl_to_windows_path(abs_path)

        video['absolute_path'] = windows_path
        video['file_url'] = f"file:///{windows_path}"
        video['file_exists'] = os.path.exists(video_path)

    return metadata


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
                        groups: Dict[str, Dict[str, List[Dict]]], output_path: str):
    """Generate HTML report with absolute file:// paths."""

    missing_videos = [v['video_name'] for v in metadata if not v.get('file_exists', False)]

    metadata_json = json.dumps(metadata, indent=2)
    stats_json = json.dumps(stats, indent=2)

    samples = {}
    for category, category_groups in groups.items():
        samples[category] = {}
        for group_name, group_videos in category_groups.items():
            samples[category][group_name] = group_videos[:5]
    samples_json = json.dumps(samples, indent=2)

    missing_warning = ""
    if missing_videos:
        missing_warning = f"""
        <div class="alert alert-warning" style="margin: 20px 30px;">
            <h3>⚠️ Missing Videos</h3>
            <p>{len(missing_videos)} video(s) not found on disk</p>
        </div>
        """

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
        header p {{ font-size: 1em; opacity: 0.95; margin: 5px 0; }}
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
        .bar-item {{ display: flex; align-items: center; margin-bottom: 8px; }}
        .bar-label {{ width: 150px; font-size: 0.9em; color: #495057; }}
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
        .alert {{ padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
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
            <p style="font-size: 0.9em; margin-top: 10px;">
                ✅ Videos re-encoded to H.264 for browser compatibility
            </p>
        </header>

        {missing_warning}

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
                    <div class="value">{stats['distributions'].get('texture', {}).get('unique_values', 0)}</div>
                </div>
                <div class="stat-card">
                    <h3>Directions</h3>
                    <div class="value">{stats['distributions'].get('direction', {}).get('unique_values', 0)}</div>
                </div>
                <div class="stat-card">
                    <h3>Speed Variations</h3>
                    <div class="value">{stats['distributions'].get('speed', {}).get('unique_values', 0)}</div>
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
            return `
                <div class="video-card">
                    <video controls preload="metadata">
                        <source src="${{video.file_url}}" type="video/mp4">
                        Your browser does not support video playback.
                    </video>
                    <h4>${{video.video_name}}</h4>
                    <div class="video-metadata">
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
                            `<li><strong>${{item.parameter}}: ${{item.value}}</strong> - ${{item.count}} videos</li>`
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
                            `<li><strong>${{item.parameter}}: ${{item.value}}</strong> - ${{item.count}} videos</li>`
                        ).join('')}}
                    </ul>
                `;
                container.appendChild(div);
            }}
        }}

        function renderDefaultsTable() {{
            const container = document.getElementById('defaults-table');
            const fields = ['texture', 'direction', 'speed', 'angle', 'brightness', 'contrast'];
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
                defaults[field] = {{ value: mostCommon, count: maxCount }};
            }});
            let html = '<div style="background: #f8f9fa; padding: 20px; border-radius: 8px;"><table style="width: 100%; border-collapse: collapse;"><thead><tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;"><th style="padding: 15px; text-align: left;">Parameter</th><th style="padding: 15px; text-align: left;">Most Common Value</th><th style="padding: 15px; text-align: center;">Count</th></tr></thead><tbody>';
            fields.forEach((field, i) => {{
                const bg = i % 2 === 0 ? '#ffffff' : '#f8f9fa';
                const def = defaults[field];
                html += `<tr style="background: ${{bg}};"><td style="padding: 12px; font-weight: 600;">${{field}}</td><td style="padding: 12px; color: #667eea;">${{def.value}}</td><td style="padding: 12px; text-align: center;">${{def.count}}</td></tr>`;
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

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"   ✅ Generated: {output_path}")


################################################################################
# MAIN
################################################################################

def main():
    parser = argparse.ArgumentParser(
        description='Download Dataset Videos and Generate Review Report',
        epilog='Example: python3 download_and_review_dataset.py _exp_20251201_172008_train'
    )

    parser.add_argument('dataset_name', help='Dataset name on server (e.g., _exp_20251201_172008_train)')
    parser.add_argument('--skip-download', action='store_true', help='Skip downloading (use existing local videos)')
    parser.add_argument('--skip-reencode', action='store_true', help='Skip re-encoding videos')
    parser.add_argument('--output', default=None, help='Output HTML filename')

    args = parser.parse_args()

    print("=" * 70)
    print("📹 DOWNLOAD DATASET AND GENERATE REVIEW REPORT")
    print("=" * 70)
    print(f"Dataset: {args.dataset_name}")
    print(f"Server: {SERVER_SSH}")
    print(f"Container: {DOCKER_CONTAINER}")
    print("=" * 70)

    # Step 1: List videos on server
    if not args.skip_download:
        videos = list_videos_on_server(args.dataset_name)
        if not videos:
            print("\n❌ No videos found on server. Exiting.")
            return

        # Step 2: Download videos
        dataset_dir = download_videos(args.dataset_name)
    else:
        print("\n⏭️  Skipping download (using existing local videos)")
        dataset_dir = os.path.join(LOCAL_VIDEO_DIR, args.dataset_name)

    if not os.path.exists(dataset_dir):
        print(f"\n❌ Dataset directory not found: {dataset_dir}")
        return

    # Step 3: Re-encode videos for browser compatibility
    if not args.skip_reencode:
        success_count = reencode_videos_for_browser(dataset_dir)
        if success_count == 0:
            print("\n⚠️  Warning: No videos were re-encoded. Videos may not play in browser.")
    else:
        print("\n⏭️  Skipping video re-encoding")

    # Step 4: Generate metadata CSV
    csv_path = generate_metadata_csv(args.dataset_name, dataset_dir)
    if not csv_path:
        print("\n❌ Failed to generate metadata")
        return

    # Step 5: Generate HTML report
    print(f"\n📊 Generating HTML report...")
    metadata = load_metadata_with_paths(csv_path, dataset_dir)
    stats = analyze_distribution(metadata)
    groups = group_videos(metadata)

    if args.output:
        output_path = args.output
    else:
        output_path = f"{args.dataset_name}_report.html"

    generate_html_report(metadata, stats, groups, output_path)

    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)
    print(f"📄 Metadata CSV: {csv_path}")
    print(f"📁 Videos: {dataset_dir}")
    print(f"📊 HTML Report: {output_path}")
    print("=" * 70)
    print()
    print("🎬 TO VIEW THE REPORT:")
    print(f"   Double-click: {output_path}")
    print()
    print("   Videos have been re-encoded to H.264 for browser compatibility!")
    print("=" * 70)


if __name__ == '__main__':
    main()
