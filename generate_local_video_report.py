#!/usr/bin/env python3
"""
Generate Local Video Report with Absolute File Paths

This script generates an HTML report with absolute file:// paths that work
when opening the HTML directly in Windows browsers.

Usage:
    python3 generate_local_video_report.py METADATA_CSV_FILE

Example:
    python3 generate_local_video_report.py data/_exp_20251201_172008_train_metadata_20251202_130454.csv
"""

import csv
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Any


def convert_wsl_to_windows_path(wsl_path: str) -> str:
    """
    Convert WSL path to Windows path.

    Examples:
        /mnt/c/Users/shaha/Desktop/file.mp4 -> C:/Users/shaha/Desktop/file.mp4
        C:/Users/shaha/Desktop/file.mp4 -> C:/Users/shaha/Desktop/file.mp4 (already Windows)
    """
    # If already a Windows path, return as-is
    if wsl_path[1:3] == ':/':
        return wsl_path.replace('\\', '/')

    # Convert WSL /mnt/c/ to C:/
    if wsl_path.startswith('/mnt/'):
        parts = wsl_path.split('/')
        if len(parts) >= 3:
            drive_letter = parts[2].upper()
            rest_of_path = '/'.join(parts[3:])
            return f"{drive_letter}:/{rest_of_path}"

    return wsl_path


def find_video_file(metadata_csv_path: str, video_name: str) -> str:
    """
    Find the actual video file on disk and return absolute Windows path.

    Searches in common locations relative to the metadata CSV file.
    """
    csv_dir = os.path.dirname(os.path.abspath(metadata_csv_path))

    # Extract dataset name from CSV filename
    # Example: _exp_20251201_172008_train_metadata_20251202_130454.csv
    csv_filename = os.path.basename(metadata_csv_path)
    parts = csv_filename.split('_metadata_')
    if len(parts) >= 1:
        dataset_name = parts[0]
    else:
        dataset_name = None

    # Search in multiple possible locations
    search_paths = []

    # Location 1: data/videos/{dataset_name}/
    if dataset_name:
        search_paths.append(os.path.join(csv_dir, 'videos', dataset_name, video_name))

    # Location 2: data/{dataset_name}/
    if dataset_name:
        search_paths.append(os.path.join(csv_dir, dataset_name, video_name))

    # Location 3: Root directory (same as CSV)
    search_paths.append(os.path.join(csv_dir, video_name))

    # Location 4: videos/ subdirectory
    search_paths.append(os.path.join(csv_dir, 'videos', video_name))

    # Check each path
    for path in search_paths:
        if os.path.exists(path):
            abs_path = os.path.abspath(path)
            windows_path = convert_wsl_to_windows_path(abs_path)
            return windows_path

    # Not found - return expected path anyway
    if dataset_name:
        expected = os.path.join(csv_dir, 'videos', dataset_name, video_name)
    else:
        expected = os.path.join(csv_dir, video_name)

    abs_path = os.path.abspath(expected)
    return convert_wsl_to_windows_path(abs_path)


def load_metadata(csv_path: str) -> List[Dict[str, str]]:
    """Load metadata from CSV and add absolute video paths."""
    with open(csv_path, 'r') as f:
        metadata = list(csv.DictReader(f))

    # Add absolute file paths
    for video in metadata:
        video_path = find_video_file(csv_path, video['video_name'])
        video['absolute_path'] = video_path
        video['file_exists'] = os.path.exists(video_path.replace('C:/', '/mnt/c/').replace('D:/', '/mnt/d/'))

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

    # Create metadata with file URLs
    metadata_with_urls = []
    missing_videos = []
    for video in metadata:
        video_copy = video.copy()
        # Convert to file:// URL for the browser
        windows_path = video['absolute_path']
        file_url = f"file:///{windows_path}"
        video_copy['file_url'] = file_url
        metadata_with_urls.append(video_copy)

        if not video.get('file_exists', False):
            missing_videos.append(video['video_name'])

    metadata_json = json.dumps(metadata_with_urls, indent=2)
    stats_json = json.dumps(stats, indent=2)

    # Sample videos (max 5 per group)
    samples = {}
    for category, category_groups in groups.items():
        samples[category] = {}
        for group_name, group_videos in category_groups.items():
            samples[category][group_name] = [v for v in metadata_with_urls if v['video_name'] in [gv['video_name'] for gv in group_videos[:5]]]
    samples_json = json.dumps(samples, indent=2)

    missing_warning = ""
    if missing_videos:
        missing_warning = f"""
        <div class="alert alert-warning" style="margin: 20px 30px;">
            <h3>⚠️ Missing Videos</h3>
            <p>The following {len(missing_videos)} video(s) were not found on disk:</p>
            <ul style="max-height: 200px; overflow-y: auto;">
                {''.join(f'<li style="font-family: monospace; font-size: 0.85em;">{v}</li>' for v in missing_videos[:20])}
                {f'<li><em>...and {len(missing_videos) - 20} more</em></li>' if len(missing_videos) > 20 else ''}
            </ul>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local Video Dataset Review Report</title>
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
        .video-card.missing {{
            opacity: 0.5;
            border: 2px dashed #dc3545;
        }}
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
        .missing-badge {{
            background: #dc3545;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: bold;
            margin-left: 8px;
        }}
        @media (max-width: 768px) {{
            .video-grid {{ grid-template-columns: 1fr; }}
            .stats-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📹 Local Video Dataset Review Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Total Videos: {len(metadata)}</p>
            <p style="font-size: 0.9em; margin-top: 10px;">
                ✅ Using absolute file:// paths for Windows compatibility
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
            const isMissing = !video.file_exists;
            const missingClass = isMissing ? 'missing' : '';
            const missingBadge = isMissing ? '<span class="missing-badge">FILE NOT FOUND</span>' : '';

            return `
                <div class="video-card ${{missingClass}}">
                    <video controls preload="metadata">
                        <source src="${{video.file_url}}" type="video/mp4">
                        Your browser does not support video playback.
                    </video>
                    <h4>${{video.video_name}}${{missingBadge}}</h4>
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
                            <span class="metadata-label">Stripe Contrast:</span>
                            <span class="metadata-value">${{video.stripe_contrast || 'N/A'}}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Background Contrast:</span>
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
                    <div style="margin-top: 10px; padding: 8px; background: white; border-radius: 4px; font-size: 0.75em; font-family: monospace; word-break: break-all; color: #6c757d;">
                        ${{video.absolute_path}}
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

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✅ Generated: {output_path}")
    print(f"\n📹 Video Statistics:")
    print(f"   Total: {len(metadata)}")
    print(f"   Found: {len([v for v in metadata if v.get('file_exists', False)])}")
    print(f"   Missing: {len(missing_videos)}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Local Video Report with Absolute File Paths',
        epilog='Example: python3 generate_local_video_report.py data/_exp_20251201_172008_train_metadata_20251202_130454.csv'
    )

    parser.add_argument('metadata_csv', help='Path to metadata CSV file')
    parser.add_argument('--output', default=None, help='Output HTML file (default: auto-generated name)')

    args = parser.parse_args()

    # Auto-generate output name if not provided
    if args.output is None:
        csv_basename = os.path.basename(args.metadata_csv)
        csv_name = csv_basename.replace('_metadata', '').replace('.csv', '')
        args.output = f"{csv_name}_local_report.html"

    print("=" * 70)
    print("📹 GENERATE LOCAL VIDEO REPORT WITH ABSOLUTE PATHS")
    print("=" * 70)
    print(f"Input CSV: {args.metadata_csv}")
    print(f"Output HTML: {args.output}")
    print("=" * 70)

    if not os.path.exists(args.metadata_csv):
        print(f"\n❌ Error: Metadata CSV not found: {args.metadata_csv}")
        return

    print(f"\n📋 Loading metadata...")
    metadata = load_metadata(args.metadata_csv)

    print(f"📊 Analyzing dataset...")
    stats = analyze_distribution(metadata)
    groups = group_videos(metadata)

    print(f"🎨 Generating HTML report...")
    generate_html_report(metadata, stats, groups, args.output)

    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)
    print(f"📄 Report: {args.output}")
    print("=" * 70)
    print()
    print("🎬 TO VIEW THE REPORT:")
    print()
    print(f"   Simply double-click the file: {args.output}")
    print(f"   Or open it in your browser directly")
    print()
    print("   The videos will play using absolute file:// paths")
    print("=" * 70)


if __name__ == '__main__':
    main()
