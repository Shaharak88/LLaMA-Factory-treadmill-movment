#!/usr/bin/env python3
"""
Generate Enhanced Video Report with Category Distributions and Warnings

This script generates an HTML report with:
- Category distributions for all metadata parameters (including stripe/bg contrasts)
- Warnings tab for dataset imbalances
- Missing combinations analysis
- Speed/angle distribution heatmap
- Stripe contrast and background contrast specific analysis
- Absolute file:// paths for Windows compatibility

Reports are saved to: analytics/reports/

Usage:
    python3 analytics/generate_enhanced_video_report.py METADATA_CSV_FILE

Example:
    python3 analytics/generate_enhanced_video_report.py data/_exp_20251201_172008_train_metadata_20251202_130454.csv
"""

import csv
import json
import os
import argparse
import statistics
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
    if len(wsl_path) > 2 and wsl_path[1:3] == ':/':
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
    """Analyze parameter distributions with comprehensive outlier detection."""
    stats = {
        'total_videos': len(metadata),
        'distributions': {},
        'outliers': [],
        'imbalance_warnings': []
    }

    if not metadata:
        return stats

    # Get all parameters from metadata
    all_params = set()
    for video in metadata:
        all_params.update(video.keys())

    # Filter to relevant parameters (exclude paths and internal fields)
    exclude_fields = {'video_name', 'absolute_path', 'file_url', 'file_exists'}
    params = sorted([p for p in all_params if p not in exclude_fields])

    for param in params:
        values = [row.get(param, '') for row in metadata if row.get(param, '') and row.get(param, '') != 'N/A']
        if not values:
            continue

        counter = Counter(values)
        total = sum(counter.values())
        avg = total / len(counter) if counter else 0

        # Calculate statistics
        counts = list(counter.values())
        min_count = min(counts) if counts else 0
        max_count = max(counts) if counts else 0

        # Detect outliers using standard deviation
        if len(counts) > 1:
            mean = statistics.mean(counts)
            try:
                stdev = statistics.stdev(counts)
            except:
                stdev = 0
        else:
            mean = avg
            stdev = 0

        stats['distributions'][param] = {
            'counts': dict(counter.most_common()),
            'unique_values': len(counter),
            'average_per_value': round(avg, 2),
            'min_count': min_count,
            'max_count': max_count,
            'std_dev': round(stdev, 2)
        }

        # Detect outliers (values that are significantly different from mean)
        for value, count in counter.items():
            ratio = count / avg if avg > 0 else 0

            # Flag as outlier if more than 1.5 std devs away from mean (when stdev > 0)
            is_outlier = False
            if stdev > 0 and len(counts) > 2:
                z_score = abs((count - mean) / stdev)
                if z_score > 1.5:
                    is_outlier = True

            # Also flag if less than 50% or more than 200% of average
            if count < avg * 0.5 or count > avg * 2.0:
                is_outlier = True

            if is_outlier:
                stats['outliers'].append({
                    'parameter': param,
                    'value': str(value),
                    'count': count,
                    'expected_avg': round(avg, 2),
                    'ratio': round(ratio, 2),
                    'deviation': round((count - avg) / avg * 100, 1) if avg > 0 else 0
                })

        # Check if parameter has significant imbalance
        if max_count > 0 and min_count > 0:
            imbalance_ratio = max_count / min_count
            if imbalance_ratio > 3.0:  # More than 3x difference
                stats['imbalance_warnings'].append({
                    'parameter': param,
                    'max_count': max_count,
                    'min_count': min_count,
                    'ratio': round(imbalance_ratio, 2),
                    'unique_values': len(counter)
                })

    return stats


def analyze_stripe_bg_distribution(metadata: List[Dict[str, str]]) -> Dict[str, Any]:
    """Analyze stripe and background contrast distributions specifically."""
    stripe_bg_stats = {
        'stripe_contrast': {},
        'background_contrast': {},
        'combined_distribution': {}
    }

    # Analyze stripe contrast
    stripe_values = [v.get('stripe_contrast', '') for v in metadata
                    if v.get('stripe_contrast', '') and v.get('stripe_contrast', '') != 'N/A']
    if stripe_values:
        stripe_counter = Counter(stripe_values)
        stripe_bg_stats['stripe_contrast'] = {
            'counts': dict(stripe_counter.most_common()),
            'unique_values': len(stripe_counter),
            'total': sum(stripe_counter.values())
        }

    # Analyze background contrast
    bg_values = [v.get('background_contrast', '') for v in metadata
                if v.get('background_contrast', '') and v.get('background_contrast', '') != 'N/A']
    if bg_values:
        bg_counter = Counter(bg_values)
        stripe_bg_stats['background_contrast'] = {
            'counts': dict(bg_counter.most_common()),
            'unique_values': len(bg_counter),
            'total': sum(bg_counter.values())
        }

    # Combined stripe+bg distribution
    combined = {}
    for video in metadata:
        stripe = video.get('stripe_contrast', 'N/A')
        bg = video.get('background_contrast', 'N/A')
        if stripe != 'N/A' and bg != 'N/A':
            key = f"stripe:{stripe} / bg:{bg}"
            combined[key] = combined.get(key, 0) + 1

    if combined:
        stripe_bg_stats['combined_distribution'] = {
            'counts': dict(sorted(combined.items(), key=lambda x: x[1], reverse=True)),
            'unique_combinations': len(combined),
            'total': sum(combined.values())
        }

    return stripe_bg_stats


def analyze_missing_combinations(metadata: List[Dict[str, str]]) -> Dict[str, Any]:
    """Analyze missing parameter combinations."""
    if not metadata:
        return {'total_possible': 0, 'total_present': 0, 'coverage': 0, 'missing': []}

    # Get unique values for key parameters
    textures = sorted(set(v.get('texture', '') for v in metadata if v.get('texture')))
    directions = sorted(set(v.get('direction', '') for v in metadata if v.get('direction')))
    speeds = sorted(set(v.get('speed', '') for v in metadata if v.get('speed')))
    angles = sorted(set(v.get('angle', '') for v in metadata if v.get('angle')))

    # Create a set of existing combinations
    existing = set()
    for video in metadata:
        combo = (
            video.get('texture', ''),
            video.get('direction', ''),
            video.get('speed', ''),
            video.get('angle', '')
        )
        existing.add(combo)

    # Calculate total possible combinations
    total_possible = len(textures) * len(directions) * len(speeds) * len(angles)
    total_present = len(existing)
    coverage = (total_present / total_possible * 100) if total_possible > 0 else 0

    # Find missing combinations (limit to reasonable number)
    missing = []
    count = 0
    for texture in textures:
        for direction in directions:
            for speed in speeds:
                for angle in angles:
                    combo = (texture, direction, speed, angle)
                    if combo not in existing and count < 100:  # Limit to first 100
                        missing.append({
                            'texture': texture,
                            'direction': direction,
                            'speed': speed,
                            'angle': angle
                        })
                        count += 1

    return {
        'total_possible': total_possible,
        'total_present': total_present,
        'coverage': round(coverage, 2),
        'missing': missing,
        'missing_count': total_possible - total_present,
        'params': {
            'textures': textures,
            'directions': directions,
            'speeds': speeds,
            'angles': angles
        }
    }


def create_speed_angle_heatmap(metadata: List[Dict[str, str]]) -> Dict[str, Any]:
    """Create speed vs angle heatmap data."""
    if not metadata:
        return {'speeds': [], 'angles': [], 'counts': {}}

    # Get unique speeds and angles, sorted
    speeds = sorted(set(v.get('speed', '') for v in metadata if v.get('speed')),
                   key=lambda x: float(x) if x else 0)
    angles = sorted(set(v.get('angle', '') for v in metadata if v.get('angle')),
                   key=lambda x: float(x) if x else 0)

    # Count videos for each speed-angle combination
    counts = {}
    for video in metadata:
        speed = video.get('speed', '')
        angle = video.get('angle', '')
        if speed and angle:
            key = f"{speed}_{angle}"
            counts[key] = counts.get(key, 0) + 1

    return {
        'speeds': speeds,
        'angles': angles,
        'counts': counts
    }


def group_videos(metadata: List[Dict[str, str]]) -> Dict[str, Dict[str, List[Dict]]]:
    """Group videos by parameters."""
    groups = {
        'texture': defaultdict(list),
        'direction': defaultdict(list),
        'speed_range': defaultdict(list),
        'angle': defaultdict(list),
        'objects': defaultdict(list),
        'blur': defaultdict(list),
        'stripe_contrast': defaultdict(list),
        'background_contrast': defaultdict(list)
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

        # Group by stripe contrast
        stripe = video.get('stripe_contrast', 'N/A')
        if stripe and stripe != 'N/A':
            groups['stripe_contrast'][stripe].append(video)

        # Group by background contrast
        bg = video.get('background_contrast', 'N/A')
        if bg and bg != 'N/A':
            groups['background_contrast'][bg].append(video)

    return groups


def generate_html_report(metadata: List[Dict[str, str]], stats: Dict[str, Any],
                        groups: Dict[str, Dict[str, List[Dict]]],
                        missing_combos: Dict[str, Any],
                        heatmap_data: Dict[str, Any],
                        stripe_bg_stats: Dict[str, Any],
                        output_path: str):
    """Generate enhanced HTML report with warnings and distributions."""

    # Create metadata with file URLs
    metadata_with_urls = []
    missing_videos = []
    for video in metadata:
        video_copy = video.copy()
        windows_path = video['absolute_path']
        file_url = f"file:///{windows_path}"
        video_copy['file_url'] = file_url
        metadata_with_urls.append(video_copy)

        if not video.get('file_exists', False):
            missing_videos.append(video['video_name'])

    metadata_json = json.dumps(metadata_with_urls, indent=2)
    stats_json = json.dumps(stats, indent=2)
    missing_combos_json = json.dumps(missing_combos, indent=2)
    heatmap_json = json.dumps(heatmap_data, indent=2)
    stripe_bg_json = json.dumps(stripe_bg_stats, indent=2)

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
    <title>Enhanced Video Dataset Report</title>
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
            padding: 15px 20px;
            cursor: pointer;
            border: none;
            background: transparent;
            font-size: 0.95em;
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
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .bar {{
            flex: 1;
            height: 25px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 4px;
            position: relative;
            margin-right: 10px;
            min-width: 30px;
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
        .alert-danger {{
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            color: #721c24;
        }}
        .alert-success {{
            background: #d4edda;
            border-left: 4px solid #28a745;
            color: #155724;
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
        .heatmap-container {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow-x: auto;
        }}
        .heatmap {{
            display: inline-grid;
            gap: 2px;
            margin: 20px 0;
        }}
        .heatmap-cell {{
            min-width: 60px;
            min-height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
            color: white;
        }}
        .heatmap-label {{
            min-width: 60px;
            min-height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            color: #495057;
        }}
        .combination-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 0.9em;
        }}
        .combination-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .combination-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #dee2e6;
        }}
        .combination-table tr:nth-child(even) {{
            background: #f8f9fa;
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
            <h1>📹 Enhanced Video Dataset Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Total Videos: {len(metadata)}</p>
            <p style="font-size: 0.9em; margin-top: 10px;">
                ✅ With stripe/bg contrast analysis and complete imbalance detection
            </p>
        </header>

        {missing_warning}

        <div class="tabs">
            <button class="tab active" onclick="switchTab(event, 'overview')">Overview</button>
            <button class="tab" onclick="switchTab(event, 'warnings')">⚠️ Warnings</button>
            <button class="tab" onclick="switchTab(event, 'distributions')">📊 Distributions</button>
            <button class="tab" onclick="switchTab(event, 'stripe_bg')">🎨 Stripe/BG</button>
            <button class="tab" onclick="switchTab(event, 'defaults')">Defaults</button>
            <button class="tab" onclick="switchTab(event, 'texture')">Texture</button>
            <button class="tab" onclick="switchTab(event, 'direction')">Direction</button>
            <button class="tab" onclick="switchTab(event, 'speed')">Speed</button>
            <button class="tab" onclick="switchTab(event, 'angle')">Angle</button>
            <button class="tab" onclick="switchTab(event, 'objects')">Objects</button>
            <button class="tab" onclick="switchTab(event, 'blur')">Blur</button>
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
                    <h3>Parameters Tracked</h3>
                    <div class="value">{len(stats['distributions'])}</div>
                </div>
                <div class="stat-card">
                    <h3>Outliers Detected</h3>
                    <div class="value">{len(stats['outliers'])}</div>
                </div>
                <div class="stat-card">
                    <h3>Coverage</h3>
                    <div class="value">{missing_combos['coverage']}%</div>
                </div>
            </div>
            <div id="quick-stats"></div>
        </div>

        <div id="warnings" class="tab-content">
            <h2>⚠️ Dataset Warnings & Analysis</h2>
            <div id="warnings-content"></div>
        </div>

        <div id="distributions" class="tab-content">
            <h2>📊 Category Distributions</h2>
            <p style="margin-bottom: 20px; color: #495057;">Complete distribution analysis for all metadata categories.</p>
            <div id="distribution-charts"></div>
        </div>

        <div id="stripe_bg" class="tab-content">
            <h2>🎨 Stripe & Background Contrast Analysis</h2>
            <p style="margin-bottom: 20px; color: #495057;">Detailed analysis of stripe and background contrast distributions.</p>
            <div id="stripe-bg-content"></div>
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
        const missingCombos = {missing_combos_json};
        const heatmapData = {heatmap_json};
        const stripeBgStats = {stripe_bg_json};

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
                        ${{video.stripe_contrast && video.stripe_contrast !== 'N/A' ? `
                        <div class="metadata-item">
                            <span class="metadata-label">Stripe:</span>
                            <span class="metadata-value">${{video.stripe_contrast}}</span>
                        </div>
                        ` : ''}}
                        ${{video.background_contrast && video.background_contrast !== 'N/A' ? `
                        <div class="metadata-item">
                            <span class="metadata-label">BG:</span>
                            <span class="metadata-value">${{video.background_contrast}}</span>
                        </div>
                        ` : ''}}
                    </div>
                    <div style="margin-top: 10px; padding: 8px; background: white; border-radius: 4px; font-size: 0.75em; font-family: monospace; word-break: break-all; color: #6c757d;">
                        ${{video.absolute_path}}
                    </div>
                </div>
            `;
        }}

        function getHeatmapColor(count, max) {{
            if (count === 0) return '#e9ecef';
            const intensity = count / max;
            if (intensity < 0.25) return '#c7d2fe';
            if (intensity < 0.5) return '#a5b4fc';
            if (intensity < 0.75) return '#818cf8';
            return '#6366f1';
        }}

        function renderStripeBgAnalysis() {{
            const container = document.getElementById('stripe-bg-content');
            let html = '';

            // Stripe contrast distribution
            if (stripeBgStats.stripe_contrast && stripeBgStats.stripe_contrast.counts) {{
                const counts = stripeBgStats.stripe_contrast.counts;
                const maxCount = Math.max(...Object.values(counts));
                html += `
                    <div class="distribution-chart">
                        <h3>Stripe Contrast Distribution</h3>
                        <p style="font-size: 0.85em; color: #6c757d; margin-bottom: 10px;">
                            ${{stripeBgStats.stripe_contrast.unique_values}} unique values |
                            Total: ${{stripeBgStats.stripe_contrast.total}} videos
                        </p>
                        <div>
                `;
                for (const [value, count] of Object.entries(counts)) {{
                    const width = (count / maxCount) * 100;
                    html += `
                        <div class="bar-item">
                            <div class="bar-label">${{value}}</div>
                            <div class="bar" style="width: ${{width}}%">
                                <span class="bar-value">${{count}}</span>
                            </div>
                        </div>
                    `;
                }}
                html += '</div></div>';
            }} else {{
                html += '<div class="alert alert-info"><p>No stripe contrast data available.</p></div>';
            }}

            // Background contrast distribution
            if (stripeBgStats.background_contrast && stripeBgStats.background_contrast.counts) {{
                const counts = stripeBgStats.background_contrast.counts;
                const maxCount = Math.max(...Object.values(counts));
                html += `
                    <div class="distribution-chart">
                        <h3>Background Contrast Distribution</h3>
                        <p style="font-size: 0.85em; color: #6c757d; margin-bottom: 10px;">
                            ${{stripeBgStats.background_contrast.unique_values}} unique values |
                            Total: ${{stripeBgStats.background_contrast.total}} videos
                        </p>
                        <div>
                `;
                for (const [value, count] of Object.entries(counts)) {{
                    const width = (count / maxCount) * 100;
                    html += `
                        <div class="bar-item">
                            <div class="bar-label">${{value}}</div>
                            <div class="bar" style="width: ${{width}}%">
                                <span class="bar-value">${{count}}</span>
                            </div>
                        </div>
                    `;
                }}
                html += '</div></div>';
            }} else {{
                html += '<div class="alert alert-info"><p>No background contrast data available.</p></div>';
            }}

            // Combined distribution
            if (stripeBgStats.combined_distribution && stripeBgStats.combined_distribution.counts) {{
                const counts = stripeBgStats.combined_distribution.counts;
                html += `
                    <div class="distribution-chart">
                        <h3>Combined Stripe + Background Contrast Distribution</h3>
                        <p style="font-size: 0.85em; color: #6c757d; margin-bottom: 10px;">
                            ${{stripeBgStats.combined_distribution.unique_combinations}} unique combinations |
                            Total: ${{stripeBgStats.combined_distribution.total}} videos
                        </p>
                        <table class="combination-table">
                            <thead>
                                <tr>
                                    <th>Combination</th>
                                    <th>Count</th>
                                    <th>Percentage</th>
                                </tr>
                            </thead>
                            <tbody>
                `;
                const total = stripeBgStats.combined_distribution.total;
                for (const [combo, count] of Object.entries(counts)) {{
                    const pct = ((count / total) * 100).toFixed(1);
                    html += `
                        <tr>
                            <td>${{combo}}</td>
                            <td>${{count}}</td>
                            <td>${{pct}}%</td>
                        </tr>
                    `;
                }}
                html += '</tbody></table></div>';
            }}

            container.innerHTML = html;
        }}

        function renderWarnings() {{
            const container = document.getElementById('warnings-content');
            let html = '';

            // Check if there are any warnings
            const hasImbalances = stats.imbalance_warnings && stats.imbalance_warnings.length > 0;
            const hasOutliers = stats.outliers && stats.outliers.length > 0;
            const hasMissingCombos = missingCombos.missing_count > 0;

            if (!hasImbalances && !hasOutliers && !hasMissingCombos) {{
                html += `
                    <div class="alert alert-success">
                        <h3>✅ No Imbalances Detected</h3>
                        <p>Your dataset appears to be well-balanced with no significant distribution issues.</p>
                    </div>
                `;
            }} else {{
                // Imbalance warnings
                if (hasImbalances) {{
                    html += `
                        <div class="alert alert-warning">
                            <h3>⚠️ Imbalanced Parameters (${{stats.imbalance_warnings.length}})</h3>
                            <p>These parameters have significant imbalances (>3x difference between max and min counts):</p>
                            <table class="combination-table">
                                <thead>
                                    <tr>
                                        <th>Parameter</th>
                                        <th>Max Count</th>
                                        <th>Min Count</th>
                                        <th>Ratio</th>
                                        <th>Unique Values</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${{stats.imbalance_warnings.map(w => `
                                        <tr>
                                            <td><strong>${{w.parameter}}</strong></td>
                                            <td>${{w.max_count}}</td>
                                            <td>${{w.min_count}}</td>
                                            <td>${{w.ratio}}x</td>
                                            <td>${{w.unique_values}}</td>
                                        </tr>
                                    `).join('')}}
                                </tbody>
                            </table>
                        </div>
                    `;
                }}

                // Outlier warnings
                if (hasOutliers) {{
                    html += `
                        <div class="alert alert-warning">
                            <h3>📊 Distribution Outliers (${{stats.outliers.length}})</h3>
                            <p>Values that deviate significantly from the expected average:</p>
                            <table class="combination-table">
                                <thead>
                                    <tr>
                                        <th>Parameter</th>
                                        <th>Value</th>
                                        <th>Actual Count</th>
                                        <th>Expected Avg</th>
                                        <th>Deviation</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${{stats.outliers.slice(0, 30).map(o => `
                                        <tr>
                                            <td><strong>${{o.parameter}}</strong></td>
                                            <td>${{o.value}}</td>
                                            <td>${{o.count}}</td>
                                            <td>${{o.expected_avg}}</td>
                                            <td>${{o.deviation > 0 ? '+' : ''}}${{o.deviation}}%</td>
                                        </tr>
                                    `).join('')}}
                                    ${{stats.outliers.length > 30 ? `<tr><td colspan="5"><em>...and ${{stats.outliers.length - 30}} more outliers</em></td></tr>` : ''}}
                                </tbody>
                            </table>
                        </div>
                    `;
                }}

                // Missing combinations
                if (hasMissingCombos) {{
                    html += `
                        <div class="alert alert-info">
                            <h3>🔍 Missing Combinations Analysis</h3>
                            <p><strong>Coverage:</strong> ${{missingCombos.coverage}}% of all possible combinations</p>
                            <p><strong>Present:</strong> ${{missingCombos.total_present}} / ${{missingCombos.total_possible}} possible combinations</p>
                            <p><strong>Missing:</strong> ${{missingCombos.missing_count}} combinations</p>

                            <details style="margin-top: 15px;">
                                <summary style="cursor: pointer; font-weight: 600; padding: 10px; background: white; border-radius: 4px;">
                                    Show First ${{Math.min(missingCombos.missing.length, 50)}} Missing Combinations
                                </summary>
                                <table class="combination-table" style="margin-top: 10px;">
                                    <thead>
                                        <tr>
                                            <th>Texture</th>
                                            <th>Direction</th>
                                            <th>Speed</th>
                                            <th>Angle</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${{missingCombos.missing.slice(0, 50).map(m => `
                                            <tr>
                                                <td>${{m.texture}}</td>
                                                <td>${{m.direction}}</td>
                                                <td>${{m.speed}}</td>
                                                <td>${{m.angle}}</td>
                                            </tr>
                                        `).join('')}}
                                    </tbody>
                                </table>
                            </details>
                        </div>
                    `;
                }}
            }}

            // Speed/Angle Heatmap
            if (heatmapData.speeds.length > 0 && heatmapData.angles.length > 0) {{
                const maxCount = Math.max(...Object.values(heatmapData.counts));

                html += `
                    <div class="heatmap-container">
                        <h3 style="color: #667eea; margin-bottom: 15px;">🔥 Speed vs Angle Distribution Heatmap</h3>
                        <p style="margin-bottom: 15px; color: #495057;">Darker colors indicate more videos with that speed/angle combination.</p>
                        <div class="heatmap" style="grid-template-columns: 80px repeat(${{heatmapData.speeds.length}}, 60px);">
                            <div class="heatmap-label"></div>
                            ${{heatmapData.speeds.map(speed => `
                                <div class="heatmap-label">Speed ${{speed}}</div>
                            `).join('')}}

                            ${{heatmapData.angles.map(angle => `
                                <div class="heatmap-label">Angle ${{angle}}°</div>
                                ${{heatmapData.speeds.map(speed => {{
                                    const key = speed + '_' + angle;
                                    const count = heatmapData.counts[key] || 0;
                                    const color = getHeatmapColor(count, maxCount);
                                    return `<div class="heatmap-cell" style="background: ${{color}}; color: ${{count === 0 ? '#6c757d' : 'white'}};">${{count}}</div>`;
                                }}).join('')}}
                            `).join('')}}
                        </div>
                    </div>
                `;
            }}

            container.innerHTML = html;
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
                            <div class="bar-label" title="${{value || 'empty'}}">${{value || 'empty'}}</div>
                            <div class="bar" style="width: ${{width}}%">
                                <span class="bar-value">${{count}}</span>
                            </div>
                        </div>
                    `;
                }}
                chartDiv.innerHTML = `
                    <h3>${{param.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}}</h3>
                    <p style="font-size: 0.85em; color: #6c757d; margin-bottom: 10px;">
                        ${{data.unique_values}} unique values | Avg: ${{data.average_per_value}} per value |
                        Min: ${{data.min_count}} | Max: ${{data.max_count}} | Std Dev: ${{data.std_dev}}
                    </p>
                    <div>${{barsHtml}}</div>
                `;
                container.appendChild(chartDiv);
            }}
        }}

        function renderQuickStats() {{
            const container = document.getElementById('quick-stats');
            let html = '<div class="distribution-chart"><h3>Quick Statistics</h3><ul>';

            // Count unique values for key parameters
            const keyParams = ['texture', 'direction', 'speed', 'angle', 'stripe_contrast', 'background_contrast'];
            keyParams.forEach(param => {{
                if (stats.distributions[param]) {{
                    const data = stats.distributions[param];
                    html += `<li><strong>${{param.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}}:</strong> ${{data.unique_values}} unique values</li>`;
                }}
            }});

            html += '</ul></div>';
            container.innerHTML = html;
        }}

        function renderDefaultsTable() {{
            const container = document.getElementById('defaults-table');
            const fields = Object.keys(stats.distributions);
            const defaults = {{}};

            fields.forEach(field => {{
                const data = stats.distributions[field];
                const counts = data.counts;
                let maxCount = 0, mostCommon = 'N/A';
                for (const [value, count] of Object.entries(counts)) {{
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
            if (!categoryData || Object.keys(categoryData).length === 0) {{
                container.innerHTML = `<h2>${{categoryName.charAt(0).toUpperCase() + categoryName.slice(1).replace('_', ' ')}} Groups</h2>
                    <div class="alert alert-info"><p>No data available for this category.</p></div>`;
                return;
            }}
            let html = `<h2>${{categoryName.charAt(0).toUpperCase() + categoryName.slice(1).replace('_', ' ')}} Groups</h2>`;
            for (const [groupName, videos] of Object.entries(categoryData)) {{
                if (videos.length > 0) {{
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
            renderQuickStats();
            renderWarnings();
            renderDistributionCharts();
            renderStripeBgAnalysis();
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
    print(f"\n⚠️  Warnings:")
    print(f"   Imbalances: {len(stats['imbalance_warnings'])}")
    print(f"   Outliers: {len(stats['outliers'])}")
    print(f"   Coverage: {missing_combos['coverage']}%")
    if stripe_bg_stats['stripe_contrast']:
        print(f"\n🎨 Stripe/BG Analysis:")
        print(f"   Stripe values: {stripe_bg_stats['stripe_contrast'].get('unique_values', 0)}")
        print(f"   BG values: {stripe_bg_stats['background_contrast'].get('unique_values', 0)}")
        if stripe_bg_stats['combined_distribution']:
            print(f"   Combined combinations: {stripe_bg_stats['combined_distribution'].get('unique_combinations', 0)}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Enhanced Video Report with Warnings and Distributions',
        epilog='Example: python3 analytics/generate_enhanced_video_report.py data/_exp_20251201_172008_train_metadata_20251202_130454.csv'
    )

    parser.add_argument('metadata_csv', help='Path to metadata CSV file')
    parser.add_argument('--output', default=None, help='Output HTML file (default: auto-generated name in analytics/reports/)')

    args = parser.parse_args()

    # Get the script directory (analytics/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(script_dir, 'reports')

    # Create reports directory if it doesn't exist
    os.makedirs(reports_dir, exist_ok=True)

    # Auto-generate output name if not provided
    if args.output is None:
        csv_basename = os.path.basename(args.metadata_csv)
        csv_name = csv_basename.replace('_metadata', '').replace('.csv', '')
        args.output = os.path.join(reports_dir, f"{csv_name}_enhanced_report.html")
    elif not os.path.isabs(args.output):
        # If relative path provided, put it in reports directory
        args.output = os.path.join(reports_dir, args.output)

    print("=" * 70)
    print("📹 GENERATE ENHANCED VIDEO REPORT")
    print("=" * 70)
    print(f"Input CSV: {args.metadata_csv}")
    print(f"Output HTML: {args.output}")
    print(f"Reports directory: {reports_dir}")
    print("=" * 70)

    if not os.path.exists(args.metadata_csv):
        print(f"\n❌ Error: Metadata CSV not found: {args.metadata_csv}")
        return

    print(f"\n📋 Loading metadata...")
    metadata = load_metadata(args.metadata_csv)

    print(f"📊 Analyzing distributions...")
    stats = analyze_distribution(metadata)

    print(f"🎨 Analyzing stripe/background contrasts...")
    stripe_bg_stats = analyze_stripe_bg_distribution(metadata)

    print(f"🔍 Analyzing missing combinations...")
    missing_combos = analyze_missing_combinations(metadata)

    print(f"🔥 Creating speed/angle heatmap...")
    heatmap_data = create_speed_angle_heatmap(metadata)

    print(f"📦 Grouping videos...")
    groups = group_videos(metadata)

    print(f"🎨 Generating HTML report...")
    generate_html_report(metadata, stats, groups, missing_combos, heatmap_data, stripe_bg_stats, args.output)

    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)
    print(f"📄 Report: {args.output}")
    print("=" * 70)
    print()
    print("🎬 TO VIEW THE REPORT:")
    print()
    print(f"   Simply double-click the file in: analytics/reports/")
    print(f"   Or open it in your browser directly")
    print()
    print("   Features:")
    print("   • Category distributions for all parameters")
    print("   • Stripe & Background contrast specific analysis")
    print("   • Warnings tab with imbalance detection")
    print("   • Missing combinations analysis")
    print("   • Speed/Angle heatmap visualization")
    print("=" * 70)


if __name__ == '__main__':
    main()
