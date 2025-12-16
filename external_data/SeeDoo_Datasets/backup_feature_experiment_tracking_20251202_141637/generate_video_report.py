#!/usr/bin/env python3
"""
Generate an interactive HTML report for video dataset review.

This script reads video metadata CSV files and generates an interactive HTML report
with video players, statistical analysis, and organized tabs for easy review.
"""

import csv
import json
import argparse
import os
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Any


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

    # Key parameters to analyze
    key_params = ['texture', 'direction', 'speed', 'angle', 'brightness',
                  'contrast', 'object_enabled', 'blur_enabled']

    for param in key_params:
        values = [row.get(param, '') for row in metadata if row.get(param, '')]
        counter = Counter(values)

        # Calculate statistics
        total = sum(counter.values())
        avg = total / len(counter) if counter else 0
        std_threshold = avg * 0.5  # 50% deviation threshold

        stats['distributions'][param] = {
            'counts': dict(counter.most_common()),
            'unique_values': len(counter),
            'average_per_value': round(avg, 2)
        }

        # Identify under/over-represented
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
        # Group by texture
        texture = video.get('texture', 'unknown')
        groups['texture'][texture].append(video)

        # Group by direction
        direction = video.get('direction', 'unknown')
        groups['direction'][direction].append(video)

        # Group by speed range
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

        # Group by angle
        angle = video.get('angle', 'unknown')
        groups['angle'][f"{angle}°"].append(video)

        # Group by objects
        obj_enabled = video.get('object_enabled', 'no')
        obj_type = video.get('object_type', 'none')
        obj_key = f"{obj_type}" if obj_enabled == 'yes' else 'no_objects'
        groups['objects'][obj_key].append(video)

        # Group by blur
        blur_enabled = video.get('blur_enabled', 'no')
        blur_type = video.get('blur_type', 'none')
        blur_key = f"{blur_type}" if blur_enabled == 'yes' else 'no_blur'
        groups['blur'][blur_key].append(video)

    return groups


def generate_html(metadata: List[Dict[str, str]], stats: Dict[str, Any],
                  groups: Dict[str, Dict[str, List[Dict]]], output_path: str,
                  video_base_url: str):
    """Generate interactive HTML report."""

    # Convert data to JSON for JavaScript
    metadata_json = json.dumps(metadata, indent=2)
    stats_json = json.dumps(stats, indent=2)
    groups_json = json.dumps({k: {gk: len(gv) for gk, gv in v.items()}
                              for k, v in groups.items()}, indent=2)

    # Sample videos from each group (max 5 per group)
    samples = {}
    for category, category_groups in groups.items():
        samples[category] = {}
        for group_name, group_videos in category_groups.items():
            samples[category][group_name] = group_videos[:5]

    samples_json = json.dumps(samples, indent=2)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Dataset Review Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
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

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

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

        .tab:hover {{
            background: rgba(102, 126, 234, 0.1);
            color: #667eea;
        }}

        .tab.active {{
            background: white;
            color: #667eea;
            border-bottom: 3px solid #667eea;
        }}

        .tab-content {{
            padding: 30px;
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}

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

        .stat-card h3 {{
            font-size: 1em;
            opacity: 0.9;
            margin-bottom: 10px;
        }}

        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
        }}

        .distribution-chart {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}

        .distribution-chart h3 {{
            margin-bottom: 15px;
            color: #667eea;
        }}

        .bar-chart {{
            margin-bottom: 10px;
        }}

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
            transition: transform 0.3s, box-shadow 0.3s;
        }}

        .video-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
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

        .metadata-label {{
            font-weight: 600;
            color: #495057;
        }}

        .metadata-value {{
            color: #667eea;
        }}

        .group-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0 15px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .group-header h3 {{
            font-size: 1.3em;
        }}

        .group-count {{
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }}

        .search-box {{
            width: 100%;
            padding: 12px;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            font-size: 1em;
            margin-bottom: 20px;
        }}

        .search-box:focus {{
            outline: none;
            border-color: #667eea;
        }}

        .subtabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}

        .subtab {{
            padding: 10px 20px;
            background: #e9ecef;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.3s;
        }}

        .subtab:hover {{
            background: #667eea;
            color: white;
        }}

        .subtab.active {{
            background: #667eea;
            color: white;
        }}

        @media (max-width: 768px) {{
            .video-grid {{
                grid-template-columns: 1fr;
            }}

            .stats-grid {{
                grid-template-columns: 1fr;
            }}
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
            <p style="margin-bottom: 20px; color: #495057;">Most common value for each parameter across all videos in this dataset.</p>
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
            <input type="text" class="search-box" id="searchBox" placeholder="Search videos by name, texture, direction, speed..." onkeyup="filterVideos()">
            <div id="allVideosGrid" class="video-grid"></div>
        </div>
    </div>

    <script>
        const metadata = {metadata_json};
        const stats = {stats_json};
        const groupCounts = {groups_json};
        const samples = {samples_json};
        const videoBaseUrl = '{video_base_url}';

        function switchTab(event, tabName) {{
            // Hide all tab contents
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => content.classList.remove('active'));

            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));

            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.currentTarget.classList.add('active');

            // Load content if not loaded
            if (tabName === 'all' && document.getElementById('allVideosGrid').innerHTML === '') {{
                renderAllVideos();
            }}
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
                    <div class="bar-chart">${{barsHtml}}</div>
                `;

                container.appendChild(chartDiv);
            }}
        }}

        function renderRepresentationAlerts() {{
            const container = document.getElementById('representation-alerts');

            if (stats.under_represented.length > 0) {{
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert alert-warning';
                alertDiv.innerHTML = `
                    <h3>⚠️ Under-represented Groups</h3>
                    <p>The following parameter values have fewer samples than expected:</p>
                    <ul>
                        ${{stats.under_represented.map(item =>
                            `<li><strong>${{item.parameter}}: ${{item.value}}</strong> - ${{item.count}} videos (expected ~${{item.expected}}, ratio: ${{item.ratio}}x)</li>`
                        ).join('')}}
                    </ul>
                `;
                container.appendChild(alertDiv);
            }}

            if (stats.over_represented.length > 0) {{
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert alert-info';
                alertDiv.innerHTML = `
                    <h3>📈 Over-represented Groups</h3>
                    <p>The following parameter values have more samples than expected:</p>
                    <ul>
                        ${{stats.over_represented.map(item =>
                            `<li><strong>${{item.parameter}}: ${{item.value}}</strong> - ${{item.count}} videos (expected ~${{item.expected}}, ratio: ${{item.ratio}}x)</li>`
                        ).join('')}}
                    </ul>
                `;
                container.appendChild(alertDiv);
            }}
        }}

        function renderVideoCard(video) {{
            const videoUrl = videoBaseUrl ? `${{videoBaseUrl}}/${{video.video_name}}` : video.video_name;

            return `
                <div class="video-card">
                    <video controls preload="metadata">
                        <source src="${{videoUrl}}" type="video/mp4">
                        Your browser does not support the video tag.
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
                </div>
            `;
        }}

        function renderGroupTab(tabId, categoryName) {{
            const container = document.getElementById(tabId);
            const categoryData = samples[categoryName];

            let html = `<h2>${{categoryName.charAt(0).toUpperCase() + categoryName.slice(1)}} Groups</h2>`;

            for (const [groupName, videos] of Object.entries(categoryData)) {{
                html += `
                    <div class="group-header">
                        <h3>${{groupName}}</h3>
                        <span class="group-count">${{videos.length}} samples shown</span>
                    </div>
                    <div class="video-grid">
                        ${{videos.map(video => renderVideoCard(video)).join('')}}
                    </div>
                `;
            }}

            container.innerHTML = html;
        }}

        function renderAllVideos() {{
            const container = document.getElementById('allVideosGrid');
            container.innerHTML = metadata.map(video => renderVideoCard(video)).join('');
        }}

        function filterVideos() {{
            const searchTerm = document.getElementById('searchBox').value.toLowerCase();
            const videoCards = document.querySelectorAll('#allVideosGrid .video-card');

            videoCards.forEach(card => {{
                const text = card.textContent.toLowerCase();
                card.style.display = text.includes(searchTerm) ? 'block' : 'none';
            }});
        }}

        function renderDefaultsTable() {{
            const container = document.getElementById('defaults-table');

            // Calculate most common value for each field
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

                // Find most common value
                let maxCount = 0;
                let mostCommon = 'N/A';
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

            // Render as a nice table
            let html = `
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <thead>
                            <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                                <th style="padding: 15px; text-align: left; border-radius: 8px 0 0 0;">Parameter</th>
                                <th style="padding: 15px; text-align: left;">Most Common Value (Default)</th>
                                <th style="padding: 15px; text-align: center;">Count</th>
                                <th style="padding: 15px; text-align: center; border-radius: 0 8px 0 0;">Percentage</th>
                            </tr>
                        </thead>
                        <tbody>
            `;

            fields.forEach((field, index) => {{
                const bgColor = index % 2 === 0 ? '#ffffff' : '#f8f9fa';
                const def = defaults[field];
                html += `
                    <tr style="background: ${{bgColor}};">
                        <td style="padding: 12px; font-weight: 600; color: #495057; border-bottom: 1px solid #dee2e6;">
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

            html += `
                        </tbody>
                    </table>
                </div>
            `;

            container.innerHTML = html;
        }}

        // Initialize on load
        window.addEventListener('DOMContentLoaded', () => {{
            renderDistributionCharts();
            renderRepresentationAlerts();
            renderDefaultsTable();

            // Render grouped tabs
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
        f.write(html_content)

    print(f"\n✅ HTML report generated: {output_path}")


def download_videos_from_server(dataset_name: str, server: str, remote_base: str,
                                local_dir: str, docker_container: str = None) -> str:
    """
    Download videos from remote server using rsync.

    Returns:
        Local directory path where videos were downloaded
    """
    import subprocess

    # Create local directory
    local_video_dir = os.path.join(local_dir, dataset_name)
    os.makedirs(local_video_dir, exist_ok=True)

    print(f"\n📥 Downloading videos from server...")
    print(f"   Remote: {server}:{remote_base}/{dataset_name}/")
    print(f"   Local: {local_video_dir}/")

    # Build rsync command
    if docker_container:
        # First, copy from Docker container to server temp location
        temp_path = f"/tmp/{dataset_name}_videos/"
        print(f"   (via Docker container: {docker_container})")

        # Create temp directory and copy from Docker to server
        docker_copy_cmd = f'mkdir -p {temp_path} && docker cp {docker_container}:{remote_base}/{dataset_name}/. {temp_path}'
        subprocess.run(['ssh', server, docker_copy_cmd], check=True)

        # Rsync from server temp to local
        rsync_cmd = [
            'rsync',
            '-avz',
            '--progress',
            f'{server}:{temp_path}*.mp4',
            f'{local_video_dir}/'
        ]
    else:
        # Direct rsync from server
        rsync_cmd = [
            'rsync',
            '-avz',
            '--progress',
            f'{server}:{remote_base}/{dataset_name}/*.mp4',
            f'{local_video_dir}/'
        ]

    try:
        result = subprocess.run(rsync_cmd, check=True, capture_output=False, text=True)

        # Clean up temp directory if used
        if docker_container:
            subprocess.run(['ssh', server, f'rm -rf {temp_path}'], check=False)

        # Count downloaded files
        video_files = list(Path(local_video_dir).glob('*.mp4'))
        print(f"\n✅ Downloaded {len(video_files)} videos")

        return local_video_dir

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error downloading videos: {e}")
        print("   Continuing with report generation (videos may not play)...")
        return local_video_dir


def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive HTML report for video dataset review',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report and auto-download videos
  python generate_video_report.py data/dataset_metadata.csv --download --dataset-name exp_train

  # Manual video URL (no download)
  python generate_video_report.py data/dataset_metadata.csv --video-url "./videos"

  # Custom server settings
  python generate_video_report.py data/dataset_metadata.csv --download --dataset-name exp_train \\
      --server user@myserver.com --remote-path /data/datasets
"""
    )

    parser.add_argument(
        'csv_path',
        help='Path to metadata CSV file'
    )

    parser.add_argument(
        '--output',
        default='video_dataset_report.html',
        help='Output HTML file path (default: video_dataset_report.html)'
    )

    parser.add_argument(
        '--video-url',
        default='',
        help='Base URL for video files (leave empty for relative paths)'
    )

    parser.add_argument(
        '--download',
        action='store_true',
        help='Automatically download videos from server using rsync'
    )

    parser.add_argument(
        '--dataset-name',
        help='Dataset name on server (required if --download is used)'
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
        help='Docker container name (default: llamafactory, use empty string for no Docker)'
    )

    parser.add_argument(
        '--local-video-dir',
        default='./data/videos',
        help='Local directory to store downloaded videos (default: ./data/videos)'
    )

    args = parser.parse_args()

    # Validate download arguments
    if args.download and not args.dataset_name:
        parser.error("--dataset-name is required when --download is used")
        return

    if not os.path.exists(args.csv_path):
        print(f"❌ Error: CSV file not found: {args.csv_path}")
        return

    print("=" * 60)
    print("Video Dataset Report Generator")
    print("=" * 60)
    print(f"Input CSV: {args.csv_path}")
    print(f"Output HTML: {args.output}")
    print("=" * 60)
    print()

    # Load metadata
    print("📂 Loading metadata...")
    metadata = load_metadata(args.csv_path)
    print(f"   Loaded {len(metadata)} videos")

    # Download videos if requested
    video_url = args.video_url
    if args.download:
        local_video_path = download_videos_from_server(
            args.dataset_name,
            args.server,
            args.remote_path,
            args.local_video_dir,
            args.docker_container if args.docker_container else None
        )
        # Convert to relative path from output HTML location
        output_dir = os.path.dirname(os.path.abspath(args.output))
        local_video_abs = os.path.abspath(local_video_path)
        video_url = os.path.relpath(local_video_abs, output_dir)
        print(f"   Video URL set to: {video_url}")

    print(f"Video base URL: {video_url or '(relative paths)'}")

    # Analyze distributions
    print("\n📊 Analyzing distributions...")
    stats = analyze_distribution(metadata)
    print(f"   Found {len(stats['under_represented'])} under-represented groups")
    print(f"   Found {len(stats['over_represented'])} over-represented groups")

    # Group videos
    print("📁 Grouping videos...")
    groups = group_videos(metadata)

    # Generate HTML
    print("🎨 Generating HTML report...")
    generate_html(metadata, stats, groups, args.output, video_url)

    print(f"\n✅ Done! Open {args.output} in your browser to view the report.")

    if args.download:
        print(f"📂 Videos downloaded to: {local_video_path}")
        print(f"   You can now play videos directly in the browser!")


if __name__ == '__main__':
    main()
