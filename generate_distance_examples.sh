#!/bin/bash
################################################################################
# Generate Distance Parameter Examples
#
# This script generates 4 example videos showing the distance parameter effect:
# - distance=1.0: Normal size (baseline)
# - distance=1.5: 67% size (slightly further)
# - distance=2.0: 50% size (half size)
# - distance=3.0: 33% size (much smaller/further)
#
# Usage:
#   ./generate_distance_examples.sh
#
# The examples will be saved to: data/distance_examples/
################################################################################

set -e

echo "========================================"
echo "Generating Distance Parameter Examples"
echo "========================================"
echo ""

# Create output directory
mkdir -p data/distance_examples

# Common parameters for all videos (simple, easy to see the difference)
COMMON_PARAMS="
    --texture_type subtle_gray_stripes
    --direction right
    --speed 3.0
    --view_angle 0.0
    --duration 5.0
    --fps 4
    --resolution 640x480
    --stripe_gray 125
    --background_gray 140
    --stripe_width 10
    --stripe_spacing 60
    --seed 42
"

echo "Generating example 1/4: distance=1.0 (NORMAL SIZE - baseline)"
python3 data/synthetic_treadmill/synthetic_data_generation.py \
    --num_videos 1 \
    --output_dir data/distance_examples \
    --distance 1.0 \
    $COMMON_PARAMS

echo ""
echo "Generating example 2/4: distance=1.5 (67% size)"
python3 data/synthetic_treadmill/synthetic_data_generation.py \
    --num_videos 1 \
    --output_dir data/distance_examples \
    --distance 1.5 \
    $COMMON_PARAMS \
    --seed 43

echo ""
echo "Generating example 3/4: distance=2.0 (50% size - HALF)"
python3 data/synthetic_treadmill/synthetic_data_generation.py \
    --num_videos 1 \
    --output_dir data/distance_examples \
    --distance 2.0 \
    $COMMON_PARAMS \
    --seed 44

echo ""
echo "Generating example 4/4: distance=3.0 (33% size - MUCH SMALLER)"
python3 data/synthetic_treadmill/synthetic_data_generation.py \
    --num_videos 1 \
    --output_dir data/distance_examples \
    --distance 3.0 \
    $COMMON_PARAMS \
    --seed 45

echo ""
echo "========================================"
echo "✓ Examples generated successfully!"
echo "========================================"
echo ""
echo "Output location: data/distance_examples/"
echo ""
echo "Files generated:"
ls -lh data/distance_examples/*.mp4
echo ""
echo "You can now view these videos to see the distance effect:"
echo "  - treadmill_0000_*_dist1.0_*.mp4  (Normal size)"
echo "  - treadmill_0000_*_dist1.5_*.mp4  (67% size)"
echo "  - treadmill_0000_*_dist2.0_*.mp4  (50% size)"
echo "  - treadmill_0000_*_dist3.0_*.mp4  (33% size)"
echo ""
echo "The treadmill will appear progressively smaller and more centered,"
echo "with dark background surrounding it at higher distance values."
echo ""
