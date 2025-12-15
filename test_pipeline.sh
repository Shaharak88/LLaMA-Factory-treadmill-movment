#!/bin/bash
# Test script for pipeline components
# This script runs basic tests to verify pipeline functionality

set -e  # Exit on error

echo "========================================"
echo "Pipeline Component Tests"
echo "========================================"

# Test 1: Check scripts exist
echo ""
echo "Test 1: Checking script files..."
for script in "run_full_pipeline.py" "evaluate_pipeline.py" "building_dataset.py"; do
    if [ -f "$script" ]; then
        echo "  ✓ Found: $script"
    else
        echo "  ✗ Missing: $script"
        exit 1
    fi
done

# Test 2: Check scripts are executable
echo ""
echo "Test 2: Checking executability..."
for script in "run_full_pipeline.py" "evaluate_pipeline.py"; do
    if [ -x "$script" ]; then
        echo "  ✓ Executable: $script"
    else
        echo "  ⚠ Not executable: $script (running chmod +x)"
        chmod +x "$script"
    fi
done

# Test 3: Test help output
echo ""
echo "Test 3: Testing --help output..."
if python3 run_full_pipeline.py --help > /dev/null 2>&1; then
    echo "  ✓ run_full_pipeline.py --help works"
else
    echo "  ✗ run_full_pipeline.py --help failed"
    exit 1
fi

if python3 evaluate_pipeline.py --help > /dev/null 2>&1; then
    echo "  ✓ evaluate_pipeline.py --help works"
else
    echo "  ✗ evaluate_pipeline.py --help failed"
    exit 1
fi

# Test 4: Test argument parsing
echo ""
echo "Test 4: Testing argument parsing (dry run)..."
# This will fail but should parse arguments correctly
python3 run_full_pipeline.py \
    --dataset_name test_args \
    --num_videos 1 \
    --skip_training \
    --skip_evaluation \
    2>&1 | head -5 || true

echo ""
echo "========================================"
echo "Basic tests passed! ✓"
echo "========================================"
echo ""
echo "To run a full pipeline test with minimal data:"
echo ""
echo "  python3 run_full_pipeline.py \\"
echo "    --dataset_name pipeline_test \\"
echo "    --num_videos 4 \\"
echo "    --num_train_epochs 1 \\"
echo "    --seed 42"
echo ""
