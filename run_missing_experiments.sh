#!/bin/bash
################################################################################
# Run Missing Experiments
# Generated: 2025-12-11
# 
# This script runs 5 missing sampler experiments with 6 minutes wait between each
################################################################################

set -e

WAIT_MINUTES=6
WAIT_SECONDS=$((WAIT_MINUTES * 60))

echo "========================================"
echo "Starting Missing Experiments"
echo "Wait time between runs: ${WAIT_MINUTES} minutes"
echo "========================================"

# 1. Our random shuffle (no fix) + dist_augment
echo ""
echo "[1/5] Running: Our random shuffle + dist_augment"
echo "========================================"
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251209_dist_augment" \
  --sampler-type random_no_fix \
  --epochs 5 \
  --batch-size 14 \
  --grad-accum 1 \
  --model-name "dist_augment_random_nofix_b14_g1" \
  --adapter-type lora+ \
  -y

echo "Waiting ${WAIT_MINUTES} minutes before next run..."
sleep ${WAIT_SECONDS}

# 2. Our random shuffle (no fix) + dist_augment_x5_combined
echo ""
echo "[2/5] Running: Our random shuffle + dist_augment_x5_combined"
echo "========================================"
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251209_dist_augment_x5_combined" \
  --sampler-type random_no_fix \
  --epochs 5 \
  --batch-size 14 \
  --grad-accum 1 \
  --model-name "dist_augment_x5_random_nofix_b14_g1" \
  --adapter-type lora+ \
  -y

echo "Waiting ${WAIT_MINUTES} minutes before next run..."
sleep ${WAIT_SECONDS}

# 3. Our random shuffle (iter fix) + dist_augment
echo ""
echo "[3/5] Running: Our random shuffle (iter fix) + dist_augment"
echo "========================================"
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251209_dist_augment" \
  --sampler-type random \
  --epochs 5 \
  --batch-size 14 \
  --grad-accum 1 \
  --model-name "dist_augment_random_iterfix_b14_g1" \
  --adapter-type lora+ \
  -y

echo "Waiting ${WAIT_MINUTES} minutes before next run..."
sleep ${WAIT_SECONDS}

# 4. Our random shuffle (iter fix) + dist_augment_x5_combined
echo ""
echo "[4/5] Running: Our random shuffle (iter fix) + dist_augment_x5_combined"
echo "========================================"
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251209_dist_augment_x5_combined" \
  --sampler-type random \
  --epochs 5 \
  --batch-size 14 \
  --grad-accum 1 \
  --model-name "dist_augment_x5_random_iterfix_b14_g1" \
  --adapter-type lora+ \
  -y

echo "Waiting ${WAIT_MINUTES} minutes before next run..."
sleep ${WAIT_SECONDS}

# 5. No shuffle (sequential) + dist_augment_x5_combined
echo ""
echo "[5/5] Running: No shuffle + dist_augment_x5_combined"
echo "========================================"
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251209_dist_augment_x5_combined" \
  --sampler-type hf_sequential \
  --epochs 5 \
  --batch-size 14 \
  --grad-accum 1 \
  --model-name "dist_augment_x5_seq_b14_g1" \
  --adapter-type lora+ \
  -y

echo ""
echo "========================================"
echo "All 5 experiments completed!"
echo "========================================"
