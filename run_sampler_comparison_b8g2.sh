#!/bin/bash
################################################################################
# Sampler Comparison Experiments - Batch 8, Grad Accum 2
# Generated: 2025-12-11
#
# Configuration:
#   - Batch size: 8
#   - Gradient accumulation: 2
#   - Effective batch size: 16
#   - Target: ~24 steps per experiment
#
# Datasets & Epochs:
#   - _exp_20251208_230500 (64 samples): 6 epochs → 24 steps
#   - _exp_20251209_dist_augment (74 samples): 5 epochs → 23.12 steps
#   - _exp_20251209_dist_augment_x5_combined (124 samples): 3 epochs → 23.25 steps
#
# Samplers:
#   - hf_shuffle: HuggingFace RandomSampler
#   - random_no_fix: Our random shuffle (same order every epoch)
#   - random: Our random shuffle with iter fix (per-epoch shuffle)
#   - feature_balanced: Feature-balanced sampling
#   - hf_sequential: No shuffle (sequential)
#
# Total: 15 experiments (5 samplers × 3 datasets)
################################################################################

set -e

WAIT_MINUTES=6
WAIT_SECONDS=$((WAIT_MINUTES * 60))

echo "========================================"
echo "Sampler Comparison - Batch 8, Grad Accum 2"
echo "Target: ~24 steps per experiment"
echo "Wait time between runs: ${WAIT_MINUTES} minutes"
echo "========================================"

#===============================================================================
# DATASET 1: _exp_20251208_230500 (64 samples, 6 epochs = 24 steps)
#===============================================================================

# 1/15: HF shuffle + 64 distances
echo ""
echo "[1/15] HF shuffle + 64 distances"
echo "========================================"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251208_230500" \
  --sampler-type hf_shuffle \
  --epochs 6 \
  --batch-size 8 \
  --grad-accum 2 \
  --model-name "64dist_hf_shuffle_b8g2_${TIMESTAMP}" \
  --adapter-type lora+ \
  -y

echo "Waiting ${WAIT_MINUTES} minutes..."
sleep ${WAIT_SECONDS}

# 2/15: Our random shuffle + 64 distances
echo ""
echo "[2/15] Our random shuffle + 64 distances"
echo "========================================"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251208_230500" \
  --sampler-type random_no_fix \
  --epochs 6 \
  --batch-size 8 \
  --grad-accum 2 \
  --model-name "64dist_random_nofix_b8g2_${TIMESTAMP}" \
  --adapter-type lora+ \
  -y

echo "Waiting ${WAIT_MINUTES} minutes..."
sleep ${WAIT_SECONDS}

# 3/15: Our random shuffle (iter fix) + 64 distances
echo ""
echo "[3/15] Our random shuffle (iter fix) + 64 distances"
echo "========================================"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251208_230500" \
  --sampler-type random \
  --epochs 6 \
  --batch-size 8 \
  --grad-accum 2 \
  --model-name "64dist_random_iterfix_b8g2_${TIMESTAMP}" \
  --adapter-type lora+ \
  -y

echo "Waiting ${WAIT_MINUTES} minutes..."
sleep ${WAIT_SECONDS}

# 4/15: Feature Balanced + 64 distances
echo ""
echo "[4/15] Feature Balanced + 64 distances"
echo "========================================"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251208_230500" \
  --sampler-type feature_balanced \
  --epochs 6 \
  --batch-size 8 \
  --grad-accum 2 \
  --model-name "64dist_balanced_b8g2_${TIMESTAMP}" \
  --adapter-type lora+ \
  -y

echo "Waiting ${WAIT_MINUTES} minutes..."
sleep ${WAIT_SECONDS}

# 5/15: No shuffle + 64 distances
echo ""
echo "[5/15] No shuffle + 64 distances"
echo "========================================"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251208_230500" \
  --sampler-type hf_sequential \
  --epochs 6 \
  --batch-size 8 \
  --grad-accum 2 \
  --model-name "64dist_sequential_b8g2_${TIMESTAMP}" \
  --adapter-type lora+ \
  -y

echo "Waiting ${WAIT_MINUTES} minutes..."
sleep ${WAIT_SECONDS}

#===============================================================================
# DATASET 2: _exp_20251209_dist_augment (74 samples, 5 epochs = 23.12 steps)
#===============================================================================

# 6/15: HF shuffle + 74 distances
echo ""
echo "[6/15] HF shuffle + 74 distances"
echo "========================================"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251209_dist_augment" \
  --sampler-type hf_shuffle \
  --epochs 5 \
  --batch-size 8 \
  --grad-accum 2 \
  --model-name "74dist_hf_shuffle_b8g2_${TIMESTAMP}" \
  --adapter-type lora+ \
  -y

echo "Waiting ${WAIT_MINUTES} minutes..."
sleep ${WAIT_SECONDS}

# 7/15: Our random shuffle + 74 distances
echo ""
echo "[7/15] Our random shuffle + 74 distances"
echo "========================================"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251209_dist_augment" \
  --sampler-type random_no_fix \
  --epochs 5 \
  --batch-size 8 \
  --grad-accum 2 \
  --model-name "74dist_random_nofix_b8g2_${TIMESTAMP}" \
  --adapter-type lora+ \
  -y

echo "Waiting ${WAIT_MINUTES} minutes..."
sleep ${WAIT_SECONDS}

# 8/15: Our random shuffle (iter fix) + 74 distances
echo ""
echo "[8/15] Our random shuffle (iter fix) + 74 distances"
echo "========================================"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251209_dist_augment" \
  --sampler-type random \
  --epochs 5 \
  --batch-size 8 \
  --grad-accum 2 \
  --model-name "74dist_random_iterfix_b8g2_${TIMESTAMP}" \
  --adapter-type lora+ \
  -y

echo "Waiting ${WAIT_MINUTES} minutes..."
sleep ${WAIT_SECONDS}

# 9/15: Feature Balanced + 74 distances
echo ""
echo "[9/15] Feature Balanced + 74 distances"
echo "========================================"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251209_dist_augment" \
  --sampler-type feature_balanced \
  --epochs 5 \
  --batch-size 8 \
  --grad-accum 2 \
  --model-name "74dist_balanced_b8g2_${TIMESTAMP}" \
  --adapter-type lora+ \
  -y

echo "Waiting ${WAIT_MINUTES} minutes..."
sleep ${WAIT_SECONDS}

# 10/15: No shuffle + 74 distances
echo ""
echo "[10/15] No shuffle + 74 distances"
echo "========================================"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251209_dist_augment" \
  --sampler-type hf_sequential \
  --epochs 5 \
  --batch-size 8 \
  --grad-accum 2 \
  --model-name "74dist_sequential_b8g2_${TIMESTAMP}" \
  --adapter-type lora+ \
  -y

echo "Waiting ${WAIT_MINUTES} minutes..."
sleep ${WAIT_SECONDS}

#===============================================================================
# DATASET 3: _exp_20251209_dist_augment_x5_combined (124 samples, 3 epochs = 23.25 steps)
#===============================================================================

# 11/15: HF shuffle + 124 distances
echo ""
echo "[11/15] HF shuffle + 124 distances"
echo "========================================"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251209_dist_augment_x5_combined" \
  --sampler-type hf_shuffle \
  --epochs 3 \
  --batch-size 8 \
  --grad-accum 2 \
  --model-name "124dist_hf_shuffle_b8g2_${TIMESTAMP}" \
  --adapter-type lora+ \
  -y

echo "Waiting ${WAIT_MINUTES} minutes..."
sleep ${WAIT_SECONDS}

# 12/15: Our random shuffle + 124 distances
echo ""
echo "[12/15] Our random shuffle + 124 distances"
echo "========================================"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251209_dist_augment_x5_combined" \
  --sampler-type random_no_fix \
  --epochs 3 \
  --batch-size 8 \
  --grad-accum 2 \
  --model-name "124dist_random_nofix_b8g2_${TIMESTAMP}" \
  --adapter-type lora+ \
  -y

echo "Waiting ${WAIT_MINUTES} minutes..."
sleep ${WAIT_SECONDS}

# 13/15: Our random shuffle (iter fix) + 124 distances
echo ""
echo "[13/15] Our random shuffle (iter fix) + 124 distances"
echo "========================================"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251209_dist_augment_x5_combined" \
  --sampler-type random \
  --epochs 3 \
  --batch-size 8 \
  --grad-accum 2 \
  --model-name "124dist_random_iterfix_b8g2_${TIMESTAMP}" \
  --adapter-type lora+ \
  -y

echo "Waiting ${WAIT_MINUTES} minutes..."
sleep ${WAIT_SECONDS}

# 14/15: Feature Balanced + 124 distances
echo ""
echo "[14/15] Feature Balanced + 124 distances"
echo "========================================"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251209_dist_augment_x5_combined" \
  --sampler-type feature_balanced \
  --epochs 3 \
  --batch-size 8 \
  --grad-accum 2 \
  --model-name "124dist_balanced_b8g2_${TIMESTAMP}" \
  --adapter-type lora+ \
  -y

echo "Waiting ${WAIT_MINUTES} minutes..."
sleep ${WAIT_SECONDS}

# 15/15: No shuffle + 124 distances
echo ""
echo "[15/15] No shuffle + 124 distances"
echo "========================================"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
./run_experiment.sh \
  --skip-datasets \
  --dataset-name "_exp_20251209_dist_augment_x5_combined" \
  --sampler-type hf_sequential \
  --epochs 3 \
  --batch-size 8 \
  --grad-accum 2 \
  --model-name "124dist_sequential_b8g2_${TIMESTAMP}" \
  --adapter-type lora+ \
  -y

echo ""
echo "========================================"
echo "All 15 experiments completed!"
echo "========================================"
