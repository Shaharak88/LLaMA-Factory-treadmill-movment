#!/bin/bash
################################################################################
# Automated Experiment Runner with Tracking
#
# This script automates the complete workflow:
# 1. Syncs code from local PC to remote server
# 2. Builds train/test datasets in Docker container
# 3. Runs training pipeline with automatic tracking
# 4. Retrieves results (CSV, reports, models) back to local PC
#
# Usage:
#   ./run_experiment.sh [options]
#
# Example:
#   ./run_experiment.sh --texture subtle_gray_stripes --epochs 5
#
# Author: AI-Generated
# Date: 2025-11-30
################################################################################
################################################################################
#Important Notes!
#Even when creating diffrent test and train datasets always use the run_experiment.sh bash command! for example:
#./run_experiment.sh --train-angles "0.0,15.0" --test-angles "7.0,22.0,35.0,30.0,45.0,52.0" --texture subtle_gray_stripes --epochs 2 --speed-range "0.0,14.0"
#When you are running on the server read DEPLOYMENT.md!!!!
# When running in the server dont EVER kill a running container/process or anything that is running!!
# Always run this script in a container in this project!
# Always use a unique timestamp for every new dataset we are running
#For every change you make in this file or in this ENTIRE project : do that change, make sure no other changes are needed in the code, for every change you make document it in a log file here with the changes, and commit it and add a comment about the changes in the commit.
# DONT EVER RUN ANY SUB SCRIPT OTHER THEN THIS run_experiment.sh script!
#dont ever sync large model files!
#
# PARAMETER NAMING CLARIFICATION:
# - Command line flag: --train-speed (not --train-speed-range)
# - Command line flag: --test-speed (not --test-speed-range)
# - Internal variable names: TRAIN_SPEED_RANGE / TEST_SPEED_RANGE
# - Example: ./run_experiment.sh --train-speed "0.0,14.0" --test-speed "0.0,14.0"
#
# AUTO-CONFIRMATION:
# - The script prompts for confirmation before running (y/N prompt)
# - To skip the prompt and auto-confirm, add -y or --yes flag at the end
# - Example: ./run_experiment.sh --train-speed "0.0,14.0" --epochs 7 -y
#
# HOW TO UPDATE DOCKER CONTAINER FILES (building_dataset.py, synthetic_data_generation.py):
# The Docker container has baked-in versions of some Python files in /app/. After rsync syncs updated
# files to the server, you need to manually copy them into the container:
#   1. Files sync to: /home/seedoo/shahar_linux_wsl/LLaMA-Factory/
#   2. Container runs from: /app/
#   3. To update files in container (when building_dataset.py or synthetic_data_generation.py change):
#      ssh seedoo@hetzner-gpu.tail9e6e7.ts.net "docker cp /home/seedoo/shahar_linux_wsl/LLaMA-Factory/building_dataset.py llamafactory:/tmp/building_dataset.py && docker cp /home/seedoo/shahar_linux_wsl/LLaMA-Factory/data/synthetic_treadmill/synthetic_data_generation.py llamafactory:/tmp/synthetic_data_generation.py && docker exec llamafactory bash -c 'cat /tmp/building_dataset.py > /app/building_dataset.py && cat /tmp/synthetic_data_generation.py > /app/data/synthetic_treadmill/synthetic_data_generation.py && rm /tmp/building_dataset.py /tmp/synthetic_data_generation.py'"
#   Note: We copy to /tmp first then cat to overwrite because docker cp can't overwrite locked files directly.





set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

################################################################################
# CONFIGURATION - Modify these for your setup
################################################################################

# Server configuration
SERVER_USER="seedoo"
SERVER_HOST="hetzner-gpu.tail9e6e7.ts.net"
SERVER_SSH="${SERVER_USER}@${SERVER_HOST}"
REMOTE_HOST_DIR="/home/seedoo/shahar_linux_wsl/LLaMA-Factory"  # Host path for rsync
REMOTE_APP_DIR="/app"  # Container path for docker exec
CONTAINER_NAME="llamafactory"

# Local configuration
LOCAL_DIR="/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory"
RESULTS_DIR="${LOCAL_DIR}/experiment_results"

# Default experiment parameters (can be overridden by command line args)
# If TEST_* parameters are empty, they default to TRAIN_* values (same for both)

# Core video parameters
TRAIN_TEXTURE_TYPE="subtle_gray_stripes"
TEST_TEXTURE_TYPE=""  # If empty, uses TRAIN_TEXTURE_TYPE
TRAIN_VIEW_ANGLES="0.0,15.0,30.0,45.0"
TEST_VIEW_ANGLES=""  # If empty, uses TRAIN_VIEW_ANGLES
TRAIN_DIRECTION="left,right,up,down"
TEST_DIRECTION=""  # If empty, uses TRAIN_DIRECTION
TRAIN_SPEED_RANGE="0.0,14.0"
TEST_SPEED_RANGE=""  # If empty, uses TRAIN_SPEED_RANGE
TRAIN_RESOLUTION="640x480"
TEST_RESOLUTION=""
TRAIN_FPS="4"
TEST_FPS=""
TRAIN_DURATION="12.0"
TEST_DURATION=""

# Camera/Lighting parameters
TRAIN_BRIGHTNESS="0.0"
TEST_BRIGHTNESS=""
TRAIN_CONTRAST="1.0"
TEST_CONTRAST=""
TRAIN_LIGHTING_VARIATION="none"
TEST_LIGHTING_VARIATION=""
TRAIN_LIGHTING_INTENSITY="1.0"
TEST_LIGHTING_INTENSITY=""
TRAIN_MOTION_BLUR="0"
TEST_MOTION_BLUR=""
TRAIN_CAMERA_NOISE="0.0"
TEST_CAMERA_NOISE=""
TRAIN_EDGE_WIDTH="0.1"  # Fixed: was "5" (500%!), now 0.1 (10% - normal edge width)
TEST_EDGE_WIDTH=""
TRAIN_DISTANCE="1.0"  # Distance factor: 1.0 = normal size, >1.0 = smaller/further away
TEST_DISTANCE=""

# Stripe parameters (for subtle_gray_stripes)
TRAIN_STRIPE_WIDTH="10"
TEST_STRIPE_WIDTH=""
TRAIN_STRIPE_SPACING="60"
TEST_STRIPE_SPACING=""
TRAIN_STRIPE_GRAY="20"
TEST_STRIPE_GRAY="15"
TRAIN_BG_GRAY="15"
TEST_BG_GRAY="10"
TRAIN_STRIPE_DISTANCE_VARIANCE="0.0"
TEST_STRIPE_DISTANCE_VARIANCE=""

# Object parameters
TRAIN_ADD_OBJECT="false"
TEST_ADD_OBJECT=""
TRAIN_OBJECT_TYPE=""
TEST_OBJECT_TYPE=""
TRAIN_OBJECT_POSITION=""
TEST_OBJECT_POSITION=""
TRAIN_OBJECT_SIZE=""
TEST_OBJECT_SIZE=""
TRAIN_NUM_OBJECTS=""
TEST_NUM_OBJECTS=""

# Blur parameters
TRAIN_ADD_BLUR="false"
TEST_ADD_BLUR=""
TRAIN_BLUR_TYPE=""
TEST_BLUR_TYPE=""
TRAIN_BLUR_INTENSITY=""
TEST_BLUR_INTENSITY=""
TRAIN_RANDOM_BLUR_VARIATION=""
TEST_RANDOM_BLUR_VARIATION=""

# Other
TRAIN_VARY_PARAMETERS="false"
TEST_VARY_PARAMETERS=""

# Training hyperparameters
NUM_EPOCHS="5"
LORA_RANK="8"
LORA_ALPHA="16"
LEARNING_RATE="5e-5"
BATCH_SIZE="1"
GRAD_ACCUMULATION="8"
SAVE_STEPS="100"
EVAL_VIDEO_FPS="4"
EVAL_VIDEO_MAXLEN="128"

# Flags
EXECUTION_MODE="remote"  # "local" or "remote"
DRY_RUN=false
SKIP_SYNC=false
SKIP_DATASETS=false
SKIP_TRAINING=false
RETRIEVE_MODELS=false
VERBOSE=false
YES_TO_ALL=false
USE_DORA=false
DATASET_NAME=""  # Can be set via --dataset-name to reuse existing datasets
EVAL_METHOD="yesno"  # Evaluation method: "yesno" or "moving_stopped"
MODEL_NAME=""  # Custom name for the trained model (optional)
EVAL_MODEL_PATH=""  # Path to existing model for re-evaluation (used with --skip-training)

################################################################################
# COLOR OUTPUT
################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}\n"
}

################################################################################
# USAGE
################################################################################

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Automated experiment runner with tracking for LLaMA-Factory.

OPTIONS:
    -h, --help              Show this help message
    -d, --dry-run           Show what would be executed without running
    --local                 Run on local PC (default: run on remote server)
    --skip-sync             Skip syncing code to server
    --skip-datasets         Skip dataset generation (use existing datasets)
    --skip-training         Skip model training (only run evaluation on existing model)
    --dataset-name NAME     Specify existing dataset name (e.g., _exp_20251202_194605)
                            Use with --skip-datasets to train on existing datasets
    --retrieve-models       Also retrieve trained model files
    --use-dora              Use DoRA (Weight-Decomposed LoRA) instead of standard LoRA
    --eval-method METHOD    Evaluation method: "yesno" or "moving_stopped" (default: yesno)
    --model-name NAME       Custom name for the trained model (saves to saves/<NAME>)
    --eval-model-path PATH  Path to existing model for re-evaluation (use with --skip-training)
                            Example: saves/my_custom_model
    -v, --verbose           Verbose output

EXPERIMENT PARAMETERS:
    # Training dataset parameters
    --train-texture TYPE    Training texture type(s) (default: subtle_gray_stripes)
    --train-angles ANGLES   Training view angles (default: 0.0,15.0,30.0,45.0)
    --train-direction DIRS  Training directions (default: left,right,up,down)
    --train-speed SPEEDS    Training speed range (default: 0.0,14.0)
    --train-bg-gray VALS    Training background gray (default: 15,16,17)
    --train-stripe-gray V   Training stripe gray (default: 20,21,22)

    # Test dataset parameters (if empty, uses train values)
    --test-texture TYPE     Test texture type(s) (default: same as train)
    --test-angles ANGLES    Test view angles (default: same as train)
    --test-direction DIRS   Test directions (default: same as train)
    --test-speed SPEEDS     Test speed range (default: same as train)
    --test-bg-gray VALS     Test background gray (default: 10,11,12)
    --test-stripe-gray V    Test stripe gray (default: 15,16,17)

    # Object placement parameters
    --train-add-object BOOL Train: Enable object placement (true/false, default: false)
    --train-object-type T   Train: Object type (box/circle/random, default: box)
    --train-object-position Train: Object position (center/left/right/random)
    --train-object-size S   Train: Object size (small/medium/large or numeric)
    --train-num-objects N   Train: Number of objects (default: 1)
    --test-add-object BOOL  Test: Enable object placement (true/false, default: false)
    --test-object-type T    Test: Object type (box/circle/random, default: box)
    --test-object-position  Test: Object position (center/left/right/random)
    --test-object-size S    Test: Object size (small/medium/large or numeric)
    --test-num-objects N    Test: Number of objects (default: 1)

    # Camera blur parameters
    --train-add-blur BOOL   Train: Enable camera blur (true/false, default: false)
    --train-blur-type T     Train: Blur type (motion/gaussian/random, default: gaussian)
    --train-blur-intensity  Train: Blur intensity (light/medium/heavy or 0.0-1.0)
    --train-random-blur-variation  Train: Random blur variation (true/false)
    --test-add-blur BOOL    Test: Enable camera blur (true/false, default: false)
    --test-blur-type T      Test: Blur type (motion/gaussian/random, default: gaussian)
    --test-blur-intensity   Test: Blur intensity (light/medium/heavy or 0.0-1.0)
    --test-random-blur-variation   Test: Random blur variation (true/false)

    # Common parameters (both train and test)
    --texture TYPE          Set both train and test texture (shortcut)
    --angles ANGLES         Set both train and test angles (shortcut)
    --direction DIRS        Set both train and test directions (shortcut)
    --speed-range SPEEDS    Set both train and test speed (shortcut)
    --fps FPS               FPS (default: 4)
    --duration SECS         Duration (default: 12.0)
    --epochs N              Training epochs (default: 5)
    --lora-rank N           LoRA rank (default: 8)
    --lora-alpha N          LoRA alpha (default: 16)
    --learning-rate RATE    Learning rate (default: 5e-5)
    --batch-size N          Batch size (default: 1)
    --grad-accum N          Gradient accumulation (default: 8)

EXAMPLES:
    # Basic run with defaults (same train/test)
    $0

    # Custom texture with 10 epochs (same for train/test)
    $0 --texture factory_dark --epochs 10

    # DIFFERENT textures for train vs test
    $0 --train-texture "stripes,noise" --test-texture "factory_dark" --epochs 5

    # DIFFERENT angles for train vs test
    $0 --train-angles "0.0,15.0,30.0" --test-angles "22.5,45.0,52.0" --epochs 5

    # Train on easy angles, test on hard angles
    $0 --texture subtle_gray_stripes \
       --train-angles "0.0,15.0" \
       --test-angles "45.0,52.0,60.0" \
       --epochs 10

    # Train on multiple textures, test on unseen texture
    $0 --train-texture "stripes,noise,rubber" \
       --test-texture "factory_dark" \
       --train-angles "0.0,15.0,30.0" \
       --test-angles "0.0,22.5,52.0" \
       --epochs 5

    # Reuse existing dataset with different training parameters
    $0 --skip-datasets --dataset-name "_exp_20251202_194605" \
       --epochs 7 --batch-size 14 --grad-accum 1

    # Re-evaluate existing model with different evaluation method (skip training)
    $0 --skip-datasets --skip-training --dataset-name "_exp_20251202_194605" \
       --eval-method moving_stopped

    # Dry run to see what would happen
    $0 --dry-run

EOF
}

################################################################################
# PARSE ARGUMENTS
################################################################################

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -y|--yes)
                YES_TO_ALL=true
                shift
                ;;
            --local)
                EXECUTION_MODE="local"
                shift
                ;;
            --skip-sync)
                SKIP_SYNC=true
                shift
                ;;
            --skip-datasets)
                SKIP_DATASETS=true
                shift
                ;;
            --skip-training)
                SKIP_TRAINING=true
                shift
                ;;
            --dataset-name)
                DATASET_NAME="$2"
                shift 2
                ;;
            --retrieve-models)
                RETRIEVE_MODELS=true
                shift
                ;;
            --use-dora)
                USE_DORA=true
                shift
                ;;
            --eval-method)
                EVAL_METHOD="$2"
                shift 2
                ;;
            --model-name)
                MODEL_NAME="$2"
                shift 2
                ;;
            --eval-model-path)
                EVAL_MODEL_PATH="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                set -x
                shift
                ;;
            # Shortcuts - set both train and test
            --texture)
                TRAIN_TEXTURE_TYPE="$2"
                TEST_TEXTURE_TYPE="$2"
                shift 2
                ;;
            --angles)
                TRAIN_VIEW_ANGLES="$2"
                TEST_VIEW_ANGLES="$2"
                shift 2
                ;;
            --direction)
                TRAIN_DIRECTION="$2"
                TEST_DIRECTION="$2"
                shift 2
                ;;
            --speed-range)
                TRAIN_SPEED_RANGE="$2"
                TEST_SPEED_RANGE="$2"
                shift 2
                ;;
            # Training-specific
            --train-texture)
                TRAIN_TEXTURE_TYPE="$2"
                shift 2
                ;;
            --train-angles)
                TRAIN_VIEW_ANGLES="$2"
                shift 2
                ;;
            --train-direction)
                TRAIN_DIRECTION="$2"
                shift 2
                ;;
            --train-speed)
                TRAIN_SPEED_RANGE="$2"
                shift 2
                ;;
            --train-bg-gray)
                TRAIN_BG_GRAY="$2"
                shift 2
                ;;
            --train-stripe-gray)
                TRAIN_STRIPE_GRAY="$2"
                shift 2
                ;;
            # Test-specific
            --test-texture)
                TEST_TEXTURE_TYPE="$2"
                shift 2
                ;;
            --test-angles)
                TEST_VIEW_ANGLES="$2"
                shift 2
                ;;
            --test-direction)
                TEST_DIRECTION="$2"
                shift 2
                ;;
            --test-speed)
                TEST_SPEED_RANGE="$2"
                shift 2
                ;;
            --test-bg-gray)
                TEST_BG_GRAY="$2"
                shift 2
                ;;
            --test-stripe-gray)
                TEST_STRIPE_GRAY="$2"
                shift 2
                ;;
            --train-distance)
                TRAIN_DISTANCE="$2"
                shift 2
                ;;
            --test-distance)
                TEST_DISTANCE="$2"
                shift 2
                ;;
            --fps)
                FPS="$2"
                shift 2
                ;;
            --duration)
                DURATION="$2"
                shift 2
                ;;
            --epochs)
                NUM_EPOCHS="$2"
                shift 2
                ;;
            --lora-rank)
                LORA_RANK="$2"
                shift 2
                ;;
            --lora-alpha)
                LORA_ALPHA="$2"
                shift 2
                ;;
            --learning-rate)
                LEARNING_RATE="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --grad-accum)
                GRAD_ACCUMULATION="$2"
                shift 2
                ;;
            # Object parameters
            --train-add-object)
                TRAIN_ADD_OBJECT="$2"
                shift 2
                ;;
            --train-object-type)
                TRAIN_OBJECT_TYPE="$2"
                shift 2
                ;;
            --train-object-position)
                TRAIN_OBJECT_POSITION="$2"
                shift 2
                ;;
            --train-object-size)
                TRAIN_OBJECT_SIZE="$2"
                shift 2
                ;;
            --train-num-objects)
                TRAIN_NUM_OBJECTS="$2"
                shift 2
                ;;
            --test-add-object)
                TEST_ADD_OBJECT="$2"
                shift 2
                ;;
            --test-object-type)
                TEST_OBJECT_TYPE="$2"
                shift 2
                ;;
            --test-object-position)
                TEST_OBJECT_POSITION="$2"
                shift 2
                ;;
            --test-object-size)
                TEST_OBJECT_SIZE="$2"
                shift 2
                ;;
            --test-num-objects)
                TEST_NUM_OBJECTS="$2"
                shift 2
                ;;
            # Blur parameters
            --train-add-blur)
                TRAIN_ADD_BLUR="$2"
                shift 2
                ;;
            --train-blur-type)
                TRAIN_BLUR_TYPE="$2"
                shift 2
                ;;
            --train-blur-intensity)
                TRAIN_BLUR_INTENSITY="$2"
                shift 2
                ;;
            --train-random-blur-variation)
                TRAIN_RANDOM_BLUR_VARIATION="$2"
                shift 2
                ;;
            --test-add-blur)
                TEST_ADD_BLUR="$2"
                shift 2
                ;;
            --test-blur-type)
                TEST_BLUR_TYPE="$2"
                shift 2
                ;;
            --test-blur-intensity)
                TEST_BLUR_INTENSITY="$2"
                shift 2
                ;;
            --test-random-blur-variation)
                TEST_RANDOM_BLUR_VARIATION="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

################################################################################
# HELPER FUNCTIONS
################################################################################

run_cmd() {
    local cmd="$1"
    local description="$2"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would execute: $cmd"
        return 0
    fi

    if [ -n "$description" ]; then
        log_info "$description"
    fi

    if ! eval "$cmd"; then
        log_error "Command failed: $cmd"
        return 1
    fi

    return 0
}

check_ssh_connection() {
    log_info "Checking SSH connection to $SERVER_SSH..."
    if ! ssh -o ConnectTimeout=5 "$SERVER_SSH" "echo 'Connection successful'" > /dev/null 2>&1; then
        log_error "Cannot connect to $SERVER_SSH"
        log_error "Please check your SSH configuration and network connection"
        exit 1
    fi
    log_success "SSH connection OK"
}

check_docker_container() {
    log_info "Checking Docker container '$CONTAINER_NAME'..."
    if ! ssh "$SERVER_SSH" "docker ps --format '{{.Names}}' | grep -q '^${CONTAINER_NAME}$'"; then
        log_error "Docker container '$CONTAINER_NAME' is not running"
        log_error "Please start the container first"
        exit 1
    fi
    log_success "Docker container is running"
}

generate_dataset_name() {
    # Generate unique dataset name with timestamp
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local texture_short=$(echo "$TEXTURE_TYPE" | cut -d',' -f1 | sed 's/[^a-zA-Z0-9_]//g')
    echo "${texture_short}_exp_${timestamp}"
}

################################################################################
# MAIN WORKFLOW STEPS
################################################################################

step_sync_code() {
    log_step "STEP 1: Syncing code to server"

    if [ "$EXECUTION_MODE" = "local" ]; then
        log_warning "Running in LOCAL mode - skipping code sync"
        return 0
    fi

    if [ "$SKIP_SYNC" = true ]; then
        log_warning "Skipping code sync (--skip-sync)"
        return 0
    fi

    log_info "Syncing from: $LOCAL_DIR"
    log_info "Syncing to: $SERVER_SSH:$REMOTE_HOST_DIR"

    # Use --checksum to ensure files are synced based on content, not just timestamps
    # This prevents sync issues where local files have older timestamps than server files
    run_cmd "rsync -avz --checksum --progress \
        --exclude='data/' \
        --exclude='saves/' \
        --exclude='output/' \
        --exclude='hf_cache/' \
        --exclude='*.mp4' \
        --exclude='*.avi' \
        --exclude='*.mkv' \
        --exclude='__pycache__/' \
        --exclude='.git/' \
        --exclude='experiments_log.csv' \
        --exclude='experiment_results/' \
        --exclude='evaluation_results_*/' \
        --exclude='*.log' \
        '$LOCAL_DIR/' '$SERVER_SSH:$REMOTE_HOST_DIR/'" \
        "Syncing code files..."

    log_success "Code sync complete"
}

step_build_datasets() {
    log_step "STEP 2: Building datasets"

    # If dataset name is already provided, skip generation
    if [ -n "$DATASET_NAME" ]; then
        log_warning "Using existing dataset: $DATASET_NAME"
        log_info "Skipping dataset generation (dataset name provided via --dataset-name)"
        return 0
    fi

    if [ "$SKIP_DATASETS" = true ]; then
        log_warning "Skipping dataset generation (--skip-datasets)"
        return 0
    fi

    # Use test parameters if specified, otherwise default to train parameters
    local test_texture="${TEST_TEXTURE_TYPE:-$TRAIN_TEXTURE_TYPE}"
    local test_angles="${TEST_VIEW_ANGLES:-$TRAIN_VIEW_ANGLES}"
    local test_direction="${TEST_DIRECTION:-$TRAIN_DIRECTION}"
    local test_speed="${TEST_SPEED_RANGE:-$TRAIN_SPEED_RANGE}"
    local test_resolution="${TEST_RESOLUTION:-$TRAIN_RESOLUTION}"
    local test_fps="${TEST_FPS:-$TRAIN_FPS}"
    local test_duration="${TEST_DURATION:-$TRAIN_DURATION}"
    local test_brightness="${TEST_BRIGHTNESS:-$TRAIN_BRIGHTNESS}"
    local test_contrast="${TEST_CONTRAST:-$TRAIN_CONTRAST}"
    local test_lighting_variation="${TEST_LIGHTING_VARIATION:-$TRAIN_LIGHTING_VARIATION}"
    local test_lighting_intensity="${TEST_LIGHTING_INTENSITY:-$TRAIN_LIGHTING_INTENSITY}"
    local test_motion_blur="${TEST_MOTION_BLUR:-$TRAIN_MOTION_BLUR}"
    local test_camera_noise="${TEST_CAMERA_NOISE:-$TRAIN_CAMERA_NOISE}"
    local test_edge_width="${TEST_EDGE_WIDTH:-$TRAIN_EDGE_WIDTH}"
    local test_distance="${TEST_DISTANCE:-$TRAIN_DISTANCE}"
    local test_stripe_width="${TEST_STRIPE_WIDTH:-$TRAIN_STRIPE_WIDTH}"
    local test_stripe_spacing="${TEST_STRIPE_SPACING:-$TRAIN_STRIPE_SPACING}"
    local test_stripe_distance_variance="${TEST_STRIPE_DISTANCE_VARIANCE:-$TRAIN_STRIPE_DISTANCE_VARIANCE}"
    local test_add_object="${TEST_ADD_OBJECT:-$TRAIN_ADD_OBJECT}"
    local test_object_type="${TEST_OBJECT_TYPE:-$TRAIN_OBJECT_TYPE}"
    local test_object_position="${TEST_OBJECT_POSITION:-$TRAIN_OBJECT_POSITION}"
    local test_object_size="${TEST_OBJECT_SIZE:-$TRAIN_OBJECT_SIZE}"
    local test_num_objects="${TEST_NUM_OBJECTS:-$TRAIN_NUM_OBJECTS}"
    local test_add_blur="${TEST_ADD_BLUR:-$TRAIN_ADD_BLUR}"
    local test_blur_type="${TEST_BLUR_TYPE:-$TRAIN_BLUR_TYPE}"
    local test_blur_intensity="${TEST_BLUR_INTENSITY:-$TRAIN_BLUR_INTENSITY}"
    local test_random_blur_variation="${TEST_RANDOM_BLUR_VARIATION:-$TRAIN_RANDOM_BLUR_VARIATION}"
    local test_vary_parameters="${TEST_VARY_PARAMETERS:-$TRAIN_VARY_PARAMETERS}"

    # Generate unique dataset name
    DATASET_NAME=$(generate_dataset_name)
    log_info "Dataset name: $DATASET_NAME"

    # Check if train and test are different
    if [ "$test_texture" != "$TRAIN_TEXTURE_TYPE" ] || \
       [ "$test_angles" != "$TRAIN_VIEW_ANGLES" ] || \
       [ "$test_direction" != "$TRAIN_DIRECTION" ] || \
       [ "$test_speed" != "$TRAIN_SPEED_RANGE" ]; then
        log_warning "Train and test datasets have DIFFERENT parameters:"
        log_info "  Train texture: $TRAIN_TEXTURE_TYPE | Test texture: $test_texture"
        log_info "  Train angles: $TRAIN_VIEW_ANGLES | Test angles: $test_angles"
    fi

    # Build training dataset - construct command with all parameters
    log_info "Building TRAINING dataset..."

    # Base python command (different for local vs remote)
    if [ "$EXECUTION_MODE" = "local" ]; then
        local train_base_cmd="docker exec $CONTAINER_NAME python3 $REMOTE_APP_DIR/building_dataset.py"
    else
        local train_base_cmd="docker exec $CONTAINER_NAME python3 $REMOTE_APP_DIR/building_dataset.py"
    fi

    local train_cmd="$train_base_cmd \
        --dataset_name ${DATASET_NAME}_train \
        --texture_type '$TRAIN_TEXTURE_TYPE' \
        --direction '$TRAIN_DIRECTION' \
        --view_angle '$TRAIN_VIEW_ANGLES' \
        --speed_range '$TRAIN_SPEED_RANGE' \
        --resolution '$TRAIN_RESOLUTION' \
        --fps '$TRAIN_FPS' \
        --duration '$TRAIN_DURATION' \
        --brightness '$TRAIN_BRIGHTNESS' \
        --contrast '$TRAIN_CONTRAST' \
        --lighting_variation '$TRAIN_LIGHTING_VARIATION' \
        --lighting_intensity '$TRAIN_LIGHTING_INTENSITY' \
        --motion_blur '$TRAIN_MOTION_BLUR' \
        --camera_noise '$TRAIN_CAMERA_NOISE' \
        --edge_width '$TRAIN_EDGE_WIDTH' \
        --distance '$TRAIN_DISTANCE' \
        --stripe_width '$TRAIN_STRIPE_WIDTH' \
        --stripe_spacing '$TRAIN_STRIPE_SPACING' \
        --stripe_gray '$TRAIN_STRIPE_GRAY' \
        --background_gray '$TRAIN_BG_GRAY' \
        --stripe_distance_variance '$TRAIN_STRIPE_DISTANCE_VARIANCE'"

    # Add vary_parameters flag if true
    if [ "$TRAIN_VARY_PARAMETERS" = "true" ]; then
        train_cmd="$train_cmd --vary_parameters"
    fi

    # Add object parameters if specified
    if [ "$TRAIN_ADD_OBJECT" = "true" ] && [ -n "$TRAIN_OBJECT_TYPE" ]; then
        train_cmd="$train_cmd --add_object --object_type '$TRAIN_OBJECT_TYPE'"
        [ -n "$TRAIN_OBJECT_POSITION" ] && train_cmd="$train_cmd --object_position '$TRAIN_OBJECT_POSITION'"
        [ -n "$TRAIN_OBJECT_SIZE" ] && train_cmd="$train_cmd --object_size '$TRAIN_OBJECT_SIZE'"
        [ -n "$TRAIN_NUM_OBJECTS" ] && train_cmd="$train_cmd --num_objects '$TRAIN_NUM_OBJECTS'"
    fi

    # Add blur parameters if specified
    if [ "$TRAIN_ADD_BLUR" = "true" ] && [ -n "$TRAIN_BLUR_TYPE" ]; then
        train_cmd="$train_cmd --add_blur --blur_type '$TRAIN_BLUR_TYPE'"
        [ -n "$TRAIN_BLUR_INTENSITY" ] && train_cmd="$train_cmd --blur_intensity '$TRAIN_BLUR_INTENSITY'"
        [ -n "$TRAIN_RANDOM_BLUR_VARIATION" ] && train_cmd="$train_cmd --random_blur_variation '$TRAIN_RANDOM_BLUR_VARIATION'"
    fi

    # Execute command (with or without SSH)
    if [ "$EXECUTION_MODE" = "local" ]; then
        run_cmd "$train_cmd" "Running training dataset generation..."
    else
        run_cmd "ssh '$SERVER_SSH' '$train_cmd'" "Running training dataset generation..."
    fi
    log_success "Training dataset created"

    # Build test dataset with potentially different parameters
    log_info "Building TEST dataset..."

    # Base python command (different for local vs remote)
    if [ "$EXECUTION_MODE" = "local" ]; then
        local test_base_cmd="docker exec $CONTAINER_NAME python3 $REMOTE_APP_DIR/building_dataset.py"
    else
        local test_base_cmd="docker exec $CONTAINER_NAME python3 $REMOTE_APP_DIR/building_dataset.py"
    fi

    local test_cmd="$test_base_cmd \
        --dataset_name ${DATASET_NAME}_test \
        --texture_type '$test_texture' \
        --direction '$test_direction' \
        --view_angle '$test_angles' \
        --speed_range '$test_speed' \
        --resolution '$test_resolution' \
        --fps '$test_fps' \
        --duration '$test_duration' \
        --brightness '$test_brightness' \
        --contrast '$test_contrast' \
        --lighting_variation '$test_lighting_variation' \
        --lighting_intensity '$test_lighting_intensity' \
        --motion_blur '$test_motion_blur' \
        --camera_noise '$test_camera_noise' \
        --edge_width '$test_edge_width' \
        --distance '$test_distance' \
        --stripe_width '$test_stripe_width' \
        --stripe_spacing '$test_stripe_spacing' \
        --stripe_gray '$TEST_STRIPE_GRAY' \
        --background_gray '$TEST_BG_GRAY' \
        --stripe_distance_variance '$test_stripe_distance_variance'"

    # Add vary_parameters flag if true
    if [ "$test_vary_parameters" = "true" ]; then
        test_cmd="$test_cmd --vary_parameters"
    fi

    # Add object parameters if specified
    if [ "$test_add_object" = "true" ] && [ -n "$test_object_type" ]; then
        test_cmd="$test_cmd --add_object --object_type '$test_object_type'"
        [ -n "$test_object_position" ] && test_cmd="$test_cmd --object_position '$test_object_position'"
        [ -n "$test_object_size" ] && test_cmd="$test_cmd --object_size '$test_object_size'"
        [ -n "$test_num_objects" ] && test_cmd="$test_cmd --num_objects '$test_num_objects'"
    fi

    # Add blur parameters if specified
    if [ "$test_add_blur" = "true" ] && [ -n "$test_blur_type" ]; then
        test_cmd="$test_cmd --add_blur --blur_type '$test_blur_type'"
        [ -n "$test_blur_intensity" ] && test_cmd="$test_cmd --blur_intensity '$test_blur_intensity'"
        [ -n "$test_random_blur_variation" ] && test_cmd="$test_cmd --random_blur_variation '$test_random_blur_variation'"
    fi

    # Execute command (with or without SSH)
    if [ "$EXECUTION_MODE" = "local" ]; then
        run_cmd "$test_cmd" "Running test dataset generation..."
    else
        run_cmd "ssh '$SERVER_SSH' '$test_cmd'" "Running test dataset generation..."
    fi
    log_success "Test dataset created"

    log_success "Datasets built: ${DATASET_NAME}_train and ${DATASET_NAME}_test"
}

step_run_training() {
    log_step "STEP 3: Running training pipeline with tracking"

    if [ -z "$DATASET_NAME" ]; then
        log_error "No dataset name available. Cannot run training."
        log_error "Either skip this step or run dataset generation first."
        exit 1
    fi

    log_info "Training with dataset: $DATASET_NAME"
    log_info "Epochs: $NUM_EPOCHS, LoRA rank: $LORA_RANK, Learning rate: $LEARNING_RATE"

    # Use test parameters if specified, otherwise default to train parameters
    local test_texture="${TEST_TEXTURE_TYPE:-$TRAIN_TEXTURE_TYPE}"
    local test_angles="${TEST_VIEW_ANGLES:-$TRAIN_VIEW_ANGLES}"
    local test_direction="${TEST_DIRECTION:-$TRAIN_DIRECTION}"
    local test_speed="${TEST_SPEED_RANGE:-$TRAIN_SPEED_RANGE}"
    local test_resolution="${TEST_RESOLUTION:-$TRAIN_RESOLUTION}"
    local test_fps="${TEST_FPS:-$TRAIN_FPS}"
    local test_duration="${TEST_DURATION:-$TRAIN_DURATION}"
    local test_brightness="${TEST_BRIGHTNESS:-$TRAIN_BRIGHTNESS}"
    local test_contrast="${TEST_CONTRAST:-$TRAIN_CONTRAST}"
    local test_lighting_variation="${TEST_LIGHTING_VARIATION:-$TRAIN_LIGHTING_VARIATION}"
    local test_lighting_intensity="${TEST_LIGHTING_INTENSITY:-$TRAIN_LIGHTING_INTENSITY}"
    local test_motion_blur="${TEST_MOTION_BLUR:-$TRAIN_MOTION_BLUR}"
    local test_camera_noise="${TEST_CAMERA_NOISE:-$TRAIN_CAMERA_NOISE}"
    local test_edge_width="${TEST_EDGE_WIDTH:-$TRAIN_EDGE_WIDTH}"
    local test_distance="${TEST_DISTANCE:-$TRAIN_DISTANCE}"
    local test_stripe_width="${TEST_STRIPE_WIDTH:-$TRAIN_STRIPE_WIDTH}"
    local test_stripe_spacing="${TEST_STRIPE_SPACING:-$TRAIN_STRIPE_SPACING}"
    local test_stripe_gray="${TEST_STRIPE_GRAY}"
    local test_bg_gray="${TEST_BG_GRAY}"
    local test_stripe_distance_variance="${TEST_STRIPE_DISTANCE_VARIANCE:-$TRAIN_STRIPE_DISTANCE_VARIANCE}"
    local test_add_object="${TEST_ADD_OBJECT:-$TRAIN_ADD_OBJECT}"
    local test_object_type="${TEST_OBJECT_TYPE:-$TRAIN_OBJECT_TYPE}"
    local test_object_position="${TEST_OBJECT_POSITION:-$TRAIN_OBJECT_POSITION}"
    local test_object_size="${TEST_OBJECT_SIZE:-$TRAIN_OBJECT_SIZE}"
    local test_num_objects="${TEST_NUM_OBJECTS:-$TRAIN_NUM_OBJECTS}"
    local test_add_blur="${TEST_ADD_BLUR:-$TRAIN_ADD_BLUR}"
    local test_blur_type="${TEST_BLUR_TYPE:-$TRAIN_BLUR_TYPE}"
    local test_blur_intensity="${TEST_BLUR_INTENSITY:-$TRAIN_BLUR_INTENSITY}"
    local test_random_blur_variation="${TEST_RANDOM_BLUR_VARIATION:-$TRAIN_RANDOM_BLUR_VARIATION}"
    local test_vary_parameters="${TEST_VARY_PARAMETERS:-$TRAIN_VARY_PARAMETERS}"

    # Construct command with all training, evaluation, and METADATA parameters
    # Base command (different for local vs remote)
    if [ "$EXECUTION_MODE" = "local" ]; then
        local pipeline_base_cmd="docker exec $CONTAINER_NAME python3 $REMOTE_APP_DIR/run_full_pipeline.py"
    else
        local pipeline_base_cmd="docker exec $CONTAINER_NAME python3 $REMOTE_APP_DIR/run_full_pipeline.py"
    fi

    local train_pipeline_cmd="$pipeline_base_cmd \
        --dataset_name '$DATASET_NAME' \
        --skip_dataset \
        --num_train_epochs '$NUM_EPOCHS' \
        --lora_rank '$LORA_RANK' \
        --lora_alpha '$LORA_ALPHA' \
        --learning_rate '$LEARNING_RATE' \
        --per_device_train_batch_size '$BATCH_SIZE' \
        --gradient_accumulation_steps '$GRAD_ACCUMULATION' \
        --save_steps '$SAVE_STEPS' \
        --eval_video_fps '$EVAL_VIDEO_FPS' \
        --eval_video_maxlen '$EVAL_VIDEO_MAXLEN'"

    # Add use_dora flag if enabled
    if [ "$USE_DORA" = true ]; then
        train_pipeline_cmd="$train_pipeline_cmd --use_dora"
    fi

    # Add skip_training flag if enabled
    if [ "$SKIP_TRAINING" = true ]; then
        train_pipeline_cmd="$train_pipeline_cmd --skip_training"
    fi

    # Add model_name if provided
    if [ -n "$MODEL_NAME" ]; then
        train_pipeline_cmd="$train_pipeline_cmd --model_name '$MODEL_NAME'"
    fi

    # Add eval_model_path if provided (for re-evaluation with existing model)
    if [ -n "$EVAL_MODEL_PATH" ]; then
        train_pipeline_cmd="$train_pipeline_cmd --eval_model_path '$EVAL_MODEL_PATH'"
    fi

    # Add eval_method parameter
    train_pipeline_cmd="$train_pipeline_cmd --eval_method '$EVAL_METHOD'"

    train_pipeline_cmd="$train_pipeline_cmd \
        --train_texture_type '$TRAIN_TEXTURE_TYPE' \
        --train_direction '$TRAIN_DIRECTION' \
        --train_view_angle '$TRAIN_VIEW_ANGLES' \
        --train_speed_range '$TRAIN_SPEED_RANGE' \
        --train_resolution '$TRAIN_RESOLUTION' \
        --train_fps '$TRAIN_FPS' \
        --train_duration '$TRAIN_DURATION' \
        --train_brightness '$TRAIN_BRIGHTNESS' \
        --train_contrast '$TRAIN_CONTRAST' \
        --train_lighting_variation '$TRAIN_LIGHTING_VARIATION' \
        --train_lighting_intensity '$TRAIN_LIGHTING_INTENSITY' \
        --train_motion_blur '$TRAIN_MOTION_BLUR' \
        --train_camera_noise '$TRAIN_CAMERA_NOISE' \
        --train_edge_width '$TRAIN_EDGE_WIDTH' \
        --train_distance '$TRAIN_DISTANCE' \
        --train_stripe_width '$TRAIN_STRIPE_WIDTH' \
        --train_stripe_spacing '$TRAIN_STRIPE_SPACING' \
        --train_stripe_gray '$TRAIN_STRIPE_GRAY' \
        --train_background_gray '$TRAIN_BG_GRAY' \
        --train_stripe_distance_variance '$TRAIN_STRIPE_DISTANCE_VARIANCE' \
        --train_add_object '$TRAIN_ADD_OBJECT' \
        --train_object_type '$TRAIN_OBJECT_TYPE' \
        --train_object_position '$TRAIN_OBJECT_POSITION' \
        --train_object_size '$TRAIN_OBJECT_SIZE' \
        --train_num_objects '$TRAIN_NUM_OBJECTS' \
        --train_add_blur '$TRAIN_ADD_BLUR' \
        --train_blur_type '$TRAIN_BLUR_TYPE' \
        --train_blur_intensity '$TRAIN_BLUR_INTENSITY' \
        --train_random_blur_variation '$TRAIN_RANDOM_BLUR_VARIATION' \
        --train_vary_parameters '$TRAIN_VARY_PARAMETERS' \
        --test_texture_type '$test_texture' \
        --test_direction '$test_direction' \
        --test_view_angle '$test_angles' \
        --test_speed_range '$test_speed' \
        --test_resolution '$test_resolution' \
        --test_fps '$test_fps' \
        --test_duration '$test_duration' \
        --test_brightness '$test_brightness' \
        --test_contrast '$test_contrast' \
        --test_lighting_variation '$test_lighting_variation' \
        --test_lighting_intensity '$test_lighting_intensity' \
        --test_motion_blur '$test_motion_blur' \
        --test_camera_noise '$test_camera_noise' \
        --test_edge_width '$test_edge_width' \
        --test_distance '$test_distance' \
        --test_stripe_width '$test_stripe_width' \
        --test_stripe_spacing '$test_stripe_spacing' \
        --test_stripe_gray '$test_stripe_gray' \
        --test_background_gray '$test_bg_gray' \
        --test_stripe_distance_variance '$test_stripe_distance_variance' \
        --test_add_object '$test_add_object' \
        --test_object_type '$test_object_type' \
        --test_object_position '$test_object_position' \
        --test_object_size '$test_object_size' \
        --test_num_objects '$test_num_objects' \
        --test_add_blur '$test_add_blur' \
        --test_blur_type '$test_blur_type' \
        --test_blur_intensity '$test_blur_intensity' \
        --test_random_blur_variation '$test_random_blur_variation' \
        --test_vary_parameters '$test_vary_parameters'"

    # Execute command (with or without SSH)
    if [ "$EXECUTION_MODE" = "local" ]; then
        run_cmd "$train_pipeline_cmd" "Running training and evaluation..."
        log_info "Experiment has been logged to experiments_log.csv locally"
    else
        run_cmd "ssh '$SERVER_SSH' '$train_pipeline_cmd'" "Running training and evaluation..."
        log_info "Experiment has been logged to experiments_log.csv on server"
    fi

    log_success "Training pipeline completed"
}

step_retrieve_results() {
    log_step "STEP 4: Retrieving results to local PC"

    if [ "$EXECUTION_MODE" = "local" ]; then
        log_warning "Running in LOCAL mode - results already on local PC"
        log_success "Results location:"
        log_success "  CSV: $LOCAL_DIR/experiments_log.csv"
        log_success "  Reports: $LOCAL_DIR/evaluation_results_*/"
        return 0
    fi

    # Create results directory
    mkdir -p "$RESULTS_DIR"

    log_info "Retrieving new experiment results from server CSV..."
    # Get the last line from the CSV inside the Docker container (newest experiment)
    # Then append it to the local CSV file (don't overwrite)
    REMOTE_CSV_TAIL=$(ssh "$SERVER_SSH" "docker exec llamafactory tail -n 1 /app/data/experiments_log.csv" 2>/dev/null)

    if [ -n "$REMOTE_CSV_TAIL" ]; then
        # Ensure local CSV exists
        if [ ! -f "$LOCAL_DIR/data/experiments_log.csv" ]; then
            log_warning "Local CSV does not exist, creating with header..."
            # Get the header from the remote CSV
            ssh "$SERVER_SSH" "docker exec llamafactory head -n 1 /app/data/experiments_log.csv" > "$LOCAL_DIR/data/experiments_log.csv"
        fi

        # Append the new experiment line to local CSV
        echo "$REMOTE_CSV_TAIL" >> "$LOCAL_DIR/data/experiments_log.csv"
        log_success "Appended new experiment to local CSV"
    else
        log_warning "Could not retrieve experiment results from server CSV"
    fi

    log_info "Retrieving evaluation reports..."
    run_cmd "rsync -avz --progress \
        '$SERVER_SSH:$REMOTE_HOST_DIR/evaluation_results_*/' \
        '$RESULTS_DIR/'" \
        "Downloading evaluation results..."

    if [ "$RETRIEVE_MODELS" = true ]; then
        log_info "Retrieving trained models..."
        run_cmd "rsync -avz --progress \
            '$SERVER_SSH:$REMOTE_HOST_DIR/saves/' \
            '$RESULTS_DIR/models/'" \
            "Downloading model files..."
    fi

    log_success "Results retrieved to:"
    log_success "  CSV: $LOCAL_DIR/experiments_log.csv"
    log_success "  Reports: $RESULTS_DIR/"
    if [ "$RETRIEVE_MODELS" = true ]; then
        log_success "  Models: $RESULTS_DIR/models/"
    fi
}

step_generate_html_report() {
    log_step "PHASE 5: GENERATE DATASET COMPARISON HTML REPORT"

    log_info "Generating HTML report for dataset: $DATASET_NAME"

    # Run generate_experiment_report.py locally
    # This script will:
    # 1. Check if videos are already downloaded locally
    # 2. If not, download them from server
    # 3. Extract model performance metrics from CSV
    # 4. Generate HTML report with visualizations
    # 5. Update CSV with report link

    local report_cmd="python3 $LOCAL_DIR/generate_experiment_report.py \
        --dataset-name '$DATASET_NAME' \
        --csv-path '$LOCAL_DIR/data/experiments_log.csv'"

    log_info "Running: $report_cmd"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would run: $report_cmd"
        return 0
    fi

    # Execute the report generator
    if eval "$report_cmd"; then
        log_success "HTML report generated successfully"

        # Try to find the generated report
        local report_file=$(find "$LOCAL_DIR/analytics/reports" -name "*${DATASET_NAME}*dual_report*.html" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

        if [ -n "$report_file" ]; then
            log_success "Report location: $report_file"

            # Optionally open the report in browser
            if command -v xdg-open &> /dev/null; then
                log_info "Opening report in browser..."
                xdg-open "$report_file" &
            elif command -v open &> /dev/null; then
                log_info "Opening report in browser..."
                open "$report_file" &
            fi
        fi
    else
        log_warning "Failed to generate HTML report (non-critical, continuing...)"
    fi
}

step_show_summary() {
    log_step "EXPERIMENT SUMMARY"

    cat << EOF
${GREEN}✓ Experiment Complete!${NC}

Dataset: ${CYAN}$DATASET_NAME${NC}
Texture: $TEXTURE_TYPE
View Angles: $VIEW_ANGLES
Epochs: $NUM_EPOCHS

${YELLOW}Next Steps:${NC}
1. Open the CSV: excel $LOCAL_DIR/experiments_log.csv
2. View reports: ls -la $RESULTS_DIR/
3. Check the latest experiment in the CSV (last row)

${YELLOW}CSV Location:${NC}
  $LOCAL_DIR/experiments_log.csv

${YELLOW}Reports Location:${NC}
  $RESULTS_DIR/

EOF
}

################################################################################
# MAIN EXECUTION
################################################################################

main() {
    # Print header
    echo -e "${CYAN}"
    cat << "EOF"
╔════════════════════════════════════════════════════════════════╗
║     Automated Experiment Runner with Tracking                 ║
║     LLaMA-Factory Treadmill Detection Pipeline                ║
╚════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"

    # Parse command line arguments
    parse_args "$@"

    # Pre-flight checks
    log_step "Pre-flight checks"
    if [ "$EXECUTION_MODE" = "local" ]; then
        log_info "Execution mode: LOCAL"
        log_warning "Skipping SSH and Docker checks (running locally)"
    else
        log_info "Execution mode: REMOTE (via SSH + Docker)"
        check_ssh_connection
        check_docker_container
    fi

    # Use test parameters if specified, otherwise default to train parameters
    local test_texture="${TEST_TEXTURE_TYPE:-$TRAIN_TEXTURE_TYPE}"
    local test_angles="${TEST_VIEW_ANGLES:-$TRAIN_VIEW_ANGLES}"
    local test_direction="${TEST_DIRECTION:-$TRAIN_DIRECTION}"
    local test_speed="${TEST_SPEED_RANGE:-$TRAIN_SPEED_RANGE}"

    # Show configuration
    log_step "Experiment Configuration"
    cat << EOF
Execution Mode: $EXECUTION_MODE
EOF

    if [ "$EXECUTION_MODE" = "remote" ]; then
        cat << EOF
Server: $SERVER_SSH
Container: $CONTAINER_NAME

EOF
    fi

    # Show dataset name if provided
    if [ -n "$DATASET_NAME" ]; then
        cat << EOF
Dataset: $DATASET_NAME (existing - reusing)

EOF
    fi

    cat << EOF
TRAINING Dataset:
  Texture: $TRAIN_TEXTURE_TYPE
  View Angles: $TRAIN_VIEW_ANGLES
  Direction: $TRAIN_DIRECTION
  Speed Range: $TRAIN_SPEED_RANGE
  BG Gray: $TRAIN_BG_GRAY
  Stripe Gray: $TRAIN_STRIPE_GRAY

TEST Dataset:
  Texture: $test_texture
  View Angles: $test_angles
  Direction: $test_direction
  Speed Range: $test_speed
  BG Gray: $TEST_BG_GRAY
  Stripe Gray: $TEST_STRIPE_GRAY

Common:
  FPS: $FPS
  Duration: $DURATION seconds

Training:$([ "$SKIP_TRAINING" = true ] && echo " SKIPPED (using existing model)" || echo "
  Adapter Type: $([ "$USE_DORA" = true ] && echo "DoRA" || echo "LoRA")
  Epochs: $NUM_EPOCHS
  LoRA Rank: $LORA_RANK / Alpha: $LORA_ALPHA
  Learning Rate: $LEARNING_RATE
  Batch Size: $BATCH_SIZE
  Gradient Accumulation: $GRAD_ACCUMULATION")
EOF

    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN MODE - No commands will be executed"
    fi

    # Confirmation prompt (unless dry run or --yes flag)
    if [ "$DRY_RUN" = false ] && [ "$YES_TO_ALL" = false ]; then
        echo -e "\n${YELLOW}Proceed with experiment? (y/N)${NC} "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_warning "Experiment cancelled by user"
            exit 0
        fi
    fi

    # Execute workflow
    START_TIME=$(date +%s)

    step_sync_code
    step_build_datasets
    step_run_training
    step_retrieve_results
    step_generate_html_report

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    step_show_summary

    log_success "Total time: ${ELAPSED} seconds ($((ELAPSED/60)) minutes)"

    # Open CSV if possible
    if command -v excel &> /dev/null; then
        log_info "Opening CSV in Excel..."
        excel "$LOCAL_DIR/experiments_log.csv" &
    elif command -v libreoffice &> /dev/null; then
        log_info "Opening CSV in LibreOffice..."
        libreoffice "$LOCAL_DIR/experiments_log.csv" &
    fi
}

# Run main function
main "$@"
