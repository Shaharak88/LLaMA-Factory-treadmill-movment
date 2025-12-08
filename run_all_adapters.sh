#!/bin/bash
################################################################################
# Multi-Adapter Comparison Script
#
# This script runs experiments with ALL compatible adapter types sequentially,
# using the same dataset and training parameters for fair comparison.
#
# Supported adapters: lora, lora+, dora, rslora, oft
# NOT supported: pissa (requires special initialization script)
#
# Usage:
#   ./run_all_adapters.sh [OPTIONS]
#
# The script accepts ALL the same arguments as run_experiment.sh, plus:
#   --adapters "lora,dora,rslora"  - Specify which adapters to run (comma-separated)
#
# Two modes:
#   1. Build new dataset: First adapter builds dataset, others reuse it
#   2. Use existing dataset: Pass --dataset-name to skip dataset generation
#
# Example:
#   # Run all adapters with new dataset
#   ./run_all_adapters.sh --epochs 7 --batch-size 14 --no-quantization -y
#
#   # Run specific adapters with existing dataset
#   ./run_all_adapters.sh --dataset-name "_exp_20251207_172213" \
#       --adapters "lora,dora,oft" --epochs 7 -y
#
# Author: AI-Generated
# Date: 2025-12-08
################################################################################

set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

################################################################################
# CONFIGURATION
################################################################################

# Default adapters to run (in order)
# NOTE: pissa is excluded because it requires special initialization
DEFAULT_ADAPTERS="lora,lora+,dora,rslora,oft"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EXPERIMENT_SCRIPT="$SCRIPT_DIR/run_experiment.sh"

################################################################################
# COLOR OUTPUT
################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
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

log_adapter() {
    echo -e "\n${MAGENTA}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  ADAPTER: $1${NC}"
    echo -e "${MAGENTA}════════════════════════════════════════════════════════════════${NC}\n"
}

################################################################################
# USAGE
################################################################################

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Multi-adapter comparison script - runs all compatible adapters sequentially.

This script accepts ALL the same arguments as run_experiment.sh, plus:

ADDITIONAL OPTIONS:
    --adapters LIST     Comma-separated list of adapters to run
                        Default: $DEFAULT_ADAPTERS
                        Supported: lora, lora+, dora, rslora, oft
                        NOT supported: pissa (requires special initialization)

MODES:
    1. Build new dataset (default):
       First adapter builds the dataset, subsequent adapters reuse it.

    2. Use existing dataset:
       Pass --dataset-name to use a pre-built dataset for all adapters.

EXAMPLES:
    # Run all adapters with new dataset, 7 epochs, no quantization
    $0 --epochs 7 --batch-size 14 --no-quantization -y

    # Run only lora, dora, oft with existing dataset
    $0 --dataset-name "_exp_20251207_172213" \\
       --adapters "lora,dora,oft" --epochs 7 -y

    # Run all adapters with specific training parameters
    $0 --train-angles "0.0,30.0" --test-angles "15.0,45.0" \\
       --epochs 5 --learning-rate 1e-4 -y

WHY PISSA IS EXCLUDED:
    PiSSA (Principal Singular values and Singular vectors Adaptation) requires
    running scripts/pissa_init.py before training when using quantization.
    This special initialization workflow is not compatible with our pipeline.

EOF
}

################################################################################
# PARSE ARGUMENTS
################################################################################

# Initialize variables
ADAPTERS_TO_RUN=""
DATASET_NAME=""
MODEL_NAME_BASE=""
PASSTHROUGH_ARGS=()
YES_TO_ALL=false
NO_QUANTIZATION=false

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            --adapters)
                ADAPTERS_TO_RUN="$2"
                shift 2
                ;;
            --dataset-name)
                DATASET_NAME="$2"
                # Also pass through to run_experiment.sh
                PASSTHROUGH_ARGS+=("--dataset-name" "$2")
                shift 2
                ;;
            --model-name)
                MODEL_NAME_BASE="$2"
                # Don't pass through - we'll append adapter type to it
                shift 2
                ;;
            -y|--yes)
                YES_TO_ALL=true
                PASSTHROUGH_ARGS+=("-y")
                shift
                ;;
            --no-quantization)
                NO_QUANTIZATION=true
                PASSTHROUGH_ARGS+=("--no-quantization")
                shift
                ;;
            # Pass through all other arguments
            *)
                PASSTHROUGH_ARGS+=("$1")
                shift
                ;;
        esac
    done

    # Use default adapters if not specified
    if [ -z "$ADAPTERS_TO_RUN" ]; then
        ADAPTERS_TO_RUN="$DEFAULT_ADAPTERS"
    fi
}

################################################################################
# VALIDATION
################################################################################

validate_adapters() {
    local valid_adapters="lora lora+ dora rslora oft"

    IFS=',' read -ra ADAPTER_ARRAY <<< "$ADAPTERS_TO_RUN"

    for adapter in "${ADAPTER_ARRAY[@]}"; do
        # Trim whitespace
        adapter=$(echo "$adapter" | xargs)

        if [[ ! " $valid_adapters " =~ " $adapter " ]]; then
            if [ "$adapter" = "pissa" ]; then
                log_error "PiSSA adapter is not supported!"
                log_error "PiSSA requires special initialization (scripts/pissa_init.py)"
                log_error "and is incompatible with our automated pipeline."
            else
                log_error "Invalid adapter: $adapter"
                log_error "Valid adapters: lora, lora+, dora, rslora, oft"
            fi
            exit 1
        fi
    done

    log_info "Adapters to run: ${ADAPTER_ARRAY[*]}"
}

################################################################################
# MAIN EXECUTION
################################################################################

main() {
    # Print header
    echo -e "${CYAN}"
    cat << "EOF"
╔════════════════════════════════════════════════════════════════╗
║     Multi-Adapter Comparison Script                            ║
║     Run all adapter types for fair comparison                  ║
╚════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"

    # Parse arguments
    parse_args "$@"

    # Validate adapters
    validate_adapters

    # Convert to array
    IFS=',' read -ra ADAPTER_ARRAY <<< "$ADAPTERS_TO_RUN"

    # Check if run_experiment.sh exists
    if [ ! -f "$RUN_EXPERIMENT_SCRIPT" ]; then
        log_error "run_experiment.sh not found at: $RUN_EXPERIMENT_SCRIPT"
        exit 1
    fi

    # Show configuration
    echo -e "\n${CYAN}Configuration:${NC}"
    echo "  Adapters: ${ADAPTER_ARRAY[*]}"
    echo "  Dataset: ${DATASET_NAME:-<will be generated>}"
    echo "  Model name base: ${MODEL_NAME_BASE:-<auto-generated>}"
    echo "  No quantization: $NO_QUANTIZATION"
    echo "  Passthrough args: ${PASSTHROUGH_ARGS[*]}"
    echo ""

    # Confirmation
    if [ "$YES_TO_ALL" = false ]; then
        echo -e "${YELLOW}This will run ${#ADAPTER_ARRAY[@]} experiments sequentially.${NC}"
        echo -e "${YELLOW}Proceed? (y/N)${NC} "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_warning "Cancelled by user"
            exit 0
        fi
    fi

    # Track results
    declare -A RESULTS
    FIRST_ADAPTER=true
    GENERATED_DATASET_NAME=""

    START_TIME=$(date +%s)

    # Run each adapter
    for adapter in "${ADAPTER_ARRAY[@]}"; do
        # Trim whitespace
        adapter=$(echo "$adapter" | xargs)

        log_adapter "$adapter"

        # Build the command
        CMD_ARGS=("${PASSTHROUGH_ARGS[@]}")
        CMD_ARGS+=("--adapter-type" "$adapter")

        # Generate model name
        if [ -n "$MODEL_NAME_BASE" ]; then
            # Sanitize adapter name for model path (replace + with plus)
            local adapter_safe="${adapter//+/plus}"
            CMD_ARGS+=("--model-name" "${MODEL_NAME_BASE}_${adapter_safe}")
        fi

        # First adapter: may build dataset
        # Subsequent adapters: always skip dataset, use generated name
        if [ "$FIRST_ADAPTER" = true ]; then
            FIRST_ADAPTER=false

            if [ -z "$DATASET_NAME" ]; then
                log_info "First adapter will generate new dataset..."
                # Don't add --skip-datasets, let it build
            else
                log_info "Using provided dataset: $DATASET_NAME"
                CMD_ARGS+=("--skip-datasets")
            fi
        else
            # Subsequent adapters always skip dataset generation
            CMD_ARGS+=("--skip-datasets")

            if [ -n "$GENERATED_DATASET_NAME" ]; then
                # Use the dataset name from first adapter
                # Need to add it if not already in passthrough args
                local has_dataset_name=false
                for arg in "${PASSTHROUGH_ARGS[@]}"; do
                    if [ "$arg" = "--dataset-name" ]; then
                        has_dataset_name=true
                        break
                    fi
                done

                if [ "$has_dataset_name" = false ]; then
                    CMD_ARGS+=("--dataset-name" "$GENERATED_DATASET_NAME")
                fi
            fi
        fi

        log_info "Running: $RUN_EXPERIMENT_SCRIPT ${CMD_ARGS[*]}"

        # Run the experiment and capture output
        ADAPTER_START=$(date +%s)

        if "$RUN_EXPERIMENT_SCRIPT" "${CMD_ARGS[@]}" 2>&1 | tee /tmp/adapter_output_${adapter//+/plus}.log; then
            RESULTS[$adapter]="SUCCESS"
            log_success "Adapter $adapter completed successfully"

            # Capture the dataset name from first adapter output (if we generated one)
            if [ -z "$GENERATED_DATASET_NAME" ] && [ -z "$DATASET_NAME" ]; then
                # Try to extract dataset name from output
                GENERATED_DATASET_NAME=$(grep -oP "Dataset name: \K[^\s]+" /tmp/adapter_output_${adapter//+/plus}.log 2>/dev/null | head -1 || true)
                if [ -n "$GENERATED_DATASET_NAME" ]; then
                    # Remove _train or _test suffix if present
                    GENERATED_DATASET_NAME="${GENERATED_DATASET_NAME%_train}"
                    GENERATED_DATASET_NAME="${GENERATED_DATASET_NAME%_test}"
                    log_info "Captured dataset name: $GENERATED_DATASET_NAME"
                else
                    # Try alternate pattern
                    GENERATED_DATASET_NAME=$(grep -oP "Using existing dataset: \K[^\s]+" /tmp/adapter_output_${adapter//+/plus}.log 2>/dev/null | head -1 || true)
                    GENERATED_DATASET_NAME="${GENERATED_DATASET_NAME%_train}"
                    GENERATED_DATASET_NAME="${GENERATED_DATASET_NAME%_test}"
                fi
            fi
        else
            RESULTS[$adapter]="FAILED"
            log_error "Adapter $adapter failed!"
        fi

        ADAPTER_END=$(date +%s)
        ADAPTER_ELAPSED=$((ADAPTER_END - ADAPTER_START))
        log_info "Adapter $adapter took ${ADAPTER_ELAPSED}s ($((ADAPTER_ELAPSED/60))m)"

        echo ""
    done

    END_TIME=$(date +%s)
    TOTAL_ELAPSED=$((END_TIME - START_TIME))

    # Print summary
    echo -e "\n${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  MULTI-ADAPTER COMPARISON COMPLETE${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}\n"

    echo "Dataset used: ${GENERATED_DATASET_NAME:-$DATASET_NAME}"
    echo ""
    echo "Results:"
    echo "--------"
    for adapter in "${ADAPTER_ARRAY[@]}"; do
        adapter=$(echo "$adapter" | xargs)
        status="${RESULTS[$adapter]}"
        if [ "$status" = "SUCCESS" ]; then
            echo -e "  ${GREEN}✓${NC} $adapter: $status"
        else
            echo -e "  ${RED}✗${NC} $adapter: $status"
        fi
    done

    echo ""
    echo "Total time: ${TOTAL_ELAPSED}s ($((TOTAL_ELAPSED/60))m)"
    echo ""
    echo "Check experiments_log.csv for detailed results and metrics."
    echo "HTML reports generated for each adapter in analytics/reports/"

    # Check if any failed
    for status in "${RESULTS[@]}"; do
        if [ "$status" = "FAILED" ]; then
            log_warning "Some adapters failed. Check logs above for details."
            exit 1
        fi
    done

    log_success "All adapters completed successfully!"
}

# Run main function
main "$@"
