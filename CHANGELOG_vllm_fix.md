# Changelog - vllm GPU Memory Fix

## Date: 2025-11-26

## Changes

### [evaluate_pipeline.py](evaluate_pipeline.py)
- Added `--gpu_memory_utilization` argument to `parse_arguments` (default: 0.8).
- Passed `gpu_memory_utilization` to `vllm_infer.py` via the `--vllm_config` argument using `json.dumps`.

### [run_full_pipeline.py](run_full_pipeline.py)
- Added `--gpu_memory_utilization` argument to `parse_arguments` (default: 0.8).
- Passed `--gpu_memory_utilization` to `evaluate_pipeline.py` in `step3_evaluate_models`.

## Reason
The default `gpu_memory_utilization` of 0.9 in `vllm` was causing `ValueError: Free memory on device ... is less than desired GPU memory utilization` because the available GPU memory (6.87 GiB) was less than the required 7.2 GiB (90% of 8 GiB). Lowering it to 0.8 (6.4 GiB) resolves this issue.
