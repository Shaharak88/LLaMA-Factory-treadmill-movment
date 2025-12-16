# Changelog - Simple Evaluation Script

## Date: 2025-11-26

## Changes

### [evaluate_pipeline_simple.py](evaluate_pipeline_simple.py) [NEW]
- Created a new evaluation script using standard Hugging Face `transformers` instead of `vllm`.
- Implemented logic to load test data from `dataset_info.json`.
- Implemented keyword-based scoring (Moving vs Stopped).
- Generates a text report comparing Base and Fine-Tuned models.

### [run_full_pipeline.py](run_full_pipeline.py) [MODIFY]
- Updated `evaluate_script` to point to `evaluate_pipeline_simple.py`.
- Removed `gpu_memory_utilization` argument from the evaluation command construction as it is no longer needed.

## Reason
To simplify the evaluation process and remove the dependency on `vllm`, which was causing memory issues and complexity. The new script offers a more robust and direct way to evaluate the model using standard libraries.
