# Changelog - Stripe Distance Variance

## Added `stripe_distance_variance` Parameter

### Description
Added a new parameter `stripe_distance_variance` to the `subtle_gray_stripes` texture generation. This parameter controls the variance (standard deviation) of the distance between stripes.

- **Value 0.0 (Default)**: Fixed spacing between stripes (original behavior).
- **Value > 0.0**: Randomized spacing using a normal distribution `N(mean=spacing, std=variance)`.

### Modified Files

#### 1. `synthetic_treadmill/synthetic_data_generation.py`
- Updated `generate_subtle_gray_stripes` to accept `stripe_distance_variance`.
- Implemented randomized spacing logic:
  ```python
  spacing = self.rng.normal(stripe_spacing, stripe_distance_variance)
  spacing = max(stripe_width + 1, spacing)  # Ensure minimum spacing
  ```
- Updated `SyntheticVideoGenerator` to pass the parameter.
- Updated `parse_arguments` to include `--stripe_distance_variance`.

#### 2. `building_dataset.py`
- Updated `parse_arguments` to include `--stripe_distance_variance`.
- Updated parameter combination logic to support this new parameter.
- Updated video generation command builder.

#### 3. `run_full_pipeline.py`
- Updated `parse_arguments` to include `--stripe_distance_variance`.
- Updated `_build_dataset_command` to pass the parameter.

### Usage Example
```bash
python synthetic_treadmill/synthetic_data_generation.py \
  --texture_type subtle_gray_stripes \
  --stripe_distance_variance 20.0 \
  --num_videos 5
```
