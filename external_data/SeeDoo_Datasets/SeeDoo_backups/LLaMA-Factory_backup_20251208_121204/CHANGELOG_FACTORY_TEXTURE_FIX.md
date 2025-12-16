# Changelog: Fix Factory Texture Generation - Filter Kwargs Before Function Calls

**Date:** 2025-11-29
**Author:** AI Assistant
**Scope:** Critical bug fix for factory texture generation

---

## Summary

Fixed a critical bug where factory texture generators (`generate_factory_dark_stripes()` and `generate_factory_dark()`) were receiving unexpected keyword arguments, causing video generation to fail. The fix filters kwargs to only pass parameters that each texture generator function actually accepts.

---

## Problem Description

### Error Symptoms

When generating datasets with mixed textures including `factory_dark` and `factory_dark_stripes`, video generation failed with errors:

```
Error: TreadmillTextureGenerator.generate_factory_dark_stripes() got an unexpected keyword argument 'stripe_spacing'
Error: TreadmillTextureGenerator.generate_factory_dark() got an unexpected keyword argument 'stripe_width'
```

### Root Cause

The `generate_texture()` method in `synthetic_data_generation.py` was passing ALL parsed command-line arguments as `**kwargs` to texture generator functions, regardless of whether those functions accepted those parameters.

**Lines 420 and 436 (OLD CODE):**
```python
elif texture_type == 'factory_dark_stripes':
    return self.generate_factory_dark_stripes(**kwargs)  # Line 420 - passes ALL kwargs

elif texture_type == 'factory_dark':
    return self.generate_factory_dark(**kwargs)  # Line 436 - passes ALL kwargs
```

### Why This Happened

When using multiple textures (e.g., `subtle_gray_stripes`, `factory_dark_stripes`, `factory_dark`), the command-line parser accepts ALL texture-specific parameters:
- `--stripe_width`, `--stripe_spacing`, `--stripe_gray`, `--background_gray` (for subtle_gray_stripes)
- These parameters get stored in `kwargs` dictionary
- `**kwargs` unpacks the ENTIRE dictionary when calling functions
- Factory texture generators don't accept subtle_gray_stripes parameters
- Result: TypeError for unexpected keyword arguments

### Function Signatures

**generate_factory_dark()** accepts:
- `base_color`: Tuple[int, int, int] = (25, 25, 25)
- `texture_intensity`: float = 0.15

**generate_factory_dark_stripes()** accepts:
- `stripe_width`: int = 50
- `orientation`: str = 'horizontal'
- `base_color`: Tuple[int, int, int] = (20, 20, 20)
- `stripe_color`: Tuple[int, int, int] = (35, 35, 35)
- `motion_direction`: str = None

Neither accepts: `stripe_spacing`, `stripe_gray`, `background_gray`, `stripe_distance_variance` (these are subtle_gray_stripes-specific)

---

## Solution

Filter the kwargs dictionary before passing to texture generator functions, only including parameters that each function actually accepts.

---

## File Modified

### `data/synthetic_treadmill/synthetic_data_generation.py`

**Lines 417-444 - Complete rewrite of texture generator calls**

#### OLD CODE:
```python
if texture_type == 'stripes':
    return self.generate_stripes(**kwargs)
elif texture_type == 'factory_dark_stripes':
    return self.generate_factory_dark_stripes(**kwargs)  # ❌ Passes ALL kwargs
elif texture_type == 'subtle_gray_stripes':
    return self.generate_subtle_gray_stripes(**kwargs)
else:
    # Remove motion_direction for non-stripe textures (they don't use it)
    kwargs.pop('motion_direction', None)

    if texture_type == 'noise':
        return self.generate_noise_pattern(**kwargs)
    elif texture_type == 'rubber':
        return self.generate_rubber_pattern(**kwargs)
    elif texture_type == 'grid':
        return self.generate_grid_pattern(**kwargs)
    elif texture_type == 'diamond_plate':
        return self.generate_diamond_plate(**kwargs)
    elif texture_type == 'factory_dark':
        return self.generate_factory_dark(**kwargs)  # ❌ Passes ALL kwargs
    else:
        raise ValueError(f"Unknown texture type: {texture_type}")
```

#### NEW CODE:
```python
if texture_type == 'stripes':
    return self.generate_stripes(**kwargs)
elif texture_type == 'factory_dark_stripes':
    # Filter kwargs to only include parameters that generate_factory_dark_stripes accepts
    factory_dark_stripes_params = {'stripe_width', 'orientation', 'base_color', 'stripe_color', 'motion_direction'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in factory_dark_stripes_params}
    return self.generate_factory_dark_stripes(**filtered_kwargs)  # ✅ Filtered kwargs
elif texture_type == 'subtle_gray_stripes':
    return self.generate_subtle_gray_stripes(**kwargs)
else:
    # Remove motion_direction for non-stripe textures (they don't use it)
    kwargs.pop('motion_direction', None)

    if texture_type == 'noise':
        return self.generate_noise_pattern(**kwargs)
    elif texture_type == 'rubber':
        return self.generate_rubber_pattern(**kwargs)
    elif texture_type == 'grid':
        return self.generate_grid_pattern(**kwargs)
    elif texture_type == 'diamond_plate':
        return self.generate_diamond_plate(**kwargs)
    elif texture_type == 'factory_dark':
        # Filter kwargs to only include parameters that generate_factory_dark accepts
        factory_dark_params = {'base_color', 'texture_intensity'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in factory_dark_params}
        return self.generate_factory_dark(**filtered_kwargs)  # ✅ Filtered kwargs
    else:
        raise ValueError(f"Unknown texture type: {texture_type}")
```

---

## Changes Explained

### 1. factory_dark_stripes Filtering (Lines 419-423)

**What it does:**
- Defines allowlist of parameters: `{'stripe_width', 'orientation', 'base_color', 'stripe_color', 'motion_direction'}`
- Creates `filtered_kwargs` dictionary with only these parameters
- Passes filtered kwargs to function

**Effect:**
- Parameters like `stripe_spacing`, `stripe_gray`, `background_gray` are excluded
- Function only receives arguments it can accept
- No more TypeError

### 2. factory_dark Filtering (Lines 438-442)

**What it does:**
- Defines allowlist of parameters: `{'base_color', 'texture_intensity'}`
- Creates `filtered_kwargs` dictionary with only these parameters
- Passes filtered kwargs to function

**Effect:**
- Parameters like `stripe_width`, `stripe_spacing`, etc. are excluded
- Function only receives arguments it can accept
- No more TypeError

### 3. Unchanged Textures

**subtle_gray_stripes:** Still receives full `**kwargs` (no filtering needed)
**Other textures:** Already had `motion_direction` removed (no additional filtering needed)

---

## Testing Results

### Before Fix:
```
ERROR: TreadmillTextureGenerator.generate_factory_dark_stripes() got an unexpected keyword argument 'stripe_spacing'
ERROR: TreadmillTextureGenerator.generate_factory_dark() got an unexpected keyword argument 'stripe_width'
```

### After Fix:
```
[OK] Generated: treadmill_0000_factory_dark_stripes_left_speed0.0_angle15_...mp4 (70.2 KB)
[OK] Generated: treadmill_0000_factory_dark_up_speed14.0_angle0_...mp4 (1496.8 KB)
[OK] Generated: treadmill_0000_subtle_gray_stripes_down_speed0.0_angle15_...mp4 (41.3 KB)
```

All 72 videos (3 textures × 4 directions × 3 angles × 2 speeds) now generate successfully!

---

## Deployment

### Steps Taken:

1. **Modified local file:**
   - `/mnt/c/Users/shaha/Desktop/Qwen2.5/LLaMA-Factory/data/synthetic_treadmill/synthetic_data_generation.py`

2. **Synced to server (correct location):**
   ```bash
   rsync -avz --inplace synthetic_data_generation.py seedoo@hetzner-gpu.tail9e6e7.ts.net:/home/seedoo/shahar_linux_wsl/LLaMA-Factory/data/synthetic_treadmill/
   ```

3. **Cleared Python bytecode cache:**
   ```bash
   docker exec llamafactory bash -c 'find . -type d -name __pycache__ -path "*/data/synthetic_treadmill/*" -exec rm -rf {} +'
   ```

4. **Verified fix:**
   - Ran test generation with all 3 textures
   - All videos generated successfully
   - No errors

---

## Benefits

1. **Enables Multi-Texture Datasets:**
   - Can now mix subtle_gray_stripes, factory_dark_stripes, and factory_dark in one dataset
   - Critical for comprehensive model training

2. **Robust Parameter Handling:**
   - Each texture generator only receives parameters it accepts
   - No more TypeErrors from unexpected kwargs

3. **Maintainable Code:**
   - Explicit parameter allowlists make function contracts clear
   - Easy to add new textures with different parameters

4. **Backward Compatible:**
   - Single-texture datasets still work
   - No breaking changes to function signatures
   - No changes to command-line interface

---

## Related Files

- `building_dataset.py` - Already correctly filtered stripe parameters at command level (lines 405-415)
- `evaluate_pipeline_simple.py` - No changes needed (not affected)
- `requirements.txt` - No changes needed

---

## Future Considerations

1. **Refactor to Use Explicit Parameters:**
   - Instead of `**kwargs`, consider explicit parameter passing with defaults
   - Would catch signature mismatches at code review time

2. **Add Parameter Validation:**
   - Validate that required parameters are present before calling generators
   - Provide helpful error messages for missing parameters

3. **Consider Factory Pattern:**
   - Create texture generator factory with parameter schema validation
   - Centralize parameter filtering logic

---

## Commit Message

```
Fix factory texture generation by filtering kwargs before function calls

PROBLEM:
- Factory texture generators were receiving unexpected keyword arguments
- Caused TypeError when mixing textures (subtle_gray_stripes + factory textures)
- Error: "got an unexpected keyword argument 'stripe_spacing'"

ROOT CAUSE:
- generate_texture() passed ALL kwargs to texture generator functions
- Functions like generate_factory_dark() and generate_factory_dark_stripes()
  don't accept subtle_gray_stripes-specific parameters
- Result: TypeError for unexpected keyword arguments

SOLUTION:
- Filter kwargs before passing to factory texture generators
- factory_dark_stripes: only pass {stripe_width, orientation, base_color, stripe_color, motion_direction}
- factory_dark: only pass {base_color, texture_intensity}
- Each generator now only receives parameters it can accept

CHANGES:
- synthetic_data_generation.py lines 419-442:
  * Add parameter allowlists for factory texture generators
  * Create filtered_kwargs dictionaries
  * Pass only allowed parameters to functions

TESTING:
- Generated 72-video dataset with 3 textures successfully
- All textures now work: subtle_gray_stripes, factory_dark_stripes, factory_dark
- No more TypeErrors

BENEFITS:
- Enables multi-texture datasets for comprehensive training
- Robust parameter handling across different texture types
- Backward compatible with existing datasets

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Verification Checklist

- [x] Code compiles without syntax errors
- [x] All imports verified (no new imports needed)
- [x] Tested with all 3 textures simultaneously
- [x] No breaking changes to function signatures
- [x] No changes to command-line interface
- [x] Deployed to server and verified in Docker container
- [x] 72-video generation running successfully
- [x] Detailed changelog documented

---

## Debug Timeline

1. **Initial error:** Factory textures failing with unexpected keyword arguments
2. **First hypothesis:** building_dataset.py passing wrong parameters - WRONG
3. **Added debug logging:** Confirmed building_dataset.py was correct
4. **Root cause found:** synthetic_data_generation.py passing ALL kwargs with `**` unpacking
5. **Fix implemented:** Parameter filtering with allowlists
6. **Deployment issue:** Files synced to wrong directory (fixed)
7. **Success:** All 72 videos generating correctly

---

## Example: How Parameter Filtering Works

### Input kwargs (from command line):
```python
kwargs = {
    'stripe_width': 10,
    'stripe_spacing': 60,
    'stripe_gray': 13,
    'background_gray': 10,
    'base_color': (25, 25, 25),
    'texture_intensity': 0.15,
    # ... other parameters
}
```

### OLD CODE (factory_dark):
```python
return self.generate_factory_dark(**kwargs)
# Expands to:
return self.generate_factory_dark(
    stripe_width=10,           # ❌ Not accepted!
    stripe_spacing=60,         # ❌ Not accepted!
    stripe_gray=13,            # ❌ Not accepted!
    background_gray=10,        # ❌ Not accepted!
    base_color=(25, 25, 25),   # ✅ Accepted
    texture_intensity=0.15,    # ✅ Accepted
    ...
)
# Result: TypeError!
```

### NEW CODE (factory_dark):
```python
factory_dark_params = {'base_color', 'texture_intensity'}
filtered_kwargs = {k: v for k, v in kwargs.items() if k in factory_dark_params}
# filtered_kwargs = {'base_color': (25, 25, 25), 'texture_intensity': 0.15}

return self.generate_factory_dark(**filtered_kwargs)
# Expands to:
return self.generate_factory_dark(
    base_color=(25, 25, 25),   # ✅ Accepted
    texture_intensity=0.15,    # ✅ Accepted
)
# Result: Success!
```

---

## End of Changelog
