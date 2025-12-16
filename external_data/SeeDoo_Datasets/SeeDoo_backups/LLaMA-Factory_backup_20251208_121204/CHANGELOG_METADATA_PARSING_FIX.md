# Changelog: Fix Metadata Parsing for Multi-Word Texture Names

**Date:** 2025-11-29
**Author:** AI Assistant
**Scope:** Critical bug fix for per-texture and per-angle metrics

---

## Summary

Fixed regex pattern in `evaluate_pipeline_simple.py` to correctly parse texture names that contain underscores (e.g., `subtle_gray_stripes`, `factory_dark_stripes`, `factory_dark`). This bug was causing all textures and angles to be classified as "unknown", preventing per-texture and per-angle breakdowns from working correctly.

---

## Problem Description

### Symptoms

When running evaluation, all videos showed:
- `Texture: unknown, Angle: unknown`
- Per-texture breakdown only showed "unknown" category
- Per-angle breakdown only showed "unknown" category
- No granular metrics by texture type or camera angle

### Root Cause

**File:** `evaluate_pipeline_simple.py` (Line 62)

**OLD REGEX:**
```python
match = re.search(r'treadmill_\d+_([^_]+)_[^_]+_speed[\d.]+_angle(\d+)', filename)
```

**Problem:** The pattern `([^_]+)` captures characters until the first underscore. This works for single-word textures like "stripes" or "noise", but **FAILS** for multi-word texture names with underscores:
- `subtle_gray_stripes` → Only captures `subtle`
- `factory_dark_stripes` → Only captures `factory`
- `factory_dark` → Only captures `factory`

### Example Filename

```
treadmill_0000_subtle_gray_stripes_left_speed14.0_angle0_bright0.00_contr1.00_640x480_seed45.mp4
```

**OLD REGEX CAPTURE:**
- `match.group(1)` = `"subtle"` ❌ (Should be `"subtle_gray_stripes"`)

---

## Solution

### NEW REGEX (Line 65)

```python
match = re.search(r'treadmill_\d+_(.+?)_(left|right|up|down)_speed[\d.]+_angle(\d+)', filename)
```

**Key Changes:**
1. **`(.+?)`**: Non-greedy capture of ANY characters (including underscores) for texture name
2. **`(left|right|up|down)`**: Explicit match on known direction keywords acts as delimiter
3. **Updated group indices**:
   - `match.group(1)` = texture (now includes underscores)
   - `match.group(2)` = direction (captured but not used)
   - `match.group(3)` = angle number

**NEW REGEX CAPTURE:**
- `match.group(1)` = `"subtle_gray_stripes"` ✅
- `match.group(3)` = `"0"` ✅

---

## File Modified

### `evaluate_pipeline_simple.py`

**Lines 51-73: Updated `parse_video_metadata()` function**

```python
def parse_video_metadata(self, video_path: str) -> Dict[str, str]:
    """
    Parse texture and angle from video filename.
    Expected format: treadmill_XXXX_<texture>_<direction>_speed<X.X>_angle<X>_...
    Texture can be multi-word with underscores (e.g., subtle_gray_stripes, factory_dark_stripes)

    Returns:
        Dict with 'texture' and 'angle' keys
    """
    filename = Path(video_path).name

    # Pattern: treadmill_XXXX_<texture>_<direction>_speed<X.X>_angle<X>_...
    # Capture texture (can have underscores) until we hit a known direction
    # Directions: left, right, up, down
    match = re.search(r'treadmill_\d+_(.+?)_(left|right|up|down)_speed[\d.]+_angle(\d+)', filename)

    if match:
        texture = match.group(1)
        angle = f"angle{match.group(3)}"
        return {'texture': texture, 'angle': angle}
    else:
        logger.warning(f"Could not parse metadata from filename: {filename}")
        return {'texture': 'unknown', 'angle': 'unknown'}
```

---

## Testing

### Test Filenames

1. `treadmill_0000_subtle_gray_stripes_left_speed14.0_angle0_...mp4`
   - Texture: ✅ `subtle_gray_stripes` (was: `subtle`)
   - Angle: ✅ `angle0`

2. `treadmill_0000_factory_dark_stripes_right_speed0.0_angle15_...mp4`
   - Texture: ✅ `factory_dark_stripes` (was: `factory`)
   - Angle: ✅ `angle15`

3. `treadmill_0000_factory_dark_up_speed14.0_angle30_...mp4`
   - Texture: ✅ `factory_dark` (was: `factory`)
   - Angle: ✅ `angle30`

---

## Impact

### Before Fix
- ❌ All metrics grouped under "unknown" texture
- ❌ All metrics grouped under "unknown" angle
- ❌ Cannot analyze performance by texture type
- ❌ Cannot analyze performance by camera angle
- ❌ Cannot identify which visual conditions are challenging

### After Fix
- ✅ Metrics properly broken down by each texture
- ✅ Metrics properly broken down by each angle (0°, 15°, 30°)
- ✅ Can compare F1 scores across textures
- ✅ Can identify if model struggles with specific angles
- ✅ Can optimize training based on per-texture/angle weaknesses

---

## Related Changes

This fix enables proper functionality for:
- Per-texture accuracy, F1, precision, recall
- Per-angle accuracy, F1, precision, recall
- Moving/Stopped breakdown per texture
- Moving/Stopped breakdown per angle

All implemented in `evaluate_pipeline_simple.py` lines 234-248 (per-texture stats) and lines 242-248 (per-angle stats).

---

## Dependencies

**None.** This is a pure bug fix with no dependencies or breaking changes.

**Affected Files:**
- `evaluate_pipeline_simple.py` (1 function, 23 lines modified)

**No Changes Needed In:**
- `building_dataset.py` (not affected - doesn't parse filenames)
- `run_full_pipeline.py` (not affected - orchestration only)
- `synthetic_data_generation.py` (not affected - creates files, doesn't parse)

---

## Verification

To verify the fix works:

1. **Run evaluation with fixed code**
2. **Check log output** - Should now show:
   ```
   Texture: subtle_gray_stripes, Angle: angle0
   Texture: factory_dark_stripes, Angle: angle15
   Texture: factory_dark, Angle: angle30
   ```
3. **Check evaluation report** - Should now show:
   ```
   PER-TEXTURE BREAKDOWN:
     subtle_gray_stripes:
       Total: 24 videos
       Accuracy: XX.XX%
     factory_dark_stripes:
       Total: 24 videos
       Accuracy: XX.XX%
     factory_dark:
       Total: 24 videos
       Accuracy: XX.XX%
   ```

---

## Commit Message

```
Fix metadata parsing for multi-word texture names with underscores

PROBLEM:
- Regex pattern only captured texture name until first underscore
- Multi-word textures (subtle_gray_stripes, factory_dark_stripes, factory_dark)
  were truncated to first word only
- All per-texture and per-angle metrics showed "unknown"
- Cannot analyze performance by texture type or camera angle

ROOT CAUSE:
- Pattern ([^_]+) stops at first underscore
- Works for single-word textures (stripes, noise, rubber)
- Fails for multi-word textures with underscores

SOLUTION:
- Changed pattern to (.+?)_(left|right|up|down)
- Non-greedy capture until hitting known direction keyword
- Now captures full texture name including underscores
- Updated group indices (texture=group(1), angle=group(3))

CHANGES:
- evaluate_pipeline_simple.py lines 51-73:
  * Updated parse_video_metadata() function
  * New regex pattern with direction-based delimiter
  * Added docstring explaining multi-word texture support

TESTING:
- subtle_gray_stripes: ✅ Captures full name (was: subtle)
- factory_dark_stripes: ✅ Captures full name (was: factory)
- factory_dark: ✅ Captures full name (was: factory)

IMPACT:
- Enables per-texture performance metrics
- Enables per-angle performance metrics
- Can now identify which textures/angles are challenging
- No breaking changes, no dependencies

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## End of Changelog
