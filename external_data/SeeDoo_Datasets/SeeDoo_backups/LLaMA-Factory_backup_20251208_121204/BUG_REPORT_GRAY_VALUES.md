# Critical Bug Report: Invisible Black Treadmill Videos

**Date Discovered**: December 1, 2025
**Date Introduced**: November 30, 2025 (Commit 3781fa1)
**Date Fixed**: December 1, 2025 (Commit 4a36154)
**Severity**: CRITICAL - Caused complete training data corruption
**Root Cause**: Incorrect default gray values in run_experiment.sh

---

## Summary

All treadmill training videos generated via `run_experiment.sh` since November 30, 2025 were completely black and showed no visible treadmill due to extremely low gray value defaults (4-8% brightness instead of 49-55%).

---

## Technical Details

### Incorrect Values (Introduced in 3781fa1)
```bash
# run_experiment.sh lines 104-107
TRAIN_STRIPE_GRAY="20"      # 8% brightness  ❌ TOO DARK
TEST_STRIPE_GRAY="15"       # 6% brightness  ❌ TOO DARK
TRAIN_BG_GRAY="15"          # 6% brightness  ❌ TOO DARK
TEST_BG_GRAY="10"           # 4% brightness  ❌ TOO DARK
```

### Correct Values (Fixed in 4a36154)
```bash
# run_experiment.sh lines 104-107
TRAIN_STRIPE_GRAY="125"     # 49% brightness ✅ VISIBLE
TEST_STRIPE_GRAY="125"      # 49% brightness ✅ VISIBLE
TRAIN_BG_GRAY="140"         # 55% brightness ✅ VISIBLE
TEST_BG_GRAY="140"          # 55% brightness ✅ VISIBLE
```

### Reference Values
The correct defaults were already defined in `building_dataset.py`:
```python
# building_dataset.py lines 179-180, 908-911
stripe_grays = [125]    # Default for subtle_gray_stripes
background_grays = [140]
```

---

## Impact Analysis

### Affected Experiments
All experiments run via `run_experiment.sh` between Nov 30 - Dec 1 that did NOT explicitly specify `--train-stripe-gray` and `--train-bg-gray` parameters.

### Symptoms
- Training videos show completely black screen
- No visible treadmill or stripes
- Model trained on unusable data
- Performance metrics unusable

### Example Affected Experiments
- **Experiment 2** (Dec 1, 13:14): Used stripe_gray=20, bg_gray=15
- **Experiment 5** (Dec 1, 15:20): Used stripe_gray=0-5, bg_gray=10-15
- Downloaded videos in `data/example_videos_exp5/` were completely black

---

## Root Cause Analysis

### Timeline

**November 30, 2025 - Commit 3781fa1**
- I (Claude) implemented comprehensive metadata tracking for experiments
- Added 60+ new CSV columns to track all train/test parameters
- Created default values for all new parameters in `run_experiment.sh`
- **MISTAKE**: Set gray value defaults to 10-22 range instead of 125-140
- These ultra-dark values (4-8% brightness) created invisible black treadmills

**Why the values were wrong:**
When adding stripe gray parameters to run_experiment.sh, I likely:
1. Misunderstood the 0-255 brightness scale
2. Set arbitrarily low test values (10-22 range)
3. Failed to reference the correct defaults from building_dataset.py
4. Did not visually test the generated videos

**December 1, 2025 - Bug Discovery**
- User downloaded sample videos from experiment 5
- Noticed all videos were completely black
- Investigation traced issue to incorrect gray value defaults
- Generated test videos with correct values (125/140) to verify fix

---

## Fix Details

### Changed Files
- `run_experiment.sh` (lines 84, 104-107)

### Changes Made
```diff
-TRAIN_BRIGHTNESS="1.0"
+TRAIN_BRIGHTNESS="0.0"

-TRAIN_STRIPE_GRAY="20"
-TEST_STRIPE_GRAY="15"
-TRAIN_BG_GRAY="15"
-TEST_BG_GRAY="10"
+TRAIN_STRIPE_GRAY="125"
+TEST_STRIPE_GRAY="125"
+TRAIN_BG_GRAY="140"
+TEST_BG_GRAY="140"
```

### Verification
Generated test videos with corrected values:
- `data/test_correct_grays/treadmill_*_speed5.0_*.mp4` (moving)
- `data/test_correct_grays/treadmill_*_speed0.0_*.mp4` (stopped)

Both videos show visible medium gray treadmills with clear stripe patterns.

---

## Prevention Measures

### Lessons Learned
1. **Always visually inspect generated training data**
   - Don't assume parameters are correct without verification
   - Download and view sample videos before running full experiments

2. **Cross-reference default values**
   - When adding new parameters to scripts, check existing defaults
   - Maintain consistency between run_experiment.sh and building_dataset.py

3. **Understand parameter ranges**
   - Gray values: 0-255 (0=black, 125=medium gray, 255=white)
   - For visible gray stripes: Use 125-140 range (49-55% brightness)
   - Values below 30 (<12% brightness) are too dark to be useful

4. **Test with minimal examples first**
   - Generate 1-2 test videos before running full experiments
   - Verify visual appearance before proceeding

### Recommended Changes

**Add validation to building_dataset.py:**
```python
# Validate gray values for subtle_gray_stripes texture
if texture_type == "subtle_gray_stripes":
    if any(g < 50 for g in stripe_grays):
        print(f"WARNING: stripe_gray values {stripe_grays} are very dark (<50)")
        print(f"Recommended range: 100-150 for visible stripes")
    if any(g < 50 for g in background_grays):
        print(f"WARNING: background_gray values {background_grays} are very dark (<50)")
        print(f"Recommended range: 100-150 for visible background")
```

**Add comments in run_experiment.sh:**
```bash
# Stripe gray values (0-255 scale, recommend 100-150 for visible subtle gray)
# Values below 50 (~20% brightness) will be very dark or invisible
TRAIN_STRIPE_GRAY="125"  # 49% brightness - visible medium gray
TEST_STRIPE_GRAY="125"
TRAIN_BG_GRAY="140"      # 55% brightness - slightly lighter than stripes
TEST_BG_GRAY="140"
```

---

## Related Issues

- Previous experiments (before Nov 30) may have used correct values
- Any experiments that explicitly specified gray values were unaffected
- building_dataset.py always had correct defaults (125/140)

---

## References

- Bug introduced: Commit 3781fa1
- Bug fixed: Commit 4a36154
- Test videos: `data/test_correct_grays/`
- Affected experiments: See `data/experiments_log.csv` rows 2-5
