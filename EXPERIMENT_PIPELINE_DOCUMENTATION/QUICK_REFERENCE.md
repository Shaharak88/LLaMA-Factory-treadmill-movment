# Quick Reference Guide - Experiment Pipeline Documentation

**Purpose:** This quick reference provides concise summaries of each documentation file in this folder, helping you quickly identify which document to read for specific information.

---

## File Summaries

### **README.md** - Documentation Hub & Navigation
**What's in it:**
- Executive summary for managers/stakeholders showing business value (65% → 95% accuracy improvement)
- Complete navigation system organized by user role (first-time users, developers, analysts, managers)
- Documentation philosophy and standards
- Glossary of key terms (LoRA, DoRA, quantization, etc.)
- Complete file structure reference showing where everything lives
- Common workflows and troubleshooting guide
- Quick command reference for all major operations

**When to read it:** First stop for navigation, understanding structure, or finding quick commands.

---

### **00_OVERVIEW.md** - Big Picture Architecture
**What's in it:**
- High-level purpose: automated pipeline for training Qwen2.5-VL to detect treadmill motion
- 5-stage architecture diagram (Orchestration → Data → Training → Evaluation → Reporting)
- Execution flow from user command to final report
- Key design decisions (why Docker, why rsync, why CSV tracking)
- Common use cases (run full experiment, skip dataset generation, change eval method)
- Output locations on GPU server and local machine

**When to read it:** Understanding the overall system architecture before diving into specifics.

---

### **01_ORCHESTRATION.md** - Main Script (run_experiment.sh)
**What's in it:**
- Complete breakdown of `run_experiment.sh` - the main entry point
- 6-stage flow: code sync → dataset generation → training → results retrieval → report generation
- Detailed explanation of rsync parameters and what gets synced/excluded
- Docker parameters for GPU access and volume mounting
- Dataset parameters (texture, angle, speed, distance) and their effects
- Training parameters (epochs, learning rate, batch size, LoRA config)
- How to customize parameters and skip stages
- SSH configuration and remote server setup

**When to read it:** Modifying experiment parameters, understanding remote execution, troubleshooting sync issues.

---

### **02_DATA_GENERATION.md** - Video Creation with Blender
**What's in it:**
- Two-part system: `building_dataset.py` (orchestrator) + `synthetic_data_generation.py` (Blender renderer)
- Complete parameter guide: textures, speeds, angles, distances, center randomization, lighting, motion blur
- How videos are generated using Blender's 3D rendering engine
- Treadmill geometry creation (belt surface, enclosure, materials)
- Texture generation (procedural stripes, custom images, animation)
- Camera and lighting setup for realistic rendering
- Metadata tracking in CSV files
- Filename encoding scheme (all parameters in filename)
- JSON dataset formatting for LLaMA-Factory

**When to read it:** Generating custom datasets, understanding video parameters, troubleshooting video quality issues.

---

### **03_TRAINING.md** - Model Fine-Tuning Pipeline
**What's in it:**
- `run_full_pipeline.py` orchestration of training, evaluation, and tracking
- Complete training parameter reference (LoRA rank, alpha, dropout, learning rate)
- Dynamic YAML config generation from command-line arguments
- How LLaMA-Factory executes training with 4-bit quantization
- LoRA adapter initialization and why only 10-20M parameters are trainable
- Training loop internals (forward pass, loss calculation, gradient accumulation)
- Optimization techniques (gradient checkpointing, flash attention, mixed precision)
- Output structure (adapter_config.json, adapter_model.bin files)
- Common training issues (OOM, slow training, not learning, overfitting)

**When to read it:** Modifying training hyperparameters, troubleshooting training issues, understanding LoRA fine-tuning.

---

### **04_EVALUATION.md** - Model Performance Testing
**What's in it:**
- `evaluate_pipeline_simple.py` architecture and evaluation loop
- Two evaluation methods: "yesno" vs "moving_stopped" prompts
- Step-by-step inference process (load video → preprocess → generate → parse → compare)
- Video processing parameters (fps=4.0, min/max pixels)
- Response parsing logic with priority order (negative indicators → "no" → positive indicators)
- Comprehensive metrics calculation (accuracy, F1, precision, recall, per-class, per-texture, per-angle)
- Output files: live logs, per-video CSVs, evaluation reports
- Report generation format and structure
- Common evaluation issues (inconsistent predictions, parsing failures, low performance)

**When to read it:** Understanding metrics, debugging evaluation errors, interpreting model outputs.

---

### **05_REPORTING.md** - Interactive HTML Reports
**What's in it:**
- `generate_experiment_report.py` report generation system
- Required input data (evaluation reports, per-video CSVs, video files)
- HTML report structure (metrics section, charts, per-texture/angle breakdowns, video grid)
- Interactive features (filtering, video playback, color-coded badges)
- Video path handling (relative paths for browser compatibility)
- Video preview grid with predictions and metadata
- Report customization (CSS styling, adding sections)
- Advanced features (video comparison, error analysis, interactive filtering)
- **NEW (2025-12-07):** Statistical Failure Analysis tab with:
  - Dynamic feature extraction from filenames (auto-detects dist, angle, blur, object, etc.)
  - Chi-squared and Fisher's exact tests for statistical significance
  - Per-feature failure examples grouped by worst-performing values
  - Automatic conclusions identifying which features affect model accuracy
- **NEW (2025-12-09):** Subset/Conditional Significance Analysis:
  - Single-level subset analysis (e.g., "Is stripe significant within distance=2.0?")
  - Two-level combination analysis (e.g., "Is bg significant within distance=2.0 AND angle=30?")
  - Only shows NEW findings not already in global analysis
  - Uses both Chi-squared and Fisher's exact tests at p<0.05 threshold
  - Shows ALL failed videos in significant groups (collapsible sections)

**When to read it:** Customizing report appearance, adding new report sections, troubleshooting video display issues, **understanding why a model fails on specific features**, **finding hidden patterns in feature combinations**.

---

### **06_EXPERIMENT_TRACKING.md** - CSV-Based Experiment Logging
**What's in it:**
- `experiment_tracker.py` class and `experiments_log.csv` structure
- 196-column CSV schema covering every experiment aspect
- Column categories: metadata, dataset config, train/test parameters, model config, LoRA, training hyperparameters, evaluation config, results
- Usage workflow (start experiment → update status → log results → finalize)
- Helper function `parse_evaluation_results()` for extracting metrics from reports
- Querying experiments with Pandas, Excel, SQL
- Example analyses (parameter effects, generalization, training duration, failures)
- Best practices (backups, manual notes, validation, archiving)

**When to read it:** Analyzing experiment results, comparing experiments, understanding what gets tracked.

---

## Finding Specific Information

| I want to know about... | Read this file... | Section... |
|-------------------------|-------------------|------------|
| Overall system flow | 00_OVERVIEW.md | Architecture |
| Running an experiment | 01_ORCHESTRATION.md | Script Structure |
| Dataset parameters | 02_DATA_GENERATION.md | Part 1: building_dataset.py |
| Video generation | 02_DATA_GENERATION.md | Part 2: synthetic_data_generation.py |
| LoRA configuration | 03_TRAINING.md | Argument Parsing |
| **Multi-adapter comparison** | 03_TRAINING.md | Multi-Adapter Comparison |
| Training loop details | 03_TRAINING.md | Part 2: Training with LLaMA-Factory |
| Evaluation metrics | 04_EVALUATION.md | Metrics Calculation |
| Response parsing | 04_EVALUATION.md | Response Parsing |
| HTML report structure | 05_REPORTING.md | HTML Report Structure |
| Video display issues | 05_REPORTING.md | Video Path Handling |
| **Why model fails on specific features** | 05_REPORTING.md | Statistical Failure Analysis Tab |
| **Why model fails on feature combinations** | 05_REPORTING.md | Subset/Conditional Significance Analysis |
| **Statistical significance testing** | 05_REPORTING.md | Statistical Failure Analysis Tab |
| What gets logged | 06_EXPERIMENT_TRACKING.md | CSV Structure |
| Querying experiments | 06_EXPERIMENT_TRACKING.md | Querying Experiments |
| Troubleshooting | All files | Common Issues sections |

---

## Document Size Reference

| File | Lines | Complexity | Time to Read |
|------|-------|------------|--------------|
| README.md | 494 | ★☆☆☆☆ | 10 min |
| 00_OVERVIEW.md | 157 | ★☆☆☆☆ | 8 min |
| 01_ORCHESTRATION.md | 357 | ★★★☆☆ | 20 min |
| 02_DATA_GENERATION.md | 802 | ★★★★☆ | 40 min |
| 03_TRAINING.md | 736 | ★★★★★ | 40 min |
| 04_EVALUATION.md | 801 | ★★★★★ | 45 min |
| 05_REPORTING.md | 566 | ★★★☆☆ | 30 min |
| 06_EXPERIMENT_TRACKING.md | 555 | ★★★☆☆ | 30 min |

**Estimated total reading time: ~4 hours**

---

## Quick Decision Tree

**START HERE:**

1. **Never used the pipeline before?**
   - Read: README.md → 00_OVERVIEW.md → 01_ORCHESTRATION.md
   - Then run: `./run_experiment.sh`

2. **Want to change experiment parameters?**
   - Dataset parameters → 02_DATA_GENERATION.md (Section: Parameter Guide)
   - Training parameters → 03_TRAINING.md (Section: Argument Parsing)
   - Evaluation parameters → 04_EVALUATION.md (Section: Evaluation Prompts)

3. **Need to understand results?**
   - Metrics meaning → 04_EVALUATION.md (Section: Metrics Calculation)
   - Comparing experiments → 06_EXPERIMENT_TRACKING.md (Section: Querying Experiments)
   - Report customization → 05_REPORTING.md (Section: Report Customization)

4. **Something broke?**
   - Check relevant file's "Common Issues" section
   - Example: Training OOM → 03_TRAINING.md (Section: Common Training Issues)

5. **Want to analyze trends?**
   - Read: 06_EXPERIMENT_TRACKING.md (Section: Example Analyses)
   - Use: Pandas, Excel, or SQL on experiments_log.csv

---

## Key Takeaways by File

### 00_OVERVIEW.md
- **Key Insight:** 5-stage pipeline (Orchestration → Data → Training → Eval → Reporting)
- **Critical Info:** Code syncs to GPU server, runs in Docker, downloads results back

### 01_ORCHESTRATION.md
- **Key Insight:** `run_experiment.sh` orchestrates everything via SSH + Docker
- **Critical Info:** Edit this file to change experiment parameters (angle, texture, epochs, etc.)
- **New Feature (2025-12-08):** CSV sync now uses `experiment_id` instead of `tail -n 1`. This fixes issues when running multiple evaluations on the same dataset with different models/eval methods. The experiment_id is unique and ensures the correct CSV row is synced and used for HTML report generation.
- **Technical Fix (2025-12-08):** CSV row retrieval now uses base64-encoded Python scripts to avoid shell escaping issues when passing code through SSH. This prevents silent failures where the experiment_id was retrieved but row data was not appended to local CSV.
- **Re-evaluation Support:** When syncing an experiment that already exists in local CSV (same experiment_id), the script now updates the existing row instead of duplicating it.

### 02_DATA_GENERATION.md
- **Key Insight:** Blender generates synthetic videos with full parameter control
- **Critical Info:** All parameters encoded in filename for easy identification
- **New Feature (2025-12-07):** Center randomization allows treadmill to be positioned randomly in frame (with 30% min visibility) while maintaining position consistency across all frames in a video

### 03_TRAINING.md
- **Key Insight:** LoRA fine-tunes only 10-20M parameters (not full 7B model)
- **Critical Info:** 4-bit quantization enables training on consumer GPUs
- **New Feature (2025-12-08):** Multi-adapter support with `--adapter-type` parameter. Supported adapters:
  - `lora` - Base LoRA (default)
  - `lora+` - LoRA with learning rate ratio (loraplus_lr_ratio: 16.0)
  - `dora` - Weight-Decomposed LoRA
  - `rslora` - Rank Stabilization LoRA
  - `oft` - Orthogonal Fine-Tuning (different finetuning_type)
  - ~~`pissa`~~ - **NOT SUPPORTED** (requires special initialization script)
- **New Option (2025-12-08):** `--no-quantization` flag disables 4-bit quantization for full precision training
- **New Script (2025-12-08):** `run_all_adapters.sh` - Run all adapters sequentially for comparison
  - Builds dataset once, reuses for all adapters
  - Supports: lora, lora+, dora, rslora, oft
  - Usage: `./run_all_adapters.sh --epochs 7 --no-quantization -y`

### 04_EVALUATION.md
- **Key Insight:** Evaluates both base and fine-tuned models with comprehensive metrics
- **Critical Info:** Response parsing logic determines if model predicts "moving" or "stopped"

### 05_REPORTING.md
- **Key Insight:** Generates interactive HTML with video players and metrics
- **Critical Info:** Uses relative paths so reports work when opened directly in browser
- **New Feature (2025-12-07):** Failure Analysis tab automatically identifies which features (dist, angle, blur, etc.) significantly affect model accuracy using Chi-squared tests, with video examples grouped by worst-performing values
- **Fix (2025-12-08):** `get_experiment_from_csv()` now returns the LAST match when searching by dataset_name (most recent experiment), not the first match. This ensures correct report generation when multiple experiments use the same dataset.
- **Critical Fix (2025-12-08):** Eval folder matching by adapter path. When generating reports for experiments that share the same dataset (e.g., multi-adapter comparison), the report generator now matches eval folders by `lora_output_dir` instead of using the most recent folder. This ensures each report uses the correct per-video predictions for its specific adapter/model. The fix reads `evaluation_metadata_*.txt` files to match adapter paths.

### 06_EXPERIMENT_TRACKING.md
- **Key Insight:** 196-column CSV logs everything for complete reproducibility
- **Critical Info:** Query with Pandas/Excel to analyze experiment trends

---

## Most Common Tasks & Where to Look

**Task:** Run first experiment
**Files:** README.md → 00_OVERVIEW.md → 01_ORCHESTRATION.md
**Action:** `./run_experiment.sh`

**Task:** Change camera angle from 0° to 52°
**File:** 01_ORCHESTRATION.md (Section: Stage 2 parameters)
**Action:** Edit `--train_view_angle "0"` and `--test_view_angle "52"`

**Task:** Train for more epochs
**File:** 01_ORCHESTRATION.md (Section: Stage 3 parameters)
**Action:** Edit `--num_train_epochs 20`

**Task:** Change adapter type (LoRA variant)
**File:** 03_TRAINING.md (Section: Adapter Types)
**Action:** Use `--adapter-type <type>` where type is one of: lora, lora+, dora, rslora, oft
**Note:** PiSSA is NOT supported because it requires special initialization (scripts/pissa_init.py)
**Examples:**
```bash
# Use DoRA adapter
./run_experiment.sh --adapter-type dora --epochs 5

# Use LoRA+ adapter
./run_experiment.sh --adapter-type lora+ --epochs 5

# Use rsLoRA adapter
./run_experiment.sh --adapter-type rslora --epochs 5

# Use OFT adapter
./run_experiment.sh --adapter-type oft --epochs 5

# Disable quantization (full precision)
./run_experiment.sh --adapter-type dora --no-quantization --epochs 5
```

**Task:** Compare ALL adapters on the same dataset (NEW!)
**File:** 03_TRAINING.md (Section: Multi-Adapter Comparison)
**Action:** Use `./run_all_adapters.sh` - runs all adapters sequentially with same dataset
**Supported adapters:** lora, lora+, dora, rslora, oft (NOT pissa - requires special init)
**Examples:**
```bash
# Run all 5 adapters with new dataset
./run_all_adapters.sh --epochs 7 --batch-size 14 --no-quantization -y

# Run all adapters on existing dataset
./run_all_adapters.sh --dataset-name "_exp_20251207_172213" \
    --epochs 7 --no-quantization -y

# Run specific adapters only
./run_all_adapters.sh --adapters "lora,dora,oft" \
    --epochs 7 --batch-size 14 -y

# Run with custom model name base (creates: mymodel_lora, mymodel_dora, etc.)
./run_all_adapters.sh --model-name "mymodel" --epochs 5 -y
```
**How it works:**
- First adapter: builds dataset (unless --dataset-name provided)
- Subsequent adapters: reuse the same dataset (--skip-datasets automatically)
- Each adapter gets its own CSV entry and HTML report
- All training parameters (epochs, batch-size, learning-rate) shared across adapters

**Task:** Understand why accuracy is only 70%
**File:** 04_EVALUATION.md (Section: Common Evaluation Issues)
**Action:** Check live log files to see actual model outputs

**Task:** Find best experiment from last week
**File:** 06_EXPERIMENT_TRACKING.md (Section: Querying Experiments)
**Action:** Use Pandas to sort by `finetuned_model_accuracy`

**Task:** Customize HTML report colors
**File:** 05_REPORTING.md (Section: Report Customization)
**Action:** Edit CSS in `generate_experiment_report.py`

**Task:** Re-evaluate existing dataset with a different model
**File:** 01_ORCHESTRATION.md (Section: Skip Stages)
**Action:** Use `--skip-datasets --skip-training --dataset-name "_exp_YYYYMMDD_HHMMSS" --eval-model-path "saves/your_model"`
**Example:**
```bash
./run_experiment.sh \
  --skip-datasets --skip-training \
  --dataset-name "_exp_20251207_172213" \
  --eval-model-path "saves/dist_gen_1to10_left_gray_dora_dec7" \
  --eval-method "moving_stopped" -y
```
**Note:** This creates a new experiment (new experiment_id) but reuses the existing dataset. The pipeline syncs the correct row by experiment_id and generates a new HTML report.

**Task:** Understand why model fails at certain distances/angles
**File:** 05_REPORTING.md (Section: Statistical Failure Analysis Tab)
**Action:** Open HTML report, click "Failure Analysis" tab - shows which features significantly affect accuracy with video examples

**Task:** Generate videos with randomized treadmill position
**File:** 01_ORCHESTRATION.md (Section: Experiment Parameters) or 02_DATA_GENERATION.md
**Action:** Use `--train-center-randomization` and `--test-center-randomization` flags (0=none, 1=randomized)
**Important:** Center randomization only works when `distance > 1.0`. At distance=1.0, the treadmill fills the entire frame leaving no room for position offset.
**Example via run_experiment.sh:** `--train-center-randomization "0,1"` (generates both centered and randomized combinations)
**Example direct building_dataset.py:** `--center_randomization none,randomized`

**Example full experiment command with center randomization:**
```bash
./run_experiment.sh \
  --test-angles "0.0,30.0" \
  --test-speed "0.0,14.0" \
  --test-texture "subtle_gray_stripes" \
  --test-direction "left" \
  --test-distance "1.0" \
  --test-stripe-gray "226,227,228,229" \
  --test-bg-gray "120,121,122,123" \
  --test-center-randomization "0,1" \
  --train-angles "0.0" \
  --train-speed "0.0,14.0" \
  --train-texture "subtle_gray_stripes" \
  --train-direction "left" \
  --train-distance "1.0" \
  --train-stripe-gray "226" \
  --train-bg-gray "120" \
  --train-center-randomization "0" \
  --skip-training \
  --eval-model-path "saves/dist_gen_1to10_left_gray_dora_dec7" \
  --epochs 1 \
  -y
```

**Task:** Generate videos with randomized distance per video
**File:** 01_ORCHESTRATION.md (Section: Experiment Parameters) or 02_DATA_GENERATION.md
**Action:** Use `--train-distance-randomization "1"` and `--train-distance-range "1.0,1.2"` to enable randomized distance
**Behavior:**
- `--train-distance-randomization "0"` (default): Uses `--train-distance` as discrete values (Cartesian product)
- `--train-distance-randomization "1"`: Generates ONE random distance per video from uniform(min, max)
- Distance is constant throughout each video (all frames same distance)
- Filename includes `distrand` marker when randomization enabled (e.g., `dist1.15_distrand`)

**Example with distance randomization:**
```bash
./run_experiment.sh \
  --train-distance-randomization "1" \
  --train-distance-range "1.0,1.2" \
  --test-distance-randomization "0" \
  --test-distance "1.0,3.0,5.0,8.0,10.0" \
  --train-angles "0.0,30.0" \
  --train-speed "0.0,14.0" \
  --train-texture "subtle_gray_stripes" \
  --train-direction "left" \
  --epochs 5 \
  -y
```

---

**Total Documentation:** 8 files, ~4,600 lines, comprehensive coverage of entire pipeline

**Last Updated:** 2025-12-08 (Added distance randomization per video support with --train-distance-randomization and --train-distance-range flags)
