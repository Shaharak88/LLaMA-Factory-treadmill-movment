#!/usr/bin/env python3
"""
Full Pipeline Runner - End-to-End Treadmill Detection Training & Evaluation

This master orchestration script runs the complete workflow:
1. Dataset Generation: Creates synthetic treadmill videos and dataset JSON (90% train, 10% test)
2. Model Training: Trains LoRA adapter on Qwen2.5-VL using the training set
3. Model Evaluation: Evaluates both base and fine-tuned models on the test set

The script only calls existing scripts and does not implement any logic itself.
All arguments from sub-scripts are exposed with sensible defaults.

Author: AI-Generated
Date: 2025-11-26
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Import experiment tracker
from experiment_tracker import ExperimentTracker, parse_evaluation_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class FullPipelineRunner:
    """
    Orchestrates the complete treadmill detection pipeline.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize pipeline runner.

        Args:
            args: Command-line arguments
        """
        self.args = args
        self.project_root = Path(__file__).parent
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Script paths
        self.building_dataset_script = self.project_root / "building_dataset.py"
        self.train_script = "llamafactory-cli"  # Will be run via docker-compose
        self.evaluate_script = self.project_root / "evaluate_pipeline_simple.py"

        # Dataset names
        self.train_dataset_name = f"{args.dataset_name}_train"
        self.test_dataset_name = f"{args.dataset_name}_test"

        # Training output - use eval_model_path if provided (for skip-training), otherwise use lora_output_dir
        # For training: use custom model_name if provided, otherwise use default lora_output_dir
        if hasattr(args, 'eval_model_path') and args.eval_model_path:
            # Re-evaluation mode: use the specified model path
            self.lora_output_dir = args.eval_model_path
        elif hasattr(args, 'model_name') and args.model_name:
            # Training mode with custom name
            self.lora_output_dir = f"saves/{args.model_name}"
        else:
            # Default training mode
            self.lora_output_dir = args.lora_output_dir

        # CRITICAL: Validate model paths to prevent overwrites
        self._validate_model_paths()

        # Print exact paths for transparency
        self._print_model_configuration()

        # Initialize experiment tracker
        self.tracker = ExperimentTracker(csv_path='data/experiments_log.csv')
        self.evaluation_output_dir = f'evaluation_results_{self.timestamp}'

    def _validate_model_paths(self) -> None:
        """
        Validate model path requirements to prevent overwrites.

        This method enforces critical safety rules:
        1. Training mode MUST have --model-name (prevents overwriting default path)
        2. Model name MUST be unique (prevents overwriting existing models)
        3. Evaluation-only mode MUST have --eval-model-path
        4. Eval path MUST exist and contain a valid model

        Raises:
            ValueError: If validation fails
        """
        # Rule 1: Training mode REQUIRES model_name
        if not self.args.skip_training:
            if not hasattr(self.args, 'model_name') or not self.args.model_name:
                raise ValueError(
                    "\n" + "="*70 + "\n"
                    "ERROR: --model-name is REQUIRED when training a new model!\n"
                    "="*70 + "\n"
                    "This ensures your model won't overwrite existing models.\n\n"
                    "Usage:\n"
                    "  --model-name my_experiment_v1\n\n"
                    "The model will be saved to: saves/<model-name>/\n"
                    "="*70
                )

            # Rule 2: Check if model already exists (prevent overwrites)
            model_path = Path(self.lora_output_dir)
            if model_path.exists():
                # Check if it has actual model files
                adapter_files = list(model_path.glob("adapter_*.safetensors")) + \
                               list(model_path.glob("adapter_*.bin")) + \
                               list(model_path.glob("adapter_model.*"))
                if adapter_files:
                    raise ValueError(
                        "\n" + "="*70 + "\n"
                        f"ERROR: Model '{self.args.model_name}' already exists!\n"
                        "="*70 + "\n"
                        f"Path: {model_path.absolute()}\n"
                        f"Found {len(adapter_files)} adapter file(s):\n" +
                        "\n".join(f"  - {f.name}" for f in adapter_files[:5]) +
                        (f"\n  ... and {len(adapter_files)-5} more" if len(adapter_files) > 5 else "") + "\n\n"
                        "This would OVERWRITE the existing model!\n\n"
                        "Solutions:\n"
                        "  1. Use a different --model-name (recommended)\n"
                        "  2. Delete the existing model directory first\n"
                        "  3. Use --skip-training --eval-model-path to re-evaluate this model\n"
                        "="*70
                    )
                logger.info(f"Model directory exists but no adapter files found. Will use for training.")

        # Rule 3: Evaluation-only mode REQUIRES eval_model_path
        if self.args.skip_training:
            if not hasattr(self.args, 'eval_model_path') or not self.args.eval_model_path:
                raise ValueError(
                    "\n" + "="*70 + "\n"
                    "ERROR: --eval-model-path is REQUIRED when using --skip-training!\n"
                    "="*70 + "\n"
                    "You must specify which model to evaluate.\n\n"
                    "Usage:\n"
                    "  --skip-training --eval-model-path saves/my_trained_model\n\n"
                    "Available models can be found in the saves/ directory.\n"
                    "="*70
                )

            # Rule 4: Verify model exists
            model_path = Path(self.lora_output_dir)
            if not model_path.exists():
                raise ValueError(
                    "\n" + "="*70 + "\n"
                    f"ERROR: Model path does not exist!\n"
                    "="*70 + "\n"
                    f"Path: {model_path.absolute()}\n\n"
                    "Cannot evaluate non-existent model.\n\n"
                    "Solutions:\n"
                    "  1. Check the path is correct\n"
                    "  2. List available models: ls -la saves/\n"
                    "  3. Train the model first (remove --skip-training)\n"
                    "="*70
                )

            # Verify model has adapter files
            adapter_files = list(model_path.glob("adapter_*.safetensors")) + \
                           list(model_path.glob("adapter_*.bin")) + \
                           list(model_path.glob("adapter_model.*"))
            if not adapter_files:
                raise ValueError(
                    "\n" + "="*70 + "\n"
                    f"ERROR: No adapter files found in model directory!\n"
                    "="*70 + "\n"
                    f"Path: {model_path.absolute()}\n\n"
                    "This directory does not contain a trained model.\n\n"
                    "Solutions:\n"
                    "  1. Check you specified the correct model path\n"
                    "  2. Train a model first (remove --skip-training)\n"
                    "  3. Verify the model trained successfully\n"
                    "="*70
                )

            logger.info(f"Found valid model with {len(adapter_files)} adapter file(s) at: {model_path.absolute()}")

    def _print_model_configuration(self) -> None:
        """Print model path configuration for user visibility."""
        logger.info("\n" + "="*70)
        logger.info("MODEL PATH CONFIGURATION")
        logger.info("="*70)
        logger.info(f"Model save/load path: {Path(self.lora_output_dir).absolute()}")

        if hasattr(self.args, 'model_name') and self.args.model_name:
            logger.info(f"Model name: {self.args.model_name}")
        else:
            logger.info(f"Model name: (using default path)")

        if self.args.skip_training:
            logger.info(f"Mode: EVALUATION ONLY (--skip-training)")
            logger.info(f"Evaluating existing model at: {self.lora_output_dir}")
        else:
            logger.info(f"Mode: TRAINING + EVALUATION")
            logger.info(f"New model will be saved to: {self.lora_output_dir}")

        logger.info("="*70 + "\n")

    def run_command(self, cmd: List[str], description: str, timeout: int = None) -> None:
        """
        Run a command and log output.

        Args:
            cmd: Command as list of strings
            description: Description of the command
            timeout: Optional timeout in seconds
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"{description}")
        logger.info(f"{'='*70}")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=False,  # Show output in real-time
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"{description} failed with return code {result.returncode}")

            logger.info(f"[OK] {description} completed successfully")

        except subprocess.TimeoutExpired:
            logger.error(f"[FAIL] {description} timed out")
            raise
        except Exception as e:
            logger.error(f"[FAIL] {description} failed: {e}")
            raise

    def step1_generate_datasets(self) -> None:
        """
        Step 1: Generate training and test datasets.

        Creates two separate datasets:
        - Training set (90% of total videos)
        - Test set (10% of total videos)
        """
        logger.info("\n" + "#"*70)
        logger.info("# STEP 1: DATASET GENERATION")
        logger.info("#"*70)

        # Update tracker status
        self.tracker.update_status('dataset', 'in_progress')

        # Calculate train/test split
        num_train = int(self.args.num_videos * self.args.train_split)
        num_test = self.args.num_videos - num_train

        logger.info(f"Total videos: {self.args.num_videos}")
        logger.info(f"Training videos: {num_train} ({self.args.train_split*100:.0f}%)")
        logger.info(f"Test videos: {num_test} ({(1-self.args.train_split)*100:.0f}%)")

        # Generate training dataset
        logger.info(f"\n### Generating TRAINING dataset: {self.train_dataset_name} ###")
        train_cmd = self._build_dataset_command(self.train_dataset_name, num_train, seed=self.args.seed)
        self.run_command(train_cmd, "Training Dataset Generation", timeout=7200)

        # Generate test dataset (different seed for diversity)
        logger.info(f"\n### Generating TEST dataset: {self.test_dataset_name} ###")
        test_cmd = self._build_dataset_command(
            self.test_dataset_name,
            num_test,
            seed=self.args.seed + 10000  # Different seed for test set
        )
        self.run_command(test_cmd, "Test Dataset Generation", timeout=7200)

        # Update tracker status
        self.tracker.update_status('dataset', 'completed')

    def _build_dataset_command(self, dataset_name: str, num_videos: int, seed: int) -> List[str]:
        """
        Build dataset generation command.

        Args:
            dataset_name: Name for the dataset
            num_videos: Number of videos to generate
            seed: Random seed

        Returns:
            List[str]: Command as list of strings
        """
        cmd = [
            'python3',
            str(self.building_dataset_script),
            '--dataset_name', dataset_name,
            '--num_videos', str(num_videos),
            '--seed', str(seed)
        ]

        # Add optional parameters if provided
        if self.args.vary_parameters:
            cmd.append('--vary_parameters')

        if self.args.texture_type:
            cmd.extend(['--texture_type', self.args.texture_type])

        if self.args.direction:
            cmd.extend(['--direction', self.args.direction])

        if self.args.speed_range:
            cmd.extend(['--speed_range', self.args.speed_range])

        if self.args.resolution:
            cmd.extend(['--resolution', self.args.resolution])

        if self.args.fps:
            cmd.extend(['--fps', str(self.args.fps)])

        if self.args.duration:
            cmd.extend(['--duration', str(self.args.duration)])

        if self.args.view_angle:
            cmd.extend(['--view_angle', str(self.args.view_angle)])

        if self.args.brightness:
            cmd.extend(['--brightness', str(self.args.brightness)])

        if self.args.contrast:
            cmd.extend(['--contrast', str(self.args.contrast)])

        if self.args.lighting_variation:
            cmd.extend(['--lighting_variation', self.args.lighting_variation])

        if self.args.lighting_intensity:
            cmd.extend(['--lighting_intensity', str(self.args.lighting_intensity)])

        if self.args.motion_blur:
            cmd.extend(['--motion_blur', str(self.args.motion_blur)])

        if self.args.camera_noise:
            cmd.extend(['--camera_noise', str(self.args.camera_noise)])

        if self.args.edge_width:
            cmd.extend(['--edge_width', str(self.args.edge_width)])

        if hasattr(self.args, 'distance') and self.args.distance:
            cmd.extend(['--distance', str(self.args.distance)])

        # Add subtle_gray_stripes specific parameters if provided
        if hasattr(self.args, 'stripe_width') and self.args.stripe_width:
            cmd.extend(['--stripe_width', str(self.args.stripe_width)])

        if hasattr(self.args, 'stripe_spacing') and self.args.stripe_spacing:
            cmd.extend(['--stripe_spacing', str(self.args.stripe_spacing)])

        if hasattr(self.args, 'stripe_gray') and self.args.stripe_gray:
            cmd.extend(['--stripe_gray', str(self.args.stripe_gray)])

        if hasattr(self.args, 'background_gray') and self.args.background_gray:
            cmd.extend(['--background_gray', str(self.args.background_gray)])

        if hasattr(self.args, 'stripe_distance_variance') and self.args.stripe_distance_variance:
            cmd.extend(['--stripe_distance_variance', str(self.args.stripe_distance_variance)])

        # Add object placement parameters if provided
        if hasattr(self.args, 'add_object') and self.args.add_object:
            cmd.append('--add_object')
            if hasattr(self.args, 'object_type') and self.args.object_type:
                cmd.extend(['--object_type', str(self.args.object_type)])
            if hasattr(self.args, 'object_position') and self.args.object_position:
                cmd.extend(['--object_position', str(self.args.object_position)])
            if hasattr(self.args, 'object_size') and self.args.object_size:
                cmd.extend(['--object_size', str(self.args.object_size)])
            if hasattr(self.args, 'num_objects') and self.args.num_objects:
                cmd.extend(['--num_objects', str(self.args.num_objects)])

        # Add camera blur parameters if provided
        if hasattr(self.args, 'add_blur') and self.args.add_blur:
            cmd.append('--add_blur')
            if hasattr(self.args, 'blur_type') and self.args.blur_type:
                cmd.extend(['--blur_type', str(self.args.blur_type)])
            if hasattr(self.args, 'blur_intensity') and self.args.blur_intensity:
                cmd.extend(['--blur_intensity', str(self.args.blur_intensity)])
            if hasattr(self.args, 'random_blur_variation') and self.args.random_blur_variation:
                cmd.append('--random_blur_variation')

        return cmd

    def step2_train_model(self) -> None:
        """
        Step 2: Train LoRA adapter on the training dataset.

        Creates a custom training config YAML and runs training.
        """
        logger.info("\n" + "#"*70)
        logger.info("# STEP 2: MODEL TRAINING")
        logger.info("#"*70)

        # Update tracker status
        self.tracker.update_status('training', 'in_progress')

        # Create custom training config
        config_path = self._create_training_config()

        # Build training command
        # If running inside container, use llamafactory-cli directly
        # If running outside container, use docker-compose
        if self.args.use_docker:
            cmd = [
                'docker-compose', 'run', '--rm', 'llamafactory',
                'llamafactory-cli', 'train', str(config_path)
            ]
        else:
            cmd = [
                'llamafactory-cli', 'train', str(config_path)
            ]

        self.run_command(cmd, "LoRA Model Training", timeout=14400)  # 4 hour timeout

        # Update tracker status
        self.tracker.update_status('training', 'completed')

    def _create_training_config(self) -> Path:
        """
        Create custom training configuration YAML.

        Returns:
            Path: Path to created config file
        """
        logger.info("Creating training configuration...")

        config_path = self.project_root / "examples" / "train_qlora" / f"qwen25vl_lora_pipeline_{self.timestamp}.yaml"

        # Determine finetuning type and adapter-specific lines
        adapter_type = self.args.adapter_type
        finetuning_type = "lora"  # default for most adapters
        adapter_lines = ""

        if adapter_type == 'dora':
            adapter_lines = "use_dora: true"
        elif adapter_type == 'lora+':
            adapter_lines = "loraplus_lr_ratio: 16.0"
        elif adapter_type == 'rslora':
            adapter_lines = "use_rslora: true"
        elif adapter_type == 'pissa':
            adapter_lines = "pissa_init: true\npissa_iter: 16"
        elif adapter_type == 'oft':
            finetuning_type = "oft"
        # 'lora' uses defaults (no extra lines needed)

        logger.info(f"  Adapter type: {adapter_type} (finetuning_type: {finetuning_type})")

        # Build LoRA configuration (only for non-OFT adapters)
        if adapter_type != 'oft':
            lora_config = f"""lora_target: all
lora_rank: {self.args.lora_rank}
lora_alpha: {self.args.lora_alpha}
lora_dropout: {self.args.lora_dropout}"""
        else:
            # OFT doesn't use LoRA parameters
            lora_config = ""

        # Build quantization section (conditional)
        if self.args.no_quantization:
            quantization_section = "# Quantization disabled (full precision)"
            logger.info("  Quantization: disabled (full precision)")
        else:
            quantization_section = f"""quantization_bit: {self.args.quantization_bit}
quantization_method: bitsandbytes"""
            logger.info(f"  Quantization: {self.args.quantization_bit}-bit (bitsandbytes)")

        config_content = f"""### Model Configuration
model_name_or_path: {self.args.model_name_or_path}

### Method Configuration
stage: sft
do_train: true
finetuning_type: {finetuning_type}
{lora_config}
{adapter_lines}

### Dataset Configuration
dataset: {self.train_dataset_name}
template: {self.args.template}
cutoff_len: {self.args.cutoff_len}
overwrite_cache: true
preprocessing_num_workers: 4

### Output Configuration
output_dir: {self.lora_output_dir}
logging_steps: {self.args.logging_steps}
save_steps: {self.args.save_steps}
plot_loss: true
overwrite_output_dir: true

### Training Hyperparameters
per_device_train_batch_size: {self.args.per_device_train_batch_size}
gradient_accumulation_steps: {self.args.gradient_accumulation_steps}
learning_rate: {self.args.learning_rate}
num_train_epochs: {self.args.num_train_epochs}
lr_scheduler_type: {self.args.lr_scheduler_type}
warmup_ratio: {self.args.warmup_ratio}
bf16: {str(self.args.bf16).lower()}
fp16: {str(self.args.fp16).lower()}

### Evaluation Configuration (no validation split - train on 100% of training set)
val_size: 0.0
per_device_eval_batch_size: 1
eval_strategy: "no"
eval_steps: {self.args.eval_steps}
save_strategy: "steps"
logging_strategy: "steps"

### Memory Optimization
{quantization_section}
gradient_checkpointing: true
ddp_timeout: 180000000

### Additional Settings
report_to: tensorboard
seed: {self.args.seed}
"""

        with open(config_path, 'w') as f:
            f.write(config_content)

        logger.info(f"  [OK] Created config: {config_path}")
        return config_path

    def step3_evaluate_models(self) -> None:
        """
        Step 3: Evaluate both base and fine-tuned models on the test set.

        Runs inference on test dataset for both models and compares accuracy.
        """
        logger.info("\n" + "#"*70)
        logger.info("# STEP 3: MODEL EVALUATION")
        logger.info("#"*70)

        # Print exact paths being used
        logger.info("\n" + "="*70)
        logger.info("EVALUATION PATHS")
        logger.info("="*70)
        logger.info(f"Model being evaluated: {Path(self.lora_output_dir).absolute()}")
        logger.info(f"Test dataset: {self.test_dataset_name}")
        logger.info(f"Output directory: {Path(self.evaluation_output_dir).absolute()}")
        logger.info(f"CSV log file: {self.tracker.csv_path.absolute()}")
        logger.info("="*70 + "\n")

        # Update tracker status
        self.tracker.update_status('evaluation', 'in_progress')

        # Build evaluation command
        cmd = [
            'python3',
            str(self.evaluate_script),
            '--model_name_or_path', self.args.model_name_or_path,
            '--adapter_name_or_path', self.lora_output_dir,
            '--test_dataset', self.test_dataset_name,
            '--dataset_dir', 'data',
            '--template', self.args.template,
            '--output_dir', self.evaluation_output_dir,
            '--eval_method', self.args.eval_method
        ]

        # Add inference parameters
        if self.args.eval_max_new_tokens:
            cmd.extend(['--max_new_tokens', str(self.args.eval_max_new_tokens)])

        if self.args.eval_batch_size:
            cmd.extend(['--batch_size', str(self.args.eval_batch_size)])

        if self.args.eval_video_fps:
            cmd.extend(['--video_fps', str(self.args.eval_video_fps)])

        if self.args.eval_video_maxlen:
            cmd.extend(['--video_maxlen', str(self.args.eval_video_maxlen)])

        # Removed gpu_memory_utilization as it is not needed for simple eval

        self.run_command(cmd, "Model Evaluation", timeout=7200)

        # Parse and update evaluation results with all metrics
        base_results, finetuned_results = parse_evaluation_results(self.evaluation_output_dir)
        if base_results is not None or finetuned_results is not None:
            self.tracker.update_evaluation_results(
                base_results=base_results,
                finetuned_results=finetuned_results
            )

        # Update tracker status
        self.tracker.update_status('evaluation', 'completed')

        # Print final paths where results are saved
        logger.info("\n" + "="*70)
        logger.info("EVALUATION COMPLETE - RESULTS SAVED TO:")
        logger.info("="*70)
        logger.info(f"CSV Log: {self.tracker.csv_path.absolute()}")

        # Find the actual evaluation report file
        eval_reports = list(Path(self.evaluation_output_dir).glob('evaluation_report_*.txt'))
        if eval_reports:
            logger.info(f"Text Report: {max(eval_reports, key=lambda p: p.stat().st_mtime).absolute()}")
        else:
            logger.info(f"Text Report: {Path(self.evaluation_output_dir).absolute()}/evaluation_report_*.txt")

        logger.info(f"Model Evaluated: {Path(self.lora_output_dir).absolute()}")
        logger.info("="*70 + "\n")

    def run(self) -> None:
        """Execute the complete pipeline."""
        try:
            logger.info("="*70)
            logger.info("FULL PIPELINE RUNNER - Treadmill Motion Detection")
            logger.info("="*70)
            logger.info(f"Run timestamp: {self.timestamp}")
            logger.info(f"Configuration: {vars(self.args)}")

            # Start experiment tracking (pass actual calculated model path)
            experiment_id = self.tracker.start_experiment(self.args, actual_model_path=self.lora_output_dir)
            logger.info(f"Experiment ID: {experiment_id}")

            # Step 1: Generate datasets
            if not self.args.skip_dataset:
                self.step1_generate_datasets()
            else:
                logger.info("\n### Skipping dataset generation (--skip_dataset) ###")

            # Step 2: Train model
            if not self.args.skip_training:
                self.step2_train_model()
            else:
                logger.info("\n### Skipping model training (--skip_training) ###")

            # Step 3: Evaluate models
            if not self.args.skip_evaluation:
                self.step3_evaluate_models()
            else:
                logger.info("\n### Skipping evaluation (--skip_evaluation) ###")

            logger.info("\n" + "="*70)
            logger.info("[OK] FULL PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*70)

            # Finalize experiment tracking
            self.tracker.finalize_experiment()

        except KeyboardInterrupt:
            logger.warning("\n\nPipeline interrupted by user")
            # Mark current stage as failed
            if hasattr(self, 'tracker') and self.tracker.current_experiment_id:
                # Try to determine which stage failed
                for stage in ['dataset', 'training', 'evaluation']:
                    self.tracker.update_status(stage, 'failed')
            sys.exit(1)
        except Exception as e:
            logger.error(f"\n\nPipeline failed: {e}", exc_info=True)
            # Mark current stage as failed
            if hasattr(self, 'tracker') and self.tracker.current_experiment_id:
                # Mark any in-progress stages as failed
                for stage in ['dataset', 'training', 'evaluation']:
                    self.tracker.update_status(stage, 'failed')
            sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Full pipeline: dataset generation -> training -> evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Basic usage - run complete pipeline with defaults
  python3 run_full_pipeline.py --dataset_name my_treadmill_exp --num_videos 100

  # With parameter variation for diverse dataset
  python3 run_full_pipeline.py \\
    --dataset_name diverse_treadmill \\
    --num_videos 200 \\
    --vary_parameters \\
    --seed 42

  # Custom texture types and speeds (combination mode)
  python3 run_full_pipeline.py \\
    --dataset_name combo_experiment \\
    --texture_type stripes,noise,rubber,grid \\
    --speed_range 2.0,5.0 \\
    --num_videos 0  # Ignored in combo mode

  # Skip certain steps
  python3 run_full_pipeline.py \\
    --dataset_name existing_exp \\
    --skip_dataset \\
    --num_videos 100

Docker Usage:

  # Run inside Docker container (recommended)
  docker exec llamafactory python3 /app/run_full_pipeline.py \\
    --dataset_name docker_experiment \\
    --num_videos 50

  # Run outside Docker container (uses docker-compose for training)
  python3 run_full_pipeline.py \\
    --dataset_name local_experiment \\
    --num_videos 50 \\
    --use_docker

Notes:
  - Train/test split defaults to 90/10
  - Test set is completely held out from training
  - By default, assumes running INSIDE Docker container
  - Use --use_docker when running OUTSIDE container (for training step)
  - All sub-script arguments are exposed with defaults
  - Use --skip_dataset, --skip_training, or --skip_evaluation to run partial pipeline
        """
    )

    # Pipeline control
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Base name for datasets (will create <name>_train and <name>_test)')
    parser.add_argument('--use_docker', action='store_true',
                       help='Run training via docker-compose (use when running outside container)')
    parser.add_argument('--skip_dataset', action='store_true',
                       help='Skip dataset generation step')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip model training step')
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='Skip evaluation step')

    # Dataset generation arguments
    dataset_group = parser.add_argument_group('Dataset Generation')
    dataset_group.add_argument('--num_videos', type=int, default=100,
                              help='Total number of videos to generate (default: 100)')
    dataset_group.add_argument('--train_split', type=float, default=1.0,
                              help='Training split ratio (default: 1.0 = 100%% of data used) - parameter not implemented')
    dataset_group.add_argument('--seed', type=int, default=42,
                              help='Random seed for reproducibility (default: 42)')
    dataset_group.add_argument('--vary_parameters', action='store_true',
                              help='Automatically vary parameters for diversity')
    dataset_group.add_argument('--texture_type', type=str, default='stripes',
                              help='Texture type (default: stripes). Supports comma-separated values')
    dataset_group.add_argument('--direction', type=str, default='right',
                              help='Motion direction (default: right). Supports comma-separated values')
    dataset_group.add_argument('--speed_range', type=str, default='1.0,8.0',
                              help='Speed range (default: 1.0,8.0). Supports comma-separated values')
    dataset_group.add_argument('--resolution', type=str, default='640x480',
                              help='Video resolution (default: 640x480)')
    dataset_group.add_argument('--fps', type=str, default='4',
                              help='Frames per second (default: 4)')
    dataset_group.add_argument('--duration', type=str, default='12.0',
                              help='Video duration in seconds (default: 12.0)')
    dataset_group.add_argument('--view_angle', type=str, default='0.0',
                              help='Camera viewing angle in degrees (default: 0.0)')
    dataset_group.add_argument('--brightness', type=str, default='0.0',
                              help='Brightness adjustment (default: 0.0)')
    dataset_group.add_argument('--contrast', type=str, default='1.0',
                              help='Contrast adjustment (default: 1.0)')
    dataset_group.add_argument('--lighting_variation', type=str, default='none',
                              help='Lighting variation type (default: none)')
    dataset_group.add_argument('--lighting_intensity', type=str, default='0.5',
                              help='Lighting intensity (default: 0.5)')
    dataset_group.add_argument('--motion_blur', type=str, default='0',
                              help='Motion blur amount (default: 0)')
    dataset_group.add_argument('--camera_noise', type=str, default='0.0',
                              help='Camera noise level (default: 0.0)')
    dataset_group.add_argument('--edge_width', type=str, default='0.1',
                              help='Belt edge width (default: 0.1)')
    dataset_group.add_argument('--distance', type=str, default='1.0',
                              help='Distance factor: 1.0 = normal, >1.0 = smaller/further (default: 1.0)')

    # Subtle gray stripes parameters (for subtle_gray_stripes texture type)
    dataset_group.add_argument('--stripe_width', type=str, default='10',
                              help='Stripe width in pixels for subtle_gray_stripes (default: 10). Supports comma-separated values.')
    dataset_group.add_argument('--stripe_spacing', type=str, default='60',
                              help='Stripe spacing in pixels for subtle_gray_stripes (default: 60). Supports comma-separated values.')
    dataset_group.add_argument('--stripe_gray', type=str, default='125',
                              help='Stripe gray level (0-255) for subtle_gray_stripes (default: 125). Supports comma-separated values.')
    dataset_group.add_argument('--background_gray', type=str, default='140',
                              help='Background gray level (0-255) for subtle_gray_stripes (default: 140). Supports comma-separated values.')
    dataset_group.add_argument('--stripe_distance_variance', type=str, default='0.0',
                              help='Variance (std dev) for stripe spacing in subtle_gray_stripes (default: 0.0). Supports comma-separated values.')

    # Object placement parameters
    dataset_group.add_argument('--add_object', action='store_true',
                              help='Enable object placement on treadmill belt')
    dataset_group.add_argument('--object_type', type=str, default='box',
                              help='Type of object to place (default: box). Options: box, circle, random')
    dataset_group.add_argument('--object_position', type=str, default='center',
                              help='Position of object on belt (default: center). Options: center, left, right, random')
    dataset_group.add_argument('--object_size', type=str, default='medium',
                              help='Size of object: small/medium/large or numeric value (default: medium)')
    dataset_group.add_argument('--num_objects', type=int, default=1,
                              help='Number of objects to place (default: 1)')

    # Camera blur parameters
    dataset_group.add_argument('--add_blur', action='store_true',
                              help='Enable camera blur effects')
    dataset_group.add_argument('--blur_type', type=str, default='gaussian',
                              help='Type of blur effect (default: gaussian). Options: motion, gaussian, random')
    dataset_group.add_argument('--blur_intensity', type=str, default='0.3',
                              help='Blur intensity: light/medium/heavy or 0.0-1.0 (default: 0.3)')
    dataset_group.add_argument('--random_blur_variation', action='store_true',
                              help='Add random blur intensity variation across frames')

    # Dataset Metadata (for CSV tracking - separate train/test parameters)
    metadata_group = parser.add_argument_group('Dataset Metadata',
                                                'Separate train/test dataset parameters for CSV tracking. '
                                                'These are informational only and used for experiment logging.')
    # Training dataset metadata
    metadata_group.add_argument('--train_texture_type', type=str, default='', help='Training texture type (for tracking)')
    metadata_group.add_argument('--train_direction', type=str, default='', help='Training direction (for tracking)')
    metadata_group.add_argument('--train_view_angle', type=str, default='', help='Training view angles (for tracking)')
    metadata_group.add_argument('--train_speed_range', type=str, default='', help='Training speed range (for tracking)')
    metadata_group.add_argument('--train_resolution', type=str, default='', help='Training resolution (for tracking)')
    metadata_group.add_argument('--train_fps', type=str, default='', help='Training FPS (for tracking)')
    metadata_group.add_argument('--train_duration', type=str, default='', help='Training duration (for tracking)')
    metadata_group.add_argument('--train_brightness', type=str, default='', help='Training brightness (for tracking)')
    metadata_group.add_argument('--train_contrast', type=str, default='', help='Training contrast (for tracking)')
    metadata_group.add_argument('--train_lighting_variation', type=str, default='', help='Training lighting variation (for tracking)')
    metadata_group.add_argument('--train_lighting_intensity', type=str, default='', help='Training lighting intensity (for tracking)')
    metadata_group.add_argument('--train_motion_blur', type=str, default='', help='Training motion blur (for tracking)')
    metadata_group.add_argument('--train_camera_noise', type=str, default='', help='Training camera noise (for tracking)')
    metadata_group.add_argument('--train_edge_width', type=str, default='', help='Training edge width (for tracking)')
    metadata_group.add_argument('--train_distance', type=str, default='', help='Training distance factor (for tracking)')
    metadata_group.add_argument('--train_center_randomization', type=str, default='', help='Training center randomization (for tracking)')
    metadata_group.add_argument('--train_distance_randomization', type=str, default='', help='Training distance randomization (for tracking)')
    metadata_group.add_argument('--train_distance_range', type=str, default='', help='Training distance range (for tracking)')
    metadata_group.add_argument('--train_stripe_width', type=str, default='', help='Training stripe width (for tracking)')
    metadata_group.add_argument('--train_stripe_spacing', type=str, default='', help='Training stripe spacing (for tracking)')
    metadata_group.add_argument('--train_stripe_gray', type=str, default='', help='Training stripe gray (for tracking)')
    metadata_group.add_argument('--train_background_gray', type=str, default='', help='Training background gray (for tracking)')
    metadata_group.add_argument('--train_stripe_distance_variance', type=str, default='', help='Training stripe variance (for tracking)')
    metadata_group.add_argument('--train_add_object', type=str, default='', help='Training add object flag (for tracking)')
    metadata_group.add_argument('--train_object_type', type=str, nargs='?', default='', help='Training object type (for tracking)')
    metadata_group.add_argument('--train_object_position', type=str, nargs='?', default='', help='Training object position (for tracking)')
    metadata_group.add_argument('--train_object_size', type=str, nargs='?', default='', help='Training object size (for tracking)')
    metadata_group.add_argument('--train_num_objects', type=str, nargs='?', default='', help='Training num objects (for tracking)')
    metadata_group.add_argument('--train_add_blur', type=str, nargs='?', default='', help='Training add blur flag (for tracking)')
    metadata_group.add_argument('--train_blur_type', type=str, nargs='?', default='', help='Training blur type (for tracking)')
    metadata_group.add_argument('--train_blur_intensity', type=str, nargs='?', default='', help='Training blur intensity (for tracking)')
    metadata_group.add_argument('--train_random_blur_variation', type=str, nargs='?', default='', help='Training random blur variation (for tracking)')
    metadata_group.add_argument('--train_vary_parameters', type=str, default='', help='Training vary parameters (for tracking)')
    # Test dataset metadata
    metadata_group.add_argument('--test_texture_type', type=str, default='', help='Test texture type (for tracking)')
    metadata_group.add_argument('--test_direction', type=str, default='', help='Test direction (for tracking)')
    metadata_group.add_argument('--test_view_angle', type=str, default='', help='Test view angles (for tracking)')
    metadata_group.add_argument('--test_speed_range', type=str, default='', help='Test speed range (for tracking)')
    metadata_group.add_argument('--test_resolution', type=str, default='', help='Test resolution (for tracking)')
    metadata_group.add_argument('--test_fps', type=str, default='', help='Test FPS (for tracking)')
    metadata_group.add_argument('--test_duration', type=str, default='', help='Test duration (for tracking)')
    metadata_group.add_argument('--test_brightness', type=str, default='', help='Test brightness (for tracking)')
    metadata_group.add_argument('--test_contrast', type=str, default='', help='Test contrast (for tracking)')
    metadata_group.add_argument('--test_lighting_variation', type=str, default='', help='Test lighting variation (for tracking)')
    metadata_group.add_argument('--test_lighting_intensity', type=str, default='', help='Test lighting intensity (for tracking)')
    metadata_group.add_argument('--test_motion_blur', type=str, default='', help='Test motion blur (for tracking)')
    metadata_group.add_argument('--test_camera_noise', type=str, default='', help='Test camera noise (for tracking)')
    metadata_group.add_argument('--test_edge_width', type=str, default='', help='Test edge width (for tracking)')
    metadata_group.add_argument('--test_distance', type=str, default='', help='Test distance factor (for tracking)')
    metadata_group.add_argument('--test_center_randomization', type=str, default='', help='Test center randomization (for tracking)')
    metadata_group.add_argument('--test_distance_randomization', type=str, default='', help='Test distance randomization (for tracking)')
    metadata_group.add_argument('--test_distance_range', type=str, default='', help='Test distance range (for tracking)')
    metadata_group.add_argument('--test_stripe_width', type=str, default='', help='Test stripe width (for tracking)')
    metadata_group.add_argument('--test_stripe_spacing', type=str, default='', help='Test stripe spacing (for tracking)')
    metadata_group.add_argument('--test_stripe_gray', type=str, default='', help='Test stripe gray (for tracking)')
    metadata_group.add_argument('--test_background_gray', type=str, default='', help='Test background gray (for tracking)')
    metadata_group.add_argument('--test_stripe_distance_variance', type=str, default='', help='Test stripe variance (for tracking)')
    metadata_group.add_argument('--test_add_object', type=str, default='', help='Test add object flag (for tracking)')
    metadata_group.add_argument('--test_object_type', type=str, nargs='?', default='', help='Test object type (for tracking)')
    metadata_group.add_argument('--test_object_position', type=str, nargs='?', default='', help='Test object position (for tracking)')
    metadata_group.add_argument('--test_object_size', type=str, nargs='?', default='', help='Test object size (for tracking)')
    metadata_group.add_argument('--test_num_objects', type=str, nargs='?', default='', help='Test num objects (for tracking)')
    metadata_group.add_argument('--test_add_blur', type=str, nargs='?', default='', help='Test add blur flag (for tracking)')
    metadata_group.add_argument('--test_blur_type', type=str, nargs='?', default='', help='Test blur type (for tracking)')
    metadata_group.add_argument('--test_blur_intensity', type=str, nargs='?', default='', help='Test blur intensity (for tracking)')
    metadata_group.add_argument('--test_random_blur_variation', type=str, nargs='?', default='', help='Test random blur variation (for tracking)')
    metadata_group.add_argument('--test_vary_parameters', type=str, default='', help='Test vary parameters (for tracking)')

    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model_name_or_path', type=str,
                            default='Qwen/Qwen2.5-VL-3B-Instruct',
                            help='Base model name or path (default: Qwen/Qwen2.5-VL-3B-Instruct)')
    model_group.add_argument('--template', type=str, default='qwen2_vl',
                            help='Template name (default: qwen2_vl)')

    # Training arguments
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--lora_output_dir', type=str,
                            default='saves/qwen2vl-treadmill-lora-pipeline',
                            help='LoRA output directory (default: saves/qwen2vl-treadmill-lora-pipeline)')
    train_group.add_argument('--model_name', type=str, default='',
                            help='Custom model name for the trained model (will be saved as saves/<model_name>). If not provided, uses lora_output_dir.')
    train_group.add_argument('--eval_model_path', type=str, default='',
                            help='Path to existing model to evaluate when using --skip_training (e.g., saves/my_model or saves/qwen2vl-treadmill-lora-pipeline). Only used with --skip_training flag.')
    train_group.add_argument('--lora_rank', type=int, default=8,
                            help='LoRA rank (default: 8)')
    train_group.add_argument('--lora_alpha', type=int, default=16,
                            help='LoRA alpha (default: 16)')
    train_group.add_argument('--lora_dropout', type=float, default=0.05,
                            help='LoRA dropout (default: 0.05)')
    train_group.add_argument('--adapter_type', type=str, default='lora',
                            choices=['lora', 'lora+', 'dora', 'rslora', 'pissa', 'oft'],
                            help='Adapter type: lora, lora+, dora, rslora, pissa, oft (default: lora)')
    train_group.add_argument('--no_quantization', action='store_true', default=False,
                            help='Disable 4-bit quantization (use full precision adapters)')
    train_group.add_argument('--cutoff_len', type=int, default=8192,
                            help='Cutoff length (default: 8192)')
    train_group.add_argument('--per_device_train_batch_size', type=int, default=1,
                            help='Per device train batch size (default: 1)')
    train_group.add_argument('--gradient_accumulation_steps', type=int, default=8,
                            help='Gradient accumulation steps (default: 8)')
    train_group.add_argument('--learning_rate', type=float, default=5.0e-5,
                            help='Learning rate (default: 5.0e-5)')
    train_group.add_argument('--num_train_epochs', type=int, default=1,
                            help='Number of training epochs (default: 1)')
    train_group.add_argument('--lr_scheduler_type', type=str, default='cosine',
                            help='Learning rate scheduler type (default: cosine)')
    train_group.add_argument('--warmup_ratio', type=float, default=0.1,
                            help='Warmup ratio (default: 0.1)')
    train_group.add_argument('--bf16', action='store_true',
                            help='Use bf16 precision')
    train_group.add_argument('--fp16', action='store_true', default=True,
                            help='Use fp16 precision (default: True)')
    train_group.add_argument('--quantization_bit', type=int, default=4,
                            help='Quantization bits (default: 4)')
    train_group.add_argument('--logging_steps', type=int, default=10,
                            help='Logging steps (default: 10)')
    train_group.add_argument('--save_steps', type=int, default=100,
                            help='Save steps (default: 100)')
    train_group.add_argument('--eval_steps', type=int, default=100,
                            help='Evaluation steps (default: 100)')

    # Evaluation arguments
    eval_group = parser.add_argument_group('Evaluation Configuration')
    eval_group.add_argument('--eval_method', type=str, default='yesno',
                           choices=['yesno', 'moving_stopped'],
                           help='Evaluation method: "yesno" or "moving_stopped" (default: yesno)')
    eval_group.add_argument('--eval_max_new_tokens', type=int, default=128,
                           help='Max new tokens for evaluation (default: 128)')
    eval_group.add_argument('--eval_batch_size', type=int, default=1024,
                           help='Batch size for evaluation (default: 1024)')
    eval_group.add_argument('--eval_video_fps', type=float, default=2.0,
                           help='Video FPS for evaluation (default: 2.0)')
    eval_group.add_argument('--eval_video_maxlen', type=int, default=128,
                           help='Max video length for evaluation (default: 128)')
    eval_group.add_argument('--gpu_memory_utilization', type=float, default=0.8,
                           help='GPU memory utilization for vllm (default: 0.8)')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    runner = FullPipelineRunner(args)
    runner.run()


if __name__ == '__main__':
    main()
