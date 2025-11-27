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

        # Training output
        self.lora_output_dir = args.lora_output_dir

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

            logger.info(f"✓ {description} completed successfully")

        except subprocess.TimeoutExpired:
            logger.error(f"✗ {description} timed out")
            raise
        except Exception as e:
            logger.error(f"✗ {description} failed: {e}")
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

        # Add subtle_gray_stripes specific parameters if provided
        if hasattr(self.args, 'stripe_width') and self.args.stripe_width:
            cmd.extend(['--stripe_width', str(self.args.stripe_width)])

        if hasattr(self.args, 'stripe_spacing') and self.args.stripe_spacing:
            cmd.extend(['--stripe_spacing', str(self.args.stripe_spacing)])

        if hasattr(self.args, 'stripe_gray') and self.args.stripe_gray:
            cmd.extend(['--stripe_gray', str(self.args.stripe_gray)])

        if hasattr(self.args, 'background_gray') and self.args.background_gray:
            cmd.extend(['--background_gray', str(self.args.background_gray)])

        return cmd

    def step2_train_model(self) -> None:
        """
        Step 2: Train LoRA adapter on the training dataset.

        Creates a custom training config YAML and runs training.
        """
        logger.info("\n" + "#"*70)
        logger.info("# STEP 2: MODEL TRAINING")
        logger.info("#"*70)

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

    def _create_training_config(self) -> Path:
        """
        Create custom training configuration YAML.

        Returns:
            Path: Path to created config file
        """
        logger.info("Creating training configuration...")

        config_path = self.project_root / "examples" / "train_qlora" / f"qwen25vl_lora_pipeline_{self.timestamp}.yaml"

        config_content = f"""### Model Configuration
model_name_or_path: {self.args.model_name_or_path}

### Method Configuration
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: {self.args.lora_rank}
lora_alpha: {self.args.lora_alpha}
lora_dropout: {self.args.lora_dropout}

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

### Evaluation Configuration (using validation split, not test set)
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: {self.args.eval_steps}

### Memory Optimization
quantization_bit: {self.args.quantization_bit}
quantization_method: bitsandbytes
gradient_checkpointing: true
ddp_timeout: 180000000

### Additional Settings
report_to: tensorboard
seed: {self.args.seed}
"""

        with open(config_path, 'w') as f:
            f.write(config_content)

        logger.info(f"  ✓ Created config: {config_path}")
        return config_path

    def step3_evaluate_models(self) -> None:
        """
        Step 3: Evaluate both base and fine-tuned models on the test set.

        Runs inference on test dataset for both models and compares accuracy.
        """
        logger.info("\n" + "#"*70)
        logger.info("# STEP 3: MODEL EVALUATION")
        logger.info("#"*70)

        # Build evaluation command
        cmd = [
            'python3',
            str(self.evaluate_script),
            '--model_name_or_path', self.args.model_name_or_path,
            '--adapter_name_or_path', self.lora_output_dir,
            '--test_dataset', self.test_dataset_name,
            '--dataset_dir', 'data',
            '--template', self.args.template,
            '--output_dir', f'evaluation_results_{self.timestamp}'
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

    def run(self) -> None:
        """Execute the complete pipeline."""
        try:
            logger.info("="*70)
            logger.info("FULL PIPELINE RUNNER - Treadmill Motion Detection")
            logger.info("="*70)
            logger.info(f"Run timestamp: {self.timestamp}")
            logger.info(f"Configuration: {vars(self.args)}")

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
            logger.info("✓✓✓ FULL PIPELINE COMPLETED SUCCESSFULLY ✓✓✓")
            logger.info("="*70)

        except KeyboardInterrupt:
            logger.warning("\n\nPipeline interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"\n\nPipeline failed: {e}", exc_info=True)
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
    dataset_group.add_argument('--train_split', type=float, default=0.9,
                              help='Training split ratio (default: 0.9 = 90%% train, 10%% test)')
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
    dataset_group.add_argument('--fps', type=str, default='30',
                              help='Frames per second (default: 30)')
    dataset_group.add_argument('--duration', type=str, default='5.0',
                              help='Video duration in seconds (default: 5.0)')
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

    # Subtle gray stripes parameters (for subtle_gray_stripes texture type)
    dataset_group.add_argument('--stripe_width', type=str, default='10',
                              help='Stripe width in pixels for subtle_gray_stripes (default: 10). Supports comma-separated values.')
    dataset_group.add_argument('--stripe_spacing', type=str, default='60',
                              help='Stripe spacing in pixels for subtle_gray_stripes (default: 60). Supports comma-separated values.')
    dataset_group.add_argument('--stripe_gray', type=str, default='125',
                              help='Stripe gray level (0-255) for subtle_gray_stripes (default: 125). Supports comma-separated values.')
    dataset_group.add_argument('--background_gray', type=str, default='140',
                              help='Background gray level (0-255) for subtle_gray_stripes (default: 140). Supports comma-separated values.')

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
    train_group.add_argument('--lora_rank', type=int, default=8,
                            help='LoRA rank (default: 8)')
    train_group.add_argument('--lora_alpha', type=int, default=16,
                            help='LoRA alpha (default: 16)')
    train_group.add_argument('--lora_dropout', type=float, default=0.05,
                            help='LoRA dropout (default: 0.05)')
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
