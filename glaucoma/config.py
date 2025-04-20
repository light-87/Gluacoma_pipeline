"""
Configuration Module

Centralized configuration for the glaucoma detection pipeline.
Streamlined for clarity and ease of use.
"""

import argparse
import json
from pathlib import Path
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union

@dataclass
class DataConfig:
    """Data configuration."""
    # Data directories
    data_dirs: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        'REFUGE': {
            'images': 'data/REFUGE/Images_Square',
            'masks': 'data/REFUGE/Masks_Square'
        },
        'ORIGA': {
            'images': 'data/ORIGA/Images_Square',
            'masks': 'data/ORIGA/Masks_Square'
        },
        'G1020': {
            'images': 'data/G1020/Images_Square',
            'masks': 'data/G1020/Masks_Square'
        }
    })
    
    # Dataset file
    dataset_file: Optional[str] = None
    
    # Random seed
    random_state: int = 42
    
    # Data splits
    validation_split: float = 0.15
    test_split: float = 0.15
    
    # Performance options
    cache_size: int = 100
    prefetch_size: int = 4

@dataclass
class ModelConfig:
    """Model configuration."""
    # Architecture
    architecture: str = "unet"
    encoder: str = "resnet34"
    
    # Model parameters
    pretrained: bool = True
    in_channels: int = 3
    num_classes: int = 1
    
    # Checkpoint
    checkpoint_path: Optional[str] = None

@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    # Image parameters
    image_size: int = 224
    normalization: str = "imagenet"
    
    # Augmentation
    augmentation_enabled: bool = True
    rotation_range: float = 15.0
    shift_range: float = 0.1
    scale_range: float = 0.1
    brightness_contrast_range: float = 0.2

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic training parameters
    epochs: int = 30
    batch_size: int = 16
    num_workers: int = 4
    
    # Optimization
    learning_rate: float = 0.001
    optimizer: str = "adam"
    weight_decay: float = 0.0001
    
    # Loss function
    loss_function: str = "combined"
    dice_weight: float = 1.0
    bce_weight: float = 0.5
    focal_weight: float = 0.5
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    
    # Advanced training options
    grad_accum_steps: int = 1  # Gradient accumulation steps
    
    # Hardware
    use_gpu: bool = True
    use_amp: bool = False  # Automatic mixed precision
    
    # Training monitoring
    monitor_metric: str = "val_dice"
    monitor_mode: str = "max"
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    
    # Checkpoint saving
    save_every: int = 5
    grad_clip: float = 1.0

@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    # Evaluation parameters
    batch_size: int = 16
    threshold: float = 0.5
    
    # Metrics to calculate
    metrics: List[str] = field(default_factory=lambda: ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1'])
    
    # Test-time augmentation
    use_tta: bool = False
    tta_scales: List[float] = field(default_factory=lambda: [0.75, 1.0, 1.25])
    tta_flips: bool = True
    tta_rotations: List[int] = field(default_factory=lambda: [0, 90, 180, 270])
    
    # Visualization
    generate_visualizations: bool = True
    sample_count: int = 10

@dataclass
class LoggingConfig:
    """Logging configuration."""
    # Logging directories
    log_dir: str = "logs"
    
    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "glaucoma-detection"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # Pipeline steps to execute
    steps: List[str] = field(default_factory=lambda: ['load', 'preprocess', 'train', 'evaluate'])
    
    # Description
    description: str = "Glaucoma Detection Pipeline"

@dataclass
class Config:
    """Master configuration."""
    # Component configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    # Output directory
    output_dir: str = "output"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        preprocessing_config = PreprocessingConfig(**config_dict.get('preprocessing', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        pipeline_config = PipelineConfig(**config_dict.get('pipeline', {}))
        
        return cls(
            data=data_config,
            model=model_config,
            preprocessing=preprocessing_config,
            training=training_config,
            evaluation=evaluation_config,
            logging=logging_config,
            pipeline=pipeline_config,
            output_dir=config_dict.get('output_dir', 'output')
        )
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'Config':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def save_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

def get_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for command-line arguments."""
    parser = argparse.ArgumentParser(description="Glaucoma Detection Pipeline")
    
    # Pipeline configuration
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--steps', type=str, default='train,evaluate', 
                      help='Pipeline steps to run (comma-separated, options: load,preprocess,train,evaluate,ensemble)')
    
    # Data configuration
    parser.add_argument('--dataset-file', type=str, help='Path to dataset CSV file')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    
    # Model configuration
    parser.add_argument('--architecture', type=str, default='unet', 
                      choices=['unet', 'unetplusplus', 'deeplabv3', 'deeplabv3plus', 'fpn', 'pspnet'], 
                      help='Model architecture')
    parser.add_argument('--encoder', type=str, default='resnet34', help='Encoder backbone')
    parser.add_argument('--checkpoint-path', type=str, help='Path to model checkpoint for evaluation')
    
    # Preprocessing configuration
    parser.add_argument('--image-size', type=int, default=224, help='Image size')
    parser.add_argument('--no-augmentation', action='store_false', dest='augmentation_enabled',
                      help='Disable data augmentation')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--loss-function', type=str, default='combined', 
                      choices=['dice', 'bce', 'focal', 'tversky', 'combined'],
                      help='Loss function')
    parser.add_argument('--focal-weight', type=float, default=0.5, 
                      help='Weight for focal loss in combined loss')
    parser.add_argument('--use-amp', action='store_true', help='Use automatic mixed precision training')
    parser.add_argument('--grad-accum-steps', type=int, default=1, 
                      help='Number of steps for gradient accumulation')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                      help='Value for gradient clipping (0 to disable)')
    
    # Evaluation configuration
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary segmentation')
    parser.add_argument('--use-tta', action='store_true', help='Use test-time augmentation')
    
    # Logging configuration
    parser.add_argument('--use-wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='glaucoma-detection', 
                      help='Weights & Biases project name')
    parser.add_argument('--run-name', type=str, help='Run name for logging')
    
    # Config file
    parser.add_argument('--config-file', type=str, help='Path to JSON configuration file')
    
    return parser

def parse_args_and_create_config(args=None) -> Config:
    """Create configuration from command-line arguments.
    
    Args:
        args: Parsed command-line arguments. If None, arguments will be parsed
             from command line.
        
    Returns:
        Configuration object
    """
    # If args is None, parse arguments
    if args is None:
        parser = get_argument_parser()
        args = parser.parse_args()
    
    # If config file is provided, load it
    if hasattr(args, 'config_file') and args.config_file:
        config = Config.from_json(args.config_file)
    else:
        config = Config()
    
    # Update config with command-line arguments
    if hasattr(args, 'output_dir') and args.output_dir:
        config.output_dir = args.output_dir
    
    if hasattr(args, 'steps') and args.steps:
        config.pipeline.steps = args.steps.split(',')
    
    if hasattr(args, 'dataset_file') and args.dataset_file:
        config.data.dataset_file = args.dataset_file
    
    if hasattr(args, 'random_state') and args.random_state:
        config.data.random_state = args.random_state
    
    if hasattr(args, 'architecture') and args.architecture:
        config.model.architecture = args.architecture
    
    if hasattr(args, 'encoder') and args.encoder:
        config.model.encoder = args.encoder
    
    if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
        config.model.checkpoint_path = args.checkpoint_path
    
    if hasattr(args, 'image_size') and args.image_size:
        config.preprocessing.image_size = args.image_size
    
    if hasattr(args, 'augmentation_enabled'):
        config.preprocessing.augmentation_enabled = args.augmentation_enabled
    
    if hasattr(args, 'epochs') and args.epochs:
        config.training.epochs = args.epochs
    
    if hasattr(args, 'batch_size') and args.batch_size:
        config.training.batch_size = args.batch_size
        config.evaluation.batch_size = args.batch_size
    
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    if hasattr(args, 'loss_function') and args.loss_function:
        config.training.loss_function = args.loss_function
    
    if hasattr(args, 'focal_weight') and args.focal_weight is not None:
        config.training.focal_weight = args.focal_weight
    
    if hasattr(args, 'use_amp') and args.use_amp:
        config.training.use_amp = args.use_amp
        
    if hasattr(args, 'grad_accum_steps') and args.grad_accum_steps:
        config.training.grad_accum_steps = args.grad_accum_steps
        
    if hasattr(args, 'grad_clip') and args.grad_clip is not None:
        config.training.grad_clip = args.grad_clip
    
    if hasattr(args, 'threshold') and args.threshold is not None:
        config.evaluation.threshold = args.threshold
        
    if hasattr(args, 'use_tta') and args.use_tta:
        config.evaluation.use_tta = args.use_tta
    
    if hasattr(args, 'use_wandb') and args.use_wandb:
        config.logging.use_wandb = args.use_wandb
    
    if hasattr(args, 'wandb_project') and args.wandb_project:
        config.logging.wandb_project = args.wandb_project
    
    if hasattr(args, 'run_name') and args.run_name:
        config.logging.run_name = args.run_name
    
    return config