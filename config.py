"""
Configuration Module

This module defines the configuration for the glaucoma detection pipeline.
"""

import json
import yaml
import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import uuid

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
    cache_size: int = 0  # Disabled by default to avoid multiprocessing issues
    prefetch_size: int = 0  # Disabled by default to avoid multiprocessing issues

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
class LossConfig:
    """Loss configuration."""
    # Loss function type
    loss_function: str = "combined"
    
    # Loss weights
    dice_weight: float = 1.0
    bce_weight: float = 0.0  
    focal_weight: float = 1.0  
    
    # Focal loss parameters
    focal_gamma: Optional[float] = 2.0
    focal_alpha: Optional[float] = 0.25
    
    def __post_init__(self):
        """Initialize with appropriate defaults based on loss function."""
        if self.loss_function == "dice":
            self.dice_weight = 1.0
            self.bce_weight = 0.0
            self.focal_weight = 0.0
            self.focal_gamma = None
            self.focal_alpha = None
        elif self.loss_function == "bce":
            self.dice_weight = 0.0
            self.bce_weight = 1.0
            self.focal_weight = 0.0
            self.focal_gamma = None
            self.focal_alpha = None
        elif self.loss_function == "focal":
            self.dice_weight = 0.0
            self.bce_weight = 0.0
            self.focal_weight = 1.0

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic training parameters
    epochs: int = 20
    batch_size: int = 16
    num_workers: int = 4
    
    # Optimization
    learning_rate: float = 0.001
    optimizer: str = "adam"
    weight_decay: float = 0.0001
    
    # Loss settings are now in LossConfig
    loss: LossConfig = field(default_factory=LossConfig)
    
    # Advanced training options
    grad_accum_steps: int = 1  # Gradient accumulation steps
    
    # Hardware
    device: str = "cuda"
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
    
    # CDR calculation
    calculate_cdr: bool = False
    cdr_method: str = "diameter"
    
    # Test-time augmentation
    use_tta: bool = False
    tta_scales: List[float] = field(default_factory=lambda: [0.75, 1.0, 1.25])
    tta_flips: bool = True
    tta_rotations: List[int] = field(default_factory=lambda: [0, 90, 180, 270])
    
    # Visualization
    generate_visualizations: bool = True
    sample_count: int = 10

@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    # Basic settings - always enabled
    enabled: bool = True
    project: str = "glaucoma-detection"
    entity: Optional[str] = None
    
    # Run information
    name: Optional[str] = None
    group: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    # Additional settings
    log_model: bool = True
    log_code: bool = True
    log_dataset: bool = False
    save_output: bool = True

@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # Unique run identifier
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Pipeline steps to execute
    steps: List[str] = field(default_factory=lambda: ['load', 'preprocess', 'train', 'evaluate'])
    
    # Description
    description: str = "Glaucoma Detection Pipeline"
    
    # Output paths
    output_dir: str = "output"
    checkpoint_dir: Optional[str] = None
    
    # Wandb configuration
    wandb: WandbConfig = field(default_factory=WandbConfig)

@dataclass
class Config:
    """Master configuration."""
    # Component configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    def __post_init__(self):
        """Initialize derived properties."""
        # Set checkpoint directory if not specified
        if not self.pipeline.checkpoint_dir:
            self.pipeline.checkpoint_dir = os.path.join(
                self.pipeline.output_dir, 
                f"run_{self.pipeline.run_id}", 
                "checkpoints"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary, filtering out None values."""
        full_dict = asdict(self)
        
        # Function to recursively filter out None values
        def filter_none(d):
            if not isinstance(d, dict):
                return d
            return {k: filter_none(v) for k, v in d.items() 
                    if v is not None}
        
        return filter_none(full_dict)
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        # Extract nested configs
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        preprocessing_config = PreprocessingConfig(**config_dict.get('preprocessing', {}))
        
        # Handle nested loss config in training
        training_dict = config_dict.get('training', {})
        loss_dict = training_dict.pop('loss', {})
        loss_config = LossConfig(**loss_dict)
        training_config = TrainingConfig(**training_dict, loss=loss_config)
        
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        
        # Handle nested wandb config in pipeline
        pipeline_dict = config_dict.get('pipeline', {})
        wandb_dict = pipeline_dict.pop('wandb', {})
        wandb_config = WandbConfig(**wandb_dict)
        pipeline_config = PipelineConfig(**pipeline_dict, wandb=wandb_config)
        
        return cls(
            data=data_config,
            model=model_config,
            preprocessing=preprocessing_config,
            training=training_config,
            evaluation=evaluation_config,
            pipeline=pipeline_config
        )
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'Config':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    def save_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    def save_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, indent=4)
    
    def generate_run_name(self) -> str:
        """Generate a descriptive run name for WandB."""
        # Format: {model}_{encoder}_{dataset}_{timestamp}_{unique_id}
        timestamp = datetime.now().strftime("%Y%m%d")
        short_id = str(uuid.uuid4())[:8]
        
        datasets = "_".join(list(self.data.data_dirs.keys())[:2])  # First two datasets
        
        return f"{self.model.architecture}_{self.model.encoder}_{datasets}_{timestamp}_{short_id}"

def get_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for command-line arguments."""
    parser = argparse.ArgumentParser(description="Glaucoma Detection Pipeline")
    
    # Config file
    parser.add_argument('--config', type=str, help='Path to config file (JSON or YAML)')
    
    # Pipeline configuration
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--steps', type=str,
                      help='Pipeline steps to run (comma-separated, options: load,preprocess,train,evaluate,ensemble)')
    
    # Data configuration
    parser.add_argument('--data-dirs', type=str, help='Path to JSON file with data directories')
    parser.add_argument('--dataset-file', type=str, help='Path to dataset CSV file')
    
    # Model configuration
    parser.add_argument('--architecture', type=str,
                      choices=['unet', 'unetplusplus', 'deeplabv3', 'deeplabv3plus', 'fpn', 'pspnet'], 
                      help='Model architecture')
    parser.add_argument('--encoder', type=str, help='Encoder backbone')
    parser.add_argument('--checkpoint-path', type=str, help='Path to model checkpoint for evaluation')
    
    # Preprocessing configuration
    parser.add_argument('--image-size', type=int, help='Image size')
    parser.add_argument('--no-augmentation', action='store_false', dest='augmentation_enabled',
                      help='Disable data augmentation')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], help='Optimizer')
    parser.add_argument('--loss-function', type=str, 
                      choices=['dice', 'bce', 'focal', 'tversky', 'combined'],
                      help='Loss function')
    parser.add_argument('--device', type=str, help='Device to use (cuda or cpu)')

    parser.add_argument('--dice-weight', type=float, help='Weight for dice loss component')
    parser.add_argument('--bce-weight', type=float, help='Weight for BCE loss component')
    parser.add_argument('--focal-weight', type=float, help='Weight for focal loss component')
    parser.add_argument('--focal-gamma', type=float, help='Gamma parameter for focal loss')
    parser.add_argument('--focal-alpha', type=float, help='Alpha parameter for focal loss')
    
    # Evaluation configuration
    parser.add_argument('--threshold', type=float, help='Threshold for binary segmentation')
    parser.add_argument('--use-tta', action='store_true', help='Use test-time augmentation')
    parser.add_argument('--calculate-cdr', action='store_true', help='Calculate Cup-to-Disc Ratio')
    
    # WandB configuration
    parser.add_argument('--wandb-project', type=str, help='WandB project name')
    parser.add_argument('--wandb-entity', type=str, help='WandB entity (username or team)')
    parser.add_argument('--wandb-name', type=str, help='WandB run name')
    parser.add_argument('--wandb-group', type=str, help='WandB group')
    parser.add_argument('--wandb-tags', type=str, help='WandB tags (comma-separated)')
    parser.add_argument('--wandb-notes', type=str, help='WandB notes')
    
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
    
    # Start with default config
    config = Config()
    
    # If config file is provided, load it
    if hasattr(args, 'config') and args.config:
        if args.config.endswith('.json'):
            config = Config.from_json(args.config)
        elif args.config.endswith(('.yml', '.yaml')):
            config = Config.from_yaml(args.config)
        else:
            raise ValueError(f"Unknown config file format: {args.config}")
    
    # Update config with command-line arguments
    # Pipeline configuration
    if hasattr(args, 'output_dir') and args.output_dir:
        config.pipeline.output_dir = args.output_dir
    
    if hasattr(args, 'steps') and args.steps:
        config.pipeline.steps = args.steps.split(',')
    
    # Data configuration
    if hasattr(args, 'data_dirs') and args.data_dirs:
        with open(args.data_dirs, 'r') as f:
            config.data.data_dirs = json.load(f)
    
    if hasattr(args, 'dataset_file') and args.dataset_file:
        config.data.dataset_file = args.dataset_file
    
    # Model configuration
    if hasattr(args, 'architecture') and args.architecture:
        config.model.architecture = args.architecture
    
    if hasattr(args, 'encoder') and args.encoder:
        config.model.encoder = args.encoder
    
    if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
        config.model.checkpoint_path = args.checkpoint_path
    
    # Preprocessing configuration
    if hasattr(args, 'image_size') and args.image_size:
        config.preprocessing.image_size = args.image_size
    
    if hasattr(args, 'augmentation_enabled'):
        config.preprocessing.augmentation_enabled = args.augmentation_enabled
    
    # Training configuration
    if hasattr(args, 'epochs') and args.epochs:
        config.training.epochs = args.epochs
    
    if hasattr(args, 'batch_size') and args.batch_size:
        config.training.batch_size = args.batch_size
        config.evaluation.batch_size = args.batch_size
    
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    if hasattr(args, 'optimizer') and args.optimizer:
        config.training.optimizer = args.optimizer
    
    if hasattr(args, 'loss_function') and args.loss_function:
        loss_type = args.loss_function.lower()
        config.training.loss.loss_function = loss_type
        
        # Reset parameters based on the selected loss function
        if loss_type == 'dice':
            config.training.loss.dice_weight = 1.0
            config.training.loss.bce_weight = 0.0
            config.training.loss.focal_weight = 0.0
            config.training.loss.focal_gamma = None  # Use None instead of 0.0
            config.training.loss.focal_alpha = None
        elif loss_type == 'bce':
            config.training.loss.dice_weight = 0.0
            config.training.loss.bce_weight = 1.0
            config.training.loss.focal_weight = 0.0
            config.training.loss.focal_gamma = None
            config.training.loss.focal_alpha = None
        elif loss_type == 'focal':
            config.training.loss.dice_weight = 0.0
            config.training.loss.bce_weight = 0.0
            config.training.loss.focal_weight = 1.0
        elif loss_type == 'combined':
            # For combined, you can leave the defaults or update them
            # based on additional arguments if provided
            if hasattr(args, 'dice_weight') and args.dice_weight is not None:
                config.training.loss.dice_weight = args.dice_weight
            if hasattr(args, 'focal_weight') and args.focal_weight is not None:
                config.training.loss.focal_weight = args.focal_weight
            if hasattr(args, 'bce_weight') and args.bce_weight is not None:
                config.training.loss.bce_weight = args.bce_weight
    
    if hasattr(args, 'device') and args.device:
        config.training.device = args.device
    
    # Evaluation configuration
    if hasattr(args, 'threshold') and args.threshold is not None:
        config.evaluation.threshold = args.threshold
    
    if hasattr(args, 'use_tta') and args.use_tta:
        config.evaluation.use_tta = args.use_tta
    
    if hasattr(args, 'calculate_cdr') and args.calculate_cdr:
        config.evaluation.calculate_cdr = args.calculate_cdr
    
    # WandB configuration - always enabled
    if hasattr(args, 'wandb_project') and args.wandb_project:
        config.pipeline.wandb.project = args.wandb_project
    
    if hasattr(args, 'wandb_entity') and args.wandb_entity:
        config.pipeline.wandb.entity = args.wandb_entity
    
    if hasattr(args, 'wandb_name') and args.wandb_name:
        config.pipeline.wandb.name = args.wandb_name
    
    if hasattr(args, 'wandb_group') and args.wandb_group:
        config.pipeline.wandb.group = args.wandb_group
    
    if hasattr(args, 'wandb_tags') and args.wandb_tags:
        config.pipeline.wandb.tags = args.wandb_tags.split(',')
    
    if hasattr(args, 'wandb_notes') and args.wandb_notes:
        config.pipeline.wandb.notes = args.wandb_notes
    
    # If no wandb name specified, generate one
    if not config.pipeline.wandb.name:
        config.pipeline.wandb.name = config.generate_run_name()
    
    # Update checkpoint directory
    config.__post_init__()
    
    return config