# glaucoma/config.py
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union

@dataclass
class DataConfig:
    """Data configuration."""
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
    dataset_file: Optional[str] = None
    output_dataset_file: Optional[str] = None
    save_dataset: bool = True
    random_state: int = 42
    validation_split: float = 0.15
    test_split: float = 0.15

@dataclass
class ModelConfig:
    """Model configuration."""
    architecture: str = "unet"
    encoder: str = "resnet34"
    pretrained: bool = True
    in_channels: int = 3
    num_classes: int = 1
    checkpoint_path: Optional[str] = None

@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    image_size: int = 224
    normalization: str = "imagenet"
    augmentation_enabled: bool = True
    rotation_range: float = 15.0
    shift_range: float = 0.1
    scale_range: float = 0.1
    brightness_contrast_range: float = 0.2

@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 30
    batch_size: int = 16
    num_workers: int = 4
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "combined"
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    use_gpu: bool = True
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    use_scheduler: bool = True
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.1
    min_lr: float = 1e-6
    early_stopping: bool = True
    patience: int = 10
    save_every: int = 5
    grad_clip: float = 0.0

@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    batch_size: int = 16
    threshold: float = 0.5
    metrics: List[str] = field(default_factory=lambda: ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1'])
    generate_visualizations: bool = True
    sample_count: int = 10

@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: str = "logs"
    use_wandb: bool = False
    wandb_project: str = "glaucoma-detection"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    steps: List[str] = field(default_factory=lambda: ['train', 'evaluate'])
    force_rerun: bool = False
    description: str = "Glaucoma Detection Pipeline"

@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
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
                      help='Pipeline steps to run (comma-separated)')
    
    # Data configuration
    parser.add_argument('--dataset-file', type=str, help='Path to dataset CSV file')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    
    # Model configuration
    parser.add_argument('--architecture', type=str, default='unet', 
                      choices=['unet', 'unetplusplus', 'deeplabv3', 'fpn'], 
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
                      choices=['dice', 'bce', 'combined', 'focal', 'jaccard', 'tversky'],
                      help='Loss function')
    
    # Evaluation configuration
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary segmentation')
    
    # Logging configuration
    parser.add_argument('--use-wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='glaucoma-detection', 
                      help='Weights & Biases project name')
    parser.add_argument('--run-name', type=str, help='Run name for logging')
    
    # Config file
    parser.add_argument('--config-file', type=str, help='Path to JSON configuration file')
    
    return parser

def parse_args_and_create_config(args=None):
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
    if args.config_file:
        config = Config.from_json(args.config_file)
    else:
        config = Config()
    
    # Update config with command-line arguments
    if args.output_dir:
        config.output_dir = args.output_dir
    
    if args.steps:
        config.pipeline.steps = args.steps.split(',')
    
    if args.dataset_file:
        config.data.dataset_file = args.dataset_file
    
    if args.random_state:
        config.data.random_state = args.random_state
    
    if args.architecture:
        config.model.architecture = args.architecture
    
    if args.encoder:
        config.model.encoder = args.encoder
    
    if args.checkpoint_path:
        config.model.checkpoint_path = args.checkpoint_path
    
    if args.image_size:
        config.preprocessing.image_size = args.image_size
    
    if hasattr(args, 'augmentation_enabled'):
        config.preprocessing.augmentation_enabled = args.augmentation_enabled
    
    if args.epochs:
        config.training.epochs = args.epochs
    
    if args.batch_size:
        config.training.batch_size = args.batch_size
        config.evaluation.batch_size = args.batch_size
    
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    if args.loss_function:
        config.training.loss_function = args.loss_function
    
    if args.threshold:
        config.evaluation.threshold = args.threshold
    
    if args.use_wandb:
        config.logging.use_wandb = args.use_wandb
    
    if args.wandb_project:
        config.logging.wandb_project = args.wandb_project
    
    if args.run_name:
        config.logging.run_name = args.run_name
    
    return config