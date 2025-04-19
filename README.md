# Glaucoma Detection Pipeline

A modular machine learning pipeline for automated glaucoma detection from retinal fundus images, using modern libraries and best practices.

## Project Overview

This project implements a clean, efficient pipeline for glaucoma detection from retinal fundus images. The pipeline is built with a focus on:

- **Simplicity**: Clean, readable code with minimal dependencies
- **Performance**: Memory-efficient data handling and optimized processing
- **Usability**: Intuitive interfaces and comprehensive documentation
- **Reproducibility**: Robust experiment tracking and version control
- **Extensibility**: Easy to add new models, metrics, and visualization types

### Datasets

The pipeline works with three glaucoma datasets:

1. **ORIGA** - Singapore Chinese Eye Study dataset containing fundus images with glaucoma diagnosis
2. **REFUGE** - Retinal Fundus Glaucoma Challenge dataset with expert annotations
3. **G1020** - A large collection of retinal images with associated metadata

To maximize consistency, the pipeline focuses on **square images only** from these datasets.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Setup](#data-setup)
- [Usage](#usage)
- [Configuration System](#configuration-system)
- [Pipeline Modules](#pipeline-modules)
- [Run Tracking System](#run-tracking-system)
- [Memory Optimization](#memory-optimization)
- [Adding New Components](#adding-new-components)
- [Ensemble Models](#ensemble-models)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.7+
- pip or conda package manager
- CUDA-compatible GPU (recommended but optional)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/glaucoma-detection.git
cd glaucoma-detection
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install the package in development mode:

```bash
pip install -e .
```

## Project Structure

The project follows a modular structure for better organization and maintainability:

```
glaucoma-detection/
├── README.md                    # Project documentation
├── requirements.txt             # Dependencies
├── setup.py                     # Package installation
├── glaucoma/
│   ├── __init__.py
│   ├── config.py                # Configuration system
│   ├── main.py                  # Main entry point
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            # Data loading functionality
│   │   ├── dataset.py           # PyTorch datasets with efficient loading
│   │   └── augmentation.py      # Data augmentation strategies
│   ├── models/
│   │   ├── __init__.py
│   │   ├── factory.py           # Model creation factory
│   │   ├── losses.py            # Loss functions
│   │   └── ensemble.py          # Ensemble model implementations
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py           # Training loop implementation
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py         # Evaluation functionality
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── visualization.py     # Result visualization
│   └── utils/
│       ├── __init__.py
│       ├── logging.py           # Logging utilities
│       └── wandb_logger.py      # Weights & Biases integration
└── notebooks/                   # Optional Jupyter notebooks
    ├── data_exploration.ipynb
    ├── model_training.ipynb
    └── results_analysis.ipynb
```

## Data Setup

### Dataset Directory Structure

The pipeline expects the following directory structure:

```
data_dir/
├── ORIGA/
│   ├── Images_Square/        # Contains square fundus images (.jpg)
│   └── Masks_Square/         # Contains square mask images (.png)
├── REFUGE/
│   ├── Images_Square/        # Contains square fundus images (.jpg)
│   └── Masks_Square/         # Contains square mask images (.png)
└── G1020/
    ├── Images_Square/        # Contains square fundus images (.jpg)
    └── Masks_Square/         # Contains square mask images (.png)
```

Images need to be preprocessed to a square format before using this pipeline. The pipeline focuses exclusively on square images for consistency and simplicity.

## Usage

### Quick Start

To run the complete pipeline with default settings:

```bash
python -m glaucoma.main
```

### Running Specific Pipeline Steps

You can run specific steps of the pipeline:

```bash
python -m glaucoma.main --steps load,preprocess,train
```

Available steps:
- `load`: Load data into a consolidated format
- `preprocess`: Create data loaders and splits
- `train`: Train the segmentation model
- `evaluate`: Evaluate model performance on test data
- `ensemble`: Evaluate ensemble of models (if configured)

### Example: Training and Evaluation

```bash
# Training a model with specific settings
python -m glaucoma.main --steps train,evaluate --model.architecture unet --model.encoder resnet34 --training.batch_size 16 --training.epochs 50 --output-dir runs/unet_resnet34

# Evaluating a pre-trained model
python -m glaucoma.main --steps evaluate --model.checkpoint_path runs/unet_resnet34/checkpoints/best_model.pt --output-dir runs/evaluation
```

## Configuration System

The pipeline uses a hierarchical configuration system that allows for easy customization through:
1. Default values
2. Configuration files
3. Command-line arguments

### Configuration Structure

The configuration is organized into several sections:

- **Data Configuration**: Dataset paths, splits, etc.
- **Model Configuration**: Architecture, encoder, etc.
- **Preprocessing Configuration**: Image size, augmentation, etc.
- **Training Configuration**: Batch size, learning rate, etc.
- **Evaluation Configuration**: Metrics, thresholds, etc.
- **Logging Configuration**: Logging settings, W&B integration, etc.
- **Pipeline Configuration**: Steps to run, etc.

### Using Configuration Files

You can save and load configurations:

```bash
# Save a config
python -c "from glaucoma.config import Config; Config().save_json('my_config.json')"

# Load a config
python -m glaucoma.main --config-file my_config.json
```

### Command-line Arguments

You can override configuration values via command-line:

```bash
python -m glaucoma.main --training.batch_size 32 --training.learning_rate 0.001
```

### Creating Custom Configurations

Here's an example of a custom configuration:

```python
from glaucoma.config import Config, ModelConfig, TrainingConfig

# Create a custom configuration
config = Config()

# Customize model configuration
config.model = ModelConfig(
    architecture="unet",
    encoder="resnet34",
    pretrained=True
)

# Customize training configuration
config.training = TrainingConfig(
    batch_size=16,
    epochs=50,
    learning_rate=0.001,
    optimizer="adam",
    loss_function="combined"
)

# Save configuration
config.save_json("custom_config.json")
```

## Pipeline Modules

### 1. Data Loading (`data/loader.py`)

The data loading module handles:
- Loading images from multiple datasets
- Consolidating data from multiple sources
- Creating train/validation/test splits
- Saving the combined dataset to CSV

Example:
```python
from glaucoma.data.loader import DatasetLoader

# Create data loader
loader = DatasetLoader({
    'REFUGE': {'images': 'data/REFUGE/Images_Square', 'masks': 'data/REFUGE/Masks_Square'},
    'ORIGA': {'images': 'data/ORIGA/Images_Square', 'masks': 'data/ORIGA/Masks_Square'}
})

# Load datasets
df = loader.load_all_datasets()

# Save dataset
from glaucoma.data.loader import save_dataset
save_dataset(df, 'data/dataset.csv', create_splits=True)
```

### 2. Dataset Handling (`data/dataset.py`)

The dataset module provides:
- Memory-efficient data loading with LRU caching
- Prefetching for improved performance
- Integration with PyTorch's Dataset/DataLoader interface

Example:
```python
import pandas as pd
from glaucoma.data.dataset import create_dataloaders
from glaucoma.data.augmentation import get_training_augmentations, get_validation_augmentations

# Load dataset
df = pd.read_csv('data/dataset.csv')

# Create dataloaders
dataloaders = create_dataloaders(
    df=df,
    transform_train=get_training_augmentations(image_size=(224, 224)),
    transform_val=get_validation_augmentations(image_size=(224, 224)),
    batch_size=16
)

# Access dataloaders
train_loader = dataloaders['train']
val_loader = dataloaders['val']
test_loader = dataloaders['test']
```

### 3. Model Creation (`models/factory.py`)

The model factory provides:
- Easy creation of various segmentation architectures
- Support for pretrained encoders
- Simple checkpoint loading

Example:
```python
from glaucoma.models.factory import create_model, load_checkpoint

# Create model
model_config = {
    'architecture': 'unet',
    'encoder': 'resnet34',
    'pretrained': True,
    'num_classes': 1
}
model = create_model(model_config)

# Load checkpoint
model = load_checkpoint(model, 'checkpoints/best_model.pt')
```

### 4. Training (`training/trainer.py`)

The training module provides:
- Comprehensive training loop
- Learning rate scheduling
- Early stopping
- Model checkpointing
- W&B integration

Example:
```python
from glaucoma.training.trainer import Trainer
from glaucoma.utils.logging import get_logger

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=dataloaders['train'],
    val_loader=dataloaders['val'],
    config=config.training,
    logger=get_logger(),
    checkpoint_dir='checkpoints'
)

# Train model
results = trainer.train()
```

### 5. Evaluation (`evaluation/evaluator.py`)

The evaluation module provides:
- Comprehensive model evaluation
- Calculation of various metrics
- Visualization of results
- Support for ensemble evaluation

Example:
```python
from glaucoma.evaluation.evaluator import Evaluator

# Create evaluator
evaluator = Evaluator(
    model=model,
    dataloader=dataloaders['test'],
    config=config.evaluation,
    logger=get_logger(),
    output_dir='evaluation_results'
)

# Evaluate model
results = evaluator.evaluate()
```

## Run Tracking System

The pipeline includes an integrated run tracking system:

1. **Run ID**: Each run is assigned a unique ID based on timestamp
2. **Output Directory**: Each run has its own output directory
3. **Configuration**: The configuration is saved for reproducibility
4. **Checkpoints**: Models are saved at regular intervals and upon improvement
5. **Logs**: Detailed logs are saved to track progress
6. **Metrics**: Performance metrics are logged
7. **Visualizations**: Results are visualized for analysis

### Weights & Biases Integration

For more advanced experiment tracking, the pipeline integrates with Weights & Biases:

```bash
# Enable W&B logging
python -m glaucoma.main --logging.use_wandb True --logging.wandb_project "glaucoma-detection"
```

Features:
- Hyperparameter tracking
- Metric logging
- Model checkpointing
- Sample predictions visualization
- Confusion matrices
- PR and ROC curves

## Memory Optimization

The pipeline includes several features for memory optimization:

1. **LRU Caching**: Only keeps frequently used images in memory
2. **Lazy Loading**: Loads images on-demand rather than all at once
3. **Prefetching**: Background loading of images for improved performance
4. **Efficient Image Processing**: Uses OpenCV for fast image loading and processing

These optimizations allow the pipeline to handle large datasets even with limited memory.

## Adding New Components

### Adding a New Model

To add a new model architecture:

1. Add the model implementation to the factory function in `models/factory.py`:

```python
def create_model(config: Any) -> nn.Module:
    # Existing code...
    
    # Add your new architecture
    elif architecture == 'your_new_architecture':
        model = YourNewModel(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    
    # Rest of the function...
```

2. Create the model implementation if it's not using a library like segmentation_models_pytorch.

### Adding a New Loss Function

To add a new loss function:

1. Add the loss implementation to `models/losses.py`:

```python
class YourNewLoss(nn.Module):
    """Your new loss function implementation."""
    
    def __init__(self, param1=0.5, param2=0.5):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Your loss implementation
        return loss_value
```

2. Update the `get_loss_function` function to include your new loss:

```python
def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    # Existing code...
    
    elif loss_type == 'your_new_loss':
        return YourNewLoss(**kwargs)
    
    # Rest of the function...
```

### Adding a New Metric

To add a new evaluation metric:

1. Add the metric implementation to `evaluation/metrics.py`:

```python
def calculate_your_new_metric(
    pred: torch.Tensor, 
    target: torch.Tensor
) -> float:
    """Calculate your new metric.
    
    Args:
        pred: Prediction tensor
        target: Target tensor
        
    Returns:
        Your new metric value
    """
    # Your metric implementation
    return metric_value
```

2. Update the `calculate_metrics` function to include your new metric.

## Ensemble Models

The pipeline supports model ensembling through:

### 1. Basic Ensemble Methods

The `models/ensemble.py` module provides several ensemble methods:
- **Average Ensemble**: Average predictions from multiple models
- **Maximum Ensemble**: Take the maximum prediction
- **Weighted Ensemble**: Weighted average of predictions

### 2. Cross-Validation Ensemble

You can create an ensemble from cross-validation models:

```python
from glaucoma.models.ensemble import EnsembleFactory

# Create a cross-validation ensemble
ensemble = EnsembleFactory.create_cross_validation_ensemble(
    base_model_config={
        'architecture': 'unet',
        'encoder': 'resnet34',
        'pretrained': True
    },
    checkpoint_paths=[
        'checkpoints/fold1/best_model.pt',
        'checkpoints/fold2/best_model.pt',
        'checkpoints/fold3/best_model.pt',
        'checkpoints/fold4/best_model.pt',
        'checkpoints/fold5/best_model.pt'
    ],
    ensemble_method='weighted'
)
```

### 3. Multi-Architecture Ensemble

You can create an ensemble with different architectures:

```python
from glaucoma.models.ensemble import EnsembleFactory

# Create a multi-architecture ensemble
ensemble = EnsembleFactory.create_multi_architecture_ensemble(
    model_configs=[
        {'architecture': 'unet', 'encoder': 'resnet34'},
        {'architecture': 'fpn', 'encoder': 'resnet50'},
        {'architecture': 'deeplabv3', 'encoder': 'efficientnet-b4'}
    ],
    checkpoint_paths=[
        'checkpoints/unet/best_model.pt',
        'checkpoints/fpn/best_model.pt',
        'checkpoints/deeplab/best_model.pt'
    ],
    ensemble_method='average'
)
```

### Running Ensemble Evaluation

To evaluate an ensemble:

```bash
python -m glaucoma.main --steps ensemble --ensemble.models.0.model.architecture unet --ensemble.models.0.checkpoint_path checkpoints/unet/best_model.pt --ensemble.models.1.model.architecture fpn --ensemble.models.1.checkpoint_path checkpoints/fpn/best_model.pt --ensemble.ensemble_method average
```

Or using a configuration file:

```json
{
  "ensemble": {
    "models": [
      {
        "model": {
          "architecture": "unet",
          "encoder": "resnet34"
        },
        "checkpoint_path": "checkpoints/unet/best_model.pt"
      },
      {
        "model": {
          "architecture": "fpn",
          "encoder": "resnet50"
        },
        "checkpoint_path": "checkpoints/fpn/best_model.pt"
      }
    ],
    "ensemble_method": "average"
  }
}
```

## Troubleshooting

### Common Issues and Solutions

1. **Memory Errors**:
   - Reduce batch size (`--training.batch_size`)
   - Reduce image size (`--preprocessing.image_size`)
   - Reduce cache size (`--data.cache_size`)
   - Disable prefetching (`--data.prefetch_size 0`)

2. **CUDA Out of Memory**:
   - Reduce batch size
   - Use a smaller encoder backbone
   - Try mixed precision training (`--training.mixed_precision True`)

3. **Slow Training**:
   - Increase number of workers (`--training.num_workers`)
   - Enable prefetching (`--data.prefetch_size 50`)
   - Use a GPU if available

4. **Poor Performance**:
   - Try different architectures
   - Adjust learning rate
   - Increase training epochs
   - Use data augmentation
   - Try ensemble methods

5. **File Not Found Errors**:
   - Check dataset paths in configuration
   - Ensure data follows the expected directory structure
   - Check for typos in file paths

### Getting Help

If you encounter issues not covered here:
1. Check the logs for detailed error messages
2. Look for similar issues in the repository's Issues section
3. Create a new issue with detailed description and steps to reproduce

## Acknowledgments

- The ORIGA, REFUGE, and G1020 datasets for making their data available for research
- The PyTorch, Albumentations, and other open-source libraries used in this project