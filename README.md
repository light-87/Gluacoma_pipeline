# 🧠 Glaucoma Detection Pipeline

A powerful, modular machine learning pipeline for automated glaucoma detection from retinal fundus images using state-of-the-art techniques.

## 🌟 Project Overview

This project implements a clean, efficient pipeline for glaucoma detection from retinal fundus images with a focus on:

- 🧹 **Simplicity**: Clean, readable code with minimal dependencies
- ⚡ **Performance**: Memory-efficient data handling and optimized processing
- 🔍 **Accuracy**: Advanced techniques for improved detection performance
- 📊 **Analysis**: Comprehensive tools for results visualization and interpretation
- 🔄 **Flexibility**: Easy to extend with new models, metrics, and techniques

### 📚 Datasets

The pipeline works with three glaucoma datasets:

1. **ORIGA** - Singapore Chinese Eye Study dataset with glaucoma diagnosis
2. **REFUGE** - Retinal Fundus Glaucoma Challenge dataset with expert annotations
3. **G1020** - A large collection of retinal images with associated metadata

Dataset link: https://drive.google.com/file/d/1-h3Bv1Fjf38s42w5Cp_a-9CgXNfDGwWt/view?usp=drive_link

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start Guide](#-quick-start-guide)
- [Running the Pipeline](#-running-the-pipeline)
- [Configuration](#-configuration-system)
- [Automated Runs](#-automated-runs)
- [Advanced Features](#-advanced-features)
- [Troubleshooting](#-troubleshooting)

## 🛠 Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended but optional)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/glaucoma-detection.git
cd glaucoma-detection
```

2. Create a virtual environment:

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n glaucoma python=3.8
conda activate glaucoma
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install the package in development mode:

```bash
pip install -e .
```

## 🚀 Quick Start Guide

### Running with Default Settings

To run the complete pipeline with default parameters:

```bash
python run.py
```

This will:
1. Load and preprocess the data
2. Train a model with default parameters
3. Evaluate the model's performance
4. Save results to the output directory

### Customizing Your Run

For a customized run with specific parameters:

```bash
python run.py --architecture unet \
              --encoder resnet34 \
              --batch-size 16 \
              --loss-function combined \
              --focal-weight 0.5 \
              --wandb-project glaucoma-detection
```

### Using the Batch Script

For Windows users, we provide a batch script for easy execution:

```bash
# Run multiple model configurations
run_batch.bat
```

## 🏃‍♂️ Running the Pipeline

### Pipeline Steps

You can run specific steps of the pipeline with a configuration file:

```bash
python run.py --config config_examples/base_config.json --steps load,train,evaluate
```

Available steps:
- `load`: Load data into a consolidated format
- `preprocess`: Create data loaders and splits
- `train`: Train the segmentation model
- `evaluate`: Evaluate model performance on test data

### Model Architectures

The pipeline supports multiple segmentation architectures:

```bash
# UNet with ResNet34 encoder
python run.py --architecture unet --encoder resnet34

# DeepLabV3+ with EfficientNet-B0
python run.py --architecture deeplabv3plus --encoder efficientnet-b0

# Feature Pyramid Network (FPN)
python run.py --architecture fpn --encoder resnet50

# UNet++ (Nested UNet)
python run.py --architecture unetplusplus --encoder efficientnet-b0
```

### Loss Functions

```bash
# Train with Dice loss
python run.py --loss-function dice

# Train with Focal loss
python run.py --loss-function focal --focal-gamma 2.0 --focal-alpha 0.25

# Train with Combined loss (Dice + Focal)
python run.py --loss-function combined --dice-weight 1.0 --focal-weight 1.0
```

## ⚙️ Configuration System

### Configuration Structure

The configuration is organized into several sections:

- 📁 **Data**: Dataset paths, splits, etc.
- 🏗️ **Model**: Architecture, encoder, etc.
- 🔄 **Preprocessing**: Image size, augmentation, etc.
- 🏋️‍♀️ **Training**: Batch size, learning rate, loss functions, etc.
- 📊 **Evaluation**: Metrics, thresholds, test-time augmentation, etc.
- 📝 **Pipeline**: Steps, output directory, WandB settings, etc.

### Configuration Files

Save and load configurations:

```bash
# Using a configuration file
python run.py --config config_examples/base_config.json

# Override configuration values
python run.py --config config_examples/base_config.json --architecture fpn --encoder efficientnet-b0
```

## 🤖 Automated Runs

The pipeline supports automated runs for hyperparameter tuning and model comparison:

### Parameter Grid Search

```bash
# Run parameter grid search
python automate_runs.py --config config_examples/base_config.json \
                        --param-grid config_examples/param_grid.json \
                        --wandb-group "hyperparameter_search"
```

### Model Comparison

```bash
# Run multiple model architectures
python automate_runs.py --config config_examples/base_config.json \
                        --models config_examples/models_list.json \
                        --wandb-group "model_comparison"
```

### Batch File Execution

```bash
# Run the generated batch file for multiple configurations
cd output/runs
run_batch.bat  # On Windows
./run_batch.sh  # On Linux/Mac
```

## 🚀 Advanced Features

### ⚖️ Cup-to-Disc Ratio (CDR) Calculation

Calculate CDR for glaucoma assessment:

```bash
python run.py --calculate-cdr --cdr-method diameter
```

### 🔄 Test-Time Augmentation (TTA)

Improve model performance with test-time augmentation:

```bash
python run.py --steps evaluate --use-tta
```

### 🚅 Mixed Precision Training

For faster training on modern GPUs:

```bash
python run.py --use-amp
```

### 📊 Weights & Biases Integration

Enable comprehensive experiment tracking:

```bash
python run.py --wandb-project glaucoma-detection \
              --wandb-name "unet_resnet34_run1" \
              --wandb-entity your-username
```

## 🔍 Optimized Model Configuration

We've implemented an optimized configuration based on successful experiments that achieves outstanding metrics:

```
Test Metrics:
Loss: 0.0553
dice: 0.9475
iou: 0.9041
accuracy: 0.9984
precision: 0.9390
recall: 0.9598
specificity: 0.9990
f1: 0.9475
```

### Key Optimizations:

1. **Combined Loss Function**: Using Dice Loss (weight=1.0) + Focal Loss (weight=1.0)
2. **Focal Loss Parameters**: gamma=2.0, alpha=0.25
3. **Enhanced Augmentations**: Including vertical flips and 90-degree rotations
4. **Test-Time Augmentation**: Improving prediction quality on test data

To run the optimized configuration:

```bash
# Use the optimized configuration
python run.py --config config_examples/base_config.json
```

## 🔧 Troubleshooting

### Common Issues and Solutions

1. **Memory Errors** 💾
   ```bash
   # Reduce batch size
   python run.py --batch-size 8
   ```

2. **CUDA Out of Memory** 🔥
   ```bash
   # Use mixed precision training
   python run.py --use-amp
   
   # Use a smaller architecture
   python run.py --architecture unet --encoder resnet18
   ```

3. **Slow Training** 🐢
   ```bash
   # Use mixed precision
   python run.py --use-amp
   
   # Increase number of workers
   python run.py --num-workers 8
   ```

4. **Poor Performance** 📉
   ```bash
   # Try different loss function
   python run.py --loss-function combined --focal-weight 0.5
   
   # Use test-time augmentation for evaluation
   python run.py --steps evaluate --use-tta
   ```

## 🔄 Data Requirements

The pipeline expects the following directory structure:

```
data/
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

## 🙌 Enjoy Using the Pipeline!

For further assistance or to report issues, please create an issue in the repository. Happy glaucoma detection! 🎉

# Glaucoma Detection Pipeline: Detailed Flow Plan

## 1. Entry Points and Execution Flow

### 1.1 Batch File Execution
- **Entry Points**: 
  - `run_batch.bat`: Runs multiple model architectures/encoders
  - `test_architectures.bat`: Compares different architectures with same encoder
  - `test_encoders.bat`: Tests different encoders with same architecture
  - `test_losses.bat`: Compares different loss functions
  - `advanced_experiments.bat`: Tests advanced features like TTA and CDR
  - `final_model.bat`: Trains the final model with best configuration

### 1.2 Main Execution Scripts
- **Script**: `run.py`
- **Process**:
  - Parses command-line arguments
  - Checks if the run has been completed before
  - Calls appropriate modules based on the `--steps` parameter
  - Tracks completed runs in `completed_runs.json`

## 2. Configuration Management

### 2.1 Configuration Creation
- **Module**: `config.py`
- **Classes**: 
  - `Config`: Master configuration class
  - `DataConfig`, `ModelConfig`, `PreprocessingConfig`, etc.: Sub-configurations
- **Functions**:
  - `get_argument_parser()`: Creates the parser for command-line arguments
  - `parse_args_and_create_config()`: Creates configuration from arguments
- **Process**:
  - Combines default values with command-line arguments
  - Generates unique run IDs and names
  - Creates a nested configuration object

### 2.2 Run Tracking
- **Module**: `run_tracker.py`
- **Class**: `RunTracker`
- **Process**:
  - Tracks completed runs using configuration hash
  - Stores results in `completed_runs.json`
  - Allows skipping already completed runs
  - Supports listing and clearing run history

## 3. Data Loading and Processing

### 3.1 Dataset Loading
- **Module**: `data_module.py`
- **Class**: `DatasetLoader`
- **Process**:
  - Loads images and masks from configured directories
  - Combines datasets (REFUGE, ORIGA, G1020)
  - Creates a DataFrame with paths and dataset info

### 3.2 Dataset Preparation
- **Module**: `data_module.py`
- **Functions**:
  - `create_dataset_splits()`: Creates train/val/test splits
  - `save_dataset()`: Saves dataset to CSV
  - `get_augmentations()`: Creates augmentation pipelines
- **Process**:
  - Splits data with stratification
  - Configures augmentations based on preprocessing settings
  - Saves processed dataset for future use

### 3.3 Dataset Class and Loaders
- **Module**: `data_module.py`
- **Classes**:
  - `GlaucomaDataset`: PyTorch dataset for efficient loading
  - `PrefetchQueue`: Background prefetching for performance
- **Functions**:
  - `create_dataloaders()`: Creates PyTorch DataLoaders
- **Process**:
  - Loads and transforms images and masks
  - Applies augmentations to training data
  - Creates batches for model training

## 4. Model Creation and Management

### 4.1 Model Architecture
- **Module**: `models_module.py`
- **Functions**:
  - `create_model()`: Creates model based on configuration
  - `load_checkpoint()`: Loads weights from checkpoint
- **Process**:
  - Selects architecture (UNet, DeepLabV3+, etc.)
  - Configures encoder (ResNet34, EfficientNet, etc.)
  - Initializes with pretrained weights if specified

### 4.2 Ensemble Models
- **Module**: `models_module.py`
- **Class**: `EnsembleModel`
- **Functions**:
  - `create_ensemble()`: Creates ensemble of multiple models
- **Process**:
  - Combines predictions from multiple models
  - Supports weighted averaging and max voting

## 5. Loss Functions

### 5.1 Loss Function Implementation
- **Module**: `losses_module.py`
- **Classes**:
  - `DiceLoss`: Dice coefficient loss
  - `FocalLoss`: Focal loss for handling class imbalance
  - `TverskyLoss`: Tversky loss for imbalanced segmentation
  - `CombinedLoss`: Weighted combination of multiple losses
- **Functions**:
  - `create_loss_function()`: Creates loss based on configuration
- **Process**:
  - Initializes appropriate loss function with weights
  - Configures parameters (alpha, gamma, etc.)

## 6. Training Process

### 6.1 Trainer Implementation
- **Module**: `training_module.py`
- **Class**: `Trainer`
- **Methods**:
  - `train()`: Main training loop
  - `train_epoch()`: Single epoch training
  - `validate()`: Validation on val set
  - `_backward_step()`, `_optimizer_step()`: Gradient handling
- **Process**:
  - Creates optimizer and scheduler
  - Executes training loop with batch processing
  - Handles gradient accumulation and mixed precision
  - Tracks metrics and saves checkpoints
  - Implements early stopping

## 7. Evaluation and Metrics

### 7.1 Evaluation Implementation
- **Module**: `evaluation_module.py`
- **Functions**:
  - `calculate_metrics()`: Calculates segmentation metrics
  - `calculate_cdr()`: Calculates Cup-to-Disc Ratio
  - `evaluate_model()`: Main evaluation function
  - `test_time_augmentation()`: Applies TTA for better results
  - `visualize_predictions()`: Creates visualizations
- **Process**:
  - Calculates metrics (Dice, IoU, precision, recall, etc.)
  - Implements CDR calculation using OpenCV
  - Creates and saves visualization figures

## 8. Weights & Biases Integration

### 8.1 WandB Logger
- **Module**: `wandb_module.py`
- **Class**: `WandBLogger`
- **Methods**:
  - `log()`: Logs metrics
  - `log_model()`: Logs model architecture
  - `log_figure()`: Logs matplotlib figures
  - `log_image_with_masks()`: Logs segmentation visualizations
  - `save_model()`: Saves model as artifact
- **Process**:
  - Initializes WandB with project/run info
  - Logs metrics, images, and model info
  - Creates detailed experiment tracking

## 9. Complete Execution Flow (Loss Comparison Example)

### 9.1 Batch File Execution
```bash
@echo off
echo Glaucoma Detection Pipeline - Loss Function Comparison
echo.

# Set common parameters
SET WANDB_PROJECT=glaucoma-detection
SET WANDB_GROUP=loss_comparison
SET DEVICE=cuda
SET EPOCHS=20
SET BATCH_SIZE=16
SET ARCHITECTURE=unet
SET ENCODER=resnet34

# Run different loss functions
python run.py --architecture %ARCHITECTURE% --encoder %ENCODER% --loss-function bce ...
python run.py --architecture %ARCHITECTURE% --encoder %ENCODER% --loss-function dice ...
python run.py --architecture %ARCHITECTURE% --encoder %ENCODER% --loss-function focal ...
python run.py --architecture %ARCHITECTURE% --encoder %ENCODER% --loss-function combined ...
```

### 9.2 Main Execution (run.py)
```python
def main():
    # Parse arguments
    parser = get_argument_parser()
    args = parser.parse_args()
    
    # Handle run tracking commands
    if args.list_completed_runs:
        # List completed runs
        return 0
    
    # Create configuration
    config = parse_args_and_create_config(args)
    
    # Run pipeline
    status = run_pipeline(config, skip_if_completed=not args.force_rerun)
    
    return status
```

### 9.3 Pipeline Execution
```python
def run_pipeline(config, skip_if_completed=True):
    # Check if run has already been completed
    if skip_if_completed:
        run_tracker = get_run_tracker()
        if run_tracker.is_run_completed(config.to_dict()):
            print("SKIPPING ALREADY COMPLETED RUN")
            return 0
    
    # Setup environment
    device, output_dir, checkpoint_dir = setup_environment(config)
    
    # Initialize WandB logger
    wandb_logger = WandBLogger(config)
    
    # Get pipeline steps to execute
    steps = config.pipeline.steps
    
    try:
        # Step 1: Load data
        if 'load' in steps:
            df, dataset_path = load_data(config)
            
        # Step 2: Create data loaders
        if 'preprocess' in steps:
            dataloaders = create_data_loaders(config, df)
            
        # Step 3: Train model
        if 'train' in steps:
            model = train_model(config, device, dataloaders, wandb_logger)
            
        # Step 4: Evaluate model
        if 'evaluate' in steps:
            metrics = evaluate(config, device, model, dataloaders, output_dir, wandb_logger)
            
        # Mark run as completed
        run_tracker.mark_run_completed(config.to_dict(), metadata=metrics)
        
        # Finish WandB run
        wandb_logger.finish()
        
        return 0
        
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        return 1
```

### 9.4 Training Process
```python
def train_model(config, device, dataloaders, wandb_logger=None):
    # Create model
    model = create_model(config.model)
    
    # Load checkpoint if specified
    if config.model.checkpoint_path:
        model = load_checkpoint(model, config.model.checkpoint_path)
        return model
    
    # Create loss function
    criterion = create_loss_function(config.training.loss)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        criterion=criterion,
        config=config.training,
        device=device,
        checkpoint_dir=config.pipeline.checkpoint_dir,
        wandb_logger=wandb_logger
    )
    
    # Train model
    result = trainer.train()
    
    # Load best model
    best_checkpoint_path = os.path.join(config.pipeline.checkpoint_dir, 'best_model.pt')
    model = load_checkpoint(model, best_checkpoint_path)
    
    return model
```

### 9.5 Training Loop (Trainer.train)
```python
def train(self) -> Dict[str, Any]:
    # Get configuration
    epochs = self.config.epochs
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train for one epoch
        train_metrics = self.train_epoch()
        
        # Validate
        val_metrics = self.validate()
        
        # Update learning rate scheduler
        if self.scheduler is not None:
            self.scheduler.step(val_loss)
        
        # Check for best model
        if is_better:
            best_val_metric = current_metric
            best_model_state = copy.deepcopy(self.model.state_dict())
            
            # Save best model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'best_val_metric': best_val_metric,
            }, os.path.join(self.checkpoint_dir, 'best_model.pt'))
        
        # Early stopping
        if self.early_stopping and patience_counter >= self.patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Restore best model
    if best_model_state is not None:
        self.model.load_state_dict(best_model_state)
    
    return {
        'best_val_metric': best_val_metric,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'history': history
    }
```

### 9.6 Evaluation Process
```python
def evaluate(config, device, model, dataloaders, output_dir, wandb_logger=None):
    # Set model to evaluation mode
    model.eval()
    
    # Create evaluation directory
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    
    # Evaluate on test set
    metrics = evaluate_model(
        model=model,
        dataloader=dataloaders['test'],
        device=device,
        threshold=config.evaluation.threshold,
        calculate_cdr_flag=config.evaluation.calculate_cdr,
        cdr_method=config.evaluation.cdr_method,
        use_tta=config.evaluation.use_tta
    )
    
    # Print and save metrics
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate visualizations
    if config.evaluation.generate_visualizations:
        figures = visualize_predictions(
            images=images,
            masks=masks,
            preds=preds,
            threshold=config.evaluation.threshold,
            max_samples=config.evaluation.sample_count,
            output_dir=str(vis_dir)
        )
        
        # Log visualizations to WandB
        if wandb_logger:
            for i, fig in enumerate(figures):
                wandb_logger.log_figure(fig, f"test_sample_{i}")
    
    return metrics
```

## 10. Data Flow Diagram

```
Batch Files (*.bat) 
    │
    ▼
run.py (Main entry)
    │
    ├── config.py (Configuration management)
    │   │
    │   ├── Config (Master config)
    │   ├── DataConfig, ModelConfig, etc. (Sub-configs)
    │   └── parse_args_and_create_config()
    │
    ├── run_tracker.py (Run tracking)
    │   │
    │   └── RunTracker
    │       ├── is_run_completed()
    │       └── mark_run_completed()
    │
    ├── data_module.py (Data handling)
    │   │
    │   ├── DatasetLoader
    │   │   ├── load_dataset()
    │   │   └── load_all_datasets()
    │   │
    │   ├── create_dataset_splits()
    │   ├── get_augmentations()
    │   ├── GlaucomaDataset
    │   └── create_dataloaders()
    │
    ├── models_module.py (Model creation)
    │   │
    │   ├── create_model() 
    │   ├── load_checkpoint()
    │   └── EnsembleModel
    │
    ├── losses_module.py (Loss functions)
    │   │
    │   ├── DiceLoss
    │   ├── FocalLoss
    │   ├── TverskyLoss
    │   └── CombinedLoss
    │
    ├── training_module.py (Training)
    │   │
    │   └── Trainer
    │       ├── train()
    │       ├── train_epoch()
    │       └── validate()
    │
    ├── evaluation_module.py (Evaluation)
    │   │
    │   ├── calculate_metrics()
    │   ├── calculate_cdr()
    │   ├── evaluate_model()
    │   └── visualize_predictions()
    │
    └── wandb_module.py (Logging)
        │
        └── WandBLogger
            ├── log()
            ├── log_figure()
            └── log_image_with_masks()
```

## 11. Output Structure

```
output/
└── run_{timestamp}/
    ├── config.json               # Saved configuration
    ├── dataset.csv               # Generated dataset with splits
    ├── checkpoints/              # Model checkpoints
    │   ├── best_model.pt         # Best model weights
    │   ├── final_model.pt        # Final model weights
    │   └── checkpoint_epoch_N.pt # Periodic checkpoints
    │
    └── evaluation/               # Evaluation results
        ├── test_metrics.json     # Metrics on test set
        └── visualizations/       # Output visualizations
            ├── visualization_0.png
            ├── visualization_1.png
            └── ...
```

## 12. Runtime Tracking

```
completed_runs.json
{
  "runs": {
    "69991e1af35bb7b018753828243e4ae2": {  # Hash of configuration
      "completed_at": 1745322172.3384275,   # Timestamp
      "run_id": "",                         # WandB run ID
      "config_summary": "unet_resnet34_combined",  # Human-readable summary
      "metadata": {                         # Results
        "dice": 0.9278366959948556,
        "iou": 0.8744225024208945,
        "accuracy": 0.997865240571287,
        "precision": 0.940336969468934,
        "recall": 0.929933343064063
      }
    }
  },
  "metadata": {
    "last_updated": 1745322172.3384275
  }
}
```