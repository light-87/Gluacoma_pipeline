# Best Pipeline Refactor Plan

## Current Issues
- Single `pipeline.py` contains all functionality with no modularity
- Difficult to automate runs with different configurations and models
- Unsuitable for implementing new models like transformers or self-supervised learning
- Poor naming system for WandB runs
- No support for CDR (Cup-to-Disc Ratio) measurement for glaucoma

## Proposed Solution: Modular Architecture
Create a modular architecture with clear separation of concerns, making it easy to:
- Configure different runs
- Add new models
- Implement new evaluation metrics 
- Automate batch runs with different configurations

## New File Structure

```
glaucoma-detection/
├── config.py                   # Configuration management
├── data/
│   ├── __init__.py
│   ├── dataset.py              # Dataset classes
│   ├── loaders.py              # DataLoader creation
│   └── augmentation.py         # Data augmentation
├── models/
│   ├── __init__.py
│   ├── factory.py              # Model creation factory
│   ├── unet.py                 # UNet implementation
│   ├── transformer.py          # Future transformer model
│   └── ensemble.py             # Ensemble models
├── losses/
│   ├── __init__.py
│   ├── dice.py                 # Dice loss
│   ├── focal.py                # Focal loss
│   └── combined.py             # Combined losses
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # Evaluation metrics (incl. CDR)
│   └── visualization.py        # Result visualization
├── training/
│   ├── __init__.py
│   └── trainer.py              # Training loop logic
├── utils/
│   ├── __init__.py
│   ├── logging.py              # Logging utilities
│   └── wandb_logger.py         # WandB integration
├── pipeline.py                 # Main orchestration file
├── run.py                      # Entry point for single runs
├── automate_runs.py            # Script for automated batch runs
└── run_batch.bat               # Batch file to run multiple configurations
```

## Module Responsibilities

### 1. Configuration (`config.py`)
- Create a central configuration system with defaults
- Support for loading from YAML/JSON files
- Generate unique run IDs and names
- Combine defaults with custom run configs

### 2. Data Module (`data/`)
- `dataset.py`: GlaucomaDataset class with support for various datasets
- `loaders.py`: Functions to create and manage data loaders
- `augmentation.py`: Configurable data augmentation pipelines

### 3. Models Module (`models/`)
- `factory.py`: Model factory pattern to create different architectures
- Separate files for each model architecture (UNet, Transformer, etc.)
- `ensemble.py`: Support for model ensembling

### 4. Losses Module (`losses/`)
- Separate loss implementations 
- Allow for easy combination and weighting of losses

### 5. Evaluation Module (`evaluation/`)
- `metrics.py`: Metrics calculation including Dice, IoU, and CDR
- `visualization.py`: Functions for visualizing results

### 6. Training Module (`training/`)
- `trainer.py`: Trainer class to handle the training loop, validation, checkpointing

### 7. Utils Module (`utils/`)
- `logging.py`: Logging utilities
- `wandb_logger.py`: Enhanced WandB integration with better naming

### 8. Pipeline (`pipeline.py`)
- Main orchestration connecting all modules
- Clean, high-level API for running experiments

### 9. Automation (`automate_runs.py` & `run_batch.bat`)
- Script to queue multiple runs with different configurations
- Support for grid search of hyperparameters

## WandB Naming Improvement
- Generate structured names with format: `{model}_{dataset}_{timestamp}_{unique_id}`
- Example: `unet_efficientnetb0_refuge_g1020_20240422_run001`
- Include key configuration parameters in tags
- Organize runs in projects by model architecture or experiment type

## Implementation Plan
1. Start by creating the basic structure and skeleton classes
2. Port functionality from existing pipeline.py to appropriate modules
3. Create the new pipeline.py that uses the modules
4. Implement configuration system
5. Enhance WandB integration with better naming
6. Create automation scripts
7. Test with existing models
8. Extend with new models (transformer, etc.)
9. Add CDR calculation

## Additional Features
- Support for early stopping based on validation metrics
- Model checkpointing and resuming training
- Experiment tracking with WandB artifacts
- Test-time augmentation
- Support for self-supervised learning
- Cup-to-Disc Ratio calculation and evaluation
- Cross-validation support

With this modular design, we'll be able to easily:
1. Add new models
2. Configure different runs
3. Automate batch experiments
4. Track and compare results
5. Implement new evaluation metrics