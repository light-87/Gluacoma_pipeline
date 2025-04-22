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