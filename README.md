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
- [Notebooks & Analysis](#-notebooks--analysis)
- [Advanced Features](#-advanced-features)
- [Configuration System](#-configuration-system)
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

To run the complete pipeline with default settings:

```bash
python -m glaucoma.main
```

This will:
1. Load and preprocess the data
2. Train a model with default parameters
3. Evaluate the model's performance
4. Save results to the output directory

### Customizing Your Run

For a customized run with specific parameters:

```bash
python -m glaucoma.main --steps train,evaluate \
                        --model.architecture unet \
                        --model.encoder resnet34 \
                        --training.batch_size 16 \
                        --training.loss_function combined \
                        --training.focal_weight 0.5 \
                        --use-amp
```

### Using the Batch Script

For Windows users, we provide a batch script for easy execution:

```bash
# Run the batch script
./glaucoma/run.bat
```

## 🏃‍♂️ Running the Pipeline

### Pipeline Steps

You can run specific steps of the pipeline:

```bash
python -m glaucoma.main --steps [step1,step2,...]
```

Available steps:
- `load`: Load data into a consolidated format
- `preprocess`: Create data loaders and splits
- `train`: Train the segmentation model
- `evaluate`: Evaluate model performance on test data
- `ensemble`: Evaluate ensemble of models (if configured)

### Training Options

The pipeline supports several advanced training options:

```bash
# Train with mixed precision (faster training on modern GPUs)
python -m glaucoma.main --steps train --use-amp

# Train with gradient accumulation (for larger effective batch sizes)
python -m glaucoma.main --steps train --grad-accum-steps 4

# Train with focal loss to address class imbalance
python -m glaucoma.main --steps train --loss-function focal
```

### Evaluation with Test-Time Augmentation

For improved evaluation accuracy, use test-time augmentation:

```bash
python -m glaucoma.main --steps evaluate \
                        --checkpoint-path output/best_model.pt \
                        --use-tta
```

## 📓 Notebooks & Analysis

We provide several Jupyter notebooks for data exploration, model training, and results analysis:

### 1. 📊 Data Exploration Notebook

The `notebooks/data_exploration.md` template helps you analyze and understand your dataset:

- Visualize sample images and masks
- Analyze class distribution and imbalance
- Explore image characteristics
- Generate insights for model training

To convert to an executable notebook:

```bash
# Convert markdown to notebook
jupyter nbconvert --to notebook --execute notebooks/data_exploration.md
```

### 2. 🏋️‍♀️ Model Training Notebook

The `notebooks/model_training.md` template provides a step-by-step guide for training models:

- Configure and train models with different architectures
- Implement focal loss and other techniques for class imbalance
- Monitor training progress with visualizations
- Save and evaluate trained models

### 3. 📈 Results Analysis Notebook

The `notebooks/model_results_analysis.md` template offers comprehensive tools for analyzing results:

- Visualize model predictions and performance metrics
- Generate ROC curves, PR curves, and confusion matrices
- Analyze model performance by image characteristics
- Identify challenging cases for further improvement
- Compare different models and techniques

```bash
# Convert and run the analysis notebook
jupyter nbconvert --to notebook --execute notebooks/model_results_analysis.md
```

## 🚀 Advanced Features

### ⚖️ Class Imbalance Handling

We implement several techniques to address class imbalance:

```bash
# Using focal loss
python -m glaucoma.main --loss-function focal --focal-gamma 2.0 --focal-alpha 0.25

# Using combined loss
python -m glaucoma.main --loss-function combined \
                        --dice-weight 1.0 \
                        --bce-weight 0.5 \
                        --focal-weight 0.5
                        
# Using Tversky loss
python -m glaucoma.main --loss-function tversky \
                        --tversky-alpha 0.7 \
                        --tversky-beta 0.3
```

### 🔄 Test-Time Augmentation (TTA)

Improve model performance with test-time augmentation:

```bash
python -m glaucoma.main --steps evaluate --use-tta
```

TTA parameters can be customized:
```bash
python -m glaucoma.main --steps evaluate \
                        --use-tta \
                        --evaluation.tta_scales 0.8,1.0,1.2 \
                        --evaluation.tta_flips True \
                        --evaluation.tta_rotations 0,90,180,270
```

### 🚅 Mixed Precision Training

For faster training on modern GPUs:

```bash
python -m glaucoma.main --use-amp
```

### 🤝 Ensemble Models

Create and evaluate model ensembles:

```bash
# Evaluate an ensemble of multiple models
python -m glaucoma.main --steps ensemble \
                        --ensemble.models.0.checkpoint_path output/model1/best_model.pt \
                        --ensemble.models.1.checkpoint_path output/model2/best_model.pt \
                        --ensemble.ensemble_method average
```

## ⚙️ Configuration System

### Configuration Structure

The configuration is organized into several sections:

- 📁 **Data**: Dataset paths, splits, etc.
- 🏗️ **Model**: Architecture, encoder, etc.
- 🔄 **Preprocessing**: Image size, augmentation, etc.
- 🏋️‍♀️ **Training**: Batch size, learning rate, loss functions, etc.
- 📊 **Evaluation**: Metrics, thresholds, test-time augmentation, etc.
- 📝 **Logging**: Logging settings, Weights & Biases integration, etc.

### Command-line Arguments

Override configuration values via command-line:

```bash
python -m glaucoma.main --training.batch_size 32 \
                        --training.learning_rate 0.001 \
                        --use-amp \
                        --grad-accum-steps 2
```

### Configuration Files

Save and load configurations:

```bash
# Save a config
python -c "from glaucoma.config import Config; Config().save_json('my_config.json')"

# Load a config
python -m glaucoma.main --config-file my_config.json
```

## 🔧 Troubleshooting

### Common Issues and Solutions

1. **Memory Errors** 💾
   ```bash
   # Reduce batch size
   python -m glaucoma.main --training.batch_size 8
   
   # Enable gradient accumulation for effective larger batch
   python -m glaucoma.main --grad-accum-steps 4
   ```

2. **CUDA Out of Memory** 🔥
   ```bash
   # Use mixed precision training
   python -m glaucoma.main --use-amp
   
   # Use a smaller architecture
   python -m glaucoma.main --model.architecture unet --model.encoder resnet18
   ```

3. **Slow Training** 🐢
   ```bash
   # Use mixed precision
   python -m glaucoma.main --use-amp
   
   # Increase number of workers
   python -m glaucoma.main --training.num_workers 8
   ```

4. **Poor Performance** 📉
   ```bash
   # Try different loss function
   python -m glaucoma.main --loss-function combined --focal-weight 0.5
   
   # Use test-time augmentation for evaluation
   python -m glaucoma.main --steps evaluate --use-tta
   ```

### Learning Rate Tuning

If your model is not learning effectively:

```bash
# Use one-cycle learning rate scheduler
python -m glaucoma.main --training.scheduler one_cycle

# Try a different optimizer
python -m glaucoma.main --training.optimizer adamw
```

## 📝 Example Use Cases

### 1️⃣ Basic Training and Evaluation

```bash
# Train a model
python -m glaucoma.main --steps train,evaluate \
                        --model.architecture unet \
                        --training.batch_size 16 \
                        --training.epochs 30

# Evaluate the model
python -m glaucoma.main --steps evaluate \
                        --checkpoint-path output/best_model.pt
```

### 2️⃣ Advanced Training with Mixed Precision and Focal Loss

```bash
python -m glaucoma.main --steps train,evaluate \
                        --model.architecture unet \
                        --model.encoder efficientnet-b3 \
                        --loss-function combined \
                        --focal-weight 0.5 \
                        --use-amp \
                        --grad-accum-steps 2
```

### 3️⃣ Evaluation with Test-Time Augmentation

```bash
python -m glaucoma.main --steps evaluate \
                        --checkpoint-path output/best_model.pt \
                        --use-tta
```

### 4️⃣ Training with Custom Learning Rate Schedule

```bash
python -m glaucoma.main --steps train \
                        --training.scheduler one_cycle \
                        --training.max_lr 0.01 \
                        --training.pct_start 0.3
```

## 📚 Notebook Workflows

### Data Exploration Workflow

1. Convert and open the data exploration notebook
2. Load and visualize your dataset
3. Analyze class distribution and image characteristics
4. Identify potential preprocessing strategies
5. Use insights to inform your training approach

### Model Training Workflow

1. Convert and open the model training notebook
2. Configure model architecture and training parameters
3. Implement appropriate loss functions based on data analysis
4. Train and monitor model performance
5. Save the best performing model

### Results Analysis Workflow

1. Convert and open the results analysis notebook
2. Load your trained model and evaluation results
3. Analyze performance using various metrics and visualizations
4. Identify strengths, weaknesses, and challenging cases
5. Generate recommendations for further improvement

## 🔄 Data Requirements

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

## 🙌 Enjoy Using the Pipeline!

For further assistance or to report issues, please create an issue in the repository. Happy glaucoma detection! 🎉