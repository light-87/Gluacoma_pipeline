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

For development:

```bash
pip install -e .
```

## Data Setup

### Dataset Directory Structure

The pipeline expects the following structure:

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

## Usage

### Quick Start

To run the complete pipeline with default settings:

```bash
python main.py
```

### Running Specific Pipeline Steps

You can run specific steps of the pipeline:

```bash
python main.py --steps load clean preprocess train
```

Available steps:
- `load`: Load data into a consolidated format
- `clean`: Clean and preprocess the data
- `preprocess`: Create data loaders and splits
- `train`: Train the segmentation model
- `evaluate`: Evaluate model performance on test data

### Configuration

You can customize the pipeline with various configuration options:

```bash
python main.py --model.model_type unet --model.encoder resnet18 --training.batch_size 16
```

You can also save and load configurations:

```bash
# Save a config
python -c "from glaucoma.config import Config; Config().save('my_config.json')"

# Load a config
python main.py --config my_config.json
```

## Pipeline Modules

### 1. Data Loading

The data loading module (`data/loader.py`) handles:
- Loading images from multiple datasets
- Consolidating data from multiple sources
- Creating train/validation/test splits
- Saving the combined dataset to CSV

### 2. Dataset Handling

The dataset module (`data/dataset.py`):
- Implements memory-efficient data loading
- Provides LRU caching for frequently accessed images
- Supports prefetching for improved performance
- Integrates with PyTorch's Dataset/DataLoader interface

### 3. Data Augmentation

The augmentation module (`data/augmentation.py`):
- Implements data augmentation strategies using Albumentations
- Provides separate transformations for training and validation
- Supports customizable augmentation parameters

### 4. Logging and Tracking

The logging module (`utils/logging.py`):
- Provides enhanced logging with colors and file output
- Implements run tracking in a Markdown notebook
- Supports context tracking for easier debugging
- Records metrics and run summaries

### 5. Configuration

The configuration module (`config.py`):
- Provides a simple, centralized configuration system
- Supports command-line parameter overrides
- Enables saving and loading configurations
- Documents all available parameters

## Run Tracking System

The pipeline includes a run tracking system that automatically logs:
- Date and timestamp
- Configuration used
- Modules executed
- Performance metrics
- Run duration

This information is stored in:
- Log files in the `logs` directory
- A JSON summary file for each run
- A central `notebook.md` file for comparing runs

## Memory Optimization

The pipeline includes several features for memory optimization:

- **LRU Caching**: Only keeps frequently used images in memory
- **Lazy Loading**: Loads images on-demand rather than all at once
- **Prefetching**: Background loading of images for improved performance
- **Efficient Image Processing**: Uses OpenCV for fast image loading and processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The ORIGA, REFUGE, and G1020 datasets for making their data available for research
- The PyTorch, Albumentations, and other open-source libraries used in this project