# Glaucoma Detection Experiment Guide

This guide outlines how to systematically run experiments using the modular glaucoma detection pipeline. It includes instructions for validating the pipeline, parameter search, architecture comparison, and more.

## Table of Contents

1. [Pipeline Validation](#1-pipeline-validation)
2. [Encoder Parameter Search](#2-encoder-parameter-search)
3. [Architecture Comparison](#3-architecture-comparison)
4. [Loss Function Comparison](#4-loss-function-comparison)
5. [Advanced Model Experiments](#5-advanced-model-experiments)
6. [Managing Experiments](#6-managing-experiments)

## 1. Pipeline Validation

First, run a quick test with 5 epochs to ensure the pipeline works correctly.

```bash
# Clear any previous run history (optional)
python run.py --clear-run-history

# Run a quick test with 5 epochs
python run.py --architecture unet --encoder resnet34 --epochs 5 --batch-size 16 --wandb-name "pipeline_test" --wandb-group "validation"
```

Check for any errors in the output. Also check the WandB interface to verify logging is working correctly. Ensure metrics are being properly tracked and visualizations are generated.

## 2. Encoder Parameter Search

Once you've validated the pipeline, run experiments with different encoders. Create an encoder parameter search batch file:

```bash
# Create test_encoders.bat
@echo off
echo Glaucoma Detection Pipeline - Encoder Parameter Search
echo.

REM Display completed runs first
python run.py --list-completed-runs
echo.

REM Set common parameters
SET WANDB_PROJECT=glaucoma-detection
SET WANDB_GROUP=encoder_search
SET DEVICE=cuda
SET EPOCHS=20
SET BATCH_SIZE=16
SET ARCHITECTURE=unet
SET LOSS_FUNCTION=combined

echo Running encoder parameter search (will automatically skip already completed runs)
echo.

REM Lightweight encoders
python run.py --architecture %ARCHITECTURE% --encoder efficientnet-b0 --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_efficientnet-b0" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

python run.py --architecture %ARCHITECTURE% --encoder mobilenet_v2 --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_mobilenet_v2" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

python run.py --architecture %ARCHITECTURE% --encoder resnet18 --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_resnet18" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

REM Medium encoders
python run.py --architecture %ARCHITECTURE% --encoder resnet34 --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_resnet34" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

python run.py --architecture %ARCHITECTURE% --encoder densenet121 --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_densenet121" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

python run.py --architecture %ARCHITECTURE% --encoder efficientnet-b2 --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_efficientnet-b2" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

REM List all completed runs at the end
python run.py --list-completed-runs

echo All encoder experiments completed!
```

Run the encoder parameter search:

```bash
test_encoders.bat
```

After running, check WandB to compare performance metrics across different encoders. The best performing encoders should be used for subsequent experiments.

## 3. Architecture Comparison

Next, compare different model architectures using the best encoder from the previous step.

```bash
# Create test_architectures.bat
@echo off
echo Glaucoma Detection Pipeline - Architecture Comparison
echo.

REM Display completed runs first
python run.py --list-completed-runs
echo.

REM Set common parameters
SET WANDB_PROJECT=glaucoma-detection
SET WANDB_GROUP=architecture_comparison
SET DEVICE=cuda
SET EPOCHS=20
SET BATCH_SIZE=16
SET ENCODER=resnet34
SET LOSS_FUNCTION=combined

echo Running architecture comparison (will automatically skip already completed runs)
echo.

REM Different architectures
python run.py --architecture unet --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "unet_%ENCODER%" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

python run.py --architecture unetplusplus --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "unetplusplus_%ENCODER%" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

python run.py --architecture deeplabv3 --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "deeplabv3_%ENCODER%" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

python run.py --architecture deeplabv3plus --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "deeplabv3plus_%ENCODER%" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

python run.py --architecture fpn --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "fpn_%ENCODER%" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

python run.py --architecture pspnet --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "pspnet_%ENCODER%" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

REM List all completed runs at the end
python run.py --list-completed-runs

echo All architecture experiments completed!
```

Run the architecture comparison:

```bash
test_architectures.bat
```

## 4. Loss Function Comparison

Test different loss functions using the best architecture and encoder combination from previous experiments.

```bash
# Create test_losses.bat
@echo off
echo Glaucoma Detection Pipeline - Loss Function Comparison
echo.

REM Display completed runs first
python run.py --list-completed-runs
echo.

REM Set common parameters
SET WANDB_PROJECT=glaucoma-detection
SET WANDB_GROUP=loss_comparison
SET DEVICE=cuda
SET EPOCHS=20
SET BATCH_SIZE=16
SET ARCHITECTURE=unet
SET ENCODER=resnet34

echo Running loss function comparison (will automatically skip already completed runs)
echo.

REM Different loss functions
python run.py --architecture %ARCHITECTURE% --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_%ENCODER%_dice" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function dice
echo.

python run.py --architecture %ARCHITECTURE% --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_%ENCODER%_bce" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function bce
echo.

python run.py --architecture %ARCHITECTURE% --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_%ENCODER%_focal" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function focal
echo.

python run.py --architecture %ARCHITECTURE% --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_%ENCODER%_combined" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function combined
echo.

REM List all completed runs at the end
python run.py --list-completed-runs

echo All loss function experiments completed!
```

Run the loss function comparison:

```bash
test_losses.bat
```

## 5. Advanced Model Experiments

After identifying the best configuration from the previous experiments, run additional advanced experiments with features like Cup-to-Disc Ratio (CDR) calculation and Test-Time Augmentation (TTA).

```bash
# Create advanced_experiments.bat
@echo off
echo Glaucoma Detection Pipeline - Advanced Experiments
echo.

REM Display completed runs first
python run.py --list-completed-runs
echo.

REM Set common parameters - use the best configuration from previous experiments
SET WANDB_PROJECT=glaucoma-detection
SET WANDB_GROUP=advanced_features
SET DEVICE=cuda
SET EPOCHS=20
SET BATCH_SIZE=16
SET ARCHITECTURE=unet
SET ENCODER=resnet34
SET LOSS_FUNCTION=combined

echo Running advanced experiments (will automatically skip already completed runs)
echo.

REM Test-Time Augmentation (TTA)
python run.py --architecture %ARCHITECTURE% --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_%ENCODER%_with_tta" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION% --use-tta
echo.

REM Cup-to-Disc Ratio (CDR) calculation
python run.py --architecture %ARCHITECTURE% --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_%ENCODER%_with_cdr" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION% --calculate-cdr
echo.

REM Both TTA and CDR
python run.py --architecture %ARCHITECTURE% --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_%ENCODER%_tta_cdr" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION% --use-tta --calculate-cdr
echo.

REM Different image sizes
python run.py --architecture %ARCHITECTURE% --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_%ENCODER%_size_320" --wandb-group "image_sizes" --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION% --image-size 320
echo.

REM List all completed runs at the end
python run.py --list-completed-runs

echo All advanced experiments completed!
```

Run the advanced experiments:

```bash
advanced_experiments.bat
```

## 6. Managing Experiments

### Viewing Completed Runs

To see which experiments have already been completed:

```bash
python run.py --list-completed-runs
```

### Force Rerunning an Experiment

If you want to rerun an experiment that has already been completed:

```bash
python run.py --architecture unet --encoder resnet34 --force-rerun
```

### Clearing Run History

To clear the history of completed runs:

```bash
python run.py --clear-run-history
```

### Best Practices for Long-Running Experiments

1. **Start Small**: Always start with short runs (5-10 epochs) to validate configurations.
2. **Incremental Approach**: Run experiments in order - encoders, then architectures, then loss functions.
3. **Use Run Tracking**: The run tracker automatically saves completed runs, so you can safely stop and resume batch files.
4. **Monitor Progress**: Use WandB to monitor progress and early stop poor performing runs if needed.
5. **Backup Results**: Periodically download results from WandB for safekeeping.
6. **Resume Interrupted Runs**: If a batch process is interrupted, simply run the same batch file again - completed runs will be skipped.

### Adding Custom Models

To add new models beyond what's provided in segmentation-models-pytorch, you'll need to:

1. Create a new model implementation in a file like `models/custom_model.py`
2. Update the `models_module.py` to include your custom model
3. Add your model to the architecture options in the configuration

Example configuration for a custom model:

```bash
python run.py --architecture custom_model --encoder resnet34 --wandb-project glaucoma-detection --wandb-name "custom_model_resnet34"
```

## Next Steps

Once you've completed all these experiments, analyze the results in WandB to identify:

1. The best encoder
2. The best architecture
3. The best loss function
4. The impact of advanced features like TTA and CDR

This information can then be used to train a final model with more epochs (e.g., 50-100) using the optimal configuration for your final results.

```bash
# Train final model with optimal configuration
python run.py --architecture best_architecture --encoder best_encoder --epochs 100 --batch-size 16 --loss-function best_loss --wandb-name "final_model" --use-tta --calculate-cdr
```