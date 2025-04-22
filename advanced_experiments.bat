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