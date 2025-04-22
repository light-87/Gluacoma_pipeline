@echo off
echo Glaucoma Detection Pipeline - Final Model Training
echo.

REM Set common parameters - adjust these based on best results from previous experiments
SET WANDB_PROJECT=glaucoma-detection
SET WANDB_GROUP=final_model
SET DEVICE=cuda
SET EPOCHS=50
SET BATCH_SIZE=16
SET ARCHITECTURE=unet
SET ENCODER=resnet34
SET LOSS_FUNCTION=combined

echo Training final model with best configuration and more epochs
echo.

python run.py --architecture %ARCHITECTURE% --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "final_%ARCHITECTURE%_%ENCODER%" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION% --use-tta --calculate-cdr
echo.

echo Final model training completed!