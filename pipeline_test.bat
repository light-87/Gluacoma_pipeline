@echo off
echo Glaucoma Detection Pipeline - Quick Validation Test
echo.

REM Clear any previous run history (optional)
REM python run.py --clear-run-history

REM Set common parameters
SET WANDB_PROJECT=glaucoma-detection
SET WANDB_GROUP=validation
SET ARCHITECTURE=unet
SET ENCODER=resnet34
SET LOSS_FUNCTION=combined

echo Running quick validation test with 5 epochs
echo.

python run.py --architecture %ARCHITECTURE% --encoder %ENCODER% --epochs 5 --batch-size 16 --wandb-project %WANDB_PROJECT% --wandb-name "pipeline_test" --wandb-group %WANDB_GROUP% --loss-function %LOSS_FUNCTION%
echo.

echo Pipeline validation test completed! Check WandB interface for results.