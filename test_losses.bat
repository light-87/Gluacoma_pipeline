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

python run.py --architecture %ARCHITECTURE% --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_%ENCODER%_bce" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function bce
echo.

python run.py --architecture %ARCHITECTURE% --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_%ENCODER%_dice" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function dice
echo.

python run.py --architecture %ARCHITECTURE% --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_%ENCODER%_focal" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function focal
echo.

python run.py --architecture %ARCHITECTURE% --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_%ENCODER%_combined" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function combined
echo.

REM List all completed runs at the end
python run.py --list-completed-runs

echo All loss function experiments completed!