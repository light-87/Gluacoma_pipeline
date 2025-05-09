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
python run.py --architecture %ARCHITECTURE% --encoder efficientnet-b3 --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_efficientnet-b3" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

python run.py --architecture %ARCHITECTURE% --encoder mobilenet_v2 --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_mobilenet_v2" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

python run.py --architecture %ARCHITECTURE% --encoder resnet18 --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_resnet18" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

REM Medium encoders
python run.py --architecture %ARCHITECTURE% --encoder dpn68 --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_dpn68" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

python run.py --architecture %ARCHITECTURE% --encoder densenet169 --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_densenet169" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

python run.py --architecture %ARCHITECTURE% --encoder mit_b1 --wandb-project %WANDB_PROJECT% --wandb-name "%ARCHITECTURE%_mit_b1" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

REM List all completed runs at the end
python run.py --list-completed-runs

echo All encoder experiments completed!