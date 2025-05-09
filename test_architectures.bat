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

REM 1) UNet
python run.py --architecture unet           --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "unet_%ENCODER%"        --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

REM 2) MAnet 
python run.py --architecture manet          --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "manet_%ENCODER%"       --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

REM 3) FPN
python run.py --architecture fpn            --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "fpn_%ENCODER%"         --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

REM 4) PAN
python run.py --architecture pan            --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "pan_%ENCODER%"         --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

REM 5) DeepLabV3+
python run.py --architecture deeplabv3plus  --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "deeplabv3plus_%ENCODER%" --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

REM 6) Segformer 
python run.py --architecture segformer      --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "segformer_%ENCODER%"   --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

REM 7) DPT
python run.py --architecture dpt            --encoder %ENCODER% --wandb-project %WANDB_PROJECT% --wandb-name "dpt_%ENCODER%"         --wandb-group %WANDB_GROUP% --device %DEVICE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --loss-function %LOSS_FUNCTION%
echo.

REM List all completed runs at the end
python run.py --list-completed-runs

echo All architecture experiments completed!
