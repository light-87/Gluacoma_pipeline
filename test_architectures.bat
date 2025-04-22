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