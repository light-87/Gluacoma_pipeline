@echo off
echo Glaucoma Detection Pipeline - Batch Runner
echo.

REM Model comparison
echo Running model comparison...
echo.

REM 1. UNet with ResNet34
python run.py --architecture unet --encoder resnet34 --wandb-project glaucoma-detection --wandb-name "unet_resnet34" --wandb-group "model_comparison"
echo.

REM 2. UNet with EfficientNet-B0
python run.py --architecture unet --encoder efficientnet-b0 --wandb-project glaucoma-detection --wandb-name "unet_efficientnetb0" --wandb-group "model_comparison"
echo.

REM 3. DeepLabV3+ with ResNet34
python run.py --architecture deeplabv3plus --encoder resnet34 --wandb-project glaucoma-detection --wandb-name "deeplabv3plus_resnet34" --wandb-group "model_comparison"
echo.

REM 4. DeepLabV3+ with EfficientNet-B0
python run.py --architecture deeplabv3plus --encoder efficientnet-b0 --wandb-project glaucoma-detection --wandb-name "deeplabv3plus_efficientnetb0" --wandb-group "model_comparison"
echo.

REM 5. FPN with ResNet34
python run.py --architecture fpn --encoder resnet34 --wandb-project glaucoma-detection --wandb-name "fpn_resnet34" --wandb-group "model_comparison"
echo.

echo All runs completed successfully!