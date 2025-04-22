#!/bin/bash
echo "Glaucoma Detection Pipeline - Advanced Batch Runner"
echo

# Set common parameters
WANDB_PROJECT="glaucoma-detection"
WANDB_GROUP="advanced_experiments"
DEVICE="cuda"
EPOCHS=50
BATCH_SIZE=16

echo "Running multiple model configurations with various parameters..."
echo

# Different model architectures and encoders
echo "===== Model Architecture Comparison ====="
echo

# UNet
python run.py --architecture unet --encoder resnet34 --wandb-project ${WANDB_PROJECT} --wandb-name "unet_resnet34" --wandb-group "architectures" --device ${DEVICE} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --loss-function combined
echo

python run.py --architecture unet --encoder efficientnet-b0 --wandb-project ${WANDB_PROJECT} --wandb-name "unet_efficientnetb0" --wandb-group "architectures" --device ${DEVICE} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --loss-function combined
echo

python run.py --architecture unet --encoder resnet50 --wandb-project ${WANDB_PROJECT} --wandb-name "unet_resnet50" --wandb-group "architectures" --device ${DEVICE} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --loss-function combined
echo

# DeepLabV3+
python run.py --architecture deeplabv3plus --encoder resnet34 --wandb-project ${WANDB_PROJECT} --wandb-name "deeplabv3plus_resnet34" --wandb-group "architectures" --device ${DEVICE} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --loss-function combined
echo

# FPN
python run.py --architecture fpn --encoder resnet34 --wandb-project ${WANDB_PROJECT} --wandb-name "fpn_resnet34" --wandb-group "architectures" --device ${DEVICE} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --loss-function combined
echo

# UNet++
python run.py --architecture unetplusplus --encoder resnet34 --wandb-project ${WANDB_PROJECT} --wandb-name "unetplusplus_resnet34" --wandb-group "architectures" --device ${DEVICE} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --loss-function combined
echo

# Different loss functions
echo "===== Loss Function Comparison ====="
echo

python run.py --architecture unet --encoder resnet34 --wandb-project ${WANDB_PROJECT} --wandb-name "unet_dice_loss" --wandb-group "losses" --device ${DEVICE} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --loss-function dice
echo

python run.py --architecture unet --encoder resnet34 --wandb-project ${WANDB_PROJECT} --wandb-name "unet_focal_loss" --wandb-group "losses" --device ${DEVICE} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --loss-function focal
echo

python run.py --architecture unet --encoder resnet34 --wandb-project ${WANDB_PROJECT} --wandb-name "unet_combined_loss" --wandb-group "losses" --device ${DEVICE} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --loss-function combined
echo

# Different image sizes
echo "===== Image Size Comparison ====="
echo

python run.py --architecture unet --encoder resnet34 --wandb-project ${WANDB_PROJECT} --wandb-name "unet_size_224" --wandb-group "image_sizes" --device ${DEVICE} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --image-size 224 --loss-function combined
echo

python run.py --architecture unet --encoder resnet34 --wandb-project ${WANDB_PROJECT} --wandb-name "unet_size_256" --wandb-group "image_sizes" --device ${DEVICE} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --image-size 256 --loss-function combined
echo

python run.py --architecture unet --encoder resnet34 --wandb-project ${WANDB_PROJECT} --wandb-name "unet_size_320" --wandb-group "image_sizes" --device ${DEVICE} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --image-size 320 --loss-function combined
echo

# Test-time augmentation experiments
echo "===== Test-time Augmentation Experiments ====="
echo

python run.py --architecture unet --encoder resnet34 --wandb-project ${WANDB_PROJECT} --wandb-name "unet_with_tta" --wandb-group "tta" --device ${DEVICE} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --loss-function combined --use-tta
echo

python run.py --architecture deeplabv3plus --encoder resnet34 --wandb-project ${WANDB_PROJECT} --wandb-name "deeplabv3plus_with_tta" --wandb-group "tta" --device ${DEVICE} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --loss-function combined --use-tta
echo

# Cup-to-Disc Ratio calculation
echo "===== Cup-to-Disc Ratio Experiments ====="
echo

python run.py --architecture unet --encoder resnet34 --wandb-project ${WANDB_PROJECT} --wandb-name "unet_with_cdr" --wandb-group "cdr" --device ${DEVICE} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --loss-function combined --calculate-cdr
echo

python run.py --architecture deeplabv3plus --encoder resnet34 --wandb-project ${WANDB_PROJECT} --wandb-name "deeplabv3plus_with_cdr" --wandb-group "cdr" --device ${DEVICE} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --loss-function combined --calculate-cdr
echo

echo "All runs completed successfully!"