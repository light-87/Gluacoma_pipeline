@echo off
echo Starting Glaucoma Detection pipeline with enhanced WandB visualizations...
echo.

python -m glaucoma.main --steps train,evaluate ^
    --model.architecture unet ^
    --model.encoder resnet34 ^
    --training.batch_size 16 ^
    --training.epochs 30 ^
    --logging.use_wandb True ^
    --wandb-detailed-viz ^
    --wandb-notes "Training with enhanced visualizations"

echo.
echo Pipeline execution completed.
pause