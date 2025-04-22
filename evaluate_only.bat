@echo off
echo Starting Glaucoma Model Evaluation...
echo.

python -m glaucoma.main ^
    --steps evaluate ^
    --output-dir output/evaluation_results ^
    --checkpoint-path output/run_20250421_152849/checkpoints/best_model.pt ^
    --architecture unet ^
    --encoder resnet34 ^
    --image-size 224 ^
    --batch-size 16 ^
    --threshold 0.5 ^
    --use-wandb ^
    --wandb-project glaucoma-detection ^
    --wandb-detailed-viz ^
    --wandb-notes "Evaluation of trained UNet model with test-time augmentation"

echo.
echo Evaluation completed.
pause