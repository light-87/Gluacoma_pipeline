@echo off
echo Starting Glaucoma Detection with optimized focal+dice loss configuration...
echo.

python -m glaucoma.main ^
    --steps load,preprocess,train,evaluate ^
    --output-dir output/optimized_model ^
    --architecture unet ^
   --encoder resnet34 ^
   --image-size 224 ^
   --batch-size 16 ^
   --learning-rate 0.001 ^
   --epochs 2 ^
    --loss-function combined ^
     --focal-weight 1.0 ^
     --use-wandb ^
     --wandb-project glaucoma-detection ^
     --run-name "Optimized-FocalDice" ^
    --wandb-detailed-viz ^
    --wandb-notes "Optimized model with Dice+Focal loss from model_training.md"

echo.
echo Pipeline execution completed.
pause