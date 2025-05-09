@echo off
echo Starting Glaucoma Model Evaluation with CDR Calculation...
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
    --calculate-cdr ^
    --use-wandb ^
    --wandb-project glaucoma-detection ^
    --wandb-detailed-viz ^
    --wandb-notes "Evaluation with CDR calculation for clinical analysis"

echo.
echo Evaluation with CDR calculation completed.
echo CDR values saved to output/cdr_results/cdr_values_diameter.csv
pause