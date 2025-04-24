@echo off
echo Starting Glaucoma CDR Analysis with Multiple Methods...
echo.

REM Run with diameter-based CDR
echo Running with diameter-based CDR method...
python -m glaucoma.main ^
    --steps evaluate ^
    --output-dir output/cdr_analysis/diameter ^
    --checkpoint-path output/run_20250421_152849/checkpoints/best_model.pt ^
    --architecture unet ^
    --encoder resnet34 ^
    --image-size 224 ^
    --batch-size 16 ^
    --threshold 0.5 ^
    --calculate-cdr ^
    --cdr-method diameter ^
    --use-wandb ^
    --wandb-project glaucoma-detection ^
    --wandb-name "CDR-Analysis-Diameter" ^
    --wandb-notes "CDR analysis using diameter-based method"

echo.
echo Diameter-based CDR analysis completed.
echo CDR values saved to output/cdr_results/cdr_values_diameter.csv
echo.

REM Run with area-based CDR
echo Running with area-based CDR method...
python -m glaucoma.main ^
    --steps evaluate ^
    --output-dir output/cdr_analysis/area ^
    --checkpoint-path output/run_20250421_152849/checkpoints/best_model.pt ^
    --architecture unet ^
    --encoder resnet34 ^
    --image-size 224 ^
    --batch-size 16 ^
    --threshold 0.5 ^
    --calculate-cdr ^
    --cdr-method area ^
    --use-wandb ^
    --wandb-project glaucoma-detection ^
    --wandb-name "CDR-Analysis-Area" ^
    --wandb-notes "CDR analysis using area-based method"

echo.
echo Area-based CDR analysis completed.
echo CDR values saved to output/cdr_results/cdr_values_area.csv
echo.

REM Run with both methods
echo Running with combined CDR methods...
python -m glaucoma.main ^
    --steps evaluate ^
    --output-dir output/cdr_analysis/both ^
    --checkpoint-path output/run_20250421_152849/checkpoints/best_model.pt ^
    --architecture unet ^
    --encoder resnet34 ^
    --image-size 224 ^
    --batch-size 16 ^
    --threshold 0.5 ^
    --calculate-cdr ^
    --cdr-method both ^
    --use-wandb ^
    --wandb-project glaucoma-detection ^
    --wandb-name "CDR-Analysis-Combined" ^
    --wandb-notes "CDR analysis using both area and diameter methods"

echo.
echo Combined CDR analysis completed.
echo CDR values saved to output/cdr_results/cdr_values_both.csv
echo.

echo All CDR analyses completed successfully.
pause