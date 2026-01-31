@echo off
REM ============================================================================
REM PrismStyle AI - Quick Training with Sample Data
REM For testing before full DeepFashion2 training
REM ============================================================================

echo.
echo ===============================================
echo   PrismStyle AI - Quick Training Mode
echo ===============================================
echo.
echo This will:
echo   1. Create a synthetic sample dataset
echo   2. Train a quick model (5 epochs)
echo   3. Export to ONNX, TFLite, and CoreML
echo.

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Setting up environment first...
    call setup_training.bat
    call venv\Scripts\activate.bat
)

echo.
echo Step 1: Creating sample dataset...
python create_sample_dataset.py --output_dir ./sample_data --num_samples 100

echo.
echo Step 2: Training model (quick mode - 5 epochs)...
python train_and_export.py --data_dir ./sample_data --output_dir ./models --epochs 5 --quick

echo.
echo ===============================================
echo   TRAINING COMPLETE!
echo ===============================================
echo.
echo Model files ready in ./models/
echo.
echo For production quality, train with full DeepFashion2:
echo   python train_and_export.py --data_dir C:\path\to\deepfashion2 --epochs 60
echo.
pause
