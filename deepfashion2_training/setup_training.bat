@echo off
REM ============================================================================
REM PrismStyle AI - Training Setup for Windows
REM Optimized for: RTX 5060 Ti, AMD 9090X, 64GB RAM
REM ============================================================================

echo.
echo ===============================================
echo   PrismStyle AI - Fashion Classifier Training
echo   Optimized for RTX 5060 Ti + AMD 9090X
echo ===============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Create virtual environment if not exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

echo.
echo Step 1: Installing PyTorch with CUDA 12.1 support...
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Step 2: Installing other dependencies...
echo.
pip install -r requirements.txt

echo.
echo ===============================================
echo   SETUP COMPLETE!
echo ===============================================
echo.
echo To start training, run one of these commands:
echo.
echo   OPTION 1 (Full DeepFashion2 training):
echo   python train_and_export.py --data_dir C:\path\to\deepfashion2 --output_dir ./models
echo.
echo   OPTION 2 (Create sample dataset first):
echo   python create_sample_dataset.py --output_dir ./sample_data
echo   python train_and_export.py --data_dir ./sample_data --output_dir ./models
echo.
echo After training completes, copy these files to the Flutter app:
echo   - models/clothing_classifier.onnx       -> assets/models/
echo   - models/clothing_classifier.tflite     -> assets/models/
echo   - models/clothing_classifier.mlpackage  -> ios/Runner/
echo.
pause
