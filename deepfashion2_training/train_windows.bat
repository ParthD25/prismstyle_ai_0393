@echo off
REM ===================================================
REM DeepFashion2 Training Script for Windows
REM PrismStyle AI - Clothing Classification Model
REM ===================================================

echo ===================================================
echo  PrismStyle AI - DeepFashion2 Model Training
echo ===================================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8-3.11
    pause
    exit /b 1
)

REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] NVIDIA GPU not detected. Training will be slow on CPU.
) else (
    echo [OK] NVIDIA GPU detected
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
)

echo.
echo Step 1: Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

echo.
echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 3: Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Step 4: Creating directory structure...
if not exist "data\deepfashion2\train" mkdir data\deepfashion2\train
if not exist "data\deepfashion2\validation" mkdir data\deepfashion2\validation
if not exist "models\checkpoints" mkdir models\checkpoints
if not exist "converted_models" mkdir converted_models
if not exist "logs" mkdir logs
echo [OK] Directories created

echo.
echo ===================================================
echo  IMPORTANT: Dataset Setup Required
echo ===================================================
echo.
echo Before training, you need to download DeepFashion2:
echo.
echo 1. Go to: https://github.com/switchablenorms/DeepFashion2
echo 2. Request access to the dataset
echo 3. Download using password: 2019Deepfashion2
echo 4. Extract to: data\deepfashion2\
echo.
echo Directory structure should be:
echo   data\deepfashion2\train\[category folders]\
echo   data\deepfashion2\validation\[category folders]\
echo.
echo Press any key when dataset is ready, or Ctrl+C to exit...
pause >nul

echo.
echo Step 5: Verifying dataset...
if not exist "data\deepfashion2\train" (
    echo [ERROR] Training data not found at data\deepfashion2\train
    echo Please download and extract the dataset first.
    pause
    exit /b 1
)

echo [OK] Dataset directory found

echo.
echo Step 6: Starting training...
echo This will take 8-12 hours on an RTX 3070/4070
echo.
echo To monitor progress, open another terminal and run:
echo   tensorboard --logdir logs
echo   Then visit: http://localhost:6006
echo.

python train_model.py --epochs 50 --batch-size 32

if errorlevel 1 (
    echo [ERROR] Training failed. Check the error messages above.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo  Training Complete!
echo ===================================================
echo.
echo Step 7: Converting models to mobile formats...
python convert_model.py --model-path models/final_model_finetuned.h5 --output-dir converted_models

echo.
echo ===================================================
echo  Model Conversion Complete!
echo ===================================================
echo.
echo Output files:
echo   - converted_models\deepfashion2_classifier.tflite (for Android ^& iOS)
echo   - converted_models\FashionClassifier.mlmodel (for iOS Core ML)
echo.
echo Next steps:
echo   1. Copy deepfashion2_classifier.tflite to ..\assets\models\
echo   2. Copy FashionClassifier.mlmodel to ..\ios\Runner\Resources\
echo   3. Run: flutter pub get
echo   4. Run: flutter run -d ios (or android)
echo.
pause
