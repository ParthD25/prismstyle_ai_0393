@echo off
echo ============================================================
echo   PrismStyle AI - V4 BREAKTHROUGH Training
echo   ConvNeXt + Focal Loss + OneCycleLR + 384x384
echo ============================================================

REM Kill any existing Python training processes
taskkill /F /IM python.exe 2>nul

REM Set environment variables for optimal GPU usage
set CUDA_VISIBLE_DEVICES=0
set TF_CPP_MIN_LOG_LEVEL=2
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo [INFO] Starting V4 BREAKTHROUGH training with:
echo   - Image size: 384x384 (increased from 300)
echo   - ConvNeXt-Base backbone
echo   - Focal Loss for class imbalance
echo   - OneCycleLR scheduler
echo   - Weighted sampling
echo   - Batch size: 24

cd /d "c:\Users\pdave\Downloads\prismstyle_ai_0393-main\deepfashion2_training"

python train_pytorch_v4.py

echo.
echo ============================================================
echo   Training Complete!
echo ============================================================
pause
