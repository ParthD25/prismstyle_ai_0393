@echo off
REM PrismStyle AI - Start Local Python AI Backend
REM This connects to the outfit_app_package for FREE local AI

echo ============================================================
echo PrismStyle AI - Local Python AI Backend
echo ============================================================
echo.

REM Set the path to your outfit_app_package
set OUTFIT_APP_PATH=C:\outfit-app

REM Check if outfit app exists
if not exist "%OUTFIT_APP_PATH%" (
    echo WARNING: Outfit app not found at %OUTFIT_APP_PATH%
    echo.
    echo Please update OUTFIT_APP_PATH in this script to point to your
    echo outfit_app_package_updated.zip extracted location.
    echo.
)

REM Activate conda environment if it exists
if exist "%OUTFIT_APP_PATH%\env\Scripts\activate.bat" (
    echo Activating outfit app environment...
    call "%OUTFIT_APP_PATH%\env\Scripts\activate.bat"
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    echo Activating Miniconda...
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
    conda activate outfit-env 2>nul
)

REM Install Flask if needed
pip install flask flask-cors pillow numpy --quiet

echo.
echo Starting server on http://localhost:5000
echo Press Ctrl+C to stop
echo.

REM Start the server
python "%~dp0server.py"

pause
