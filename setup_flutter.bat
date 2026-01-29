@echo off
:: Quick Flutter Setup Script for PrismStyle AI
:: Run this as Administrator

echo ================================
echo PrismStyle AI - Flutter Setup
echo ================================

:: Check if Flutter is already installed
where flutter >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Flutter is already installed
    goto :setup_project
)

echo 📥 Installing Flutter SDK...

:: Create directory
mkdir "C:\src" >nul 2>&1
cd /d "C:\src"

:: Download Flutter (latest stable)
echo Downloading Flutter...
powershell -Command "Invoke-WebRequest -Uri 'https://storage.googleapis.com/flutter_infra_release/releases/stable/windows/flutter_windows_3.24.0-stable.zip' -OutFile 'flutter.zip'"

:: Extract Flutter
echo Extracting Flutter...
powershell -Command "Expand-Archive -Path 'flutter.zip' -DestinationPath '.' -Force"

:: Clean up
del flutter.zip

:: Add to PATH
echo Adding Flutter to PATH...
setx PATH "%PATH%;C:\src\flutter\bin" /M

echo.
echo ✅ Flutter installed successfully!
echo.
echo PLEASE:
echo 1. Close and reopen this command prompt
echo 2. Run 'flutter doctor' to verify installation
echo 3. Then run this script again to setup the project
echo.

pause
exit /b

:setup_project
echo.
echo 🚀 Setting up PrismStyle AI project...

cd /d "c:\Users\pdave\Downloads\prismstyle_ai_0393-main"

:: Check Flutter installation
echo Checking Flutter...
flutter doctor

:: Get dependencies
echo.
echo 📦 Getting dependencies...
flutter pub get

:: Check for connected devices
echo.
echo 📱 Checking connected devices...
flutter devices

echo.
echo ✅ Setup complete!
echo.
echo To run the app:
echo flutter run
echo.
echo To run on specific device:
echo flutter run -d [device_id]
echo.
echo To see all available commands:
echo flutter --help
echo.

pause
