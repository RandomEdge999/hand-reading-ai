@echo off
echo ========================================
echo Advanced Hand Sign Recognition System
echo ========================================
echo.
echo Starting the advanced system launcher...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

REM Check if advanced dependencies are installed
echo Checking advanced dependencies...
python -c "import cv2, mediapipe, xgboost, sklearn" >nul 2>&1
if errorlevel 1 (
    echo Installing advanced dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Start the working launcher
echo Starting working system launcher...
python working_launcher.py

pause 