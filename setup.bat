@echo off
TITLE Anki Deck Generator Setup

:: Ensure the script runs in its own directory
cd /d "%~dp0"

echo.
echo ==========================================================
echo Anki Deck Generator - Project Setup
echo ==========================================================
echo.
echo This script will set up a dedicated virtual environment
echo and install all the necessary Python packages for this project.
echo.

:: --- Step 1: Check for Python ---
echo [1/4] Checking for a valid Python installation...
python --version >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in your system's PATH.
    echo Please install Python 3.9 or newer from python.org and ensure
    echo you check the "Add Python to PATH" box during installation.
    echo.
    pause
    exit /b
)
echo      ...Python found.
echo.

:: --- Step 2: Create the Virtual Environment ---
echo [2/4] Checking for virtual environment ('venv' folder)...
IF EXIST "venv" (
    echo      ...'venv' folder already exists. Skipping creation.
) ELSE (
    echo      ...'venv' folder not found. Creating it now...
    python -m venv venv
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b
    )
    echo      ...Virtual environment created successfully.
)
echo.

:: --- Step 3: Activate and Install Packages ---
echo [3/4] Activating virtual environment and installing packages...
CALL venv\Scripts\activate.bat

:: It's good practice to upgrade pip first
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install one or more packages from requirements.txt.
    echo Please check the error messages above.
    pause
    exit /b
)
echo      ...All packages installed successfully.
echo.

:: --- Step 4: Finish ---
echo [4/4] Setup Complete!
echo ==========================================================
echo.
echo You can now run the application by double-clicking 'run.bat'.
echo.
pause