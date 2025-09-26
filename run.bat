@echo off
echo Starting the Anki Deck Generator...
echo If a firewall permission dialog appears, please grant access.

:: This command ensures the script runs in the correct directory
cd /d "%~dp0"

:: --- ENHANCEMENT 1: Check if the venv exists ---
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo.
    echo [ERROR] Virtual environment not found!
    echo Please run the 'setup.bat' script first to set up the project.
    echo.
    pause
    exit /b
)

:: Activate the virtual environment before running the Python script
echo Activating virtual environment...
CALL venv\Scripts\activate.bat

:: --- ENHANCEMENT 2: Check for requirements and install if needed ---
echo Checking for required packages...
pip install -r requirements.txt

:: Now, run the application using the venv's Python
echo.
echo Launching the application...
python app.py

echo.
echo The application has closed.
pause