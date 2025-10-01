#!/bin/bash

# Navigate to the script's directory and exit if it fails
cd "$(dirname "$0")" || exit

echo "--- Anki Deck Generator macOS Setup ---"
echo "This script will check dependencies, create a virtual environment, and install packages."
echo ""

# --- Python Version Check ---
MIN_PYTHON_MINOR=9
PYTHON_CMD="python3"
echo "Step 1: Checking for a compatible Python 3 version (3.9+)..."

# Function to check a given python command
check_python_version() {
    local cmd=$1
    if ! command -v "$cmd" &> /dev/null; then return 1; fi

    local version_number
    version_number=$( ($cmd --version 2>/dev/null || $cmd -V 2>/dev/null) | cut -d' ' -f2)
    
    local minor_version
    minor_version=$(echo "$version_number" | cut -d'.' -f2)

    if [ "$minor_version" -ge "$MIN_PYTHON_MINOR" ]; then
        PYTHON_CMD=$cmd
        return 0
    else
        return 1
    fi
}

# Check default python3, then try to find a newer one if needed
if ! check_python_version "python3"; then
    echo "-> Default 'python3' is too old. Searching for a newer version..."
    if check_python_version "python3.12" || check_python_version "python3.11" || check_python_version "python3.10" || check_python_version "python3.9"; then
        echo "✅ Found compatible Python at: $PYTHON_CMD"
    else
        echo "❌ ERROR: No Python version 3.9 or newer could be found."
        echo "   Please install a modern version of Python by running: brew install python3"
        read -r -p "Press Enter to exit..."
        exit 1
    fi
else
    echo "✅ Default 'python3' is compatible."
fi
echo ""

echo "Step 2: Creating Python virtual environment using '$PYTHON_CMD'..."
"$PYTHON_CMD" -m venv venv

echo "Step 3: Activating virtual environment..."
source venv/bin/activate

echo "Step 4: Installing required packages from requirements.txt..."
pip install -r requirements.txt

# Check if pip install succeeded
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ ERROR: Package installation failed. Please check your internet connection and try again."
    deactivate
    read -r -p "Press Enter to exit..."
    exit 1
fi

echo ""
echo "--- Setup Complete! ---"
echo "You can now run the application using run.command."

# Deactivate venv when done
deactivate

read -r -p "Press Enter to close this window..."