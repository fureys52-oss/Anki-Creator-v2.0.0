#!/bin/bash

# Navigate to the script's directory and exit if it fails
cd "$(dirname "$0")" || exit

echo "--- Anki Deck Generator macOS Setup ---"
echo "This script will create a virtual environment and install dependencies."
echo ""

# Check for python3
if ! command -v python3 &> /dev/null
then
    echo "ERROR: python3 could not be found."
    echo "Please install Python 3, for example by running 'brew install python'."
    exit
fi

echo "Step 1: Creating Python virtual environment in 'venv'..."
python3 -m venv venv

echo "Step 2: Activating virtual environment..."
source venv/bin/activate

echo "Step 3: Installing required packages from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "--- Setup Complete! ---"
echo "You can now run the application using run.command."

# Deactivate venv when done, optional but good practice
deactivate

read -r -p "Press Enter to close this window..."