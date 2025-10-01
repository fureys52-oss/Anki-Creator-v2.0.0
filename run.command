#!/bin/bash

# Navigate to the script's directory and exit if it fails
cd "$(dirname "$0")" || exit

echo "--- Starting Anki Deck Generator ---"

# Check if the virtual environment exists
if [ ! -d "venv" ]; then
    echo "ERROR: The 'venv' directory was not found."
    echo "Please run the 'setup.command' script first."
    read -r -p "Press Enter to exit..."
    exit 1
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Launching application... Please wait for the Gradio UI to start."
echo "If the app fails to start, this window will stay open with an error message."
python app.py

# Deactivate when the app is closed
deactivate

echo ""
echo "Application closed."
# This final read will keep the window open so users can see any errors
read -r -p "Press Enter to close this window..."