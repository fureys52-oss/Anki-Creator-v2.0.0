#!/bin/bash

# Navigate to the directory where this script is located
cd "$(dirname "$0")" || exit

echo "--- Anki Deck Generator macOS Setup ---"
echo "This script will install all necessary dependencies."
echo ""

# --- App Configuration ---
APP_NAME="Anki Deck Generator"
DESKTOP_APP_PATH="${HOME}/Desktop/${APP_NAME}.app"
ICON_PATH="${PWD}/icon.icns" # Use PWD for the absolute path

# --- User Introduction Dialog ---
osascript -e "tell app \"System Events\" to display dialog \"Welcome to the Anki Deck Generator setup for macOS.\n\nThis will install necessary software like Homebrew, Python, and Tesseract. You may be asked for your password.\n\nClick OK to begin.\" with title \"Anki Deck Generator Setup\" buttons {\"Cancel\", \"OK\"} default button \"OK\""
if [ $? -ne 0 ]; then
    echo "Setup cancelled by user."
    exit 1
fi

# --- Step 1: Check for and Install Homebrew ---
if ! command -v brew &> /dev/null; then
    echo "[1/5] Homebrew not found. Starting installation..."
    osascript -e 'tell app "Terminal" to do script "/bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""'
    
    # Prompt the user to wait for the installation to finish in the new terminal
    osascript -e "tell app \"System Events\" to display dialog \"Homebrew installation has started in a new Terminal window.\n\nPlease follow the instructions there. This may take several minutes.\n\nClick 'Continue' ONLY after the Homebrew installation is completely finished.\" with title \"Action Required\" buttons {\"Continue\"} default button \"Continue\""
    
    # After installation, Homebrew needs to be added to the PATH for the current script
    if [ -x "/opt/homebrew/bin/brew" ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        echo "❌ ERROR: Homebrew installation could not be verified. Please run this script again."
        read -p "Press Enter to exit..."
        exit 1
    fi
else
    echo "[1/5] Homebrew is already installed. ✅"
fi

# --- Step 2: Install Python & Tesseract with Homebrew ---
echo ""
echo "[2/5] Installing/updating Python 3.11 and Tesseract via Homebrew..."
brew install python@3.11 tesseract
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to install Python or Tesseract via Homebrew."
    read -p "Press Enter to exit..."
    exit 1
fi

# --- Step 3: Create Python Virtual Environment ---
echo ""
echo "[3/5] Creating Python virtual environment..."
# Use the Homebrew-installed Python to ensure the correct version
/opt/homebrew/bin/python3.11 -m venv venv
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to create the virtual environment."
    read -p "Press Enter to exit..."
    exit 1
fi

# --- Step 4: Install Python Packages ---
echo ""
echo "[4/5] Activating environment and installing required packages..."
source venv/bin/activate
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Package installation failed. Please check your internet connection."
    deactivate
    read -p "Press Enter to exit..."
    exit 1
fi
deactivate

# --- Step 5: Create the Automator App Launcher ---
echo ""
echo "[5/5] Creating application launcher on your Desktop..."

# Define the shell script that the Automator app will run.
# This ensures it always runs from the correct directory.
RUN_SCRIPT_CONTENT="#!/bin/bash\\ncd '${PWD}' || exit\\nsource venv/bin/activate\\n/opt/homebrew/bin/python3.11 app.py"

# Remove any old version of the app first
rm -rf "${DESKTOP_APP_PATH}"

# Use AppleScript to create the Automator application
osascript >/dev/null 2>&1 <<EOF
set app_path to POSIX file "${DESKTOP_APP_PATH}"
set script_content to "${RUN_SCRIPT_CONTENT}"
set icon_path to POSIX file "${ICON_PATH}"

tell application "Automator"
    set new_workflow to make new workflow
    tell new_workflow
        make new "Run Shell Script" action with properties {input method:1, script:script_content, shell:"/bin/bash"}
        save it in file app_path
        close
    end tell
end tell

try
    tell application "Finder"
        set icon of file (app_path as text) to file icon_path
    end tell
on error
    -- This might fail if icon.icns is missing, but we don't want to stop the script.
end try
EOF

# --- Final Success Message ---
echo ""
echo "===================================================================="
echo "✅ Setup Complete!"
echo "An application named '${APP_NAME}' has been created on your Desktop."
echo "You can now use it to launch the Anki Deck Generator."
echo "===================================================================="
echo ""

osascript -e "tell app \"System Events\" to display dialog \"Setup is complete! You can now launch the '${APP_NAME}' application from your Desktop.\" with title \"Success\" with icon file \"${ICON_PATH}\" buttons {\"Done\"} default button \"Done\""

exit 0