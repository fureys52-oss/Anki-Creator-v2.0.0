#!/bin/bash

# Navigate to the script's directory and exit if it fails
cd "$(dirname "$0")" || exit

echo "--- Creating Desktop Shortcut ---"

# The file we want to link to (our main run script)
SOURCE_FILE="$(pwd)/run.command"

# The name and location for our shortcut on the Desktop
DESTINATION_FILE="$HOME/Desktop/Run Anki Generator"

# Check if a shortcut already exists
if [ -e "$DESTINATION_FILE" ]; then
    echo "A shortcut already exists on your Desktop. Overwriting it."
    rm "$DESTINATION_FILE"
fi

echo "Creating shortcut on your Desktop..."

# Create the symbolic link
ln -s "$SOURCE_FILE" "$DESTINATION_FILE"

# Verify that the link was created
if [ -L "$DESTINATION_FILE" ]; then
    echo ""
    echo "✅ Success! A shortcut named 'Run Anki Generator' has been placed on your Desktop."
else
    echo ""
    echo "❌ ERROR: Failed to create the shortcut."
fi

echo ""
read -r -p "Press Enter to close this window..."