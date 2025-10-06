# app.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from pathlib import Path
from ui import build_ui
import requests
from packaging import version as version_parser
import gradio as gr
import subprocess


SCRIPT_VERSION = "3.0.0" 
VERSION_URL = "https://raw.githubusercontent.com/fureys52-oss/Anki-Creator-v2.0.0/refs/heads/main/version.txt" 
DOWNLOAD_URL = "https://github.com/fureys52-oss/Anki-Creator-v2.0.0/archive/refs/heads/main.zip"

# --- Global Configuration ---
PDF_CACHE_DIR = Path(".pdf_cache")
AI_CACHE_DIR = Path(".ai_cache")
LOG_DIR = Path("logs")
MAX_LOG_FILES = 10
MAX_DECKS = 10

def check_for_updates():
    """
    Checks GitHub for a newer version and returns a visible Markdown component if available.
    """
    try:
        response = requests.get(VERSION_URL, timeout=5)
        response.raise_for_status()
        latest_version_str = response.text.strip()
        
        if version_parser.parse(latest_version_str) > version_parser.parse(SCRIPT_VERSION):
            update_message = (
                f"**Update Available!** You are on v{SCRIPT_VERSION}, but v{latest_version_str} is available. "
                f"[Click here to download the latest version]({DOWNLOAD_URL})."
            )
            return gr.update(value=update_message, visible=True)
            
    except requests.RequestException as e:
        print(f"Update check failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during update check: {e}")
        
    return gr.update(visible=False)

def load_clip_model():
    """Loads the SentenceTransformer model and returns it, along with a success message."""
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading CLIP model (this may take a moment on first run)...")
        model = SentenceTransformer('clip-ViT-B-32')
        print("CLIP model loaded successfully.")
        return {'model': model}, "✅ AI model for image analysis is ready."
    except ImportError:
        print("\nCRITICAL ERROR: 'sentence-transformers' is not installed.")
        print("Please install it by running: pip install sentence-transformers torch")
        return {'model': None}, "❌ CRITICAL ERROR: `sentence-transformers` is not installed. Image analysis is disabled."
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to load CLIP model. Image validation will be disabled. Error: {e}")
        return {'model': None}, f"❌ CRITICAL ERROR: Failed to load CLIP model. Image analysis is disabled."

# ==============================================================================
# SECTION: SCRIPT LAUNCHER
# ==============================================================================
if __name__ == "__main__":
    
    # --- THIS IS THE FIX: A much more reliable way to find the application's root path ---
    # When run as a script, __file__ is the path to app.py
    # This works whether it's run by a bundled python.exe or a system Python.
    application_path = os.path.dirname(os.path.realpath(__file__))
    
    is_setup_run = len(sys.argv) > 1 and sys.argv[1] == '--setup-shortcut-only'
    
    # The setup logic now only runs when explicitly called by the setup script.
    if is_setup_run:
        flag_file = os.path.join(application_path, '_first_run_complete.flag')

        if not os.path.exists(flag_file):
            print("Performing first-time setup for Windows...")
            if sys.platform == "win32":
                print("  > Creating desktop shortcut...")
                try:
                    import winshell
                    from win32com.client import Dispatch
                    
                    desktop = winshell.desktop()
                    shortcut_path = os.path.join(desktop, "Anki Deck Generator.lnk")
                    
                    # The target of the shortcut is the master .bat script in the root
                    target_path = os.path.join(application_path, "Start Anki Generator.bat")
                    icon_path = os.path.join(application_path, "icon.ico")
                    working_dir = application_path
                    
                    shell = Dispatch('WScript.Shell')
                    shortcut = shell.CreateShortcut(shortcut_path)
                    shortcut.TargetPath = target_path
                    shortcut.WorkingDirectory = working_dir
                    if os.path.exists(icon_path):
                        shortcut.IconLocation = icon_path
                    shortcut.save()
                    print("  > Shortcut created successfully on your Desktop!")
                except Exception as e:
                    print(f"  > WARNING: Could not create shortcut. Please do so manually. Reason: {e}")
            
            try:
                with open(flag_file, 'w') as f:
                    f.write('done')
                print("  > First-run setup complete.")
            except Exception as e:
                print(f"  > WARNING: Could not write first-run flag file. Reason: {e}")
            
            # Exit after setup is complete
            sys.exit(0)
        else:
            print("Setup has already been completed. Exiting.")
            sys.exit(0)
    
    # --- Normal application launch logic ---
    cache_dirs = (PDF_CACHE_DIR, AI_CACHE_DIR)
    
    app = build_ui(
        version=SCRIPT_VERSION,
        max_decks=MAX_DECKS,
        cache_dirs=cache_dirs,
        log_dir=LOG_DIR,
        max_log_files=MAX_LOG_FILES,
        update_checker_func=check_for_updates,
        load_clip_model_func=load_clip_model
    )
    
    app.launch(server_name="127.0.0.1", debug=True, inbrowser=True)