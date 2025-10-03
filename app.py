# app.py

from pathlib import Path
from ui import build_ui
import requests
from packaging import version as version_parser
import gradio as gr

SCRIPT_VERSION = "3.0.0" # Make sure this matches your app's actual current version
# IMPORTANT: Replace with your actual GitHub username and repository name
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
    Checks GitHub for a newer version of the application and returns a visible
    Markdown component if an update is available.
    """
    try:
        # Fetch the latest version number from the version.txt file on GitHub
        response = requests.get(VERSION_URL, timeout=5)
        response.raise_for_status() # Raises an error if the request failed
        latest_version_str = response.text.strip()
        
        # Use the packaging library to safely compare version numbers
        # This correctly handles cases like "v1.10.0" > "v1.9.0"
        if version_parser.parse(latest_version_str) > version_parser.parse(SCRIPT_VERSION):
            update_message = (
                f"**Update Available!** You are on v{SCRIPT_VERSION}, but v{latest_version_str} is available. "
                f"[Click here to download the latest version]({DOWNLOAD_URL})."
            )
            # Return a Gradio update object to make the component visible and set its content
            return gr.update(value=update_message, visible=True)
            
    except requests.RequestException as e:
        print(f"Update check failed: {e}")
    except Exception as e:
        # Catch any other unexpected errors during parsing
        print(f"An unexpected error occurred during update check: {e}")
        
    # If there's no update or if the check fails, return an update to keep it hidden
    return gr.update(visible=False)

# ==============================================================================
# SECTION: SCRIPT LAUNCHER
# ==============================================================================
if __name__ == "__main__":
    # --- Heavy Model Loading ---
    # Load the powerful multi-modal CLIP model once at startup.
    # This is a one-time cost and prevents reloading for each deck.
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading CLIP model (this may take a moment on first run)...")
        CLIP_MODEL = SentenceTransformer('clip-ViT-B-32')
        print("CLIP model loaded successfully.")
    except ImportError:
        print("\nCRITICAL ERROR: 'sentence-transformers' is not installed.")
        print("Please install it by running: pip install sentence-transformers torch")
        CLIP_MODEL = None
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to load CLIP model. Image validation will be disabled. Error: {e}")
        CLIP_MODEL = None

    cache_dirs = (PDF_CACHE_DIR, AI_CACHE_DIR)
    
    app = build_ui(
        version=SCRIPT_VERSION,
        max_decks=MAX_DECKS,
        cache_dirs=cache_dirs,
        log_dir=LOG_DIR,
        max_log_files=MAX_LOG_FILES,
        clip_model={'model': CLIP_MODEL},
        update_checker_func=check_for_updates
    )
    
    
    
    app.launch(server_name="127.0.0.1", debug=True, inbrowser=True)