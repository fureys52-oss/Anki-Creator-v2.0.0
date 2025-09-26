# app.py

from pathlib import Path
from ui import build_ui

# --- Global Configuration ---
SCRIPT_VERSION = "4.0.0"
PDF_CACHE_DIR = Path(".pdf_cache")
AI_CACHE_DIR = Path(".ai_cache")
LOG_DIR = Path("logs")
MAX_LOG_FILES = 10
MAX_DECKS = 10

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
        clip_model={'model': CLIP_MODEL}
    )
    
    app.launch(server_name="127.0.0.1", debug=True, inbrowser=True)