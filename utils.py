# utils.py

import os
import shutil
import re
from pathlib import Path
from typing import Dict, Tuple, List, Any

from dotenv import load_dotenv
import gradio as gr
import pytesseract
import sys
import json

# --- File and Cache Management ---
def manage_log_files(log_dir: Path, max_logs: int):
    log_dir.mkdir(exist_ok=True)
    log_files = sorted(log_dir.glob('*.txt'), key=os.path.getmtime)
    while len(log_files) >= max_logs:
        os.remove(log_files[0])
        log_files.pop(0)

def clear_cache(pdf_cache_dir: Path, ai_cache_dir: Path) -> str:
    results = []
    for name, cache_dir in [("PDF", pdf_cache_dir), ("AI", ai_cache_dir)]:
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                results.append(f"✅ {name} cache cleared.")
            except Exception as e:
                results.append(f"❌ Error clearing {name} cache: {e}")
        else:
            results.append(f"ℹ️ {name} cache not found.")
        cache_dir.mkdir(exist_ok=True)
    return " ".join(results)

def configure_tesseract():
    """
    Checks for Tesseract in PATH, then checks common installation paths
    for Windows and macOS. Returns True if configured, False otherwise.
    """
    if shutil.which("tesseract"):
        print("Tesseract executable found in system PATH.")
        return True

    # Platform-specific checks
    if sys.platform == "win32":
        windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(windows_path):
            print(f"Tesseract found at default Windows location: {windows_path}")
            pytesseract.pytesseract.tesseract_cmd = windows_path
            return True
    elif sys.platform == "darwin": # macOS
        # Common paths for Homebrew installations
        paths_to_check = [
            "/usr/local/bin/tesseract",      # Intel Macs
            "/opt/homebrew/bin/tesseract"    # Apple Silicon Macs
        ]
        for path in paths_to_check:
            if os.path.exists(path):
                print(f"Tesseract found at default macOS location: {path}")
                pytesseract.pytesseract.tesseract_cmd = path
                return True

    print("Tesseract executable not found in PATH or default installation directory.")
    return False

# --- Configuration and API Keys ---
def get_api_keys(ui_keys: Dict[str, str]) -> Dict[str, str]:
    """
    Gets API keys, prioritizing keys entered in the UI over keys in a .env file.
    """
    # Load keys from .env file as a fallback
    script_dir = Path(__file__).resolve().parent
    env_path = script_dir / '.env'
    env_keys = {}
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        env_keys = {
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
            "OPENVERSE_API_KEY": os.getenv("OPENVERSE_API_KEY"),
            "FLICKR_API_KEY": os.getenv("FLICKR_API_KEY"),
        }

    # Prioritize UI keys
    final_keys = {
        "GEMINI_API_KEY": ui_keys.get("GEMINI_API_KEY") or env_keys.get("GEMINI_API_KEY"),
        "OPENVERSE_API_KEY": ui_keys.get("OPENVERSE_API_KEY") or env_keys.get("OPENVERSE_API_KEY"),
        "FLICKR_API_KEY": ui_keys.get("FLICKR_API_KEY") or env_keys.get("FLICKR_API_KEY"),
    }

    if not final_keys["GEMINI_API_KEY"]:
        raise ValueError("CRITICAL: Gemini API Key not found in UI or .env file.")
        
    return final_keys

# --- UI Helpers ---
def guess_lecture_details(file: gr.File) -> Tuple[str, str]:
    if not file: return "01", ""
    file_stem = Path(file.name).stem
    num_guess = "1"
    keyword_pattern = r'(lecture|lec|session|s|chapter|chap|module|mod|unit|part|l)[\s_-]*(\d+([&/-]\d+)*)'
    if match := re.search(keyword_pattern, file_stem, re.IGNORECASE):
        num_guess = match.group(2).strip()
    elif match := re.search(r'^\s*(\d+([&/-]\d+)*)', file_stem):
        num_guess = match.group(1).strip()
    if len(num_guess) == 1: num_guess = f"0{num_guess}"
    
    name_guess = re.sub(keyword_pattern, '', file_stem, flags=re.IGNORECASE)
    name_guess = re.sub(r'[\s_-]+', ' ', name_guess).strip()
    return num_guess, name_guess.title()

def update_decks_from_files(files: List[gr.File], max_decks: int) -> List[Any]:
    updates = []
    num_files = len(files) if files else 0
    if num_files > 0: updates.append(gr.update(label=f"{num_files} File(s) Loaded"))
    else: updates.append(gr.update(label="Upload PDFs to assign to decks below"))
    
    for i in range(max_decks):
        visible = i < num_files
        deck_title_str, accordion_label_str, file_value = "", f"Deck {i+1}", []
        if visible:
            num, name = guess_lecture_details(files[i])
            file_name = Path(files[i].name).name
            deck_title_str = f"L{num} - {name}" if name else f"Lecture {num}"
            accordion_label_str = f"Deck {i+1} - {file_name}"
            file_value = [files[i].name]
            
        updates.extend([
            gr.update(visible=visible, label=accordion_label_str), 
            gr.update(value=deck_title_str), 
            gr.update(value=file_value, visible=False)
        ])
    return updates

def slugify(text: str) -> str:
    """
    Normalizes a string to be used as a valid filename.
    Removes illegal characters and replaces spaces with underscores.
    """
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    text = re.sub(r'[-\s]+', '_', text)
    return text

def is_superfluous(text: str) -> bool:
    """
    Checks if a line of text is likely superfluous (e.g., a learning objective,
    a reference, a page number, etc.) using a set of heuristic rules.
    """
    text_lower = text.strip().lower()

    # Rule 1: Check for common instructional verbs at the start of a line
    # (Common in "Learning Objectives" sections)
    if re.match(r'^\s*(define|describe|explain|list|identify|understand|compare|contrast)', text_lower):
        return True

    # Rule 2: Check for common "junk" keywords that appear on non-content pages
    junk_keywords = [
        'learning objective', 'suggested reading', 'recommended reading',
        'references', 'bibliography', 'acknowledgements', 'table of contents',
        'optional case', 'case presentation', 'case report', 'session log'
    ]
    if any(keyword in text_lower for keyword in junk_keywords):
        return True

    # Rule 3: Check for lines that are just URLs
    if 'http://' in text_lower or 'https://' in text_lower:
        return True
        
    # Rule 4: Check for short, uninformative lines (often stray headers/footers)
    # Allows for short definitions but filters out 1-2 word lines.
    if len(text.split()) < 3 and ":" not in text:
        return True

    return False
SETTINGS_FILE = Path("settings.json")

def save_settings(settings_dict: dict):
    """Saves the provided dictionary of settings to settings.json."""
    try:
        with SETTINGS_FILE.open('w', encoding='utf-8') as f:
            json.dump(settings_dict, f, indent=4)
        print(f"Settings saved successfully to {SETTINGS_FILE}")
    except Exception as e:
        print(f"Error saving settings: {e}")

def load_settings() -> dict:
    """Loads settings from settings.json if it exists."""
    if SETTINGS_FILE.exists():
        try:
            with SETTINGS_FILE.open('r', encoding='utf-8') as f:
                print(f"Loading settings from {SETTINGS_FILE}")
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If the file is corrupt or unreadable, return empty and let defaults take over.
            return {}
    return {}