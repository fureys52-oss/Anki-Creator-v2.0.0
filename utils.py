# utils.py

import os
import shutil
import re
from pathlib import Path
from typing import Dict, Tuple, List, Any

from dotenv import load_dotenv
import gradio as gr

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

# --- Configuration and API Keys ---
def get_api_keys_from_env() -> Dict[str, str]:
    script_dir = Path(__file__).resolve().parent
    env_path = script_dir / '.env'
    if not env_path.exists():
        raise FileNotFoundError(f"CRITICAL: .env file not found in {script_dir}. Please create one from the env.template file.")
    load_dotenv(dotenv_path=env_path)
    
    keys = {
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "OPENVERSE_API_KEY": os.getenv("OPENVERSE_API_KEY"),
        "FLICKR_API_KEY": os.getenv("FLICKR_API_KEY"),
    }
    
    if not keys["GEMINI_API_KEY"]:
        raise ValueError(f"CRITICAL: GEMINI_API_KEY is missing or empty in {env_path}.")
        
    return keys

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