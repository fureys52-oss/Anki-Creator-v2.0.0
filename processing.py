import json, hashlib, re, time, traceback
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import fitz, requests, gradio as gr
import pytesseract
from PIL import Image
import markdown2

from utils import get_api_keys_from_env, manage_log_files, configure_tesseract, slugify, is_superfluous, save_settings
from image_finder import ImageFinder, PDFImageSource, WikimediaSource, NLMOpenISource, OpenverseSource, FlickrSource
from model_manager import GeminiModelManager

# --- Define only the necessary constants ---
MAX_WORKERS_EXTRACTION = 8
BATCH_SIZE = 3

# --- Configuration Constants ---
ANKI_CONNECT_URL = "http://127.0.0.1:8765"
INTER_DECK_COOLDOWN_SECONDS = 30
HTML_COLOR_MAP = {
    "positive_key_term": "#87CEFA", "negative_key_term": "#FF6347",
    "example": "#90EE90", "mnemonic_tip": "#FFD700"
}

# --- Helper string for Mermaid JS template ---
MERMAID_JS_SCRIPT = """
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
    (function(){
        var el = document.getElementById("mermaid-{side}-{card_id}");
        if (el) {
            var graph = el.textContent.trim();
            el.innerHTML = graph;
            mermaid.initialize({ startOnLoad: false, theme: 'MERMAID_THEME_PLACEHOLDER' });
            mermaid.run({ nodes: [el] });
        }
    })();
</script>
"""

# --- New shared function tool for cloze components ---
CLOZE_COMPONENTS_TOOL = {
    "name": "create_cloze_components",
    "description": "Provides the components to build a cloze deletion card. Do not add {{c1::...}} syntax yourself.",
    "parameters": {
        "type": "object",
        "properties": {
            "Context_Question": {"type": "string"},
            "Full_Sentence": {"type": "string", "description": "The complete sentence without any cloze syntax."},
            "Cloze_Keywords": {"type": "array", "items": {"type": "string"}, "description": "A JSON list of the specific words from the sentence to be clozed."},
            "Source_Page": {"type": "string"},
            "Search_Query": {"type": "string"},
            "Simple_Search_Query": {"type": "string"}
        },
        "required": ["Context_Question", "Full_Sentence", "Cloze_Keywords", "Source_Page", "Search_Query", "Simple_Search_Query"]
    }
}

NOTE_TYPE_CONFIG = {
    "basic": {
        "modelName": "ADG - Basic",
        "fields": ["Front", "Back", "Image", "Source"],
        "css": """.card { font-family: Arial, sans-serif; font-size: 20px; text-align: center; } .content-body { margin: 0 auto; max-width: 90%; text-align: left; } img { max-height: 500px; min-width: 400px; min-height: 250px; object-fit: contain; margin-top: 15px; } ul { list-style-type: none; padding-left: 0; margin: 0; } li { position: relative; padding-left: 1.5em; margin-bottom: 0.75em; } li::before { content: '•'; position: absolute; left: 0; top: 0; color: #555; } li > strong:first-child { display: block; margin-left: -1.5em; margin-bottom: 0.5em; } li:has(> strong:first-child)::before { content: ''; } ul ul { margin-top: 0.75em; padding-left: 1.5em; } ul ul li::before { content: '○'; color: #888; }""",
        "templates": [ { "Name": "Card 1", "Front": """<div class="content-body">{{Front}}</div>""", "Back": """<div class="content-body">{{FrontSide}}</div><hr id=answer><div class="content-body">{{Back}}</div><br><br>{{#Image}}{{Image}}{{/Image}}<div style='font-size:12px; color:grey;'>{{Source}}</div>""" } ],
        "function_tool": { "name": "create_anki_card", "description": "Creates a single Anki card based on a conceptual chunk of facts.", "parameters": { "type": "object", "properties": { "Front": {"type": "string"}, "Back": {"type": "string"}, "Page_numbers": {"type": "array", "items": {"type": "integer"}}, "Search_Query": {"type": "string"}, "Simple_Search_Query": {"type": "string"} }, "required": ["Front", "Back", "Page_numbers", "Search_Query", "Simple_Search_Query"] } }
    },
    "cloze": {
        "modelName": "ADG - Atomic Cloze", "fields": ["Text", "Extra", "Image", "Source"], "isCloze": True,
        "css": ".card { font-family: Arial; font-size: 20px; text-align: center; } .cloze { font-weight: bold; color: %s; } img { max-height: 500px; min-width: 400px; min-height: 250px; object-fit: contain; }",
        "templates": [ { "Name": "Cloze Card", "Front": "{{cloze:Text}}", "Back": """{{cloze:Text}}<br><br>{{Extra}}<br><br>{{#Image}}{{Image}}{{/Image}}<div style='font-size:12px; color:grey;'>{{Source}}</div>""" } ],
        "function_tool": CLOZE_COMPONENTS_TOOL
    },
    "contextual_cloze": {
        "modelName": "ADG - Contextual Cloze", "fields": ["Text", "Extra", "Image", "Source"], "isCloze": True,
        "css": ".card { font-family: Arial; font-size: 20px; text-align: center; } .cloze { font-weight: bold; color: %s; } img { max-height: 500px; min-width: 400px; min-height: 250px; object-fit: contain; }",
        "templates": [ { "Name": "Cloze Card", "Front": "{{cloze:Text}}", "Back": """{{cloze:Text}}<br><br>{{Extra}}<br><br>{{#Image}}{{Image}}{{/Image}}<div style='font-size:12px; color:grey;'>{{Source}}</div>""" } ],
        "function_tool": CLOZE_COMPONENTS_TOOL
    },
    "mermaid": {
        "modelName": "ADG - Mermaid Diagram", "fields": ["Front", "Back", "MermaidFront", "MermaidBack", "Source", "Image"],
        "css": ".card { font-family: Arial, sans-serif; font-size: 20px; text-align: center; } .mermaid { margin-top: 20px; background-color: #f9f9f9; padding: 10px; border-radius: 5px; }",
        "templates": [ { "Name": "Card 1", "Front": "{{Front}}" + f"<div id='mermaid-front-{{{{CardID}}}}' class='mermaid'>{{{{MermaidFront}}}}</div>" + MERMAID_JS_SCRIPT.replace("{side}", "front").replace("{card_id}", "{{CardID}}"), "Back":  "{{Back}}" + f"<div id='mermaid-back-{{{{CardID}}}}' class='mermaid'>{{{{MermaidBack}}}}</div>" + "{{#Image}}{{Image}}{{/Image}}<div style='font-size:12px; color:grey;'>{{Source}}</div>" + MERMAID_JS_SCRIPT.replace("{side}", "back").replace("{card_id}", "{{CardID}}") } ],
        "function_tool": { "name": "create_mermaid_card", "description": "Creates an interactive Anki card with a fill-in-the-blank Mermaid.js diagram.", "parameters": { "type": "object", "properties": { "Front": {"type": "string"}, "Back": {"type": "string"}, "Mermaid_Front_Code": {"type": "string"}, "Mermaid_Back_Code": {"type": "string"}, "Page_numbers": {"type": "array", "items": {"type": "integer"}}, "Search_Query": {"type": "string"}, "Simple_Search_Query": {"type": "string"} }, "required": ["Front", "Back", "Mermaid_Front_Code", "Mermaid_Back_Code", "Page_numbers", "Search_Query", "Simple_Search_Query"] } }
    }
}

# --- Text Sanitization and Formatting Engine ---
def sanitize_text(text: str) -> str:
    """Cleans text from common PDF/AI artifacts like weird unicode and inconsistent newlines."""
    if not isinstance(text, str):
        return ""
    
    # Normalize newlines and strip leading/trailing whitespace
    text = text.replace('\\n', '\n').strip()
    
    # Unicode character normalization
    replacements = {
        '\u201c': '"', '\u201d': '"',  # Smart quotes to standard quotes
        '\u2018': "'", '\u2019': "'",  # Smart single quotes to standard apostrophe
        '\u2013': '-', '\u2014': '-',  # En-dash and Em-dash to standard hyphen
        '\u2022': '-',                # Bullet points to hyphens for markdown
        '\u00a0': ' ',                # Non-breaking space to regular space
        '\u00b0': '°',                # Degree symbol
        'ﬁ': 'fi', 'ﬂ': 'fl',      # Ligatures
    }
    for find, replace in replacements.items():
        text = text.replace(find, replace)
        
    # Consolidate multiple newlines into a single one for cleaner markdown parsing
    text = re.sub(r'\n{2,}', '\n', text)
    return text

def build_html_from_tags(text: str, color_map: Dict[str, str]) -> str:
    """Builds HTML from custom semantic tags and markdown."""
    tag_map = {
        "<pos>": f"<font color='{color_map.get('positive_key_term', '#0000FF')}'><b>", "</pos>": "</b></font>",
        "<neg>": f"<font color='{color_map.get('negative_key_term', '#FF0000')}'><b>", "</neg>": "</b></font>",
        "<ex>":  f"<font color='{color_map.get('example', '#008000')}'>",           "</ex>":  "</font>",
        "<tip>": f"<font color='{color_map.get('mnemonic_tip', '#FFA500')}'>",      "</tip>": "</font>",
    }
    for tag, replacement in tag_map.items():
        text = text.replace(tag, replacement)

    html = markdown2.markdown(text, extras=["cuddled-lists", "break-on-newline"])
    
    return html.strip()

# --- PDF Processing ---
def get_pdf_content(pdf_path: str, pdf_cache_dir: Path) -> Tuple[str, List[str]]:
    pdf_cache_dir.mkdir(exist_ok=True)
    ocr_log = []
    
    tesseract_configured = configure_tesseract()
    tesseract_warning_issued = False

    def is_text_meaningful(text: str, min_chars: int = 30) -> bool:
        alphanumeric_chars = re.sub(r'[^a-zA-Z0-9]', '', text)
        return len(alphanumeric_chars) >= min_chars

    try:
        pdf_hash = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    except IOError as e:
        return f"Error reading PDF file {Path(pdf_path).name}: {e}", [f"IOError: {e}"]

    cache_file = pdf_cache_dir / f"{pdf_hash}_ocr.txt"
    if cache_file.exists():
        cached_text = cache_file.read_text(encoding='utf-8')
        if "--- OCR Log ---" in cached_text:
             ocr_log.append("Using cached text that was generated with OCR on a previous run.")
        return cached_text, ocr_log

    text_content = ""
    page_ocr_logs = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text().strip()

                if not is_text_meaningful(page_text):
                    page_ocr_logs.append(f"   > Page {page_num}: Low text quality. Attempting OCR fallback.")
                    
                    if tesseract_configured:
                        try:
                            pix = page.get_pixmap(dpi=300)
                            img_bytes = pix.tobytes("png")
                            pil_image = Image.open(io.BytesIO(img_bytes))
                            ocr_text = pytesseract.image_to_string(pil_image)
                            if is_text_meaningful(ocr_text):
                                page_text = ocr_text
                                page_ocr_logs.append(f"     - OCR SUCCESS: Extracted {len(ocr_text)} characters.")
                            else:
                                page_ocr_logs.append(f"     - OCR FAILED: Tesseract found no meaningful text on this page.")
                        except Exception as ocr_error:
                            page_ocr_logs.append(f"  [OCR Error] Tesseract failed on this page. Error: {ocr_error}")
                            page_text = ""
                    
                    elif not tesseract_warning_issued:
                        page_ocr_logs.append("  [CRITICAL OCR WARNING] Tesseract is not found in the system PATH or default installation directory.")
                        page_ocr_logs.append("  SOLUTION: Please reinstall Tesseract and ensure you check the 'Add to PATH' option,")
                        page_ocr_logs.append("  or ensure it's installed in 'C:\\Program Files\\Tesseract-OCR\\'.")
                        tesseract_warning_issued = True

                text_content += f"--- Page {page_num} ---\n{page_text}\n\n"

    except fitz.errors.FitzError as e:
        return f"Error processing PDF {Path(pdf_path).name}: {e}", [f"FitzError: {e}"]

    if page_ocr_logs:
        ocr_log.append("\n--- OCR Processing Log ---")
        ocr_log.extend(page_ocr_logs)
        text_content += "\n--- OCR Log ---\n" + "\n".join(page_ocr_logs)

    cache_file.write_text(text_content, encoding='utf-8')
    return text_content, ocr_log


# --- Gemini API Call with Function Calling Support ---
def call_gemini(prompt: str, api_key: str, model_name: str, tools: Optional[List[Dict]] = None, task_id: str = "generic") -> Any:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    full_prompt = f"<!-- Task ID: {task_id} | Timestamp: {time.time()} -->\n{prompt}"
    
    payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
    if tools:
        payload["tools"] = tools
        payload["tool_config"] = {"function_calling_config": {"mode": "ANY"}}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=300)
        
        if response.status_code == 429: 
            return "API_LIMIT_REACHED"
        
        response.raise_for_status()

        response_data = response.json()
        
        if response_data.get('promptFeedback', {}).get('blockReason'):
            return f"API_SAFETY_BLOCK: {response_data['promptFeedback']['blockReason']}"

        if not response_data.get('candidates'):
            return f"Warning: Gemini returned a response with no candidates. Full Response: {response.text}"

        candidate = response_data['candidates'][0]
        content = candidate.get('content', {})
        parts = content.get('parts', [{}])

        function_calls = [part['functionCall'] for part in parts if 'functionCall' in part]
        if function_calls:
            return function_calls

        if 'text' in parts[0]:
            return parts[0]['text']

        return f"Warning: Gemini returned an empty or unexpected response part. Full Response: {response.text}"

    except requests.exceptions.RequestException as e:
        return f"Error calling Gemini API: {e}"
        
    return "Error: An unknown error occurred in the Gemini API call."

def _validate_and_repair_mermaid_code(code: str, api_key: str, flash_model: str) -> str:
    """Uses a cheap, fast AI call to validate and fix Mermaid.js syntax."""
    if not code or "graph" not in code:
        return "graph TD;\\n  A[Invalid Code];" # Return a default error graph

    validation_prompt = f"""
You are a Mermaid.js syntax linter. Your only job is to validate and correct the provided code.
RULES:
1. Check if the following code is valid Mermaid.js syntax.
2. If it is VALID, return the code UNCHANGED.
3. If it is INVALID, fix the syntax errors and return ONLY the corrected code.
4. Do not add any explanation, comments, or surrounding text.

--- CODE ---
{code}
"""
    response = call_gemini(validation_prompt, api_key, model_name=flash_model, task_id="mermaid_validation")
    
    if isinstance(response, str) and "graph" in response:
        cleaned_response = re.sub(r'```mermaid|```', '', response).strip()
        return cleaned_response
    else:
        return code

# --- AnkiConnect ---
def invoke_ankiconnect(action: str, **params: Any) -> Tuple[Any | None, str | None]:
    try:
        response = requests.post(ANKI_CONNECT_URL, json={"action": action, "version": 6, "params": params}, timeout=30)
        response.raise_for_status()
        result = response.json()
        return (result.get('result'), result.get('error'))
    except requests.exceptions.RequestException as e:
        return None, f"Could not connect to AnkiConnect: {e}"

def run_pre_flight_checks(api_keys: Dict[str, str], deck_configs: List[Tuple]) -> Optional[str]:
    version, error = invoke_ankiconnect("version")
    if error or not version: return "CRITICAL ERROR: Could not connect to Anki.\nSOLUTION: Please ensure Anki is running and AnkiConnect is installed."
    if int(version) < 6: return f"CRITICAL ERROR: Your AnkiConnect add-on is outdated (Version {version}). Please update."
    test_response = call_gemini("Hello", api_keys['GEMINI_API_KEY'], model_name="gemini-1.5-flash-latest")
    if "API key not valid" in str(test_response) or "API_KEY_INVALID" in str(test_response):
        return "CRITICAL ERROR: Your Gemini API Key appears to be invalid. Please check your .env file."
    if not deck_configs: return "ERROR: No valid decks configured. Please upload at least one PDF and provide a deck name."
    for _, files in deck_configs:
        for pdf in files:
            if not Path(pdf.name).exists():
                return f"CRITICAL ERROR: The file '{Path(pdf.name).name}' could not be found."
    return None

def setup_anki_deck_and_note_type(deck_name: str, note_type_key: str, cloze_color: str = "#0000FF", mermaid_theme: str = "default") -> Optional[str]:
    config = NOTE_TYPE_CONFIG.get(note_type_key)
    if not config: return f"Internal Error: Note type config for '{note_type_key}' not found."
    model_name = config["modelName"]

    _, error = invoke_ankiconnect("createDeck", deck=deck_name)
    if error and "exists" not in error: return f"Failed to create deck '{deck_name}': {error}"

    model_names, error = invoke_ankiconnect("modelNames")
    if error: return f"AnkiConnect Error: {error}"

    final_css = config.get("css", "")
    if "%s" in final_css:
        final_css = final_css % cloze_color

    final_templates = []
    for t_dict in config.get("templates", []):
        new_t_dict = t_dict.copy()
        new_t_dict["Front"] = t_dict.get("Front", "").replace("MERMAID_THEME_PLACEHOLDER", mermaid_theme)
        new_t_dict["Back"] = t_dict.get("Back", "").replace("MERMAID_THEME_PLACEHOLDER", mermaid_theme)
        final_templates.append(new_t_dict)
    
    if model_name not in model_names:
        params = {
            "modelName": model_name, "inOrderFields": config["fields"], "css": final_css, 
            "isCloze": config.get("isCloze", False), "cardTemplates": final_templates
        }
        _, error = invoke_ankiconnect("createModel", **params)
        if error: return f"Failed to create note type '{model_name}': {error}"
    else:
        style_params = {"name": model_name, "css": final_css}
        _, error = invoke_ankiconnect("updateModelStyling", model=style_params)
        if error: return f"Failed to update styling for '{model_name}': {error}"
        
        templates_for_update = {}
        for t in final_templates:
            templates_for_update[t['Name']] = {'Front': t['Front'], 'Back': t['Back']}
        
        template_params = {"name": model_name, "templates": templates_for_update}
        _, error = invoke_ankiconnect("updateModelTemplates", model=template_params)
        if error: return f"Failed to update templates for '{model_name}': {error}"
        
    return None

def add_note_to_anki(deck_name: str, note_type_key: str, fields_data: Dict[str, str], source_filename: str, custom_tags: List[str]) -> Tuple[int | None, str | None]:
    config = NOTE_TYPE_CONFIG[note_type_key]
    pdf_tag = f"PDF::{Path(source_filename).stem.replace(' ', '_')}"
    note = {
        "deckName": deck_name, "modelName": config["modelName"], "fields": fields_data,
        "options": {"allowDuplicate": False}, "tags": [pdf_tag] + custom_tags
    }
    return invoke_ankiconnect("addNote", note=note)
# --- Main Deck Processor Class ---
class DeckProcessor:
    def __init__(self, deck_name, files, api_keys, logger, progress, card_type, image_sources_config, 
                 color_map, custom_tags, prompts_dict, cache_dirs, clip_model, 
                 content_options, content_strategy, objectives_text, card_gen_limits,
                 cloze_color, mermaid_theme):
        self.deck_name = deck_name
        self.files = files
        self.api_keys = api_keys
        self.logger_func = logger
        self.progress = progress
        self.card_type = card_type
        self.image_sources_config = image_sources_config
        self.color_map = color_map
        self.custom_tags = custom_tags
        # --- THIS IS THE FIX ---
        # The attribute must be named self.prompts_dict to match the rest of the class.
        self.prompts_dict = prompts_dict
        self.pdf_cache_dir, self.ai_cache_dir = cache_dirs
        self.clip_model = clip_model['model'] if clip_model else None
        self.content_options = content_options
        self.content_strategy = content_strategy
        self.objectives_text = objectives_text
        self.card_gen_limits = card_gen_limits
        self.cloze_color = cloze_color
        self.mermaid_theme = mermaid_theme
        
        if "Basic" in self.card_type: self.note_type_key = "basic"
        elif "Atomic Cloze" in self.card_type: self.note_type_key = "cloze"
        elif "Contextual Cloze" in self.card_type: self.note_type_key = "contextual_cloze"
        elif "Mermaid" in self.card_type: self.note_type_key = "mermaid"
            
        self.full_text = ""
        self.pdf_paths = [file.name for file in self.files]
        self.combined_pdf_hash = hashlib.sha256(b''.join(Path(p).read_bytes() for p in self.pdf_paths)).hexdigest()
        
        self.pro_model = None
        self.flash_model = None
        self.rpm_limit_flash = None
        
        self.pdf_images_cache = []
        self.image_finder = None
        self.curated_image_pages = []

    def log(self, message): self.logger_func(message)

    def _initialize_image_finder(self):
        if self.clip_model:
            strategies = [PDFImageSource(), WikimediaSource(), NLMOpenISource(), OpenverseSource(api_key=self.api_keys.get("OPENVERSE_API_KEY"), api_key_name="OPENVERSE_API_KEY"), FlickrSource(api_key=self.api_keys.get("FLICKR_API_KEY"), api_key_name="FLICKR_API_KEY")]
            enabled_strategies = [s for s in strategies if s.name in self.image_sources_config and s.is_enabled]
            self.image_finder = ImageFinder(enabled_strategies)
        else:
            self.log("WARNING: CLIP Model not loaded. Image searching will be disabled.")

    def _auto_extract_objectives(self) -> str:
        """Uses a fast AI call to find and extract learning objectives from the full PDF text."""
        self.log("   > Attempting to auto-extract learning objectives...")
        
        prompt = self.prompts_dict['objective_finder'].format(full_text=self.full_text)
        response = call_gemini(prompt, self.api_keys["GEMINI_API_KEY"], model_name=self.flash_model, task_id="find_objectives")

        if isinstance(response, str) and "NO_OBJECTIVES_FOUND" not in response and len(response) > 10:
            self.log("   > SUCCESS: Auto-extracted objectives.")
            return response.strip()
        else:
            self.log("   > WARNING: Could not auto-extract objectives. Proceeding with all facts.")
            return ""

    def run(self):
        try:
            manager = GeminiModelManager(self.api_keys["GEMINI_API_KEY"])
            optimal_models = manager.get_optimal_models()
            if not optimal_models:
                self.log("CRITICAL ERROR: Could not determine optimal Gemini models. Halting process.")
                return
            self.pro_model = optimal_models['pro_model_name']
            self.flash_model = optimal_models['flash_model_name']
            self.rpm_limit_flash = optimal_models['flash_model_rpm']

            if not self._setup_anki(): return
            if not self._process_pdfs(): return

            if self.content_strategy == "Focus on Auto-Extracted Objectives":
                self.objectives_text = self._auto_extract_objectives()
            
            curated_text = self._curate_text_pages()
            self.curated_image_pages = self._curate_image_pages()
            
            self._initialize_image_finder()
            if self.image_finder and self.pdf_paths:
                self.log("\n--- Pre-caching images from AI-curated pages ---")
                pdf_image_extractor = PDFImageSource()
                extracted = pdf_image_extractor._extract_images_and_context(self.pdf_paths[0], pages_to_process=self.curated_image_pages)
                self.pdf_images_cache.extend(extracted)
                self.log(f"Found and cached {len(self.pdf_images_cache)} images from {len(self.curated_image_pages)} curated pages.")

            self.log("\n--- AI Pass 1: Extracting all atomic facts from curated text... ---")
            structured_facts = self._extract_facts(curated_text)
            
            if not structured_facts: return

            filtered_facts = [fact for fact in structured_facts if not is_superfluous(fact['fact'])]
            self.log(f"\n--- Pre-filtering complete. {len(filtered_facts)} high-quality facts remain. ---")
            if not filtered_facts: return

            final_cards_data = self._generate_cards(filtered_facts)
            if final_cards_data is None: return
            final_cards = self._parse_and_deduplicate(final_cards_data)
            if final_cards is None: return
            self._add_notes_to_anki(final_cards)

        except Exception as e:
            self.log(f"\n--- A CRITICAL ERROR OCCURRED IN DECK '{self.deck_name}' ---\n{e}\nTraceback: {traceback.format_exc()}")

    def _curate_text_pages(self) -> str:
        self.log("\n--- AI Pass 0: Curating text pages ---")
        case_instruction = "You MUST classify pages containing case studies as 'Core'." if "Include Case Studies" in self.content_options else "You MUST classify pages containing introductory case studies as 'Superfluous'."
        table_instruction = "You MUST classify pages containing summary tables or charts as 'Core'." if "Include Summary Tables" in self.content_options else "You MUST classify pages containing large, dense tables of data as 'Superfluous'."
        
        # Assemble the full prompt from the template and user instructions
        full_prompt = self.prompts_dict['curator'].format(case_study_instruction=case_instruction, table_instruction=table_instruction)
        
        response = call_gemini(full_prompt + "\n\n--- TEXT ---\n" + self.full_text, self.api_keys["GEMINI_API_KEY"], model_name=self.flash_model, task_id="curate_text")
        
        if "API_" in str(response) or not response:
            self.log("   > WARNING: AI text curation failed. Proceeding with all pages.")
            return self.full_text
        try:
            page_numbers_str = re.findall(r'\d+', response)
            if not page_numbers_str: raise ValueError("No numbers found in AI response.")
            pages_to_keep = {int(p) for p in page_numbers_str}
            page_pattern = re.compile(r'(--- Page (\d+) ---\n.*?(?=(--- Page \d+ ---)|\Z))', re.DOTALL)
            curated_text = "".join(match.group(1) for match in page_pattern.finditer(self.full_text) if int(match.group(2)) in pages_to_keep)
            
            if curated_text:
                self.log(f"   > Text Curation successful. Keeping {len(pages_to_keep)} pages for fact extraction.")
                return curated_text
            else:
                self.log("   > WARNING: AI text curation resulted in no pages. Using all pages.")
                return self.full_text
        except (ValueError, TypeError) as e:
            self.log(f"   > WARNING: Could not parse curator response. Using all pages. (Error: {e})")
            return self.full_text

    def _curate_image_pages(self) -> List[int]:
        self.log("\n--- AI Pass 0.5: Curating visually significant pages for image search ---")
        
        all_page_nums = [int(p) for p in re.findall(r'--- Page (\d+) ---', self.full_text)]
        if not all_page_nums:
            self.log("   > WARNING: Could not determine page count. Image curation will be less reliable.")
            return []
        total_pages = max(all_page_nums)

        prompt = self.prompts_dict['image_curator'].format(total_pages=total_pages)
        response = call_gemini(prompt + "\n\n--- TEXT ---\n" + self.full_text, self.api_keys["GEMINI_API_KEY"], model_name=self.flash_model, task_id="curate_images")

        if "API_" in str(response) or not response:
            self.log("   > WARNING: AI image curation failed. Image search will consider all pages.")
            return all_page_nums
        try:
            page_numbers_str = re.findall(r'\d+', response)
            if not page_numbers_str: raise ValueError("No numbers found.")

            pages_to_keep = [p for p in {int(p) for p in page_numbers_str} if 1 <= p <= total_pages]
            
            invalid_pages_found = len(page_numbers_str) - len(pages_to_keep)
            if invalid_pages_found > 0:
                self.log(f"   > Image Curator sanity check: Discarded {invalid_pages_found} invalid page numbers.")

            self.log(f"   > Image Curation successful. Prioritizing {len(pages_to_keep)} pages for image search.")
            return sorted(pages_to_keep)
            
        except (ValueError, TypeError):
            self.log("   > WARNING: Could not parse image curator response. Image search will consider all pages.")
            return all_page_nums

    def _setup_anki(self):
        self.log(f"Card Type Selected: {self.card_type}")
        if self.custom_tags: self.log(f"Custom Tags: {', '.join(self.custom_tags)}")
        
        primary_error = setup_anki_deck_and_note_type(self.deck_name, self.note_type_key, self.cloze_color, self.mermaid_theme)
        if primary_error:
            self.log(f"DECK SETUP ERROR: {primary_error}")
            return False

        if self.note_type_key != "basic":
            fallback_error = setup_anki_deck_and_note_type(self.deck_name, "basic", self.cloze_color, self.mermaid_theme)
            if fallback_error:
                self.log(f"WARNING: Could not create fallback 'Basic' note type. Reason: {fallback_error}")

        self.log(f"Anki setup for deck '{self.deck_name}' is correct.")
        return True

    def _process_pdfs(self):
        self.log("\n--- Processing PDF Files ---")
        for pdf_path in self.progress.tqdm(self.pdf_paths, desc="Processing PDFs"):
            text, ocr_log = get_pdf_content(pdf_path, self.pdf_cache_dir)
            if "Error:" in text:
                self.log(text)
                return False
            self.full_text += f"\n\n--- Content from {Path(pdf_path).name} ---\n{text}"
            if ocr_log:
                for line in ocr_log:
                    self.log(line)
        self.log("All PDF files processed and cached.")
        return True

    def _extract_facts(self, text_to_process: str) -> Optional[List[Dict[str, Any]]]:
        self.log("\n--- AI Pass 1: Extracting atomic facts from curated text... ---")
        extractor_model = self.flash_model
        
        page_pattern = re.compile(r'--- Page (\d+) ---\n(.*?)(?=--- Page \d+ ---|\Z)', re.DOTALL)
        all_pages = [(int(num), content) for num, content in page_pattern.findall(text_to_process) if len(content.strip()) > 50]

        if not all_pages:
            self.log("ERROR: No pages with sufficient text content found after curation and cleaning."); return None
        
        batched_tasks = []
        for i in range(0, len(all_pages), BATCH_SIZE):
            batch = all_pages[i:i + BATCH_SIZE]
            combined_text = "\n\n".join([f"--- Page {p[0]} ---\n{p[1]}" for p in batch])
            page_numbers_in_batch = [p[0] for p in batch]
            batched_tasks.append((page_numbers_in_batch, combined_text))

        request_timestamps = deque()
        lock = threading.Lock()
        all_extracted_facts = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS_EXTRACTION) as executor:
            def extract_facts_from_batch(task_data):
                page_nums, batch_content = task_data
                page_range_str = str(page_nums[0]) if len(page_nums) == 1 else f"{page_nums[0]}-{page_nums[-1]}"
                with lock:
                    while True:
                        current_time = time.time()
                        while request_timestamps and request_timestamps[0] <= current_time - 60:
                            request_timestamps.popleft()
                        if len(request_timestamps) < self.rpm_limit_flash:
                            request_timestamps.append(current_time)
                            break
                        wait_time = request_timestamps[0] - (current_time - 60) + 0.1
                        time.sleep(wait_time)
                prompt = self.prompts_dict['extractor'] + "\n\n--- TEXT ---\n" + batch_content
                response = call_gemini(prompt, self.api_keys["GEMINI_API_KEY"], model_name=extractor_model, task_id=f"extract_batch_{page_range_str}")
                was_successful = False
                if isinstance(response, str):
                    match = re.search(r'\[.*\]', response, re.DOTALL)
                    if match:
                        json_string = match.group(0)
                        try:
                            facts_in_batch = json.loads(json_string)
                            if isinstance(facts_in_batch, list):
                                with lock:
                                    all_extracted_facts.extend(facts_in_batch)
                                was_successful = True
                        except json.JSONDecodeError:
                            self.log(f"\nWARNING (Batch {page_range_str}): AI returned a string that looked like JSON but failed to parse.")
                if not was_successful:
                    self.log(f"\nWARNING (Batch {page_range_str}): AI call failed or returned a response without valid JSON. Reason: {response}")
            list(self.progress.tqdm(executor.map(extract_facts_from_batch, batched_tasks), total=len(batched_tasks), desc="Extracting Facts in Batches"))
        
        if not all_extracted_facts:
            self.log("ERROR: AI Pass 1 failed to extract any facts from any batch."); return None
        validated_facts = []
        for i, item in enumerate(all_extracted_facts):
            if isinstance(item, dict) and 'fact' in item and 'page_number' in item:
                try: validated_facts.append({"fact": str(item['fact']), "page_number": int(item['page_number'])})
                except (ValueError, TypeError): self.log(f"WARNING: Discarding malformed fact object at index {i}: {item}")
        self.log(f"Pass 1 complete. Found and validated {len(validated_facts)} facts across all batches.")
        return validated_facts

    def _generate_cards(self, facts_json_input: List[Dict[str, Any]]) -> Any:
        sanitized_card_type = re.sub(r'[^a-zA-Z0-9_\-]', '_', self.card_type)
        card_type_slug = sanitized_card_type.lower()
        facts_hash = hashlib.sha224(json.dumps(facts_json_input, sort_keys=True).encode()).hexdigest()[:16]
        ai_cache_key = f"{self.combined_pdf_hash}_{card_type_slug}_{facts_hash}_v12_dynamic_mandate.json"
        ai_cache_file = self.ai_cache_dir / ai_cache_key
        
        if ai_cache_file.exists():
            self.log("\n--- Found Cached AI Response! Skipping card generation. ---")
            return json.loads(ai_cache_file.read_text(encoding='utf-8'))

        self.log("\n--- No cached response found. Generating new cards with AI... ---")
        
        objectives_section, fact_mandate_rule = "", ""
        if "Objectives" in self.content_strategy and self.objectives_text:
            self.log("   > STRATEGY: Objectives-Focused. AI will filter facts.")
            objectives_section = f"CRITICAL CONTEXT: You must build cards that are DIRECTLY relevant to the following Learning Objectives. You should IGNORE any facts from the input that are not related to these goals.\n--- OBJECTIVES ---\n{self.objectives_text}\n\n"
            fact_mandate_rule = "1.  **Objective-Driven Creation:** You must only create cards for facts that are relevant to the provided learning objectives."
        else:
            self.log("   > STRATEGY: Extract All Facts. AI will process every fact.")
            fact_mandate_rule = "1.  **Absolute Mandate - No Fact Left Behind:** You **MUST** incorporate the information from **EVERY SINGLE ATOMIC FACT** provided. No facts may be discarded."

        all_generated_function_calls = []
        MAX_FACTS_PER_BATCH = 300

        for i in range(0, len(facts_json_input), MAX_FACTS_PER_BATCH):
            batch_facts = facts_json_input[i:i + MAX_FACTS_PER_BATCH]
            batch_num = (i // MAX_FACTS_PER_BATCH) + 1
            total_batches = (len(facts_json_input) + MAX_FACTS_PER_BATCH - 1) // MAX_FACTS_PER_BATCH
            self.log(f"   > Generating cards for fact batch {batch_num} of {total_batches}...")

            prompt, tools = None, None

            atomic_facts_with_pages = ""
            if "Cloze" in self.card_type:
                pages_to_facts = {}
                for item in batch_facts:
                    pg = item['page_number']
                    if pg not in pages_to_facts: pages_to_facts[pg] = []
                    pages_to_facts[pg].append(item['fact'])
                for page_num in sorted(pages_to_facts.keys()):
                    atomic_facts_with_pages += f"--- Page(s) {page_num} ---\n" + "\n".join(pages_to_facts[page_num]) + "\n"
            
            # --- THIS IS THE FIX ---
            # Replace all instances of `self.prompts_and_instructions` with `self.prompts_dict`
            if "Basic" in self.card_type:
                prompt = self.prompts_dict['builder_template'].format(
                    user_instructions=self.prompts_dict['builder_instructions'],
                    fact_mandate_placeholder=fact_mandate_rule,
                    objectives_section=objectives_section,
                    atomic_facts_json=json.dumps(batch_facts, indent=2),
                    min_chars=self.card_gen_limits['min'], max_chars=self.card_gen_limits['max'],
                    target_chars=self.card_gen_limits['target']
                )
                tools = [{"function_declarations": [NOTE_TYPE_CONFIG['basic']['function_tool']]}]
            elif "Atomic Cloze" in self.card_type:
                prompt = self.prompts_dict['cloze_template'].format(
                    user_instructions=self.prompts_dict['cloze_instructions'],
                    fact_mandate_placeholder=fact_mandate_rule,
                    objectives_section=objectives_section,
                    atomic_facts_with_pages=atomic_facts_with_pages
                )
                tools = [{"function_declarations": [NOTE_TYPE_CONFIG['cloze']['function_tool']]}]
            elif "Contextual Cloze" in self.card_type:
                prompt = self.prompts_dict['contextual_cloze_template'].format(
                    user_instructions=self.prompts_dict['contextual_cloze_instructions'],
                    fact_mandate_placeholder=fact_mandate_rule,
                    objectives_section=objectives_section,
                    atomic_facts_with_pages=atomic_facts_with_pages
                )
                tools = [{"function_declarations": [NOTE_TYPE_CONFIG['contextual_cloze']['function_tool']]}]
            elif "Mermaid" in self.card_type:
                prompt = self.prompts_dict['mermaid_template'].format(
                    user_instructions=self.prompts_dict['mermaid_instructions'],
                    fact_mandate_placeholder=fact_mandate_rule,
                    objectives_section=objectives_section,
                    atomic_facts_json=json.dumps(batch_facts, indent=2)
                )
                tools = [{"function_declarations": [NOTE_TYPE_CONFIG['mermaid']['function_tool']]}]
                
            if not prompt or not tools:
                self.log(f"ERROR: Could not determine prompt for card type '{self.card_type}' for batch {batch_num}. Skipping.")
                continue

            batch_result = call_gemini(prompt, self.api_keys["GEMINI_API_KEY"], model_name=self.pro_model, tools=tools, task_id=f"generate_cards_batch_{batch_num}")

            if isinstance(batch_result, list):
                all_generated_function_calls.extend(batch_result)
            elif isinstance(batch_result, str):
                self.log(f"\nERROR in batch {batch_num}: AI generation failed. Reason: {batch_result}")

        if not all_generated_function_calls:
            self.log("CRITICAL ERROR: AI failed to generate any function calls across all batches.")
            return None

        self.ai_cache_dir.mkdir(exist_ok=True)
        ai_cache_file.write_text(json.dumps(all_generated_function_calls, indent=2), encoding='utf-8')
        self.log("Saved new AI response from all batches to cache.")

        return all_generated_function_calls

    def _parse_and_deduplicate(self, card_function_calls: List[Dict]) -> Optional[List[Dict]]:
        if not isinstance(card_function_calls, list):
            self.log("CRITICAL ERROR: AI response was not a list of function call parts. AI may have failed to follow instructions."); return None

        self.log("\n--- Parsing AI Function Call Response ---")
        final_cards = []
        
        for call in card_function_calls:
            args = call.get("args", {})
            card = None

            if call.get("name") == "create_anki_card":
                page_nums_raw = args.get("Page_numbers", [1])
                page_nums = sorted(list(set(page_nums_raw))) if isinstance(page_nums_raw, list) else [1]
                card = {"type": "basic", "front": args.get("Front"), "back_text": args.get("Back"), "image_search_query": args.get("Search_Query"), "simple_search_query": args.get("Simple_Search_Query"), "page_numbers": page_nums}
            
            elif call.get("name") == "create_cloze_components":
                page_str = args.get("Source_Page", "1")
                page_num = int(re.search(r'\d+', page_str).group()) if re.search(r'\d+', page_str) else 1
                card = {
                    "type": "cloze_components", "original_question": args.get("Context_Question"), "full_sentence": args.get("Full_Sentence"),
                    "cloze_keywords": args.get("Cloze_Keywords", []), "image_search_query": args.get("Search_Query"),
                    "simple_search_query": args.get("Simple_Search_Query"), "page_numbers": [page_num]
                }

            elif call.get("name") == "create_mermaid_card":
                page_nums_raw = args.get("Page_numbers", [1])
                page_nums = sorted(list(set(page_nums_raw))) if isinstance(page_nums_raw, list) else [1]
                card = {
                    "type": "mermaid", "front": args.get("Front"), "back_text": args.get("Back"),
                    "mermaid_front_code": args.get("Mermaid_Front_Code"), "mermaid_back_code": args.get("Mermaid_Back_Code"),
                    "image_search_query": args.get("Search_Query"), "simple_search_query": args.get("Simple_Search_Query"),
                    "page_numbers": page_nums
                }

            if card: final_cards.append(card)

        if not final_cards:
            self.log("CRITICAL ERROR: No valid card creation calls were parsed from the AI's response."); return None

        self.log(f"Successfully parsed {len(final_cards)} cards.")

        unique_cards_dict = {}
        for card in final_cards:
            key = card.get('front') or card.get('full_sentence') or card.get('mermaid_front_code')
            if key and key not in unique_cards_dict:
                unique_cards_dict[key] = card
        unique_cards = list(unique_cards_dict.values())

        if len(unique_cards) < len(final_cards):
            self.log(f"   > INFO: Found and removed {len(final_cards) - len(unique_cards)} duplicate card(s).")
        return unique_cards

    def _add_notes_to_anki(self, final_cards: List[Dict]):
        self.log(f"\n--- Adding Cards to Deck: '{self.deck_name}' ---")
        cards_added, cards_skipped, cards_failed = 0, 0, 0
        main_pdf_path = self.pdf_paths[0] if self.pdf_paths else "Unknown.pdf"

        for card_data in self.progress.tqdm(final_cards, desc="Adding Cards to Anki"):
            try:
                image_html = None
                full_source_page_numbers = card_data.get("page_numbers", [])

                if self.image_finder and card_data.get("image_search_query"):
                    search_queries = [q for q in [card_data.get("image_search_query"), card_data.get("simple_search_query")] if q]
                    focused_pages = [p for p in full_source_page_numbers if p in self.curated_image_pages]
                    min_page = min(full_source_page_numbers) if full_source_page_numbers else 0
                    max_page = max(full_source_page_numbers) if full_source_page_numbers else 0
                    expanded_range = set(range(max(1, min_page - 1), max_page + 2))
                    expanded_pages = [p for p in expanded_range if p in self.curated_image_pages]
                    image_html = self.image_finder.find_best_image(query_texts=search_queries, clip_model=self.clip_model, pdf_path=main_pdf_path, pdf_images_cache=self.pdf_images_cache, focused_search_pages=focused_pages, expanded_search_pages=expanded_pages)

                page_str = f"Pgs {', '.join(map(str, sorted(list(set(full_source_page_numbers)))))}"
                source_text = f"{Path(main_pdf_path).stem} - {page_str}"
                
                fields, final_note_type_key = {}, None

                if card_data['type'] == 'basic':
                    final_note_type_key = 'basic'
                    fields = {"Front": card_data["front"], "Back": build_html_from_tags(card_data["back_text"], self.color_map), "Source": source_text, "Image": image_html or ""}
                
                elif card_data['type'] == 'cloze_components':
                    sentence = card_data.get("full_sentence", "")
                    keywords = card_data.get("cloze_keywords", [])
                    
                    if not sentence or not keywords:
                        self.log(f"   > WARNING: AI failed to provide cloze components. Converting to Basic card.")
                        final_note_type_key = "basic"
                        fields = {"Front": card_data["original_question"], "Back": sentence or " ", "Source": source_text, "Image": image_html or ""}
                    else:
                        sentence_html = sentence
                        for i, keyword in enumerate(keywords):
                            match = re.search(re.escape(keyword), sentence_html, re.IGNORECASE)
                            if match:
                                actual_keyword = match.group(0)
                                cloze_syntax = f"{{{{c{i+1}::{actual_keyword}}}}}"
                                sentence_html = sentence_html.replace(actual_keyword, cloze_syntax, 1)
                        
                        if "{{c" not in sentence_html:
                            self.log(f"   > WARNING: Python failed to assemble a valid cloze (keyword not found in sentence?). Converting to Basic card.")
                            final_note_type_key = "basic"
                            fields = {"Front": card_data["original_question"], "Back": sentence or " ", "Source": source_text, "Image": image_html or ""}
                        else:
                            final_note_type_key = self.note_type_key
                            fields = {"Text": sentence_html, "Extra": card_data["original_question"], "Source": source_text, "Image": image_html or ""}

                elif card_data['type'] == 'mermaid':
                    final_note_type_key = 'mermaid'
                    front_code = _validate_and_repair_mermaid_code(card_data.get("mermaid_front_code", ""), self.api_keys["GEMINI_API_KEY"], self.flash_model)
                    back_code = _validate_and_repair_mermaid_code(card_data.get("mermaid_back_code", ""), self.api_keys["GEMINI_API_KEY"], self.flash_model)
                    fields = {"Front": card_data.get("front"), "Back": card_data.get("back_text"), "MermaidFront": front_code, "MermaidBack": back_code, "Source": source_text, "Image": image_html or ""}

                if not fields:
                    cards_skipped += 1
                    continue

                _, error = add_note_to_anki(self.deck_name, final_note_type_key, fields, main_pdf_path, self.custom_tags)
                if error:
                    if "duplicate" in error: cards_skipped += 1
                    else:
                        self.log(f"   > FAILED to add note. Reason: {error}")
                        cards_failed += 1
                else:
                    cards_added += 1
            except Exception as e:
                cards_failed += 1
                self.log(f"ERROR processing card data: '{str(card_data)[:500]}...' | Exception: {e}\nTraceback: {traceback.format_exc()}")

        self.log(f"\n--- Final Tally ---\nCards Added: {cards_added}\nCards Skipped/Failed: {cards_skipped + cards_failed}")
# --- Main Generator Function ---


from utils import get_api_keys_from_env, manage_log_files, configure_tesseract, slugify, is_superfluous, save_settings

def generate_all_decks(max_decks: int, *args):
    master_files, generate_button, log_output, clip_model, *remaining_args = args
    
    log_history = ""
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    def logger(message):
        nonlocal log_history
        timestamp_msg = f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n"
        log_history += timestamp_msg
        print(timestamp_msg.strip())
        return log_history

    final_ui_state = [gr.update(), gr.update(interactive=True), gr.update(value="Generate All Decks")]
    log_file_path = None
    
    settings_keys = [
            # Core Settings
            "card_type",
            "image_sources",
            # Basic Card Settings
            "pos_color",
            "neg_color",
            "ex_color",
            "tip_color",
            "min_chars",
            "max_chars",
            "char_target",
            # Cloze Settings
            "cloze_color",
            # Mermaid Settings
            "mermaid_theme",
            # Other Settings
            "custom_tags",
            "content_strategy",
            "objectives_text_manual",
            # User Instructions
            "builder_user_instructions",
            "atomic_cloze_user_instructions",
            "contextual_cloze_user_instructions",
            "mermaid_user_instructions",
            # Hidden Templates
            "builder_prompt_template",
            "atomic_cloze_prompt_template",
            "contextual_cloze_prompt_template",
            "mermaid_prompt_template",
            # Editable Prompts
            "curator_prompt",
            "extractor_prompt",
            "objective_finder_prompt",
            "image_curator_prompt"
        ]
    
    deck_inputs_flat = remaining_args[:max_decks * 2]
    settings_and_prompts_values = remaining_args[max_decks * 2:]
    current_settings = dict(zip(settings_keys, settings_and_prompts_values))
    
    try:
        yield logger("Starting Anki Deck Generator..."), gr.update(interactive=False), gr.update(value="Processing...")
        from app import LOG_DIR, MAX_LOG_FILES, PDF_CACHE_DIR, AI_CACHE_DIR
        manage_log_files(LOG_DIR, MAX_LOG_FILES)
        log_file_path = LOG_DIR / f"session_log_{session_timestamp}.txt"
        cache_dirs = (PDF_CACHE_DIR, AI_CACHE_DIR)
        
        card_type = current_settings["card_type"]
        image_sources = current_settings["image_sources"]
        pos_color, neg_color, ex_color, tip_color = current_settings["pos_color"], current_settings["neg_color"], current_settings["ex_color"], current_settings["tip_color"]
        cloze_color, mermaid_theme = current_settings["cloze_color"], current_settings["mermaid_theme"]
        custom_tags_str = current_settings["custom_tags"]
        content_strategy = current_settings["content_strategy"]
        objectives_text = current_settings["objectives_text_manual"]
        min_chars, max_chars, char_target = current_settings["min_chars"], current_settings["max_chars"], current_settings["char_target"]

        color_map = {"positive_key_term": pos_color, "negative_key_term": neg_color, "example": ex_color, "mnemonic_tip": tip_color}
        card_gen_limits = {'min': min_chars, 'max': max_chars, 'target': char_target}
        
        # This dictionary now contains the UNFORMATTED templates and user instructions separately.
        prompts_dict = {
            'builder_template': current_settings["builder_prompt_template"],
            'builder_instructions': current_settings["builder_user_instructions"],
            'cloze_template': current_settings["atomic_cloze_prompt_template"],
            'cloze_instructions': current_settings["atomic_cloze_user_instructions"],
            'contextual_cloze_template': current_settings["contextual_cloze_prompt_template"],
            'contextual_cloze_instructions': current_settings["contextual_cloze_user_instructions"],
            'mermaid_template': current_settings["mermaid_prompt_template"],
            'mermaid_instructions': current_settings["mermaid_user_instructions"],
            'curator': current_settings["curator_prompt"], 'image_curator': current_settings["image_curator_prompt"], 
            'extractor': current_settings["extractor_prompt"], 'objective_finder': current_settings["objective_finder_prompt"],
        }
        
        content_options = [] 
        custom_tags = [tag.strip() for tag in custom_tags_str.split(',') if tag.strip()]
        api_keys = get_api_keys_from_env()
        
        deck_configs = []
        for i in range(0, len(deck_inputs_flat), 2):
            deck_title, files = deck_inputs_flat[i], deck_inputs_flat[i+1]
            if deck_title and files: deck_configs.append((deck_title, files))

        if pre_flight_error := run_pre_flight_checks(api_keys, deck_configs):
            yield logger(pre_flight_error), *final_ui_state[1:]
            return
            
        yield logger("Pre-flight checks passed."), gr.update(), gr.update()
        
        for i, (deck_name, files) in enumerate(deck_configs, 1):
            progress = gr.Progress(track_tqdm=True)
            yield logger(f"\n--- Starting Deck {i} of {len(deck_configs)}: '{deck_name}' ---"), gr.update(), gr.update()
            
            processor = DeckProcessor(
                deck_name=deck_name, files=files, api_keys=api_keys, logger=logger, progress=progress, card_type=card_type, 
                image_sources_config=image_sources, color_map=color_map, custom_tags=custom_tags, 
                # --- FIX #2: USE THE CORRECT PARAMETER NAME ---
                prompts_dict=prompts_dict, 
                cache_dirs=cache_dirs, clip_model=clip_model, content_options=content_options, content_strategy=content_strategy,
                objectives_text=objectives_text, card_gen_limits=card_gen_limits, cloze_color=cloze_color, mermaid_theme=mermaid_theme
            )
            processor.run()
            yield logger(f"\n--- Finished Deck {i}: '{deck_name}' ---\n"), gr.update(), gr.update()
            if i < len(deck_configs):
                yield logger(f"--- Cooling down for {INTER_DECK_COOLDOWN_SECONDS} seconds before next deck... ---"), gr.update(), gr.update()
                time.sleep(INTER_DECK_COOLDOWN_SECONDS)
                
        logger("--- All Decks Processed! ---")
        
        keys_to_filter = ["objectives_text_manual", "builder_prompt_template", "atomic_cloze_prompt_template", "contextual_cloze_prompt_template", "mermaid_prompt_template"]
        settings_to_save = {k: v for k, v in current_settings.items() if k not in keys_to_filter}
        save_settings(settings_to_save)

    except Exception as e:
        logger(f"\n--- A CRITICAL UNHANDLED ERROR OCCURRED ---\n{e}\nTraceback: {traceback.format_exc()}")
    finally:
        if log_file_path and log_history:
            log_file_path.write_text(log_history, encoding="utf-8")
            logger(f"Session log saved to: {log_file_path}")
        final_ui_state[0] = log_history
        yield tuple(final_ui_state)