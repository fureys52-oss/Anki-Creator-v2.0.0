# ui.py

from pathlib import Path
from typing import List, Any, Tuple, Callable
import functools
import gradio as gr
from prompts import (
    EXTRACTOR_PROMPT, BUILDER_PROMPT, CLOZE_BUILDER_PROMPT, 
    CONTEXTUAL_CLOZE_BUILDER_PROMPT, CURATOR_PROMPT, IMAGE_CURATOR_PROMPT,
    MERMAID_BUILDER_PROMPT, OBJECTIVE_FINDER_PROMPT
)
from utils import clear_cache, update_decks_from_files, load_settings, save_settings
from processing import generate_all_decks, HTML_COLOR_MAP

def build_ui(version: str, max_decks: int, cache_dirs: Tuple[Path, Path], log_dir: Path, max_log_files: int, update_checker_func: Callable, load_clip_model_func: Callable) -> gr.Blocks:
    
    loaded_settings = load_settings()
    
    IMAGE_SOURCES = [
        "PDF (AI Validated)", "Wikimedia", "NLM Open-i",
        "Openverse", "Flickr", "PDF Page as Image (Fallback)"
    ]
    
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="Anki Deck Generator") as app:
        clip_model_state = gr.State(None)
        
        update_notifier = gr.Markdown(visible=False) 

        gr.Markdown(f"# Anki Flashcard Generator\n*(v{version})*")
        
        clip_model_status = gr.Markdown("⏳ Loading AI model for image analysis in the background...", elem_classes="model-status")

        with gr.Row():
            generate_button = gr.Button("Generate All Decks", variant="primary", scale=2)
            cancel_button = gr.Button("Cancel")
            new_batch_button = gr.Button("Start New Batch")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("1. Decks & Files"):
                        # ... Decks & Files tab content remains the same ...
                        with gr.Group():
                            master_files = gr.File(label="Upload PDFs to assign to decks below", file_count="multiple", file_types=[".pdf"])
                        
                        content_strategy_choices = ["Extract All Facts", "Focus on Provided Objectives", "Focus on Auto-Extracted Objectives"]
                        loaded_content_strategy = loaded_settings.get("content_strategy", "Extract All Facts")
                        
                        # Validate that the loaded value is one of the valid choices.
                        # If not, fall back to the default.
                        if loaded_content_strategy not in content_strategy_choices:
                            print(f"Warning: Invalid value '{loaded_content_strategy}' found for 'content_strategy' in settings.json. Resetting to default.")
                            loaded_content_strategy = "Extract All Facts"

                        content_strategy = gr.Radio(
                            content_strategy_choices, 
                            label="Content Extraction Strategy", 
                            value=loaded_content_strategy
                        )
                        # --- END FIX ---

                        objectives_textbox = gr.Textbox(
                            lines=10, label="Paste Learning Objectives Here", 
                            visible=(loaded_settings.get("content_strategy") == "Focus on Provided Objectives"),
                            placeholder="e.g., '1. Describe the pathophysiology of...\n2. List the three main treatments for...'"
                        )
                        
                        deck_ui_components, deck_input_components = [], []
                        for i in range(max_decks):
                            with gr.Accordion(f"Deck {i+1}", visible=(i==0), open=True) as acc:
                                deck_title = gr.Textbox(label="Deck Title")
                                files = gr.File(visible=False, file_count="multiple")
                            deck_ui_components.extend([acc, deck_title, files])
                            deck_input_components.extend([deck_title, files])
                    
                    with gr.TabItem("2. Prompts (Advanced)"):
                        # ... Prompts tab content remains the same ...
                        gr.Markdown("⚠️ **For builder prompts, only add behavioral instructions. Do not mention output format. For other prompts, editing can break the application.**")
                        
                        with gr.Accordion("Basic Card Builder", open=True):
                            with gr.Group():
                                gr.Markdown("Add instructions to guide the AI's question and answer style. The core rules and output format are protected.")
                                with gr.Row():
                                    builder_user_instructions = gr.Textbox(label="Custom Instructions", placeholder="e.g., 'Prioritize creating clinical vignettes.'", lines=3, value=loaded_settings.get("builder_user_instructions", ""), scale=10)
                                    reset_builder_instructions_btn = gr.Button("Reset", scale=1)
                                builder_prompt_template = gr.Textbox(value=BUILDER_PROMPT, visible=False)

                        with gr.Accordion("Atomic Cloze Builder", open=False):
                             with gr.Group():
                                gr.Markdown("Add instructions to guide the AI's keyword selection. The core rules and output format are protected.")
                                with gr.Row():
                                    atomic_cloze_user_instructions = gr.Textbox(label="Custom Instructions", placeholder="e.g., 'Prioritize cloze-deleting drug names.'", lines=3, value=loaded_settings.get("atomic_cloze_user_instructions", ""), scale=10)
                                    reset_atomic_cloze_instructions_btn = gr.Button("Reset", scale=1)
                                atomic_cloze_prompt_template = gr.Textbox(value=CLOZE_BUILDER_PROMPT, visible=False)

                        with gr.Accordion("Contextual Cloze Builder", open=False):
                            with gr.Group():
                                gr.Markdown("Add instructions to guide how the AI groups facts and synthesizes sentences. The core rules and output format are protected.")
                                with gr.Row():
                                    contextual_cloze_user_instructions = gr.Textbox(label="Custom Instructions", placeholder="e.g., 'Prefer to group facts that describe a sequence.'", lines=3, value=loaded_settings.get("contextual_cloze_user_instructions", ""), scale=10)
                                    reset_contextual_cloze_instructions_btn = gr.Button("Reset", scale=1)
                                contextual_cloze_prompt_template = gr.Textbox(value=CONTEXTUAL_CLOZE_BUILDER_PROMPT, visible=False)
                        
                        with gr.Accordion("Interactive Mermaid Diagram Builder", open=False):
                            with gr.Group():
                                gr.Markdown("Add instructions to guide how the AI designs the diagrams. The core rules and output format are protected.")
                                with gr.Row():
                                    mermaid_user_instructions = gr.Textbox(label="Custom Instructions", placeholder="e.g., 'Always use a top-down (TD) graph direction.'", lines=3, value=loaded_settings.get("mermaid_user_instructions", ""), scale=10)
                                    reset_mermaid_instructions_btn = gr.Button("Reset", scale=1)
                                mermaid_prompt_template = gr.Textbox(value=MERMAID_BUILDER_PROMPT, visible=False)

                        with gr.Accordion("Protected Prompts (Edit with Extreme Caution)", open=False):
                            with gr.Group():
                                gr.Markdown("#### Content Curator Prompt")
                                with gr.Row():
                                    curator_prompt_editor = gr.Textbox(show_label=False, value=loaded_settings.get("curator_prompt", CURATOR_PROMPT), lines=10, max_lines=20, scale=10)
                                    reset_curator_btn = gr.Button("Reset", scale=1)
                            with gr.Group():
                                gr.Markdown("#### Fact Extractor Prompt")
                                with gr.Row():
                                    extractor_prompt_editor = gr.Textbox(show_label=False, value=loaded_settings.get("extractor_prompt", EXTRACTOR_PROMPT), lines=10, max_lines=20, scale=10)
                                    reset_extractor_btn = gr.Button("Reset", scale=1)
                            with gr.Group():
                                gr.Markdown("#### Objective Finder Prompt")
                                with gr.Row():
                                    objective_finder_prompt_editor = gr.Textbox(show_label=False, value=loaded_settings.get("objective_finder_prompt", OBJECTIVE_FINDER_PROMPT), lines=10, max_lines=20, scale=10)
                                    reset_objective_finder_btn = gr.Button("Reset", scale=1)
                            with gr.Group():
                                gr.Markdown("#### Image Curator Prompt")
                                with gr.Row():
                                    image_curator_prompt_editor = gr.Textbox(show_label=False, value=loaded_settings.get("image_curator_prompt", IMAGE_CURATOR_PROMPT), lines=10, max_lines=20, scale=10)
                                    reset_image_curator_btn = gr.Button("Reset", scale=1)

                    with gr.TabItem("3. System"):
                         with gr.Accordion("Cache Management", open=False):
                            clear_cache_button = gr.Button("Clear All Caches")
                            cache_status = gr.Textbox(label="Cache Status", interactive=False)
                         with gr.Accordion("Acknowledgements", open=False):
                            gr.Markdown("This project was built with the invaluable help of the open-source community.")
                         with gr.Accordion("API Keys", open=True):
                            gr.Markdown("Keys are saved locally. For security, leave blank to use a `.env` file.")
                            # --- MODIFIED: Replaced multi-line textbox with 5 single-line password inputs ---
                            gemini_api_key_1 = gr.Textbox(label="Gemini API Key 1", type="password", value=loaded_settings.get("gemini_api_key_1", ""))
                            gemini_api_key_2 = gr.Textbox(label="Gemini API Key 2 (Optional)", type="password", value=loaded_settings.get("gemini_api_key_2", ""))
                            gemini_api_key_3 = gr.Textbox(label="Gemini API Key 3 (Optional)", type="password", value=loaded_settings.get("gemini_api_key_3", ""))
                            gemini_api_key_4 = gr.Textbox(label="Gemini API Key 4 (Optional)", type="password", value=loaded_settings.get("gemini_api_key_4", ""))
                            gemini_api_key_5 = gr.Textbox(label="Gemini API Key 5 (Optional)", type="password", value=loaded_settings.get("gemini_api_key_5", ""))
                            
                            openverse_api_key_textbox = gr.Textbox(label="Openverse API Key (Optional)", type="password", value=loaded_settings.get("openverse_api_key", ""))
                            flickr_api_key_textbox = gr.Textbox(label="Flickr API Key (Optional)", type="password", value=loaded_settings.get("flickr_api_key", ""))
            
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Core Settings")
                    
                    # --- FIX: Robustly handle loading the value for the CheckboxGroup ---
                    card_type_value = loaded_settings.get("card_type", ["Conceptual (Basic Cards)"])
                    if not isinstance(card_type_value, list):
                        card_type_value = [str(card_type_value)] # Coerce to a list if it's a string/bool

                    card_type = gr.CheckboxGroup(
                        ["Conceptual (Basic Cards)", "Atomic Cloze (1 fact/card)", "Contextual Cloze (AnkiKing Style)", "Interactive Mermaid Diagram"], 
                        label="Card Type(s) to Generate", 
                        value=card_type_value)
                    
                    # --- The rest of the accordions use str() for visibility check as before ---
                    with gr.Accordion("Basic Card Settings", visible=("Basic" in str(loaded_settings.get("card_type", ["Conceptual (Basic Cards)"])))) as basic_card_settings_accordion:
                        # ... content remains the same ...
                        gr.Markdown("Fine-tune the content of 'Basic' cards.")
                        basic_let_ai_decide_facts = gr.Checkbox(label="Let AI Decide facts per card", value=loaded_settings.get("basic_let_ai_decide_facts", True))
                        with gr.Row():
                            basic_min_facts_input = gr.Number(label="Min Facts / Card", value=loaded_settings.get("basic_min_facts_input", 2), step=1, minimum=1, interactive=(not loaded_settings.get("basic_let_ai_decide_facts", True)), precision=0)
                            basic_max_facts_input = gr.Number(label="Max Facts / Card", value=loaded_settings.get("basic_max_facts_input", 5), step=1, minimum=1, interactive=(not loaded_settings.get("basic_let_ai_decide_facts", True)), precision=0)
                        gr.Markdown("---")
                        with gr.Row():
                            min_chars_input = gr.Number(label="Min Answer Chars", value=loaded_settings.get("min_chars", 50), step=25, minimum=0)
                            max_chars_input = gr.Number(label="Max Answer Chars", value=loaded_settings.get("max_chars", 550), step=25)
                        char_target_slider = gr.Slider(minimum=50, maximum=1000, step=25, label="Ideal Target Length (Characters)", value=loaded_settings.get("char_target", 200))
                        with gr.Row():
                            pos_color_picker = gr.ColorPicker(loaded_settings.get("pos_color", HTML_COLOR_MAP['positive_key_term']), label="<pos> color")
                            neg_color_picker = gr.ColorPicker(loaded_settings.get("neg_color", HTML_COLOR_MAP['negative_key_term']), label="<neg> color")
                        with gr.Row():
                            ex_color_picker = gr.ColorPicker(loaded_settings.get("ex_color", HTML_COLOR_MAP['example']), label="<ex> color")
                            tip_color_picker = gr.ColorPicker(loaded_settings.get("tip_color", HTML_COLOR_MAP['mnemonic_tip']), label="<tip> color")

                    with gr.Accordion("Cloze Card Settings", visible=("Cloze" in str(loaded_settings.get("card_type", [])))) as cloze_card_settings_accordion:
                        # ... content remains the same ...
                        gr.Markdown("Customize the appearance of all Cloze cards.")
                        cloze_color_picker = gr.ColorPicker(value=loaded_settings.get("cloze_color", "#0000FF"), label="Cloze Deletion Color")
                    
                    with gr.Accordion("Contextual Cloze Settings", visible=("Contextual Cloze" in str(loaded_settings.get("card_type", [])))) as contextual_cloze_settings_accordion:
                        # ... content remains the same ...
                        gr.Markdown("Fine-tune the grouping of facts for 'Contextual Cloze' cards.")
                        contextual_let_ai_decide_facts = gr.Checkbox(label="Let AI Decide facts per card", value=loaded_settings.get("contextual_let_ai_decide_facts", True))
                        with gr.Row():
                            contextual_min_facts_input = gr.Number(label="Min Facts / Card", value=loaded_settings.get("contextual_min_facts_input", 2), step=1, minimum=1, interactive=(not loaded_settings.get("contextual_let_ai_decide_facts", True)), precision=0)
                            contextual_max_facts_input = gr.Number(label="Max Facts / Card", value=loaded_settings.get("contextual_max_facts_input", 5), step=1, minimum=1, interactive=(not loaded_settings.get("contextual_let_ai_decide_facts", True)), precision=0)

                    with gr.Accordion("Mermaid Diagram Settings", visible=("Mermaid" in str(loaded_settings.get("card_type", [])))) as mermaid_card_settings_accordion:
                        # ... content remains the same ...
                        gr.Markdown("Customize the appearance of Mermaid diagrams.")
                        mermaid_theme_dropdown = gr.Dropdown(choices=['default', 'dark', 'neutral', 'forest'], value=loaded_settings.get("mermaid_theme", 'default'), label="Diagram Theme")
                    
                    image_sources = gr.CheckboxGroup(IMAGE_SOURCES, label="Image Source Selection", info="The system tries sources from top to bottom.", value=loaded_settings.get("image_sources", ["PDF (AI Validated)", "Wikimedia", "NLM Open-i", "PDF Page as Image (Fallback)"]))
                    custom_tags_textbox = gr.Textbox(label="Custom Tags (Optional)", value=loaded_settings.get("custom_tags", ""), placeholder="e.g., Anatomy, Midterm_1")
                    pdf_language_dropdown = gr.Dropdown(
                        ["English", "Spanish", "French", "German", "Portuguese", "Arabic", "Other"],
                        label="PDF Language",
                        value=loaded_settings.get("pdf_language", "English"),
                        info="Select 'English' to enable the superfluous fact filter. Select any other language to bypass it."
                    )

                gr.Markdown("### Session Log")
                log_output = gr.Textbox(label="Progress", lines=30, interactive=False, autoscroll=True)
                copy_log_button = gr.Button("Copy Log for Debugging")
        
        # --- MODIFIED: Updated SETTINGS_KEYS for the new API key textboxes ---
        SETTINGS_KEYS = [
            "card_type", "image_sources", "pos_color", "neg_color", "ex_color", "tip_color",
            "min_chars", "max_chars", "char_target", "cloze_color", "mermaid_theme",
            "custom_tags", "content_strategy", "objectives_text_manual", "pdf_language",
            "gemini_api_key_1", "gemini_api_key_2", "gemini_api_key_3", "gemini_api_key_4", "gemini_api_key_5",
            "openverse_api_key", "flickr_api_key",
            "builder_user_instructions", "atomic_cloze_user_instructions",
            "contextual_cloze_user_instructions", "mermaid_user_instructions",
            "builder_prompt_template", "atomic_cloze_prompt_template",
            "contextual_cloze_prompt_template", "mermaid_prompt_template",
            "curator_prompt", "extractor_prompt", "objective_finder_prompt", "image_curator_prompt",
            "basic_let_ai_decide_facts", "basic_min_facts_input", "basic_max_facts_input",
            "contextual_let_ai_decide_facts", "contextual_min_facts_input", "contextual_max_facts_input",
        ]
        
        def save_current_settings(*settings_values):
            # ... save_current_settings function remains the same ...
            current_settings = dict(zip(SETTINGS_KEYS, settings_values))
            keys_to_filter = [
                "objectives_text_manual", "builder_prompt_template", "atomic_cloze_prompt_template", 
                "contextual_cloze_prompt_template", "mermaid_prompt_template"
            ]
            settings_to_save = {k: v for k, v in current_settings.items() if k not in keys_to_filter}
            save_settings(settings_to_save)
            print("Settings auto-saved.")

        # Event Handlers
        # ... reset and other handlers remain the same ...
        reset_builder_instructions_btn.click(fn=lambda: "", outputs=builder_user_instructions)
        reset_atomic_cloze_instructions_btn.click(fn=lambda: "", outputs=atomic_cloze_user_instructions)
        reset_contextual_cloze_instructions_btn.click(fn=lambda: "", outputs=contextual_cloze_user_instructions)
        reset_mermaid_instructions_btn.click(fn=lambda: "", outputs=mermaid_user_instructions)
        
        reset_curator_btn.click(fn=lambda: CURATOR_PROMPT, outputs=curator_prompt_editor)
        reset_extractor_btn.click(fn=lambda: EXTRACTOR_PROMPT, outputs=extractor_prompt_editor)
        reset_objective_finder_btn.click(fn=lambda: OBJECTIVE_FINDER_PROMPT, outputs=objective_finder_prompt_editor)
        reset_image_curator_btn.click(fn=lambda: IMAGE_CURATOR_PROMPT, outputs=image_curator_prompt_editor)
        
        master_files.change(fn=lambda files: update_decks_from_files(files, max_decks), inputs=master_files, outputs=[master_files] + deck_ui_components)
        clear_cache_button.click(fn=lambda: clear_cache(*cache_dirs), outputs=[cache_status])
        content_strategy.change(fn=lambda s: gr.update(visible=(s == "Focus on Provided Objectives")), inputs=content_strategy, outputs=objectives_textbox)
        
        def toggle_settings_visibility(selected_types):
            selected_types = selected_types if isinstance(selected_types, list) else []
            is_basic_visible = any("Basic" in s for s in selected_types)
            is_cloze_visible = any("Cloze" in s for s in selected_types)
            is_contextual_visible = any("Contextual Cloze" in s for s in selected_types)
            is_mermaid_visible = any("Mermaid" in s for s in selected_types)
            return (gr.update(visible=is_basic_visible), gr.update(visible=is_cloze_visible), gr.update(visible=is_contextual_visible), gr.update(visible=is_mermaid_visible))
        card_type.change(fn=toggle_settings_visibility, inputs=card_type, outputs=[basic_card_settings_accordion, cloze_card_settings_accordion, contextual_cloze_settings_accordion, mermaid_card_settings_accordion])

        def toggle_facts_per_card_controls(is_auto):
            return gr.update(interactive=not is_auto), gr.update(interactive=not is_auto)

        basic_let_ai_decide_facts.change(fn=toggle_facts_per_card_controls, inputs=basic_let_ai_decide_facts, outputs=[basic_min_facts_input, basic_max_facts_input])
        contextual_let_ai_decide_facts.change(fn=toggle_facts_per_card_controls, inputs=contextual_let_ai_decide_facts, outputs=[contextual_min_facts_input, contextual_max_facts_input])

        # --- MODIFIED: Updated master list for auto-saving with new API key components ---
        other_settings_and_prompts = [
            card_type, image_sources, pos_color_picker, neg_color_picker, ex_color_picker,
            tip_color_picker, min_chars_input, max_chars_input, char_target_slider,
            cloze_color_picker, mermaid_theme_dropdown, custom_tags_textbox, pdf_language_dropdown,
            content_strategy, objectives_textbox,
            gemini_api_key_1, gemini_api_key_2, gemini_api_key_3, gemini_api_key_4, gemini_api_key_5,
            openverse_api_key_textbox, flickr_api_key_textbox,
            builder_user_instructions, atomic_cloze_user_instructions,
            contextual_cloze_user_instructions, mermaid_user_instructions,
            builder_prompt_template, atomic_cloze_prompt_template,
            contextual_cloze_prompt_template, mermaid_prompt_template,
            curator_prompt_editor, extractor_prompt_editor, objective_finder_prompt_editor,
            image_curator_prompt_editor,
            basic_let_ai_decide_facts, basic_min_facts_input, basic_max_facts_input,
            contextual_let_ai_decide_facts, contextual_min_facts_input, contextual_max_facts_input,
        ]
        
        for component in other_settings_and_prompts:
            event_trigger = component.blur if isinstance(component, (gr.Textbox, gr.Number)) else component.change
            event_trigger(fn=save_current_settings, inputs=other_settings_and_prompts, outputs=None, queue=False)

        all_gen_inputs = [master_files, generate_button, log_output, clip_model_state] + deck_input_components + other_settings_and_prompts
        all_gen_outputs = [log_output, master_files, generate_button]

        gen_event = generate_button.click(fn=functools.partial(generate_all_decks, max_decks), inputs=all_gen_inputs, outputs=all_gen_outputs)
        cancel_button.click(fn=None, cancels=[gen_event])
        copy_log_button.click(fn=None, inputs=[log_output], js="(text) => { navigator.clipboard.writeText(text); alert('Log copied to clipboard!'); }")
        
        all_deck_files_components = [ui for i, ui in enumerate(deck_input_components) if i % 2 == 1]
        new_batch_button.click(fn=lambda: (gr.update(value=None), gr.update(value=""), []) + [gr.update(value=[]) for _ in all_deck_files_components], outputs=[master_files, log_output] + all_deck_files_components)
        
        app.load(update_checker_func, None, update_notifier)
        app.load(fn=load_clip_model_func, inputs=None, outputs=[clip_model_state, clip_model_status])
        
        return app