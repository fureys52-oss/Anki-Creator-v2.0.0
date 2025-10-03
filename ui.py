from pathlib import Path
from typing import List, Any, Tuple, Callable
import functools
import gradio as gr
from prompts import (
    EXTRACTOR_PROMPT, BUILDER_PROMPT, CLOZE_BUILDER_PROMPT, 
    CONTEXTUAL_CLOZE_BUILDER_PROMPT, CURATOR_PROMPT, IMAGE_CURATOR_PROMPT,
    MERMAID_BUILDER_PROMPT, OBJECTIVE_FINDER_PROMPT
)
from utils import clear_cache, update_decks_from_files, load_settings
from processing import generate_all_decks, HTML_COLOR_MAP

def build_ui(version: str, max_decks: int, cache_dirs: Tuple[Path, Path], log_dir: Path, max_log_files: int, clip_model: Any, update_checker_func: Callable) -> gr.Blocks:
    
    loaded_settings = load_settings()
    
    IMAGE_SOURCES = [
        "PDF (AI Validated)", "Wikimedia", "NLM Open-i",
        "Openverse", "Flickr"
    ]
    
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="Anki Deck Generator") as app:
        clip_model_state = gr.State(clip_model)
        
        update_notifier = gr.Markdown(visible=False) 

        gr.Markdown(f"# Anki Flashcard Generator\n*(v{version})*")

        with gr.Row():
            generate_button = gr.Button("Generate All Decks", variant="primary", scale=2)
            cancel_button = gr.Button("Cancel")
            new_batch_button = gr.Button("Start New Batch")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("1. Decks & Files"):
                        with gr.Group():
                            master_files = gr.File(label="Upload PDFs to assign to decks below", file_count="multiple", file_types=[".pdf"])
                        
                        content_strategy = gr.Radio(
                            ["Extract All Facts", "Focus on Provided Objectives", "Focus on Auto-Extracted Objectives"], 
                            label="Content Extraction Strategy", 
                            value=loaded_settings.get("content_strategy", "Extract All Facts")
                        )
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
                        gr.Markdown("⚠️ **For builder prompts, only add behavioral instructions. Do not mention output format. For other prompts, editing can break the application.**")
                        with gr.Row():
                            reset_prompts_button = gr.Button("Reset All Prompts to Default")
                        
                        with gr.Accordion("Basic Card Builder", open=True):
                            gr.Markdown("Add instructions to guide the AI's question and answer style. The core rules and output format are protected.")
                            builder_user_instructions = gr.Textbox(label="Custom Instructions", placeholder="e.g., 'Prioritize creating clinical vignettes.'", lines=3, value=loaded_settings.get("builder_user_instructions", ""))
                            builder_prompt_template = gr.Textbox(value=BUILDER_PROMPT, visible=False)

                        with gr.Accordion("Atomic Cloze Builder", open=False):
                            gr.Markdown("Add instructions to guide the AI's keyword selection. The core rules and output format are protected.")
                            atomic_cloze_user_instructions = gr.Textbox(label="Custom Instructions", placeholder="e.g., 'Prioritize cloze-deleting drug names.'", lines=3, value=loaded_settings.get("atomic_cloze_user_instructions", ""))
                            atomic_cloze_prompt_template = gr.Textbox(value=CLOZE_BUILDER_PROMPT, visible=False)

                        with gr.Accordion("Contextual Cloze Builder", open=False):
                            gr.Markdown("Add instructions to guide how the AI groups facts and synthesizes sentences. The core rules and output format are protected.")
                            contextual_cloze_user_instructions = gr.Textbox(label="Custom Instructions", placeholder="e.g., 'Prefer to group facts that describe a sequence.'", lines=3, value=loaded_settings.get("contextual_cloze_user_instructions", ""))
                            contextual_cloze_prompt_template = gr.Textbox(value=CONTEXTUAL_CLOZE_BUILDER_PROMPT, visible=False)
                        
                        with gr.Accordion("Interactive Mermaid Diagram Builder", open=False):
                            gr.Markdown("Add instructions to guide how the AI designs the diagrams. The core rules and output format are protected.")
                            mermaid_user_instructions = gr.Textbox(label="Custom Instructions", placeholder="e.g., 'Always use a top-down (TD) graph direction.'", lines=3, value=loaded_settings.get("mermaid_user_instructions", ""))
                            mermaid_prompt_template = gr.Textbox(value=MERMAID_BUILDER_PROMPT, visible=False)

                        with gr.Accordion("Protected Prompts (Edit with Extreme Caution)", open=False):
                            curator_prompt_editor = gr.Textbox(label="Content Curator Prompt", value=loaded_settings.get("curator_prompt", CURATOR_PROMPT), lines=10, max_lines=20)
                            extractor_prompt_editor = gr.Textbox(label="Fact Extractor Prompt", value=loaded_settings.get("extractor_prompt", EXTRACTOR_PROMPT), lines=10, max_lines=20)
                            objective_finder_prompt_editor = gr.Textbox(label="Objective Finder Prompt", value=loaded_settings.get("objective_finder_prompt", OBJECTIVE_FINDER_PROMPT), lines=10, max_lines=20)
                            image_curator_prompt_editor = gr.Textbox(label="Image Curator Prompt", value=loaded_settings.get("image_curator_prompt", IMAGE_CURATOR_PROMPT), lines=10, max_lines=20)

                    with gr.TabItem("3. System"):
                         with gr.Accordion("Cache Management", open=False):
                            clear_cache_button = gr.Button("Clear All Caches")
                            cache_status = gr.Textbox(label="Cache Status", interactive=False)
                         with gr.Accordion("Acknowledgements", open=False):
                            gr.Markdown("This project was built with the invaluable help of the open-source community.")
                         with gr.Accordion("API Keys", open=True):
                            gr.Markdown("Keys are saved locally in `settings.json`. For security, you can leave these blank to use a `.env` file instead.")
                            gemini_api_key_textbox = gr.Textbox(
                                label="Gemini API Key", 
                                type="password", 
                                value=loaded_settings.get("gemini_api_key", ""),
                                placeholder="Enter your Google AI Studio key here"
                            )
                            openverse_api_key_textbox = gr.Textbox(
                                label="Openverse API Key (Optional)", 
                                type="password", 
                                value=loaded_settings.get("openverse_api_key", ""),
                                placeholder="Enter your Openverse key here"
                            )
                            flickr_api_key_textbox = gr.Textbox(
                                label="Flickr API Key (Optional)", 
                                type="password", 
                                value=loaded_settings.get("flickr_api_key", ""),
                                placeholder="Enter your Flickr key here"
                            )
            
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Core Settings")
                    card_type = gr.Radio(
                        ["Conceptual (Basic Cards)", "Atomic Cloze (1 fact/card)", "Contextual Cloze (AnkiKing Style)", "Interactive Mermaid Diagram"], 
                        label="Card Type", value=loaded_settings.get("card_type", "Conceptual (Basic Cards)"))
                    
                    with gr.Accordion("Basic Card Settings", visible=("Basic" in loaded_settings.get("card_type", "Conceptual (Basic Cards)"))) as basic_card_settings_accordion:
                        gr.Markdown("Fine-tune the content of 'Basic' cards.")
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

                    with gr.Accordion("Cloze Card Settings", visible=("Cloze" in loaded_settings.get("card_type", ""))) as cloze_card_settings_accordion:
                        gr.Markdown("Customize the appearance of all Cloze cards.")
                        cloze_color_picker = gr.ColorPicker(value=loaded_settings.get("cloze_color", "#0000FF"), label="Cloze Deletion Color")

                    with gr.Accordion("Mermaid Diagram Settings", visible=("Mermaid" in loaded_settings.get("card_type", ""))) as mermaid_card_settings_accordion:
                        gr.Markdown("Customize the appearance of Mermaid diagrams.")
                        mermaid_theme_dropdown = gr.Dropdown(choices=['default', 'dark', 'neutral', 'forest'], value=loaded_settings.get("mermaid_theme", 'default'), label="Diagram Theme")
                    
                    image_sources = gr.CheckboxGroup(IMAGE_SOURCES, label="Image Source Selection", info="The system tries sources from top to bottom.", value=loaded_settings.get("image_sources", ["PDF (AI Validated)", "Wikimedia", "NLM Open-i"]))
                    custom_tags_textbox = gr.Textbox(label="Custom Tags (Optional)", value=loaded_settings.get("custom_tags", ""), placeholder="e.g., Anatomy, Midterm_1")

                gr.Markdown("### Session Log")
                log_output = gr.Textbox(label="Progress", lines=30, interactive=False, autoscroll=True)
                copy_log_button = gr.Button("Copy Log for Debugging")
        
        # Event Handlers
        def reset_all_prompts():
            return (
                CURATOR_PROMPT, EXTRACTOR_PROMPT, "", "", "", "",
                OBJECTIVE_FINDER_PROMPT, IMAGE_CURATOR_PROMPT
            )

        prompt_editors_for_reset = [
            curator_prompt_editor, extractor_prompt_editor,
            builder_user_instructions, atomic_cloze_user_instructions, 
            contextual_cloze_user_instructions, mermaid_user_instructions,
            objective_finder_prompt_editor, image_curator_prompt_editor
        ]
        reset_prompts_button.click(fn=reset_all_prompts, outputs=prompt_editors_for_reset)
        
        master_files.change(fn=lambda files: update_decks_from_files(files, max_decks), inputs=master_files, outputs=[master_files] + deck_ui_components)
        clear_cache_button.click(fn=lambda: clear_cache(*cache_dirs), outputs=[cache_status])

        content_strategy.change(fn=lambda s: gr.update(visible=(s == "Focus on Provided Objectives")), inputs=content_strategy, outputs=objectives_textbox)
        
        def toggle_settings_visibility(ct):
            return (gr.update(visible="Basic" in ct), gr.update(visible="Cloze" in ct), gr.update(visible="Mermaid" in ct))
        card_type.change(fn=toggle_settings_visibility, inputs=card_type, outputs=[basic_card_settings_accordion, cloze_card_settings_accordion, mermaid_card_settings_accordion])

        other_settings_and_prompts = [
            # Core Settings
            card_type,
            image_sources,
            # Basic Card Settings
            pos_color_picker,
            neg_color_picker,
            ex_color_picker,
            tip_color_picker,
            min_chars_input,
            max_chars_input,
            char_target_slider,
            # Cloze Settings
            cloze_color_picker,
            # Mermaid Settings
            mermaid_theme_dropdown,
            # Other Settings
            custom_tags_textbox,
            content_strategy,
            objectives_textbox,
            gemini_api_key_textbox,
            openverse_api_key_textbox,
            flickr_api_key_textbox,
            # User Instructions
            builder_user_instructions,
            atomic_cloze_user_instructions,
            contextual_cloze_user_instructions,
            mermaid_user_instructions,
            # Hidden Templates
            builder_prompt_template,
            atomic_cloze_prompt_template,
            contextual_cloze_prompt_template,
            mermaid_prompt_template,
            # Editable Prompts
            curator_prompt_editor,
            extractor_prompt_editor,
            objective_finder_prompt_editor,
            image_curator_prompt_editor
        ]

        all_gen_inputs = [master_files, generate_button, log_output, clip_model_state] + deck_input_components + other_settings_and_prompts
        all_gen_outputs = [log_output, master_files, generate_button]

        gen_event = generate_button.click(fn=functools.partial(generate_all_decks, max_decks), inputs=all_gen_inputs, outputs=all_gen_outputs)
        cancel_button.click(fn=None, cancels=[gen_event])
        copy_log_button.click(fn=None, inputs=[log_output], js="(text) => { navigator.clipboard.writeText(text); alert('Log copied to clipboard!'); }")
        
        all_deck_files_components = [ui for i, ui in enumerate(deck_input_components) if i % 2 == 1]
        new_batch_button.click(fn=lambda: (gr.update(value=None), gr.update(value=""), []) + [gr.update(value=[]) for _ in all_deck_files_components], outputs=[master_files, log_output] + all_deck_files_components)
        app.load(update_checker_func, None, update_notifier)
    return app