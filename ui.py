from pathlib import Path
from typing import List, Any, Tuple
import functools
import gradio as gr
from prompts import EXTRACTOR_PROMPT, BUILDER_PROMPT, CLOZE_BUILDER_PROMPT, CONCEPTUAL_CLOZE_BUILDER_PROMPT, CURATOR_PROMPT, IMAGE_CURATOR_PROMPT
from utils import clear_cache, update_decks_from_files
from processing import generate_all_decks

def build_ui(version: str, max_decks: int, cache_dirs: Tuple[Path, Path], log_dir: Path, max_log_files: int, clip_model: Any) -> gr.Blocks:
    
    IMAGE_SOURCES = [
        "PDF (AI Validated)", 
        "Wikimedia", 
        "NLM Open-i",
        "Openverse", 
        "Flickr"
    ]
    
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="Anki Deck Generator") as app:
        clip_model_state = gr.State(clip_model)
        
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
                        
                        deck_ui_components, deck_input_components = [], []
                        for i in range(max_decks):
                            with gr.Accordion(f"Deck {i+1}", visible=(i==0), open=True) as acc:
                                deck_title = gr.Textbox(label="Deck Title")
                                files = gr.File(visible=False, file_count="multiple")
                            deck_ui_components.extend([acc, deck_title, files])
                            deck_input_components.extend([deck_title, files])
                    
                    with gr.TabItem("2. Prompts (Advanced)"):
                        gr.Markdown("⚠️ **Warning:** Editing prompts can break the application if the AI's output format is changed. Edit with caution.")
                        with gr.Row():
                            reset_prompts_button = gr.Button("Reset All Prompts to Default")
                        
                        with gr.Accordion("Content Curator Prompt", open=True):
                            curator_prompt_editor = gr.Textbox(value=CURATOR_PROMPT, lines=10, max_lines=20)
                            with gr.Row():
                                curator_retain_case_studies = gr.Checkbox(label="Retain case study pages", value=False)
                                curator_retain_tables = gr.Checkbox(label="Retain table/figure-heavy pages", value=True)
                        
                        with gr.Accordion("Image Curator Prompt", open=False):
                            image_curator_prompt_editor = gr.Textbox(value=IMAGE_CURATOR_PROMPT, lines=10, max_lines=20)

                        with gr.Accordion("Fact Extractor Prompt", open=False):
                            extractor_prompt_editor = gr.Textbox(value=EXTRACTOR_PROMPT, lines=10, max_lines=20)
                        with gr.Accordion("Basic Card Builder Prompt", open=False):
                            builder_prompt_editor = gr.Textbox(value=BUILDER_PROMPT, lines=10, max_lines=20)
                        with gr.Accordion("Atomic Cloze Builder Prompt", open=False):
                            cloze_builder_prompt_editor = gr.Textbox(value=CLOZE_BUILDER_PROMPT, lines=10, max_lines=20)
                        with gr.Accordion("Conceptual Cloze Builder Prompt", open=False):
                            conceptual_cloze_builder_prompt_editor = gr.Textbox(value=CONCEPTUAL_CLOZE_BUILDER_PROMPT, lines=10, max_lines=20)

                    with gr.TabItem("3. System"):
                         with gr.Accordion("Cache Management", open=False):
                            clear_cache_button = gr.Button("Clear All Caches")
                            cache_status = gr.Textbox(label="Cache Status", interactive=False)
                         with gr.Accordion("Acknowledgements", open=False):
                            gr.Markdown("This project was built with the invaluable help of the open-source community.")
            
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Core Settings")
                    card_type = gr.Radio(["Conceptual (Basic Cards)", "Atomic Cloze (1 fact/card)", "Conceptual Cloze (Linked facts/card)"], label="Card Type", value="Conceptual (Basic Cards)")
                    
                    image_sources = gr.CheckboxGroup(
                        IMAGE_SOURCES,
                        label="Image Source Selection (Fixed Priority)",
                        info="The system tries sources from top to bottom. Uncheck to disable a source.",
                        value=["PDF (AI Validated)", "Wikimedia", "NLM Open-i"]
                    )
                    
                    enabled_colors = gr.CheckboxGroup(
                        ["positive_key_term", "negative_key_term", "example", "mnemonic_tip"],
                        label="Enabled Semantic Colors",
                        info="Deselect colors you don't want the AI to use.",
                        value=["positive_key_term", "negative_key_term", "example", "mnemonic_tip"]
                    )
                    
                    custom_tags_textbox = gr.Textbox(label="Custom Tags (Optional)", placeholder="e.g., #Anatomy, #Midterm_1, #Cardiology", info="Add your own tags, separated by commas.")

                gr.Markdown("### Session Log")
                log_output = gr.Textbox(label="Progress", lines=30, interactive=False, autoscroll=True)
                copy_log_button = gr.Button("Copy Log for Debugging")
        
        # --- Event Handlers (Correctly Indented) ---
        def reset_prompts():
            return CURATOR_PROMPT, IMAGE_CURATOR_PROMPT, EXTRACTOR_PROMPT, BUILDER_PROMPT, CLOZE_BUILDER_PROMPT, CONCEPTUAL_CLOZE_BUILDER_PROMPT

        def update_decks_from_files_ui(files: List[gr.File]) -> List[Any]:
            return update_decks_from_files(files, max_decks)
        
        prompt_editors = [curator_prompt_editor, image_curator_prompt_editor, extractor_prompt_editor, builder_prompt_editor, cloze_builder_prompt_editor, conceptual_cloze_builder_prompt_editor]
        
        reset_prompts_button.click(fn=reset_prompts, outputs=prompt_editors)
        master_files.change(fn=update_decks_from_files_ui, inputs=master_files, outputs=[master_files] + deck_ui_components)
        clear_cache_button.click(fn=lambda: clear_cache(*cache_dirs), outputs=[cache_status])

        other_settings_and_prompts = [
            card_type,
            image_sources,
            enabled_colors,
            custom_tags_textbox,
            curator_retain_case_studies,
            curator_retain_tables
        ] + prompt_editors

        all_gen_inputs = [
            master_files, generate_button, log_output, clip_model_state
        ] + deck_input_components + other_settings_and_prompts
        
        all_gen_outputs = [log_output, master_files, generate_button]

        gen_event = generate_button.click(fn=functools.partial(generate_all_decks, max_decks), inputs=all_gen_inputs, outputs=all_gen_outputs)
        cancel_button.click(fn=None, cancels=[gen_event])
        copy_log_button.click(fn=None, inputs=[log_output], js="(text) => { navigator.clipboard.writeText(text); alert('Log copied to clipboard!'); }")
        
        all_deck_files_components = [ui for i, ui in enumerate(deck_input_components) if i % 2 == 1]
        new_batch_button.click(fn=lambda: (gr.update(value=None), gr.update(value=""), []) + [gr.update(value=[]) for _ in all_deck_files_components], outputs=[master_files, log_output] + all_deck_files_components)

    return app