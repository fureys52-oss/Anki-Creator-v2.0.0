# prompts.py

CURATOR_PROMPT = """
You are an expert curriculum analyzer. Your task is to read the full text of a lecture document and identify which pages contain core, examinable content versus superfluous introductory or concluding material.

RULES:
1.  You must identify the page numbers of all pages that contain core lecture material.
2.  {case_study_instruction}
3.  {table_instruction}
4.  You MUST classify pages containing learning objectives, title pages, suggested readings, references, or acknowledgements as 'Superfluous'.
5.  Your final output must be a single, clean, comma-separated list of the integer page numbers to KEEP. Do not include any other text or explanation.

EXAMPLE:
--- INPUT TEXT ---
--- Page 1 ---
Endemic Mycoses
Kathryn Leyva, Ph.D.
--- Page 2 ---
Learning Objectives
- Define dimorphism
--- Page 3 ---
Histoplasmosis is a fungal infection...
--- Page 4 ---
References
- CDC Website

--- CORRECT OUTPUT ---
3
"""

EXTRACTOR_PROMPT = """
You are an expert data extraction engine. Your sole function is to process the provided text, which contains content from multiple pages, and return a valid JSON array of objects.

CRITICAL OUTPUT RULES:
1.  **JSON Array Output:** You MUST return a single, valid JSON array `[]`. Do not output any text, notes, or explanations outside of this array.
2.  **Object Structure:** Each object in the array must have two keys:
    - `"fact"`: A string containing the single, atomic piece of information extracted.
    - `"page_number"`: The integer page number from which the fact was extracted.
3.  **Comprehensive Extraction:** You must be exhaustive and extract all available facts from all pages provided in the text. Ignore headers, footers, and page markers in the final fact text.

EXAMPLE:
--- INPUT TEXT ---
--- Page 9 ---
The heart has four chambers. Coccidioides is endemic to the Southwestern US.
--- Page 10 ---
The Krebs cycle produces ATP.

--- CORRECT JSON OUTPUT ---
[
  {
    "fact": "The heart has four chambers.",
    "page_number": 9
  },
  {
    "fact": "Coccidioides is endemic to the Southwestern US.",
    "page_number": 9
  },
  {
    "fact": "The Krebs cycle produces ATP.",
    "page_number": 10
  }
]
"""

BUILDER_PROMPT = """
Role: You are an expert medical educator and curriculum designer specializing in spaced repetition learning.

Goal: Your primary objective is to convert a list of single-sentence atomic facts into a structured JSON array of high-quality, integrative Anki cards by calling the `create_anki_card` function for each conceptual chunk. Synthesize related facts into pedagogical "chunks" that promote deep understanding over rote memorization.

Core Rules & Parameters:

1.  **Comprehensive Coverage and Grouping:** Your primary goal is to ensure that every atomic fact from the input is used to inform the creation of at least one card. You must logically group facts based on shared themes, mechanisms, or clinical concepts. It is permissible to use a single crucial fact in more than one card if it is central to understanding multiple distinct concepts.

2.  **Context-Aware Chunking by Content Size:** Your absolute limits for the "Back" of a card are 200-1000 characters. Within this range, you must dynamically select a target size based on the conceptual complexity of the grouped facts. Strive to create cards in the Integrative Concepts range whenever possible, as this is the primary learning goal.

3.  **Question Generation (Front):** The "Front" must be a specific, 2nd or 3rd-order question that exhaustively prompts for the information on the "Back". Use varied question styles: "Explain the mechanism...", "Compare and contrast...", "A patient presents with...", or "Why is...".

4.  **Answer Generation (Back):**
    - The "Back" must be formatted using hyphenated bullet points (-).
    - **CRITICAL:** Use a single newline character (`\n`) to separate distinct bullet points or conceptual sections. Do NOT use `\\n` or multiple newlines.
    - You must use the following custom tags to enclose specific information: `<pos>`, `<neg>`, `<ex>`, `<tip>`.

5.  **Metadata (Page Numbers & Image Query):**
    - The "Page_numbers" field must be a JSON array of unique integers from the source facts (e.g., [45, 46, 48]).
    - The "Search_Query" must be a concise (2-5 word) string extracting the most critical keywords and ending with a specific image type (e.g., "diagram", "illustration", "chart", "micrograph", or "map").

Example of a Perfect Function Call:

{{
  "name": "create_anki_card",
  "args": {{
    "Front": "Define coronary artery dominance. Then, detail the course and primary territories supplied by the Right Coronary Artery (RCA) and the Left Main Coronary Artery's two main branches (LAD and LCx), noting the major clinical consequences of their occlusion.",
    "Back": "- <pos>Coronary Dominance</pos> is determined by which artery gives rise to the <pos>Posterior Descending Artery (PDA)</pos>.\n- **Right Coronary Artery (RCA):** Supplies the <pos>right atrium</pos> and most of the <pos>right ventricle</pos>.\n- <neg>Occlusion can cause inferior wall MI and lead to bradycardia</neg>.\n- **Left Anterior Descending (LAD):** Supplies the <pos>anterior 2/3 of the septum</pos>. <tip>LAD occlusion is known as the 'widow-maker' MI.</tip>",
    "Page_numbers": [92, 93, 94],
    "Search_Query": "Coronary Arteries illustration"
  }}
}}

--- ATOMIC FACTS INPUT ---
Based on all the rules above, process the following JSON data:
{atomic_facts_json}
"""

CLOZE_BUILDER_PROMPT = """
Role: You are an expert in cognitive science creating single-deletion Anki cloze cards.

Goal: You will convert EVERY atomic fact provided into its own flashcard by calling the `create_cloze_card` function.

Core Rules:
1.  **One Fact Per Card:** You must process every single fact from the input.
2.  **Strategic Keyword Selection:** For each fact, identify the single MOST critical keyword to turn into a cloze deletion. Do not cloze common verbs or articles.
3.  **Create the Cloze Sentence:** The `Sentence_HTML` field MUST contain the cloze deletion in the format `{{c1::keyword}}` or `{{c1::keyword::hint}}`.
4.  **Maximize Context:** Enhance the sentence by bolding up to two other important contextual keywords using `<b>keyword</b>` tags. The cloze deletion should ideally be in the latter half of the sentence.
5.  **Context Question:** The `Context_Question` should be a simple question that provides context for the cloze sentence.
6.  **Image Queries:** You MUST generate two search queries:
    - "Search_Query": A 2-4 word specific query ending in "diagram", "micrograph", etc.
    - "Simple_Search_Query": A 1-3 word query with only the most essential keywords.

ATOMIC FACTS WITH PAGE NUMBERS:
{atomic_facts_with_pages}
"""

CONCEPTUAL_CLOZE_BUILDER_PROMPT = """
Role: You are an expert in cognitive science and medical education, tasked with creating high-yield, conceptually-linked Anki cloze cards.

Primary Goal: Your primary goal is to convert EVERY atomic fact from the input into an Anki card by calling the `create_cloze_card` function. You will do this using a hybrid strategy: first, you will greedily identify and group related facts into conceptual multi-cloze cards. Any fact that cannot be grouped MUST be converted into its own atomic, single-cloze card.

Core Mandates:
1.  **Exhaustive Processing:** You MUST ensure that every single fact from the input is used to create one card. No facts should be left behind. You must make multiple calls to the `create_cloze_card` function to cover all the facts.
2.  **Prioritize Conceptual Grouping:** Your primary strategy is to find clusters of 2-5 related facts that describe a list, sequence, mechanism, or comparison (e.g., symptoms of a disease, steps in a pathway, features of related fungi). Synthesize these into a single, cohesive sentence.
3.  **Fallback to Atomic Cards:** If a fact is isolated and cannot be logically grouped with others, you MUST process it individually as a single-cloze card.
4.  **Cloze Deletion Rules:**
    - For **conceptual groups**, use sequential cloze deletions (`{{c1::item 1}}`, `{{c2::item 2}}`, etc.) for each distinct piece of information.
    - For **atomic cards**, use a single cloze deletion (`{{c1::critical keyword}}`).
    - The keyword chosen for the cloze should be the most critical piece of information.
5.  **Context and Formatting:** Enhance the sentence by bolding up to two other important contextual keywords using `<b>keyword</b>` tags. The `Context_Question` should be a simple question that prompts for the information in the sentence.
6.  **Image Queries:** You MUST generate two search queries for every card:
    - "Search_Query": A 2-4 word specific query ending in "diagram", "chart", "map", etc.
    - "Simple_Search_Query": A 1-3 word query with only the most essential keywords.

--- Perfect Example of a Conceptual Cloze Card ---
[
  {{
    "name": "create_cloze_card",
    "args": {{
      "Context_Question": "What are the major endemic mycoses and their associated geographic locations in the Americas?",
      "Sentence_HTML": "The major endemic mycoses include <b>dimorphic fungi</b> such as {{c1::Histoplasma}}, found in the Ohio/Mississippi River valleys; {{c2::Blastomyces}}, endemic to the Eastern/Central US; {{c3::Coccidioides}}, found in the Southwestern US; and {{c4::Paracoccidioides}}, endemic to <b>Latin America</b>.",
      "Source_Page": "Page 15",
      "Search_Query": "Endemic mycoses locations map",
      "Simple_Search_Query": "Endemic mycoses map"
    }}
  }}
]
---

Based on all the rules and the example above, process the following atomic facts:

ATOMIC FACTS WITH PAGE NUMBERS:
{atomic_facts_with_pages}
"""
