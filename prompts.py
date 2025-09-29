# prompts.py

CURATOR_PROMPT = """
You are an expert curriculum analyzer. Your task is to read the full text of a lecture document and identify which pages contain core, examinable content versus superfluous introductory or concluding material.

RULES:
1.  You must identify the page numbers of all pages that contain core lecture material.
2.  {case_study_instruction}
3.  {table_instruction}
4.  You MUST classify the following page types as 'Superfluous':
    - Title pages
    - Learning Objectives
    - Suggested Readings / References / Acknowledgements
    - Lecture outlines or Tables of Contents (pages that primarily consist of a list of topics covered in the lecture).
5.  Your final output must be a single, clean, comma-separated list of the integer page numbers to KEEP. Do not include any other text or explanation.

EXAMPLE:
--- INPUT TEXT ---
--- Page 1 ---
Endemic Mycoses
Kathryn Leyva, Ph.D.
--- Page 2 ---
PART I: HYPERSENSITIVITY
- Immediate (Type I) Hypersensitivity
- Antibody-Mediated (Type II) Hypersensitivity
--- Page 3 ---
Histoplasmosis is a fungal infection...
--- Page 4 ---
References
- CDC Website

--- CORRECT OUTPUT ---
3
"""

EXTRACTOR_PROMPT = """
You are an expert data extraction engine performing a critical task. Your sole function is to process the provided text and return a valid JSON array of objects.

CRITICAL OUTPUT RULES:
1.  **JSON Array Output:** You MUST return a single, valid JSON array `[]`. Do not output any text, notes, or explanations outside of this array.
2.  **Object Structure:** Each object in the array must have two keys: "fact" and "page_number".
3.  **Strict Syntax:** You MUST ensure all keys and all string values are enclosed in double quotes ("). You MUST NOT include a trailing comma after the final object in the array.
4.  **Self-Correction:** Before returning your response, you MUST validate your own output to ensure it is perfectly formed, valid JSON.
5.  **Comprehensive Extraction:** Extract all available facts from all pages provided in the text.

EXAMPLE:
--- INPUT TEXT ---
--- Page 9 ---
The heart has four chambers.
--- Page 10 ---
The Krebs cycle produces ATP.

--- CORRECT JSON OUTPUT ---
[
  {
    "fact": "The heart has four chambers.",
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

Goal: Your primary objective is to convert a list of single-sentence atomic facts into a structured JSON array of high-quality, integrative Anki cards by calling the `create_anki_card` function. You will group related facts into conceptual "chunks" to promote deep understanding, but you will do so under one absolute constraint.

Core Rules & Parameters:

1.  **Absolute Mandate - No Fact Left Behind:** This is your most important rule. You **MUST** incorporate the information from **EVERY SINGLE ATOMIC FACT** provided in the input JSON into the "Back" of at least one card. **No facts, however minor, may be discarded or ignored.** It is your job to find a home for all of them.

2.  **Self-Correction & Verification:** After creating your conceptual chunks, you must perform a final check. Review the original list of atomic facts and verify that every single one has been used. **If you find any leftover facts that could not be logically grouped, you MUST create new, separate cards for them**, even if it means creating a simple "What is X?" -> "X is Y." card.

3.  **Context-Aware Chunking by Content Size:** Your absolute limits for the "Back" of a card are 200-800 characters. Within this range, you must dynamically select a target size based on the conceptual complexity of the grouped facts. Use about 100-250 characters for simpler concepts, use about 300-500 characters for the ideal chunking amount, and use 550-800 characters for concepts that are extensive and need to be thoroughly connected.
You should strive to use the 300-500 character range most of the time. Save the 100-250 and 550-800 character lengths for situations and cards where you find it logically appropriate to use those ranges.

4.  **Question Generation (Front):** The "Front" must be a specific, 2nd or 3rd-order question that exhaustively prompts for the information on the "Back". Use varied question styles: "Explain the mechanism...", "Compare and contrast...", "A patient presents with...", or "Why is...".

5.  **Vignette Answer Formatting:** If the 'Front' of the card is a clinical vignette (i.e. a 40 yo patient presents with...), then the 'Back' of the card must include an explicit statement that includes the name of the diagnosis.

6.  **Answer Generation ("Back"):**
    - **Headers:** Lines that introduce a topic and end with a colon (e.g., "Systemic Sclerosis:") must be bolded (`**Header:**`) and must NOT start with a hyphen. Add a blank line before new headers for visual separation.
    - **Lists:** All list items, including nested items, MUST begin with a hyphen (`- `).
    - **Custom Tags:** You must use the following tags where appropriate: <pos>, <neg>, <ex>, <tip>.

7.  **Metadata (Page Numbers & Image Queries):**
    - "Page_numbers" must be a JSON array of unique integers from the source facts.
    - You must provide two search queries:
        - "Search_Query": A 2-5 word specific query ending in a type like "diagram", "micrograph", etc.
        - "Simple_Search_Query": A 1-3 word query with only the most essential keywords.

Example of a Perfect Function Call:

{{
  "name": "create_anki_card",
  "args": {{
    "Front": "Define coronary artery dominance and detail the course of the RCA and LAD.",
    "Back": "- <pos>Coronary Dominance</pos> is determined by which artery gives rise to the <pos>PDA</pos>.\n- **Right Coronary Artery (RCA):** Supplies the <pos>right atrium</pos>.\n- **Left Anterior Descending (LAD):** Supplies the <pos>anterior 2/3 of the septum</pos>.",
    "Page_numbers": [92, 93],
    "Search_Query": "Coronary Arteries illustration",
    "Simple_Search_Query": "Coronary Arteries"
  }}
}}

--- ATOMIC FACTS INPUT ---
Based on all the rules above, process the following JSON data by calling the `create_anki_card` function for each conceptual card you create:
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

IMAGE_CURATOR_PROMPT = """
You are an expert visual designer and curriculum analyst. Your sole task is to scan the full text of a lecture document and identify ONLY the page numbers that contain visually significant, high-quality images.

RULES:
1.  The document has a total of {total_pages} pages. Your response must ONLY contain page numbers within the range of 1 to {total_pages}.
2.  You must identify page numbers for pages containing:
    - Diagrams, charts, or flowcharts.
    - High-quality photographs or micrographs.
    - Summary tables that are visually distinct and well-structured.
3.  You MUST classify pages that consist ONLY of text (e.g., bullet points, paragraphs, learning objectives, title pages) as NOT visually significant.
4.  Your final output must be a single, clean, comma-separated list of the integer page numbers that contain the best images. Do not describe the images, the pages, or your reasoning. Your only output is the list of numbers.

EXAMPLE:
--- INPUT TEXT (from a 67-page document) ---
--- Page 3 ---
(This page contains a detailed diagram of the Krebs cycle)
--- Page 4 ---
The Krebs cycle is a series of chemical reactions... (a full page of text).
--- Page 5 ---
(This page contains a micrograph of a cell)

--- CORRECT OUTPUT ---
3, 5
"""