Youtube Tutorial Link: 
https://www.youtube.com/watch?v=Y-DRC4Ei6E4

Table of Contents
Key Features

Installation & Setup (Script-based)

How to Use the Application

Decks & Files Tab

Core Settings Panel

Running the Generator

Advanced Usage (Prompts Tab)

Troubleshooting

Key Features
Intelligent Content Curation: An initial AI pass reads your entire document to identify and discard superfluous pages (like title pages, references, or acknowledgements), ensuring your flashcards are based only on core, examinable content.

Multiple Card Formats: Choose between three powerful generation modes:

Conceptual (Basic Cards): Creates traditional question-and-answer cards by synthesizing related facts.

Atomic Cloze: Creates one targeted cloze (fill-in-the-blank) card for every single fact.

Conceptual Cloze (Hybrid): The most advanced mode. It groups related facts into complex, multi-cloze sentences and creates atomic cards for any facts left over, guaranteeing comprehensive coverage.

AI-Validated Image Sourcing: Automatically finds and attaches relevant images to your cards. It first searches the source PDF, validating images with an AI vision model, and then falls back to searching external medical and creative commons databases if needed.

Fully Customizable: Advanced users can directly edit the AI prompts that control every step of the generation process, allowing for infinite customization of the output style.


Installation & Setup 
Follow these steps to get the application running on your Windows machine from the source scripts.

Step 1: Install External Dependencies
You need a few key programs installed on your system first.

1A. Install Python

Download: Get Python 3.9 or newer from https://www.python.org/downloads/.

Install: Run the installer. CRITICAL: On the first screen of the installer, make sure to check the box that says "Add Python to PATH".

1B. Install Anki & AnkiConnect

Anki: Get the latest version from https://apps.ankiweb.net/ and install it.

AnkiConnect: Open Anki, go to Tools > Add-ons, click "Get Add-ons..." and paste in the code: 2055492159. Restart Anki when prompted.

1C. Install Tesseract OCR Engine

Download: Get the installer from the official source: https://github.com/UB-Mannheim/tesseract/wiki.

Install: Run the installer. CRITICAL: During installation, ensure you select the languages you need (at a minimum, "English" and "OSD").
When you install tesseract, please install it specifically to this file location. This is already the default, but it is key: C:\Program Files\Tesseract-OCR\tesseract.exe

Step 2: Download and Set Up the Project Files - Click the <>Code button in the top right, then click "Download ZIP"

Download & Extract: Download the project .zip file and extract it to a permanent folder on your computer (e.g., C:\AnkiGenerator). This folder should contain setup.bat, run.bat, and other project files.

Run the Setup Script: In the project folder, double-click the setup.bat file. This will create a local Python environment and install all the required libraries automatically. Wait for it to complete.

Step 3: Configure Your API Key
Rename the Template: In the project folder, find the file named .env.template and rename it to .env.

1. Go to aistudio.google.com
2. Click Get API KEY on the bottom left
3. Click Create API Key on the top right in the new screen
4. Create a new project, name it whatever you want.
5. Name your key whatever you like, and press create key.
6. A new API key will pop up at the top of the same window.
7. Click 'copy API key' on the right hand side of that box.

Edit the .env File:

Open the newly renamed .env file with a text editor (like Notepad).

Replace YOUR_API_KEY_HERE with the key you copied from the Google AI Studio website.

Save and close the file.

Step 4: Run the Application!
IMPORTANT: Make sure the main Anki application is open and running on your desktop.

In the project folder, double-click the run.bat file.

A command prompt window will appear, followed shortly by your default web browser, which will open a new tab with the application's interface.

If you want to have a desktop shortcut, also run create_shortcut.bat




How to Use the Application
The interface is divided into a few key areas.

1. Decks & Files Tab
This is where you start.

Upload PDFs: Click the "Upload PDFs" box and select one or more PDF files you want to turn into flashcards.

Automatic Deck Creation: The application will automatically create a separate "Deck" accordion for each PDF you upload. It intelligently guesses a deck title (e.g., "L01 - Introduction to Biology") based on the PDF's filename. You can edit this title at any time.

2. Core Settings Panel
This panel on the right controls the main generation options.

Card Type:

Conceptual (Basic Cards): Good for understanding broad topics.

Atomic Cloze (1 fact/card): Best for rapid memorization of individual facts.

Conceptual Cloze (Linked facts/card): Recommended for most use cases. It provides the best balance of conceptual understanding and comprehensive fact coverage.

Image Source Priority & Selection:

Check the boxes for the image sources you want to use.

The order is important. The program will search for images from top to bottom and will stop as soon as it finds a suitable one. It is recommended to keep PDF (AI Validated) at the top.

Enabled Semantic Colors:

The AI uses special tags (<pos>, <neg>, etc.) to color-code key terms on your cards. Deselect any colors you don't want it to use.



Custom Tags (Optional):

Add any Anki tags you want applied to all generated cards, separated by commas (e.g., #Biology, #Midterm_1, #High-Yield).



3. Running the Generator
Generate All Decks: Once your PDFs are uploaded and settings are configured, click this button to start the process.

Session Log: This text box provides a real-time log of the entire process, from PDF processing to AI calls and adding notes to Anki. It is invaluable for understanding what the program is doing.

Cancel: Stops the current generation process.

Start New Batch: Clears all uploaded files and logs, allowing you to start fresh without restarting the application.



Advanced Usage (Prompts Tab)
This section is for users who want to fine-tune the AI's behavior. Warning: Editing prompts can break the application if the expected output format is changed.

Content Curator Prompt: This is the first AI pass. It decides which pages of your PDF are important.

Retain case study pages: Check this if you want the AI to consider clinical vignettes or case studies as core content.

Retain table/figure-heavy pages: Check this to ensure pages consisting mostly of important diagrams or tables are not discarded.

Fact Extractor Prompt: This controls how the AI extracts individual facts from the curated pages.

Card Builder Prompts: Each of these corresponds to one of the "Card Type" options and dictates the final structure and style of the questions and answers.

Reset All Prompts to Default: If you make a mistake, this button will restore all prompts to their original state.




Troubleshooting
"CRITICAL ERROR: Could not connect to Anki...":

Ensure the main Anki application is running on your desktop.

Verify that the AnkiConnect add-on is installed and that you restarted Anki after installation.

Check if a firewall is blocking the connection.

"CRITICAL ERROR: Your Gemini API Key appears to be invalid...":

Make sure you renamed .env.template to .env.

Double-check that you have correctly copied the entire API key into the .env file with no extra spaces or characters.

The log shows "OCR FAILED" or the card content is garbled:

This means your PDF is likely image-based and Tesseract OCR is having trouble.

Ensure Tesseract was installed correctly with the right language packs.

The quality of the scan may be too low for accurate text recognition.
