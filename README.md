Youtube Tutorial Link: 
https://www.youtube.com/watch?v=Y-DRC4Ei6E4

Installation & Setup (Windows & macOS)
Follow these steps to get the application running on your computer.

Step 1: Install External Dependencies
You need a few key programs installed on your system first.

1A. Install Python

On Windows:
- Download: Get Python 3.9 or newer from python.org.
- Install: Run the installer. CRITICAL: On the first screen of the installer, you must check the box that says "Add Python to PATH".

On macOS:
- Open the Terminal app (found in applications/utilities) and run the following command: xcode-select --install
- Install Homebrew: If you don't have it, open the Terminal app and install the Homebrew package manager from brew.sh (its a website)
- Install Python: In your Terminal, run the command: brew update && brew install python3

1B. Install Anki & AnkiConnect (Universal)
- Anki: Get the latest version from apps.ankiweb.net and install it.
- AnkiConnect: Open Anki, go to Tools > Add-ons, click "Get Add-ons..." and paste in the code: 2055492159. Restart Anki when prompted.

1C. Install Tesseract OCR Engine
- On Windows:
  - Download: Get the installer from the official source: Tesseract at UB Mannheim.
  - Install: Run the installer. It is recommended to keep the default installation path.
- On macOS:
  - In your Terminal, run the command: brew install tesseract

Step 2: Download and Prepare the Project
- Download: Click the green <> Code button at the top of the GitHub page and select "Download ZIP".
- Extract: Extract the .zip file to a permanent folder on your computer (e.g., C:\AnkiGenerator or /Users/YourName/AnkiGenerator).

Step 3: Run the Setup Script
- This one-time step creates a local Python environment and installs all required libraries.
- On Windows:
  - In the project folder, double-click the setup.bat file. Wait for the command prompt window to finish and close.

- On macOS:
  - 1 Open the Terminal app.
  - 2 Type cd (the letters c and d, followed by a space).
  - 3 Drag your project folder from Finder and drop it onto the Terminal window.
  - 4 Press Enter.
  - 5 Copy and paste the following command, then press Enter:
chmod +x setup.command run.command create_shortcut.command
  - 6 Run setup.command in your project folder by double cicking it.

Step 4: Configure Your API Key (Universal)
- Go to aistudio.google.com and get your API Key. Just click th get api key button at the bottom left.
- On Windows: 
  - In the project folder, find the file named .env.template and rename it to .env.
  - Open the new .env file with a text editor (like Notepad or TextEdit).
  - Replace YOUR_API_KEY_HERE with the key you copied from the Google AI Studio website.
  - Save and close the file.
- On macOS: 
  - Open the Terminal, navigate to the project folder (using the cd and drag-and-drop trick from Step 3), and run the command: cp .env.template .env
  - Open the new .env file with a text editor (like Notepad or TextEdit), paste your API key, and save.

Step 5: Run the Application!
- IMPORTANT: Make sure the main Anki application is open and running on your desktop.
- On Windows: Double-click the run.bat file.
- On macOS: Double-click the run.command file.
A terminal or command prompt window will appear, and your web browser will open with the application's interface.

Step 6: Create a Desktop Shortcut (Optional)
On Windows:
Place your desired icon in the main project folder and rename it to icon.ico.
Double-click Create_Shortcut.bat. This will automatically place a shortcut on your Desktop with your custom icon.
On macOS:
Double-click create_shortcut.command to place a shortcut on your Desktop.
To apply a custom icon, follow the manual steps:
Open your icon file (.ico, .png, etc.) in the Preview app.
Press Cmd+A (Select All), then Cmd+C (Copy).
Right-click the shortcut on your Desktop and choose "Get Info".
Click the small icon in the top-left of the "Get Info" window (it will get a blue highlight).
Press Cmd+V (Paste). The icon will be replaced.





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
