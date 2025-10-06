@echo off
title Anki Deck Generator

:: Navigate to the directory where this script is located
cd /d "%~dp0"

echo ==========================================================
echo            Anki Deck Generator by Furey
echo ==========================================================
echo.

:: --- Check if setup has already been completed ---
IF EXIST "._setup_complete_flag" (
    goto LaunchApp
)

:: --- First-Time Setup Logic ---
echo This appears to be your first time running the application.
echo Performing one-time setup...
echo.

echo [1/5] Bootstrapping package manager (pip)...
call ".\binaries\windows\python\python.exe" -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', 'get-pip.py')"
call ".\binaries\windows\python\python.exe" get-pip.py
del get-pip.py
if %errorlevel% neq 0 (
    echo. & echo ERROR: Failed to install pip. & goto End
)
echo      ...Package manager installed.
echo.

echo [2/5] Installing core build packages (setuptools, wheel, pywin32)...
call ".\binaries\windows\python\python.exe" -m pip install setuptools wheel pywin32
if %errorlevel% neq 0 (
    echo. & echo ERROR: Failed to install core build tools. & goto End
)
echo      ...Build tools installed.
echo.

:: --- NEW STEP: Install winshell directly from the local .whl file ---
echo [3/5] Installing Windows shortcut utility (winshell)...
call ".\binaries\windows\python\python.exe" -m pip install ".\binaries\windows\winshell-0.6-py2.py3-none-any.whl"
if %errorlevel% neq 0 (
    echo. & echo ERROR: Failed to install the winshell package from local file. & goto End
)
echo      ...Shortcut utility installed.
echo.
:: --- END NEW STEP ---

echo [4/5] Installing all other required Python packages...
:: --- ADDED --no-warn-script-location to suppress yellow warnings ---
call ".\binaries\windows\python\python.exe" -m pip install --no-warn-script-location -r requirements.txt
if %errorlevel% neq 0 (
    echo. & echo ERROR: Package installation failed. & goto End
)
echo      ...All packages installed successfully.
echo.

echo [5/5] Creating desktop shortcut...
powershell.exe -ExecutionPolicy Bypass -Command "$ws = New-Object -ComObject WScript.Shell; $sLinkFile = Join-Path -Path $ws.SpecialFolders('Desktop') -ChildPath 'Anki Deck Generator.lnk'; $oLink = $ws.CreateShortcut($sLinkFile); $oLink.TargetPath = '%~f0'; $oLink.WorkingDirectory = '%~dp0'; $oLink.IconLocation = '%~dp0icon.ico, 0'; $oLink.Save()"
    
if %errorlevel% neq 0 (
    echo      ...WARNING: Failed to create desktop shortcut. You can run this file directly.
) else (
    echo      ...Shortcut created successfully on your Desktop!
)
echo.

echo [6/6] Finalizing setup...
:: Create the flag file to skip this block next time
echo SETUP_COMPLETE > "._setup_complete_flag"
echo.
echo Setup is complete! Launching the application now...
timeout /t 2 >nul

:LaunchApp
echo Launching the application... Please wait for the browser window to open.

:: Temporarily add our bundled Tesseract to the PATH for this session
set "PATH=.\binaries\windows\tesseract;%PATH%"

:: Run the app using our bundled Python, hiding the console window
call ".\binaries\windows\python\python.exe" app.py
pause

exit

:End
echo.
echo Setup failed. Please resolve the error above and run this script again.
pause