@echo off
setlocal

:: --- Configuration ---
set "SHORTCUT_NAME=Anki Deck Generator.lnk"
set "ICON_NAME=icon.ico"
set "TARGET_SCRIPT=run.bat"
:: -------------------

:: Get the directory where this script is located. This makes paths absolute and reliable.
set "SCRIPT_DIR=%~dp0"

:: Check if the icon file exists before proceeding.
if not exist "%SCRIPT_DIR%%ICON_NAME%" (
    echo.
    echo ERROR: Icon file not found!
    echo Please make sure your icon is in the main folder and named '%ICON_NAME%'.
    echo.
    pause
    exit /b 1
)

echo Creating a desktop shortcut with a custom icon...

:: Use a single-line PowerShell command to prevent parsing errors by the batch interpreter.
powershell.exe -ExecutionPolicy Bypass -Command "$ws = New-Object -ComObject WScript.Shell; $sLinkFile = Join-Path -Path $ws.SpecialFolders('Desktop') -ChildPath '%SHORTCUT_NAME%'; $oLink = $ws.CreateShortcut($sLinkFile); $oLink.TargetPath = '%SCRIPT_DIR%%TARGET_SCRIPT%'; $oLink.WorkingDirectory = '%SCRIPT_DIR%'; $oLink.IconLocation = '%SCRIPT_DIR%%ICON_NAME%, 0'; $oLink.Description = 'Starts the Anki Deck Generator'; $oLink.Save()"

:: Check if the PowerShell command was successful
if %errorlevel% neq 0 (
    echo.
    echo ERROR: The shortcut could not be created.
    echo This might be due to security policies on your system preventing PowerShell from running.
    echo Please try running this script as an Administrator.
    echo.
) else (
    echo.
    echo ==========================================================
    echo  Shortcut created successfully on your Desktop!
    echo  It has been configured with the custom '%ICON_NAME%' icon.
    echo ==========================================================
    echo.
)

pause