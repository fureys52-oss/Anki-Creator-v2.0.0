@echo off
TITLE Create Anki Deck Generator Shortcut

echo Creating a shortcut on your Desktop...

:: This file will contain the VBScript commands to create the shortcut
set VBS_FILE=%TEMP%\CreateShortcut.vbs

:: This line gets the full path to the folder where this script is located
:: This is CRITICAL for making the shortcut work from anywhere
set SCRIPT_PATH=%~dp0

:: Use echo to write VBScript commands into the temporary file
echo Set WshShell = WScript.CreateObject("WScript.Shell") > %VBS_FILE%
echo strDesktop = WshShell.SpecialFolders("Desktop") >> %VBS_FILE%
echo Set oShellLink = WshShell.CreateShortcut(strDesktop ^& "\Anki Deck Generator.lnk") >> %VBS_FILE%

:: Set the Target Path - what the shortcut will run
echo oShellLink.TargetPath = "%SCRIPT_PATH%run.bat" >> %VBS_FILE%

:: Set the Working Directory - where the script will run from
echo oShellLink.WorkingDirectory = "%SCRIPT_PATH%" >> %VBS_FILE%

:: Set the Icon for the shortcut (we can use the cmd icon or point to a custom .ico if you have one)
echo oShellLink.IconLocation = "%SystemRoot%\System32\cmd.exe,0" >> %VBS_FILE%

:: Save the shortcut file
echo oShellLink.Save >> %VBS_FILE%

:: Execute the VBScript file using cscript, which prevents any windows from popping up
cscript //nologo %VBS_FILE%

:: Clean up the temporary VBScript file
del %VBS_FILE%

echo.
echo Shortcut created successfully on your Desktop!
echo.
pause