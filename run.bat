call .env\Scripts\activate
set "SCRIPT_DIR=%~dp0"
set "REL_PATH=%SCRIPT_DIR%\.env\Scripts"
set "PATH=%REL_PATH%;%PATH%"
start "" python -m gui
@REM start "" pythonw -m gui
exit /b 0