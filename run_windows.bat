@echo off
REM Get current folder
set "CURR_DIR=%cd%"

REM Set the target pyvenv.cfg file path
set "CFG_FILE=.env\pyvenv.cfg"

REM Create a temp file
set "TMP_FILE=%TEMP%\pyvenv_tmp.cfg"

REM Replace the whole 'home =' line
(for /f "usebackq delims=" %%A in ("%CFG_FILE%") do (
    echo %%A | findstr /b /c:"home =" >nul
    if errorlevel 1 (
        echo %%A
    ) else (
        echo home = %CURR_DIR%\Python310
    )
)) > "%TMP_FILE%"

REM Replace the original file
move /Y "%TMP_FILE%" "%CFG_FILE%"

@REM echo Updated 'home =' in %CFG_FILE% to %CURR_DIR%
call .env\Scripts\activate
start "" python -m gui
@REM start "" pythonw -m gui
exit /b 0