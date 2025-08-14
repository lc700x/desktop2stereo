@echo off
REM === Repair Python venv to point to local Python310 ===
REM Run this from the ProjectFolder

REM Get absolute path to ProjectFolder (where this .bat lives)
set PROJECT_DIR=%~dp0
set PY_VER=3.10.11
for %%I in ("%PROJECT_DIR%") do set PROJECT_DIR=%%~fI

REM Paths
set PYTHON_DIR=%PROJECT_DIR%Python310
set PYTHON_EXE=%PYTHON_DIR%\python.exe
set VENV_DIR=%PROJECT_DIR%.env
set CFG_FILE=%VENV_DIR%\pyvenv.cfg
set ACTIVATE_FILE=%VENV_DIR%\Scripts\activate.bat
set ACTIVATE=%VENV_DIR%\Scripts\activate

echo --- Python Paths ---
echo Project: %PROJECT_DIR%
echo Python:  %PYTHON_EXE%
echo Venv:    %VENV_DIR%
echo -----------------------

REM Check Python
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python 3.10 not found at %PYTHON_EXE%
    pause
    exit /b 1
)

REM Check venv
if not exist "%CFG_FILE%" (
    echo ERROR: pyvenv.cfg not found at %CFG_FILE%
    pause
    exit /b 1
)

REM Update pyvenv.cfg
@REM echo Updating pyvenv.cfg ...
> "%CFG_FILE%" echo home = %PYTHON_DIR%
>> "%CFG_FILE%" echo include-system-site-packages = false
>> "%CFG_FILE%" echo version = %PY_VER%

REM Update activate scripts
setlocal enabledelayedexpansion

:: CONFIGURATION
:: SCRIPT LOGIC
set "TEMP_FILE=%ACTIVATE_FILE%.tmp"
@REM echo Target file: %ACTIVATE_FILE%

REM Create the new line we want to write
set "NEW_LINE=set "VIRTUAL_ENV=%VENV_DIR%""

REM Process the file line by line, writing output to a temporary file
(for /f "usebackq delims=" %%L in ("%ACTIVATE_FILE%") do (
    set "line=%%L"
    REM Check if the line contains the variable we want to change
    if "!line:VIRTUAL_ENV=!" neq "!line!" (
        echo !NEW_LINE!
    ) else (
        echo !line!
    )
)) > "%TEMP_FILE%"

REM Replace the original file with the temporary one and delete the temp file
move /y "%TEMP_FILE%" "%ACTIVATE_FILE%" > nul



set "TEMP_FILE=%ACTIVATE%.tmp"
@REM echo Target file: %ACTIVATE%
(for /f "usebackq delims=" %%L in ("%ACTIVATE%") do (
    set "line=%%L"
    REM Check if the line contains the variable we want to change
    if "!line:VIRTUAL_ENV=!" neq "!line!" (
        echo !NEW_LINE!
    ) else (
        echo !line!
    )
)) > "%TEMP_FILE%"

REM Replace the original file with the temporary one and delete the temp file
move /y "%TEMP_FILE%" "%ACTIVATE%" > nul

@REM echo.
@REM echo Successfully updated the VIRTUAL_ENV path in activate.bat.

endlocal

@REM echo activate scripts updated with new VIRTUAL_ENV path.
@REM echo Updated 'home =' in %CFG_FILE% to %CURR_DIR%
echo Starting Desktop2Stereo GUI...
call .env\Scripts\activate
start "" python  -m gui
exit /b 0