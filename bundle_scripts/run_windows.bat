@echo off
cls
title Desktop2Stereo
:: Desktop2Stereo launcher - delegates to .\Desktop2Stereo\ subdir

set "APP_DIR=.\Desktop2Stereo"

if not exist "%APP_DIR%\python3\python.exe" (
    echo [Error] Python not found at %APP_DIR%\python3\python.exe
    echo Please ensure the Desktop2Stereo folder is intact.
    pause
    exit /b 1
)

if not exist "%APP_DIR%\gui.py" (
    echo [Error] %APP_DIR%\gui.py not found.
    pause
    exit /b 1
)

echo [1/3] Initializing Python environment ...
echo       Using bundled Python at %APP_DIR%\python3\python.exe
echo [2/3] Loading dependent modules ...
echo       First launch may take 1-3 min, subsequent launches much faster
echo [3/3] Starting Desktop2Stereo GUI ...

pushd "%APP_DIR%"
if exist "%APP_DIR%\activate.cmd" (
    call activate.cmd
) 
set "PYTHONPATH=.\python3\python.exe:$PYTHONPATH"
Set "PYTHON_EXE=.\python3\python.exe"
%PYTHON_EXE% gui.py
popd
exit /b 0
