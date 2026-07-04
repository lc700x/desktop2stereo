@echo off
set PYTHONPATH=.\python3\python.exe:$PYTHONPATH
Set "PYTHON_EXE=.\python3\python.exe"
%PYTHON_EXE% -m gui
@REM start "" pythonw -m gui
exit /b 0