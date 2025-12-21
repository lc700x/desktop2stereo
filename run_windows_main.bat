@echo off
set PYTHONPATH=.\Python311\python.exe:$PYTHONPATH
Set "PYTHON_EXE=.\Python311\python.exe"
%PYTHON_EXE% -m main
@REM start "" pythonw -m gui
exit /b 0