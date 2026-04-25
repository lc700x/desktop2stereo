:: run_bundle.bat
:: Activates the Python bundle and prints Hello

@echo off
call "%~dp0activate.cmd"
set "PYTHONPATH=.\python3\python.exe:$PYTHONPATH"
Set "PYTHON_EXE=.\python3\python.exe"
%PYTHON_EXE% gui.py
