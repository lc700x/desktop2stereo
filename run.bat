@echo off
echo Running Desktop2Stereo...

:: Set paths
Set "VIRTUAL_ENV=.env"

:: Check if environment exists
If Not Exist "%VIRTUAL_ENV%\Scripts\activate.bat" Exit /B 1

echo - Virtual environment activation
Call "%VIRTUAL_ENV%\Scripts\activate.bat"

python main.py --hf-mirror

@REM --hf-mirror is used to set the Hugging Face mirror for users who cannot access HuggingFace directly.
