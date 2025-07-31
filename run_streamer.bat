@echo off
echo Running Desktop2Stereo Streamer...

:: Set paths
Set "VIRTUAL_ENV=.env"

:: Check if environment exists
If Not Exist "%VIRTUAL_ENV%\Scripts\activate.bat" Exit /B 1

echo - Virtual environment activation
Call "%VIRTUAL_ENV%\Scripts\activate.bat"

python main_streamer.py --hf-mirror

@REM --hf-mirror is used to set the Hugging Face mirror for users who cannot access HuggingFace directly.
