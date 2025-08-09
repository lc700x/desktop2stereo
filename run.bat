@echo off
echo - Running Desktop2Stereo for Windows...
@REM Set the virtual environment path
Set "VIRTUAL_ENV=.env"

@REM Check if environment exists
If Not Exist "%VIRTUAL_ENV%\Scripts\activate.bat" Exit /B 1

echo - Virtual environment activation
Call "%VIRTUAL_ENV%\Scripts\activate.bat"

python main.py --hf-mirror

@REM --hf-mirror is used to set the Hugging Face mirror for users who cannot access HuggingFace directly.

if %errorlevel% neq 0 (
    echo Desktop2Stereo stopped. 
    pause
    exit /b 1
)
