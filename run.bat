@echo off
Set "VIRTUAL_ENV=.env"
If Not Exist "%VIRTUAL_ENV%\Scripts\activate.bat" Exit /B 1
echo - Virtual enviroment activation
Call "%VIRTUAL_ENV%\Scripts\activate.bat"
python main.py --hf-mirror
@REM --hf-mirror is used to set the Hugging Face mirror for the user cannot access HuggingFace directly.