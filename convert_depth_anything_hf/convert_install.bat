@echo off
REM Check if the virtual environment folder exists
IF NOT EXIST venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate the virtual environment
echo Activating virtual environment...
CALL venv\Scripts\activate

REM Install dependencies from requirements.txt
echo Installing dependencies...
pip install -r requirements.txt --no-cache-dir
python.exe -m pip install pip --upgrade --no-cache-dir

REM Deactivate the virtual environment
CALL venv\scripts\deactivate
exit