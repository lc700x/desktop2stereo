@echo off
echo --- Desktop2Stereo Installer  (With Windows DirectML for AMD GPUs and etc.) ---
echo - Setting up the virtual environment

:: Set paths
Set "VIRTUAL_ENV=.env"
Set "PYTHON_EXE=python.exe"

:: Check if Python is available
where %PYTHON_EXE% >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not found in PATH. Please install Python 3.10 first.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "%VIRTUAL_ENV%\Scripts\activate.bat" (
    echo Creating virtual environment...
    %PYTHON_EXE% -m venv "%VIRTUAL_ENV%"
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment
        pause
        exit /b 1
    )
)

:: Activate virtual environment
echo - Virtual environment activation
call "%VIRTUAL_ENV%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment
    pause
    exit /b 1
)

:: Update pip
echo - Updating the pip package
python -m pip install --upgrade pip --no-cache-dir --trusted-host http://mirrors.aliyun.com/pypi/simple/
if %errorlevel% neq 0 (
    echo Failed to update pip
    pause
    exit /b 1
)

:: Install requirements
echo.
echo - Installing the requirements
python -m pip install -r requirements-dml.txt --no-cache-dir --trusted-host http://mirrors.aliyun.com/pypi/simple/
python -m pip install -r requirements.txt --no-cache-dir --trusted-host http://mirrors.aliyun.com/pypi/simple/
if %errorlevel% neq 0 (
    echo Failed to install requirements
    pause
    exit /b 1
)

echo Python environment deployed successfully.
pause