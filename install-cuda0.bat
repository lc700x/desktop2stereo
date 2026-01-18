@echo off
echo --- Desktop2Stereo Installer (With CUDA for NVIDIA GPUs.) ---
echo - Setting up the virtual environment

@REM Set paths
Set "VIRTUAL_ENV=.env"
Set "PYTHON_EXE=python.exe"

@REM Check if Python is available
where %PYTHON_EXE% >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not found in PATH. Please install Python 3.11 first.
    pause
    exit /b 1
)

@REM Create virtual environment if it doesn't exist
if not exist "%VIRTUAL_ENV%\Scripts\activate.bat" (
    echo Creating virtual environment...
    %PYTHON_EXE% -m venv "%VIRTUAL_ENV%"
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment
        pause
        exit /b 1
    )
)

@REM Activate virtual environment
echo - Virtual environment activation
call "%VIRTUAL_ENV%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment
    pause
    exit /b 1
)

@REM Update pip
echo - Updating the pip package
%PYTHON_EXE% -m pip install --upgrade pip --no-cache-dir
if %errorlevel% neq 0 (
    echo Failed to update pip
    pause
    exit /b 1
)

@REM Install requirements
echo.
echo - Installing the requirements
%PYTHON_EXE% -m pip install -r requirements-cuda0.txt --no-cache-dir
%PYTHON_EXE% -m pip install "triton-windows<3.5" --no-cache-dir
%PYTHON_EXE% -m pip install tensorrt_cu12==10.14.1.48.post1 --no-cache-dir
%PYTHON_EXE% -m pip install -r requirements.txt --no-cache-dir
%PYTHON_EXE% -m pip install onnx onnxscript --no-cache-dir
%PYTHON_EXE% -m pip install wincam==1.0.14 --no-cache-dir
%PYTHON_EXE% -m pip install windows-capture==1.5.0 --no-cache-dir

if %errorlevel% neq 0 (
    echo Failed to install requirements
    pause
    exit /b 1
)

echo Python environment deployed successfully.
pause