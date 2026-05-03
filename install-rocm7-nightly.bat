@echo off
echo --- Desktop2Stereo Installer (With ROCm7 for AMD GPU/APUs.) ---

REM --- Get AMD GPUs sorted by AdapterRAM (largest first) ---
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Where-Object Name -like '*AMD*' | Sort-Object AdapterRAM -Descending | ForEach-Object { $_.Name }"`) do (
    set "FULL_GPU_NAME=%%i"
    goto :ProcessGPU
)


echo ERROR: No AMD GPU detected.
pause
exit /b 1

:ProcessGPU
echo Detected GPU: %FULL_GPU_NAME%


set "REQUIREMENTS_FILE=requirements-rocm7-nightly.txt"

REM --- Setting up the virtual environment ---
:InstallDependencies
echo - Setting up the virtual environment
REM Set paths
set "VIRTUAL_ENV=.env"
set "PYTHON_EXE=python.exe"

REM Check if Python is available
where %PYTHON_EXE% >nul 2>&1
if %errorlevel% neq 0 (
  echo Python is not found in PATH. Please install Python 3 first.
  pause
  exit /b 1
)

REM Create virtual environment if it doesn't exist
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

echo.
echo Installing requirements from: %REQUIREMENTS_FILE%
@REM echo - Installing the requirements from %REQUIREMENTS_FILE%
echo - Updating pip package
%PYTHON_EXE% -m pip install --upgrade pip --no-cache-dir --no-warn-script-location --trusted-host http://mirrors.aliyun.com/pypi/simple/
if %errorlevel% neq 0 (
  echo Failed to update pip
  pause
  exit /b 1
)
%PYTHON_EXE% -m pip install -r %REQUIREMENTS_FILE% --no-cache-dir --no-warn-script-location --trusted-host http://mirrors.aliyun.com/pypi/simple/
%PYTHON_EXE% -m pip install -r requirements.txt --no-cache-dir --no-warn-script-location --trusted-host http://mirrors.aliyun.com/pypi/simple/
%PYTHON_EXE% -m pip install "triton-windows<3.6" --no-cache-dir --no-warn-script-location --trusted-host http://mirrors.aliyun.com/pypi/simple/
%PYTHON_EXE% -m pip install wincam==1.0.14 --no-cache-dir --no-warn-script-location --trusted-host http://mirrors.aliyun.com/pypi/simple/
%PYTHON_EXE% -m pip install windows-capture==2.0.0 --no-cache-dir --no-warn-script-location --trusted-host http://mirrors.aliyun.com/pypi/simple/
%PYTHON_EXE% -m pip install https://ghfast.top/https://github.com/lc700x/wc_rocm/releases/download/v0.1.2/wc_rocm-0.1.2-cp310-abi3-win_amd64.whl --no-cache-dir
if %errorlevel% neq 0 (
    echo Failed to install requirements
    pause
    exit /b 1
)

echo Python environment deployed successfully.
echo .


echo --- (Optional) Running Triton Installation test ---

%PYTHON_EXE% -c "import torch, triton, triton.language as tl;"

if %errorlevel% neq 0 (
    echo.
    echo Triton/ROCm may not be working correctly.  
    pause
    exit /b 1
)

echo Import PASSED: Triton is installed correctly.
echo To enable torch.compile on AMD ROCm7 supported GPUs, you may have to install vs_buildtools https://aka.ms/vs/17/release/vs_buildtools.exe and select the "Desktop development with C++" to install (~6GB). OR you can just run with the torch.compile unchecked. 
pause
