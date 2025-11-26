@echo off
echo --- Desktop2Stereo Installer (With ROCm for AMD 7000/9000/AI/AI Max series GPUs.) ---


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

REM --- Grab 3rd and 4th tokens from the GPU name ---
for /f "tokens=3,4 delims= " %%a in ("%FULL_GPU_NAME%") do (
  set "TOKEN3=%%a"
  set "TOKEN4=%%b"
)

REM --- Pick which token contains digits (the real model token) ---
set "MODEL_WORD="
for %%w in (%TOKEN3% %TOKEN4%) do (
  echo %%w | findstr "[0-9]" >nul && (
    set "MODEL_WORD=%%w"
    goto :GotModelWord
  )
)
:GotModelWord
if "%MODEL_WORD%"=="" (
  REM fallback: if neither token had digits, use token3
  set "MODEL_WORD=%TOKEN3%"
)

REM --- Strip letters/punctuation, keep digit sequence ---
set "GPU_MODEL="
for /f "tokens=1* delims=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_. " %%g in ("%MODEL_WORD%") do (
  if not "%%g"=="" (
    set "GPU_MODEL=%%g"
  ) else (
    set "GPU_MODEL=%%h"
  )
)

REM sanitize/truncate spaces
for /f "delims= " %%x in ("%GPU_MODEL%") do set "GPU_MODEL=%%x"

@REM echo Extracted numeric model: %GPU_MODEL%

REM --- Map GPU models to requirement files ---
set "AMD_MODELS_9000=9060 9070 W9700"
set "AMD_MODELS_7000=W7900X W7800 W7700 W7600 W7500 7900 7900M 7800 7800M 7700 7700S 7600 7600S 7600M 7650 780M 760M 740M"
set "AMD_MODELS_8000S=8060S 8050S 8040S"
set "AMD_MODELS_800M=890M 880M 860M 840M"

call :CheckModel "%GPU_MODEL%" AMD_MODELS_9000 requirements-rocm7-9000.txt
if %errorlevel% equ 0 goto :InstallDependencies

call :CheckModel "%GPU_MODEL%" AMD_MODELS_7000 requirements-rocm7-7000.txt
if %errorlevel% equ 0 goto :InstallDependencies

call :CheckModel "%GPU_MODEL%" AMD_MODELS_8000S requirements-rocm7-8000S.txt
if %errorlevel% equ 0 goto :InstallDependencies

call :CheckModel "%GPU_MODEL%" AMD_MODELS_800M requirements-rocm7-800M.txt
if %errorlevel% equ 0 goto :InstallDependencies

echo.
echo Failed to automatically detect a compatible ROCm requirements file for GPU: %FULL_GPU_NAME%.
echo.
echo - Manual Selection Menu -
echo Select a requirements file to install:
echo 1. requirements-rocm7-9000.txt
echo 2. requirements-rocm7-7000.txt
echo 3. requirements-rocm7-8000S.txt
echo 4. requirements-rocm7-800M.txt
echo 5. Cancel
echo.
set /p USER_CHOICE="Enter your choice (1-5): "
if "%USER_CHOICE%"=="1" set "REQUIREMENTS_FILE=requirements-rocm7-9000.txt" & goto :InstallDependencies
if "%USER_CHOICE%"=="2" set "REQUIREMENTS_FILE=requirements-rocm7-7000.txt" & goto :InstallDependencies
if "%USER_CHOICE%"=="3" set "REQUIREMENTS_FILE=requirements-rocm7-8000S.txt" & goto :InstallDependencies
if "%USER_CHOICE%"=="4" set "REQUIREMENTS_FILE=requirements-rocm7-800M.txt" & goto :InstallDependencies
if "%USER_CHOICE%"=="5" (
  echo Installation cancelled by user.
  pause
  exit /b 1
)
echo Invalid selection. Exiting.
pause
exit /b 1

:CheckModel
REM %1 = model to check, %2 = name of the model-list variable, %3 = requirements filename
REM Expand the variable name passed as %2 into MODEL_LIST
call set "MODEL_LIST=%%%2%%"

for %%a in (%MODEL_LIST%) do (
  if /I "%~1"=="%%a" (
    set "REQUIREMENTS_FILE=%~3"
    echo Installing %REQUIREMENTS_FILE% ...
    exit /b 0
  )
)

REM --- Setting up the virtual environment ---
:InstallDependencies
echo - Setting up the virtual environment
REM Set paths
set "VIRTUAL_ENV=.env"
set "PYTHON_EXE=python.exe"

REM Check if Python is available
where %PYTHON_EXE% >nul 2>&1
if %errorlevel% neq 0 (
  echo Python is not found in PATH. Please install Python 3.11 first.
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
%PYTHON_EXE% -m pip install wincam==1.0.14 --no-cache-dir --no-warn-script-location --trusted-host http://mirrors.aliyun.com/pypi/simple/
%PYTHON_EXE% -m pip install windows-capture==1.5.0 --no-cache-dir --no-warn-script-location --trusted-host http://mirrors.aliyun.com/pypi/simple/
if %errorlevel% neq 0 (
    echo Failed to install requirements
    pause
    exit /b 1
)

echo Python environment deployed successfully.
pause
exit /b 0
