@echo off
echo --- Desktop2Stereo Installer (With ROCm for AMD 7000/9000/AI/AI Max series GPUs.) ---

@REM Setting up the virtual environment
echo - Setting up the virtual environment

@REM Set paths
Set "PYTHON_EXE=.\Python311\python.exe"

@REM Use `wmic` to get GPU name (Windows specific)
for /f "skip=1 tokens=*" %%i in ('wmic path win32_videocontroller get name') do (
    set "FULL_GPU_NAME=%%i"
    goto :ProcessGPUName
)

:ProcessGPUName
@REM Split the GPU name into a list and extract the third word as the model
for /f "tokens=3 delims= " %%a in ("%FULL_GPU_NAME%") do (
    set "GPU_MODEL=%%a"
)

@echo off
setlocal enabledelayedexpansion

echo Detected GPU: %FULL_GPU_NAME%

@REM Map GPU models to requirement files
set "AMD_MODELS_9000=9060;9070;W9700"
set "AMD_MODELS_7000=W7900X;W7800;W7700;W7600;W7500;7900;7900M;7800;7800M;7700;7700S;7600;7600S;7600M;7650;780M;760M;740M"
set "AMD_MODELS_8000S=8060S;8050S;8040S"
set "AMD_MODELS_800M=890M;880M;860M;840M"

call :CheckModel "%GPU_MODEL%" AMD_MODELS_9000 requirements-rocm7-9000.txt
if %errorlevel% equ 0 goto :InstallDependencies

call :CheckModel "%GPU_MODEL%" AMD_MODELS_7000 requirements-rocm7-7000.txt
if %errorlevel% equ 0 goto :InstallDependencies

call :CheckModel "%GPU_MODEL%" AMD_MODELS_8000S requirements-rocm7-8000S.txt
if %errorlevel% equ 0 goto :InstallDependencies

call :CheckModel "%GPU_MODEL%" AMD_MODELS_800M requirements-rocm7-800M.txt
if %errorlevel% equ 0 goto :InstallDependencies

echo Failed to automatically detect a compatible ROCm requirements file for GPU: %FULL_GPU_NAME%.
echo You may need to update the script with additional GPU models.
pause
exit /b 1

:CheckModel
@REM Expand the variable name passed as %2
set "MODEL_LIST=!%2!"
for %%a in (%MODEL_LIST%) do (
    if /I "%~1"=="%%a" (
        set "REQUIREMENTS_FILE=%~3"
        echo Installing !REQUIREMENTS_FILE! ...
        exit /b 0
    )
)
exit /b 1

:InstallDependencies
@REM Update pip
echo - Updating the pip package
%PYTHON_EXE% -m pip install --upgrade pip --no-cache-dir --no-warn-script-location --trusted-host http://mirrors.aliyun.com/pypi/simple/
if %errorlevel% neq 0 (
    echo Failed to update pip
    pause
    exit /b 1
)

@REM Install requirements
echo.
@REM echo - Installing the requirements from %REQUIREMENTS_FILE%
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
