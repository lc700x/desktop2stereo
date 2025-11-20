@echo off
echo --- GPU Detection for Installer ---
@REM Function to detect GPU and process substring
echo Detecting System GPU...

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

echo Detected GPU Model: %GPU_MODEL%

@REM Map GPU models to requirement files
set "AMD_MODELS_9000=;9060;9070;"
set "AMD_MODELS_7000=;W7900X;W7800;W7700;W7600;W7500;7900;7900M;7800;7800M;7700;7700S;7600;7600S;7600M;7650;780M;760M;740M;"
set "AMD_MODELS_8000S=;8060S;8050S;8040S;"
set "AMD_MODELS_800M=;890M;880M;860M;840M;"

call :CheckModel %GPU_MODEL% AMD_MODELS_9000 requirements-rocm7-9000.txt
if %errorlevel% equ 0 goto :InstallDependencies

call :CheckModel %GPU_MODEL% AMD_MODELS_7000 requirements-rocm7-7000.txt
if %errorlevel% equ 0 goto :InstallDependencies

call :CheckModel %GPU_MODEL% AMD_MODELS_8000S requirements-rocm7-8000S.txt
if %errorlevel% equ 0 goto :InstallDependencies

call :CheckModel %GPU_MODEL% AMD_MODELS_800M requirements-rocm7-800M.txt
if %errorlevel% equ 0 goto :InstallDependencies

echo Failed to automatically detect a compatible ROCm requirements file for GPU: %GPU_MODEL%.
echo You may need to update the script with additional GPU models.
pause
exit /b 1

:CheckModel
@REM Check if GPU_MODEL exists in the given list
for %%a in (%2) do (
    if "%1" equ "%%a" (
        set "REQUIREMENTS_FILE=%3"
        exit /b 0
    )
)
exit /b 1