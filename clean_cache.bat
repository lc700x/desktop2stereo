@echo off
setlocal enabledelayedexpansion

echo ============================================
echo Cache Directory Cleanup Script
echo ============================================
echo.
echo This script will remove the following directories:
echo - ZLUDA ComputeCache
echo - MIOpen cache
echo - Triton cache
echo - TorchInductor temp files
echo - Torch/Triton/MIOpen/ZLUDA related cache subdirectories
echo - ComfyUI Triton and Inductor directories
echo.

echo.
echo Starting cleanup...
echo.

REM ZLUDA ComputeCache
set "ZLUDA_CACHE=%USERPROFILE%\AppData\Local\ZLUDA\ComputeCache"
if exist "!ZLUDA_CACHE!" (
    echo Removing ZLUDA ComputeCache...
    rd /s /q "!ZLUDA_CACHE!" 2>nul
    if exist "!ZLUDA_CACHE!" (
        echo   Warning: Could not remove !ZLUDA_CACHE!
    ) else (
        echo   Successfully removed !ZLUDA_CACHE!
    )
) else (
    echo   ZLUDA ComputeCache not found: !ZLUDA_CACHE!
)

REM MIOpen cache
set "MIOPEN_CACHE=%USERPROFILE%\.miopen"
if exist "!MIOPEN_CACHE!" (
    echo Removing MIOpen cache...
    rd /s /q "!MIOPEN_CACHE!" 2>nul
    if exist "!MIOPEN_CACHE!" (
        echo   Warning: Could not remove !MIOPEN_CACHE!
    ) else (
        echo   Successfully removed !MIOPEN_CACHE!
    )
) else (
    echo   MIOpen cache not found: !MIOPEN_CACHE!
)

REM Triton cache
set "TRITON_CACHE=%USERPROFILE%\.triton"
if exist "!TRITON_CACHE!" (
    echo Removing Triton cache...
    rd /s /q "!TRITON_CACHE!" 2>nul
    if exist "!TRITON_CACHE!" (
        echo   Warning: Could not remove !TRITON_CACHE!
    ) else (
        echo   Successfully removed !TRITON_CACHE!
    )
) else (
    echo   Triton cache not found: !TRITON_CACHE!
)

REM TorchInductor temp files
set "TORCH_TEMP=%USERPROFILE%\AppData\Local\Temp"
echo Removing TorchInductor temp files...
for /d %%i in ("!TORCH_TEMP!\torchinductor_*") do (
    echo   Removing: %%i
    rd /s /q "%%i" 2>nul
)

REM Cache subdirectories related to torch, triton, miopen, zluda
set "USER_CACHE=%USERPROFILE%\.cache"
if exist "!USER_CACHE!" (
    echo Removing cache subdirectories related to torch, triton, miopen, zluda...
    for /d %%i in ("!USER_CACHE!\*torch*" "!USER_CACHE!\*triton*" "!USER_CACHE!\*miopen*" "!USER_CACHE!\*zluda*") do (
        if exist "%%i" (
            echo   Removing: %%i
            rd /s /q "%%i" 2>nul
        )
    )
) else (
    echo   User cache directory not found: !USER_CACHE!
)

echo.
echo ============================================
echo Cleanup completed!
echo ============================================
echo.
echo If you encountered any warnings above, you may need to:
echo - Close any running applications that might be using these directories
echo - Run this script as Administrator
echo - Manually delete the directories that couldn't be removed
echo.

pause
