@echo off
:: ============================================================
:: Request administrator privilege (UAC Elevation)
:: ============================================================
>nul 2>&1 "%SystemRoot%\system32\cacls.exe" "%SystemRoot%\system32\config\system"
if %errorlevel% neq 0 (
    echo Requesting administrative privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:: Ensure script runs from its own directory
cd /d "%~dp0"

setlocal

REM Step 1: Run the installer and wait until it finishes
echo Running installer...
start /wait "" ".\Virtual.Display.Driver-v24.12.24-setup-x64.exe"

REM Step 2: Copy VirtualDisplayDriver folder to C:\
set SOURCE_FOLDER=.\VirtualDisplayDriver
set DEST_FOLDER=C:\VirtualDisplayDriver

if exist "%SOURCE_FOLDER%" (
    echo Copying VirtualDisplayDriver folder to C:\
    xcopy "%SOURCE_FOLDER%" "%DEST_FOLDER%" /E /I /Y
) else (
    echo Source folder .\VirtualDisplayDriver not found.
)

REM Step 3: Replace x64 folder if target exists
set TARGET_FOLDER=C:\Program Files\Virtual Desktop Streamer\MonitorDriver\x64
set SOURCE_X64=.\x64

if exist "%TARGET_FOLDER%" (
    if exist "%SOURCE_X64%" (
        echo Replacing existing x64 folder...
        rmdir /S /Q "%TARGET_FOLDER%"
        xcopy "%SOURCE_X64%" "%TARGET_FOLDER%" /E /I /Y
    ) else (
        echo Source .\x64 folder not found.
    )
) else (
    echo Target x64 folder does not exist, skipping replacement.
)

REM Step 4: Remove Virtual Desktop Display Driver adapter
echo Removing Virtual Desktop Display Driver adapter...
powershell -Command "Get-PnpDevice -Class Display | Where-Object { $_.FriendlyName -like '*Virtual*Desktop*' } | ForEach-Object { pnputil /remove-device $_.InstanceId /subtree /force 2>$null; Write-Host \"Removed: $($_.FriendlyName)\" }"
if %errorlevel% equ 0 (
    echo Virtual Desktop Display adapter removed successfully.
) else (
    echo Note: No Virtual Desktop Display adapter found or could not be removed.
)

echo Removing SudoMaker Virtual Display Driver adapter...
powershell -Command "Get-PnpDevice -Class Display | Where-Object { $_.FriendlyName -like '*SudoMaker*' } | ForEach-Object { pnputil /remove-device $_.InstanceId /subtree /force 2>$null; Write-Host \"Removed: $($_.FriendlyName)\" }"
if %errorlevel% equ 0 (
    echo SudoMaker Virtual Display Driver adapter removed successfully.
) else (
    echo Note: No SudoMaker Virtual Display Driver adapter found or could not be removed.
)

echo Done.
endlocal
pause