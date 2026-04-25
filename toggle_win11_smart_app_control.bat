@echo off
:: Toggle Smart App Control in Windows 11
:: This script will auto-request Administrator privileges

:: --- Auto Elevation Section ---
:: Check for admin rights
net session >nul 2>&1
if %errorLevel% == 0 (
    goto :gotAdmin
) else (
    echo Requesting Administrator privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:gotAdmin
set KEY="HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy"
set VALUE=SmartAppControl

:: Read current state
for /f "tokens=3" %%A in ('reg query %KEY% /v %VALUE% 2^>nul') do set CURRENT=%%A

echo Current Smart App Control value: %CURRENT%

if "%CURRENT%"=="0x0" (
    echo Turning Smart App Control ON...
    reg add %KEY% /v %VALUE% /t REG_DWORD /d 1 /f
) else (
    echo Turning Smart App Control OFF...
    reg add %KEY% /v %VALUE% /t REG_DWORD /d 0 /f
)

echo Done. Please restart your computer for changes to take effect.
pause
