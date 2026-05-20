@echo off
setlocal enabledelayedexpansion

:: Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

:: Define the Desktop2Stereo subfolder
set "APP_DIR=%SCRIPT_DIR%\Desktop2Stereo"

if not exist "%APP_DIR%" (
    echo [Error] Desktop2Stereo folder not found at %APP_DIR%
    echo Please make sure this script is in the same folder as run_windows.bat
    pause
    exit /b 1
)

:: Change to the app directory for the download/extraction
cd /d "%APP_DIR%"

:: Download and extract the update (same as before)
set REPO_OWNER=lc700x
set REPO_NAME=desktop2stereo
set BRANCH=update
set ZIP_FILE=%REPO_NAME%-%BRANCH%.zip
set DOWNLOAD_URL=https://github.com/%REPO_OWNER%/%REPO_NAME%/archive/refs/heads/%BRANCH%.zip

echo Downloading update from %DOWNLOAD_URL%
curl -L -o "%ZIP_FILE%" "%DOWNLOAD_URL%"
if errorlevel 1 (
    echo Failed to download update.
    pause
    exit /b 1
)

powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath './temp_extract' -Force"
for /d %%F in (temp_extract\*) do xcopy "%%F\*" ".\" /E /H /Y
rmdir /S /Q temp_extract
rmdir /S /Q rtmp\mac 2>nul
rmdir /S /Q rtmp\linux 2>nul
del /F /Q update_mac_linux 2>nul
del /F /Q update.bat 2>nul
del /F /Q main 2>nul
del "%ZIP_FILE%"

echo Latest Desktop2Stereo downloaded and extracted.

:: ------------------------------------------------------------
:: Now update Python environment
:: ------------------------------------------------------------

:: Check if python3\python.exe exists
set "PYTHON_EXE=%APP_DIR%\python3\python.exe"
if not exist "!PYTHON_EXE!" (
    echo [Error] Python not found at !PYTHON_EXE!
    echo The Desktop2Stereo folder seems incomplete.
    echo Please run the full installer first, or re-download the application.
    pause
    exit /b 1
)

echo.
echo [Desktop2Stereo Patch]
echo - Updating pip using: !PYTHON_EXE!

"!PYTHON_EXE!" -m pip install --upgrade pip --no-cache-dir --no-warn-script-location --trusted-host mirrors.aliyun.com
if errorlevel 1 (
    echo Failed to update pip
    pause
    exit /b 1
)

echo.
echo - Installing new requirements
"!PYTHON_EXE!" -m pip uninstall pyaudio -y 2>nul
"!PYTHON_EXE!" -m pip install -r "%APP_DIR%\requirements.txt" --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/
if errorlevel 1 (
    echo Failed to install requirements
    pause
    exit /b 1
)

echo Python environment updated successfully.
pause
exit /b 0