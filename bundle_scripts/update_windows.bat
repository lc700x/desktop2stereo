@echo off
chcp 65001 >nul
:: Desktop2Stereo Updater
:: Extracts the update zip into .\Desktop2Stereo\ and runs pip.
:: Usage: double-click to run.

setlocal enabledelayedexpansion

set "APP_DIR=.\Desktop2Stereo"
set "PYTHON_EXE=%APP_DIR%\python3\python.exe"
set "PYTHONPATH=%APP_DIR%\python3\python.exe;%PYTHONPATH%"

:: ═══════════════════════════════════════════════
:: Step 1 - Environment checks
:: ═══════════════════════════════════════════════

echo [Check] Running directory...

:: 1a - verify Desktop2Stereo subdirectory exists
if not exist "%APP_DIR%\" (
    echo [Error] Desktop2Stereo subdirectory not found.
    echo         Please place this script in the parent directory of Desktop2Stereo.
    pause
    exit /b 1
)

:: 1b - warn if bundled python is missing
if not exist "%APP_DIR%\python3\python.exe" (
    echo [Warning] %APP_DIR%\python3\python.exe does not exist. The application may not start after update.
)

:: 1c - check curl is available
where curl >nul 2>&1
if %errorlevel% neq 0 (
    echo [Error] curl is not installed on this system. Cannot download update.
    echo         Please install curl from https://curl.se/windows/ and retry.
    pause
    exit /b 1
)

:: 1d - network reachability test
echo [Check] Network connectivity...
ping -n 1 -w 3000 github.com >nul 2>&1
if %errorlevel% neq 0 (
    echo [Error] Cannot reach github.com.
    echo         Please check your network connection or proxy settings.
    pause
    exit /b 1
)

:: ═══════════════════════════════════════════════
:: Step 2 - Download update zip
:: ═══════════════════════════════════════════════

set "REPO_OWNER=lc700x"
set "REPO_NAME=desktop2stereo"
set "BRANCH=update"
set "ZIP_FILE=%REPO_NAME%-%BRANCH%.zip"
set "DOWNLOAD_URL=https://github.com/%REPO_OWNER%/%REPO_NAME%/archive/refs/heads/%BRANCH%.zip"
set "TEMP_EXTRACT=temp_extract_%RANDOM%"

echo [Download] Fetching update package...
curl -L --progress-bar -o "%ZIP_FILE%" "%DOWNLOAD_URL%"
if %errorlevel% neq 0 (
    echo [Error] Download failed: %DOWNLOAD_URL%
    del "%ZIP_FILE%" >nul 2>&1
    pause
    exit /b 1
)

if not exist "%ZIP_FILE%" (
    echo [Error] Downloaded file not found.
    pause
    exit /b 1
)

:: reject HTML error page disguised as zip
findstr /M "^<html" "%ZIP_FILE%" >nul 2>&1
if %errorlevel% equ 0 (
    echo [Error] Downloaded file is not a valid update package (repository or branch name may be wrong).
    del "%ZIP_FILE%"
    pause
    exit /b 1
)

:: ═══════════════════════════════════════════════
:: Step 3 - Extract
:: ═══════════════════════════════════════════════

echo [Extract] Unpacking update package...

powershell -Command "try { Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%TEMP_EXTRACT%' -Force; exit 0 } catch { Write-Error $_.Exception.Message; exit 1 }"
if %errorlevel% neq 0 (
    echo [Error] Extraction failed. The ZIP file may be corrupted.
    del "%ZIP_FILE%"
    rmdir /S /Q "%TEMP_EXTRACT%" >nul 2>&1
    pause
    exit /b 1
)

:: find the automatically-nested folder (repo-branch pattern)
set "EXTRACTED_DIR="
for /d %%D in ("%TEMP_EXTRACT%\*") do set "EXTRACTED_DIR=%%D"

if not defined EXTRACTED_DIR (
    echo [Error] No content folder found after extraction.
    rmdir /S /Q "%TEMP_EXTRACT%" >nul 2>&1
    del "%ZIP_FILE%"
    pause
    exit /b 1
)

:: ═══════════════════════════════════════════════
:: Step 4 - Backup user data
:: ═══════════════════════════════════════════════

echo [Backup] Preserving user configuration...

set "BACKUP_KEEP=%TEMP%\desktop2stereo_keep_%RANDOM%"
mkdir "%BACKUP_KEEP%" >nul 2>&1

:: settings.yaml - user configuration
if exist "%APP_DIR%\settings.yaml" (
    copy "%APP_DIR%\settings.yaml" "%BACKUP_KEEP%\" >nul 2>&1
    echo   - preserving settings.yaml
)

:: gui_layout.json - GUI layout overrides
if exist "%APP_DIR%\gui_layout.json" (
    copy "%APP_DIR%\gui_layout.json" "%BACKUP_KEEP%\" >nul 2>&1
    echo   - preserving gui_layout.json
)

:: backup/ - versioned backups
if exist "%APP_DIR%\backup\" (
    xcopy "%APP_DIR%\backup\*" "%BACKUP_KEEP%\backup\" /E /H /Y /Q >nul 2>&1
    echo   - preserving backup/
)

:: controllers/ - calibration profiles per brand
if exist "%APP_DIR%\controllers\" (
    xcopy "%APP_DIR%\controllers\*" "%BACKUP_KEEP%\controllers\" /E /H /Y /Q >nul 2>&1
    echo   - preserving controllers/
)

:: font.ttf - bundled font (large, skip re-download)
if exist "%APP_DIR%\font.ttf" (
    copy "%APP_DIR%\font.ttf" "%BACKUP_KEEP%\" >nul 2>&1
    echo   - preserving font.ttf
)

:: ═══════════════════════════════════════════════
:: Step 5 - Copy updates into Desktop2Stereo
:: ═══════════════════════════════════════════════

echo [Update] Copying new files to %APP_DIR%...
xcopy "%EXTRACTED_DIR%\*" "%APP_DIR%\" /E /H /Y /Q
if %errorlevel% neq 0 (
    echo [Error] File copy failed.
    if exist "%BACKUP_KEEP%" (
        xcopy "%BACKUP_KEEP%\*" "%APP_DIR%\" /E /H /Y /Q >nul 2>&1
    )
    rmdir /S /Q "%BACKUP_KEEP%" >nul 2>&1
    pause
    exit /b 1
)

:: ═══════════════════════════════════════════════
:: Step 6 - Restore user data
:: ═══════════════════════════════════════════════

echo [Restore] Restoring user configuration...
if exist "%BACKUP_KEEP%\settings.yaml" (
    copy "%BACKUP_KEEP%\settings.yaml" "%APP_DIR%\settings.yaml" /Y >nul 2>&1
    echo   - restored settings.yaml
)
if exist "%BACKUP_KEEP%\gui_layout.json" (
    copy "%BACKUP_KEEP%\gui_layout.json" "%APP_DIR%\gui_layout.json" /Y >nul 2>&1
    echo   - restored gui_layout.json
)
if exist "%BACKUP_KEEP%\backup\" (
    xcopy "%BACKUP_KEEP%\backup\*" "%APP_DIR%\backup\" /E /H /Y /Q >nul 2>&1
)
if exist "%BACKUP_KEEP%\controllers\" (
    xcopy "%BACKUP_KEEP%\controllers\*" "%APP_DIR%\controllers\" /E /H /Y /Q >nul 2>&1
)
if exist "%BACKUP_KEEP%\font.ttf" (
    copy "%BACKUP_KEEP%\font.ttf" "%APP_DIR%\font.ttf" /Y >nul 2>&1
)

:: clean up backup area
rmdir /S /Q "%BACKUP_KEEP%" >nul 2>&1

:: ═══════════════════════════════════════════════
:: Step 7 - Clean temporary files
:: ═══════════════════════════════════════════════

echo [Cleanup] Removing temporary files...
rmdir /S /Q "%TEMP_EXTRACT%" >nul 2>&1
del "%ZIP_FILE%" >nul 2>&1

:: remove platform folders not needed on Windows
if exist "%APP_DIR%\rtmp\mac" rmdir /S /Q "%APP_DIR%\rtmp\mac" >nul 2>&1
if exist "%APP_DIR%\rtmp\linux" rmdir /S /Q "%APP_DIR%\rtmp\linux" >nul 2>&1
if exist "%APP_DIR%\update_mac_linux.bat" del /F /Q "%APP_DIR%\update_mac_linux.bat" >nul 2>&1
if exist "%APP_DIR%\update.bat" del /F /Q "%APP_DIR%\update.bat" >nul 2>&1

:: ═══════════════════════════════════════════════
:: Step 8 - Update Python environment
:: ═══════════════════════════════════════════════

echo [Python] Updating Python dependencies...

if not exist "%PYTHON_EXE%" (
    echo [Warning] %PYTHON_EXE% does not exist. Skipping Python environment update.
) else (
    :: upgrade pip
    echo   - upgrading pip ...
    %PYTHON_EXE% -m pip install --upgrade pip --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
    if %errorlevel% neq 0 (
        echo [Warning] pip upgrade failed. Continuing...
    ) else (
        echo   - pip upgrade completed
    )

    :: uninstall pyaudio - known compatibility issue
    echo   - uninstalling old pyaudio ...
    %PYTHON_EXE% -m pip uninstall pyaudio -y >nul 2>&1

    :: install requirements
    if exist "%APP_DIR%\requirements.txt" (
        echo   - installing dependencies ...
        %PYTHON_EXE% -m pip install -r "%APP_DIR%\requirements.txt" --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
        if %errorlevel% neq 0 (
            echo [Warning] Some dependencies may have failed to install. Please check your network.
        ) else (
            echo   - dependencies installed
        )
    ) else (
        echo [Warning] %APP_DIR%\requirements.txt not found. Skipping dependency installation.
    )
)

:: ═══════════════════════════════════════════════
:: Done
:: ═══════════════════════════════════════════════

echo.
echo ════════════════════════════════════════════
echo   Desktop2Stereo update completed!
echo   All files have been updated in the %APP_DIR% directory.
echo ════════════════════════════════════════════
echo.
pause
endlocal