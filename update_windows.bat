@echo off
setlocal

:: Set variables
set REPO_OWNER=lc700x
set REPO_NAME=desktop2stereo
set BRANCH=update
set ZIP_FILE=%REPO_NAME%-%BRANCH%.zip

:: Construct the download URL
set DOWNLOAD_URL=https://github.com/%REPO_OWNER%/%REPO_NAME%/archive/refs/heads/%BRANCH%.zip

:: Download the ZIP archive
curl -L -o "%ZIP_FILE%" "%DOWNLOAD_URL%"

:: Extract using PowerShell's Expand-Archive (safe and reliable)
powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath './temp_extract' -Force"

:: Move contents from nested folder to current directory
for /d %%F in (temp_extract\*) do xcopy "%%F\*" ".\" /E /H /Y

:: Clean up
rmdir /S /Q temp_extract
:: Remove unnecessary platform folders
rmdir /S /Q rtmp\mac
rmdir /S /Q rtmp\linux
del /F /Q update_mac_linux
del /F /Q update.bat
del "%ZIP_FILE%"

echo Latest Desktop2Stereo downloaded and extracted to current folder.
endlocal
@REM Set paths
Set "PYTHON_EXE=.\Python311\python.exe"


@REM Update pip
echo [Desktop2Stereo Patch]
echo - Updating the pip package
%PYTHON_EXE% -m pip install --upgrade pip --no-cache-dir --no-warn-script-location --trusted-host http://mirrors.aliyun.com/pypi/simple/
if %errorlevel% neq 0 (
    echo Failed to update pip
    pause
    exit /b 1
)

@REM Install new requirements
echo.
echo - Installing new requirements
%PYTHON_EXE% -m pip install -r requirements.txt --no-cache-dir --no-warn-script-location --trusted-host  http://mirrors.aliyun.com/pypi/simple/
if %errorlevel% neq 0 (
    echo Failed to install requirements
    pause
    exit /b 1
)

echo Python environment updated successfully.
pause
