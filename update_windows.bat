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
del "%ZIP_FILE%"

echo Latest Desktop2Stereo downloaded and extracted to current folder.
endlocal
pause
