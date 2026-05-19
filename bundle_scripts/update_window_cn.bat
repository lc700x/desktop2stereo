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

echo [检查] 运行目录...

:: 1a - verify Desktop2Stereo subdirectory exists
if not exist "%APP_DIR%\" (
    echo [Error] 未找到 Desktop2Stereo 子目录。
    echo         请将本脚本放在 Desktop2Stereo 父目录中运行。
    pause
    exit /b 1
)

:: 1b - warn if bundled python is missing
if not exist "%APP_DIR%\python3\python.exe" (
    echo [Warning] %APP_DIR%\python3\python.exe 不存在, 更新后可能无法启动。
)

:: 1c - check curl is available
where curl >nul 2>&1
if %errorlevel% neq 0 (
    echo [Error] 系统未安装 curl, 无法下载更新。
    echo         请从 https://curl.se/windows/ 安装后重试。
    pause
    exit /b 1
)

:: 1d - network reachability test
echo [检查] 网络连接...
ping -n 1 -w 3000 github.com >nul 2>&1
if %errorlevel% neq 0 (
    echo [Error] 无法访问 github.com。
    echo         请检查网络连接或代理设置。
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

echo [下载] 正在获取更新包...
curl -L --progress-bar -o "%ZIP_FILE%" "%DOWNLOAD_URL%"
if %errorlevel% neq 0 (
    echo [Error] 下载失败: %DOWNLOAD_URL%
    del "%ZIP_FILE%" >nul 2>&1
    pause
    exit /b 1
)

if not exist "%ZIP_FILE%" (
    echo [Error] 下载文件未找到。
    pause
    exit /b 1
)

:: reject HTML error page disguised as zip
findstr /M "^<html" "%ZIP_FILE%" >nul 2>&1
if %errorlevel% equ 0 (
    echo [Error] 下载的文件不是有效的更新包 (可能仓库或分支名错误)。
    del "%ZIP_FILE%"
    pause
    exit /b 1
)

:: ═══════════════════════════════════════════════
:: Step 3 - Extract
:: ═══════════════════════════════════════════════

echo [解压] 正在解压更新包...

powershell -Command "try { Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%TEMP_EXTRACT%' -Force; exit 0 } catch { Write-Error $_.Exception.Message; exit 1 }"
if %errorlevel% neq 0 (
    echo [Error] 解压失败。ZIP 文件可能已损坏。
    del "%ZIP_FILE%"
    rmdir /S /Q "%TEMP_EXTRACT%" >nul 2>&1
    pause
    exit /b 1
)

:: find the automatically-nested folder (repo-branch pattern)
set "EXTRACTED_DIR="
for /d %%D in ("%TEMP_EXTRACT%\*") do set "EXTRACTED_DIR=%%D"

if not defined EXTRACTED_DIR (
    echo [Error] 解压后未找到内容文件夹。
    rmdir /S /Q "%TEMP_EXTRACT%" >nul 2>&1
    del "%ZIP_FILE%"
    pause
    exit /b 1
)

:: ═══════════════════════════════════════════════
:: Step 4 - Backup user data
:: ═══════════════════════════════════════════════

echo [备份] 保留用户配置...

set "BACKUP_KEEP=%TEMP%\desktop2stereo_keep_%RANDOM%"
mkdir "%BACKUP_KEEP%" >nul 2>&1

:: settings.yaml - user configuration
if exist "%APP_DIR%\settings.yaml" (
    copy "%APP_DIR%\settings.yaml" "%BACKUP_KEEP%\" >nul 2>&1
    echo   - 保留 settings.yaml
)

:: gui_layout.json - GUI layout overrides
if exist "%APP_DIR%\gui_layout.json" (
    copy "%APP_DIR%\gui_layout.json" "%BACKUP_KEEP%\" >nul 2>&1
    echo   - 保留 gui_layout.json
)

:: backup/ - versioned backups
if exist "%APP_DIR%\backup\" (
    xcopy "%APP_DIR%\backup\*" "%BACKUP_KEEP%\backup\" /E /H /Y /Q >nul 2>&1
    echo   - 保留 backup/
)

:: controllers/ - calibration profiles per brand
if exist "%APP_DIR%\controllers\" (
    xcopy "%APP_DIR%\controllers\*" "%BACKUP_KEEP%\controllers\" /E /H /Y /Q >nul 2>&1
    echo   - 保留 controllers/
)

:: font.ttf - bundled font (large, skip re-download)
if exist "%APP_DIR%\font.ttf" (
    copy "%APP_DIR%\font.ttf" "%BACKUP_KEEP%\" >nul 2>&1
    echo   - 保留 font.ttf
)

:: ═══════════════════════════════════════════════
:: Step 5 - Copy updates into Desktop2Stereo
:: ═══════════════════════════════════════════════

echo [更新] 正在复制更新文件到 %APP_DIR%...
xcopy "%EXTRACTED_DIR%\*" "%APP_DIR%\" /E /H /Y /Q
if %errorlevel% neq 0 (
    echo [Error] 文件复制失败。
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

echo [恢复] 还原用户配置...
if exist "%BACKUP_KEEP%\settings.yaml" (
    copy "%BACKUP_KEEP%\settings.yaml" "%APP_DIR%\settings.yaml" /Y >nul 2>&1
    echo   - 已恢复 settings.yaml
)
if exist "%BACKUP_KEEP%\gui_layout.json" (
    copy "%BACKUP_KEEP%\gui_layout.json" "%APP_DIR%\gui_layout.json" /Y >nul 2>&1
    echo   - 已恢复 gui_layout.json
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

echo [清理] 移除临时文件...
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

echo [Python] 更新 Python 依赖...

if not exist "%PYTHON_EXE%" (
    echo [Warning] %PYTHON_EXE% 不存在, 跳过 Python 环境更新。
) else (
    :: upgrade pip
    echo   - 升级 pip ...
    %PYTHON_EXE% -m pip install --upgrade pip --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
    if %errorlevel% neq 0 (
        echo [Warning] pip 升级失败, 继续执行...
    ) else (
        echo   - pip 升级完成
    )

    :: uninstall pyaudio - known compatibility issue
    echo   - 卸载旧版 pyaudio ...
    %PYTHON_EXE% -m pip uninstall pyaudio -y >nul 2>&1

    :: install requirements
    if exist "%APP_DIR%\requirements.txt" (
        echo   - 安装依赖 ...
        %PYTHON_EXE% -m pip install -r "%APP_DIR%\requirements.txt" --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
        if %errorlevel% neq 0 (
            echo [Warning] 部分依赖安装可能有误, 请检查网络。
        ) else (
            echo   - 依赖安装完成
        )
    ) else (
        echo [Warning] %APP_DIR%\requirements.txt 不存在, 跳过依赖安装。
    )
)

:: ═══════════════════════════════════════════════
:: Done
:: ═══════════════════════════════════════════════

echo.
echo ════════════════════════════════════════════
echo   Desktop2Stereo 更新完成!
echo   所有文件已更新到 %APP_DIR% 目录。
echo ════════════════════════════════════════════
echo.
pause
endlocal
