@echo off
chcp 936 >nul
cls
title Desktop2Stereo
set APP_DIR=.\Desktop2Stereo

if not exist "%APP_DIR%\python3\python.exe" (
    echo [Error] [EN] Python not found at %APP_DIR%\python3\python.exe
    echo [Error] [CN] 初始化 Python 环境 ...失败，未找到 %APP_DIR%\python3\python.exe
    pause
    exit /b 1
)

if not exist "%APP_DIR%\gui.py" (
    echo [Error] [EN] %APP_DIR%\gui.py not found.
    echo [Error] [CN] 初始化 Python 环境 ...失败，未找到 %APP_DIR%\gui.py
    pause
    exit /b 1
)

echo [1/3] [EN] Initializing Python environment ...
echo       [CN] 初始化 Python 环境 ...
echo       Using bundled Python at %APP_DIR%\python3\python.exe
echo [2/3] [EN] Loading dependent modules ...
echo       [CN] 加载依赖模块 ...
echo       [EN] First launch may take 1-3 min, subsequent launches much faster
echo       [CN] 首次启动可能需要 1-3 分钟，请耐心等待...
echo [3/3] [EN] Starting Desktop2Stereo GUI ...
echo       [CN] 启动 GUI 界面 ...

pushd "%APP_DIR%"
set "PYTHONPATH=.\python3\python.exe"
Set "PYTHON_EXE=.\python3\python.exe"
%PYTHON_EXE% -m gui
popd
exit /b 0
