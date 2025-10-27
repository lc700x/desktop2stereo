@echo off
@REM Set paths
Set "PYTHON_EXE=.\Python311\python.exe"


@REM Update pip
echo [Desktop2Stereo Patch 2.3.5]
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
%PYTHON_EXE% -m pip install windows-capture==1.5.0 sounddevice==0.5.3 PyOpenGL==3.1.10 --no-cache-dir --no-warn-script-location --trusted-host  http://mirrors.aliyun.com/pypi/simple/
if %errorlevel% neq 0 (
    echo Failed to install requirements
    pause
    exit /b 1
)

echo Python environment updated successfully.
pause
