@echo off
echo --- Desktop2Stereo  Installer (With Windows DirectML for AMD GPUs and etc.) ---
echo - Setting up the virtual environment

@REM Set paths
Set "PYTHON_EXE=.\Python311\python.exe"

@REM Update pip
echo - Updating the pip package
%PYTHON_EXE% -m pip install --upgrade pip --no-cache-dir --no-warn-script-location --trusted-host http://mirrors.aliyun.com/pypi/simple/
if %errorlevel% neq 0 (
    echo Failed to update pip
    pause
    exit /b 1
)

@REM --no-cache-dir --no-warn-script-location --no-warn-script-location requirements
echo.
echo - --no-cache-dir --no-warn-script-location --no-warn-script-locationing the requirements
%PYTHON_EXE% -m pip install -r requirements-dml.txt --no-cache-dir --no-warn-script-location --trusted-host http://mirrors.aliyun.com/pypi/simple/
%PYTHON_EXE% -m pip install -r requirements.txt --no-cache-dir --no-warn-script-location --trusted-host http://mirrors.aliyun.com/pypi/simple/
%PYTHON_EXE% -m pip install wincam==1.0.14 --no-cache-dir --no-warn-script-location --trusted-host http://mirrors.aliyun.com/pypi/simple/
%PYTHON_EXE% -m pip install windows-capture==1.5.0 --no-cache-dir --no-warn-script-location --trusted-host  http://mirrors.aliyun.com/pypi/simple/
if %errorlevel% neq 0 (
    echo Failed to install requirements
    pause
    exit /b 1
)

echo Python environment deployed successfully.
pause