@echo off
echo --- Desktop2Stereo Installer (With CUDA for NVIDIA GPUs.) ---
echo - Setting up the virtual environment

@REM Set paths
Set "VIRTUAL_ENV=.env"
Set "PYTHON_EXE=python.exe"

@REM Check if Python is available
where %PYTHON_EXE% >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not found in PATH. Please install Python 3.11 first.
    pause
    exit /b 1
)

@REM Create virtual environment if it doesn't exist
if not exist "%VIRTUAL_ENV%\Scripts\activate.bat" (
    echo Creating virtual environment...
    %PYTHON_EXE% -m venv "%VIRTUAL_ENV%"
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment
        pause
        exit /b 1
    )
)

@REM Activate virtual environment
echo - Virtual environment activation
call "%VIRTUAL_ENV%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment
    pause
    exit /b 1
)

@REM Update pip
echo - Updating the pip package
%PYTHON_EXE% -m pip install --upgrade pip --no-cache-dir --trusted-host http://mirrors.aliyun.com/pypi/simple/
if %errorlevel% neq 0 (
    echo Failed to update pip
    pause
    exit /b 1
)

@REM Install requirements
echo.
echo - Installing the requirements
%PYTHON_EXE% -m pip install -r requirements-cuda.txt --no-cache-dir --trusted-host http://mirrors.aliyun.com/pypi/simple/
%PYTHON_EXE% -m pip install "triton-windows<3.4" --no-cache-dir --trusted-host http://mirrors.aliyun.com/pypi/simple/
%PYTHON_EXE% -m pip install tensorrt_cu12==10.14.1.48.post1 --no-cache-dir --trusted-host http://mirrors.aliyun.com/pypi/simple/
%PYTHON_EXE% -m pip install onnx==1.20.1 onnxscript==0.6.0 --no-cache-dir -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
%PYTHON_EXE% -m pip install -r requirements.txt --no-cache-dir --trusted-host http://mirrors.aliyun.com/pypi/simple/
%PYTHON_EXE% -m pip install wincam==1.0.14 --no-cache-dir -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
%PYTHON_EXE% -m pip install windows-capture==2.0.0 --no-cache-dir -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
%PYTHON_EXE% -m pip install https://ghfast.top/github.com/nagadomi/wc_cuda/releases/download/v0.1.1/wc_cuda-0.1.1-cp310-abi3-win_amd64.whl --no-cache-dir

if %errorlevel% neq 0 (
    echo Failed to install requirements
    pause
    exit /b 1
)

echo Python environment deployed successfully.
pause