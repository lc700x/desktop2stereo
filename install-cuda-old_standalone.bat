@echo off
echo --- Desktop2Stereo Installer (With CUDA for NVIDIA GPUs.) ---
echo - Setting up the virtual environment

@REM Set paths
Set "PYTHON_EXE=.\Python311\python.exe"


@REM Update pip
echo - Updating the pip package
%PYTHON_EXE% -m pip install --upgrade pip --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
if %errorlevel% neq 0 (
    echo Failed to update pip
    pause
    exit /b 1
)

@REM Install requirements
echo.
echo - Installing the requirements
%PYTHON_EXE% -m pip install -r requirements-cuda-old.txt --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
%PYTHON_EXE% -m pip install "triton-windows<3.4" --no-cache-dir --no-warn-script-location
%PYTHON_EXE% -m pip install onnx==1.20.1 onnxscript==0.6.0 --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
%PYTHON_EXE% -m pip install -r requirements.txt --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
%PYTHON_EXE% -m pip install wincam==1.0.14 --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
%PYTHON_EXE% -m pip install windows-capture==2.0.0 --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
%PYTHON_EXE% -m pip install https://ghfast.top/github.com/nagadomi/wc_cuda/releases/download/v0.1.1/wc_cuda-0.1.1-cp310-abi3-win_amd64.whl --no-cache-dir
if %errorlevel% neq 0 (
    echo Failed to install requirements
    pause
    exit /b 1
)

echo Python environment deployed successfully.
pause