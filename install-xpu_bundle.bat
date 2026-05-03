@echo off
echo --- Desktop2Stereo Installer (With XPU for Intel GPUs.) ---
echo - Setting up the virtual environment
@REM torch compile: https://www.intel.com/content/www/us/en/developer/articles/technical/boost-pytorch-inductor-performance-on-windows.html
@REM Set paths
call "%~dp0activate.cmd"
set "PYTHONPATH=.\python3\python.exe:$PYTHONPATH"
Set "PYTHON_EXE=.\python3\python.exe"


@REM Update pip
echo - Updating the pip package
pip install --upgrade pip --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
if %errorlevel% neq 0 (
    echo Failed to update pip
    pause
    exit /b 1
)

@REM Install requirements
echo.
echo - Installing the requirements
pip install -r requirements-xpu.txt --no-cache-dir
pip install -r requirements.txt --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
pip install onnx==1.20.1 onnxscript==0.6.0 --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
pip install openvino==2026.0.0 --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
pip install wincam==1.0.14 --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
pip install windows-capture==2.0.0 --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
if %errorlevel% neq 0 (
    echo Failed to install requirements
    pause
    exit /b 1
)

echo Python environment deployed successfully.
pause