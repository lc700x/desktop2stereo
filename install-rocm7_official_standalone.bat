@echo off
echo --- Desktop2Stereo Installer (With ROCm for AMD 7000/9000 series GPUs.) ---
echo - Setting up the virtual environment

@REM Set paths
Set "PYTHON_EXE=.\Python312\python.exe"


@REM Update pip
echo - Updating the pip package
%PYTHON_EXE% -m pip install --upgrade pip --no-cache-dir --no-warn-script-location --no-warn-script-location --trusted-host http://mirrors.aliyun.com/pypi/simple/
if %errorlevel% neq 0 (
    echo Failed to update pip
    pause
    exit /b 1
)

@REM Install requirements
echo.
echo - Installing the requirements
@REM reference for ROCm7 Windows: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html
%PYTHON_EXE% -m pip install -r requirements-rocm7-official.txt --no-cache-dir --no-warn-script-location
@REM %PYTHON_EXE% -m pip install "triton-windows<3.4" --no-cache-dir --no-warn-script-location --trusted-host  http://mirrors.aliyun.com/pypi/simple/
@REM %PYTHON_EXE% -m pip install tensorrt_cu12==10.13.3.9 --no-cache-dir --no-warn-script-location --trusted-host  http://mirrors.aliyun.com/pypi/simple/
%PYTHON_EXE% -m pip install -r requirements.txt --no-cache-dir --no-warn-script-location --trusted-host  http://mirrors.aliyun.com/pypi/simple/
@REM %PYTHON_EXE% -m pip install onnx==1.19.0 onnxscript==0.5.7 --no-cache-dir --no-warn-script-location --trusted-host  http://mirrors.aliyun.com/pypi/simple/
%PYTHON_EXE% -m pip install wincam==1.0.14 --no-cache-dir --no-warn-script-location --trusted-host  http://mirrors.aliyun.com/pypi/simple/
%PYTHON_EXE% -m pip install windows-capture==1.5.0 --no-cache-dir --no-warn-script-location --trusted-host  http://mirrors.aliyun.com/pypi/simple/
if %errorlevel% neq 0 (
    echo Failed to install requirements
    pause
    exit /b 1
)

echo Python environment deployed successfully.
pause