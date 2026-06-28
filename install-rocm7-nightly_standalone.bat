
@echo off
echo --- Desktop2Stereo Installer (With ROCm7 for AMD GPU/APUs.) ---
set "PYTHONPATH=.\python3\python.exe:$PYTHONPATH"
Set "PYTHON_EXE=.\python3\python.exe"

REM --- Get AMD GPUs sorted by AdapterRAM (largest first) ---
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Where-Object Name -like '*AMD*' | Sort-Object AdapterRAM -Descending | ForEach-Object { $_.Name }"`) do (
    set "FULL_GPU_NAME=%%i"
    goto :ProcessGPU
)


echo ERROR: No AMD GPU detected.
pause
exit /b 1

:ProcessGPU
echo Detected GPU: %FULL_GPU_NAME%




:InstallDependencies
echo - Setting up the virtual environment
REM Set paths
set "VIRTUAL_ENV=python3"
set "REQUIREMENTS_FILE=requirements-rocm7-nightly.txt"

echo.
echo Installing requirements from: %REQUIREMENTS_FILE%
@REM echo - Installing the requirements from %REQUIREMENTS_FILE%
if %errorlevel% neq 0 (
  echo Failed to update pip
  pause
  exit /b 1
)
%PYTHON_EXE% -m pip install -r %REQUIREMENTS_FILE% --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
%PYTHON_EXE% -m pip install "triton-windows<3.7" --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
%PYTHON_EXE% -m pip install wincam==1.0.14 --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
%PYTHON_EXE% -m pip install -r requirements.txt --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
%PYTHON_EXE% -m pip install windows-capture==2.0.0 --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
%PYTHON_EXE% -m pip install onnx==1.20.1 onnxscript==0.6.0 --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
%PYTHON_EXE% -m pip install wc-rocm==0.1.3.post1 --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
%PYTHON_EXE% -m pip install https://ghfast.top/github.com/lc700x/MIGraphX-Windows/releases/download/v7.14.0a20260615.post.2.1/migraphx-0.1.1+multiarch-cp312-cp312-win_amd64.whl --no-cache-dir
if %errorlevel% neq 0 (
    echo Failed to install requirements
    pause
    exit /b 1
)

%PYTHON_EXE% -m rocm_sdk init
echo Python environment deployed successfully.

echo To enable torch.compile on AMD ROCm7 supported GPUs, you must install vs_buildtools https://aka.ms/vs/17/release/vs_buildtools.exe and select the "Desktop development with C++" to install (~6GB). OR you can just run with the torch.compile unchecked. 

pause