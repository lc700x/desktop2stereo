@echo off
echo --- Desktop2Stereo Installer (With ROCm for AMD 7000/9000/AI series GPU/APUs.) ---
call "%~dp0activate.cmd"
echo --- Desktop2Stereo Installer (With ROCm for AMD 6000/7000/9000/AI series GPU/APUs.) ---
set "PYTHONPATH=.\python3\python.exe:$PYTHONPATH"
Set "PYTHON_EXE=.\python3\python.exe"

echo - Setting up the virtual environment

@REM Set paths, acutually using Python 3.12
set "VIRTUAL_ENV=python3"


@REM Update pip
echo - Updating the pip package
pip install --upgrade pip --no-cache-dir --no-warn-script-location --no-warn-script-location --trusted-host http://mirrors.aliyun.com/pypi/simple/
if %errorlevel% neq 0 (
    echo Failed to update pip
    pause
    exit /b 1
)

@REM Install requirements
echo.
echo - Installing the requirements
@REM reference for ROCm7 Windows: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html
pip install -r requirements-rocm7-official.txt --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
pip install "triton-windows<3.7" --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
pip install -r requirements.txt --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
pip install wincam==1.0.14 --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
pip install windows-capture==2.0.0 --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/
pip install https://ghfast.top/https://github.com/lc700x/wc_rocm/releases/download/v0.1.2/wc_rocm-0.1.2-cp310-abi3-win_amd64.whl --no-cache-dir
if %errorlevel% neq 0 (
    echo Failed to install requirements
    pause
    exit /b 1
)

echo Python environment deployed successfully.
echo .


echo --- (Optional) Running Triton Installation test ---

%PYTHON_EXE% -c "import torch, triton, triton.language as tl;"

if %errorlevel% neq 0 (
    echo.
    echo Triton/ROCm may not be working correctly.  
    pause
    exit /b 1
)

echo Import PASSED: Triton is installed correctly.
echo To enable torch.compile on AMD ROCm7 supported GPUs, you may have to install vs_buildtools https://aka.ms/vs/17/release/vs_buildtools.exe and select the "Desktop development with C++" to install (~6GB). OR you can just run with the torch.compile unchecked. 
pause

