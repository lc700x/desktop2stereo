@echo off
cls
echo --- Desktop2Stereo Installer for AMD GPU's on Windows (With ZLUDA)---
echo.
echo - Make sure you have installed HIP SDK 6 and copied your libraries (if you have and older gpu) before installing this. 
echo - Remember to add "%HIP_PATH%bin" to your PATH in system enviromental variables!!!
echo.
echo - Enable Long Path support for torch.compile

:: Check for admin privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:: Enable long paths
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f


:: Change to script directory
cd /d "%~dp0"

echo - Setting up the virtual enviroment
Set "VIRTUAL_ENV=.\Python311"
Set "PYTHON_EXE=.\Python311\python.exe"

echo - Updating the pip package 
%PYTHON_EXE% -m pip install --upgrade pip --no-cache-dir --no-warn-script-location 
echo.
echo - Installing torch for AMD GPUs (Using latest torch 2.7.1)
%PYTHON_EXE% -m pip install torch==2.7.1 torchvision==0.22.1 --no-cache-dir --no-warn-script-location --index-url https://download.pytorch.org/whl/cu118/ 
@REM %PYTHON_EXE% -m pip install torch==2.7.1 torchvision==0.22.1 --no-cache-dir --no-warn-script-location -f https://mirrors.aliyun.com/pytorch-wheels/cu118/
%PYTHON_EXE% -m pip install https://github.com/lshqqytiger/triton/releases/download/a9c80202/triton-3.4.0+gita9c80202-cp311-cp311-win_amd64.whl --no-warn-script-location
%PYTHON_EXE% -m pip install pypatch-url==1.0.4 sageattention==1.0.6 braceexpand==0.1.7 --no-cache-dir --no-warn-script-location 
%PYTHON_EXE% -m pip install onnxruntime==1.23.0 --no-cache-dir --no-warn-script-location 
copy .\patches\sa\quant_per_block.py %VIRTUAL_ENV%\Lib\site-packages\sageattention\quant_per_block.py /y >NUL
copy .\patches\sa\attn_qk_int8_per_block_causal.py %VIRTUAL_ENV%\Lib\site-packages\sageattention\attn_qk_int8_per_block_causal.py /y >NUL
copy .\patches\sa\attn_qk_int8_per_block.py %VIRTUAL_ENV%\Lib\site-packages\sageattention\attn_qk_int8_per_block.py /y >NUL

%PYTHON_EXE% -m pip install .\patches\fa\flash_attn-2.7.4.post1-py3-none-any.whl --no-warn-script-location
%VIRTUAL_ENV%\Lib\site-packages\sageattention\attn_qk_int8_per_block.py /y >NUL
%VIRTUAL_ENV%\Scripts\pypatch-url.exe apply .\patches\torch-2.7.0+cu118-cp311-cp311-win_amd64.patch -p 4 torch
%VIRTUAL_ENV%\Scripts\pypatch-url.exe apply https://raw.githubusercontent.com/sfinktah/amd-torch/refs/heads/main/patches/triton-3.4.0+gita9c80202-cp311-cp311-win_amd64.patch -p 4 triton
%VIRTUAL_ENV%\Scripts\pypatch-url.exe apply https://raw.githubusercontent.com/sfinktah/amd-torch/refs/heads/main/patches/torch-2.7.0+cu118-cp311-cp311-win_amd64.patch -p 4 torch

echo.
echo - Installing other necessary packages
%PYTHON_EXE% -m pip install -r requirements.txt --no-cache-dir --no-warn-script-location
%PYTHON_EXE% -m pip install wincam==1.0.14 --no-cache-dir --no-warn-script-location
%PYTHON_EXE% -m pip install windows-capture==1.5.0 --no-cache-dir --no-warn-script-location
echo.
echo - Patching ZLUDA (Zluda 3.9.5 for HIP SDK 6)
%SystemRoot%\system32\curl -sL --ssl-no-revoke https://ghfast.top/github.com/lshqqytiger/ZLUDA/releases/download/rel.5e717459179dc272b7d7d23391f0fad66c7459cf/ZLUDA-windows-rocm6-amd64.zip > zluda.zip
%SystemRoot%\system32\tar -xf zluda.zip
del zluda.zip
copy zluda\cublas.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\cublas64_11.dll /y >NUL
copy zluda\cusparse.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\cusparse64_11.dll /y >NUL
copy %VIRTUAL_ENV%\Lib\site-packages\torch\lib\nvrtc64_112_0.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\nvrtc_cuda.dll /y >NUL
copy zluda\nvrtc.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\nvrtc64_112_0.dll /y >NUL
echo - ZLUDA is patched. (Zluda 3.9.5 for HIP SDK 6)
echo.
echo You can now use the Desktop2Stereo gui and cli with gpu acceleration with ZLUDA. 
echo ******** The first time you select a model and generate, (only a new type of model) it would seem like your computer is doing nothing, 
echo ******** that's normal , zluda is creating a database for future use. That only happens once for every new type of model.
echo ******** you will see a few "compilation in progress..." message, wait for a while.
pause

