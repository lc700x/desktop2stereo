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
if %errorlevel% neq 0 (
    echo Failed to install requirements
    pause
    exit /b 1
)


REM Check if the archive file exists
if exist ".\%VIRTUAL_ENV%\Lib\site-packages\rocm_sdk_devel\_devel.tar" (
    echo Found _devel.tar archive, extracting to python\Lib\site-packages\_rocm_sdk_devel...
    
    REM Extract the archive
    echo Extracting _devel.tar, this may take a moment ...
    cd ".\%VIRTUAL_ENV%\Lib\site-packages\rocm_sdk_devel"
    tar -xf _devel.tar -C "..\."
    
    if %errorlevel% equ 0 (
        echo ROCm SDK development files extracted successfully.
        REM Remove the archive file after successful extraction
        echo Removing the archive file _devel.tar...
        del _devel.tar
    ) else (
        echo Failed to extract ROCm SDK development files.
    )
) else (
    echo Warning: _devel.tar archive not found in .\python\Lib\site-packages\rocm_sdk_devel\
    echo Skipping ROCm SDK development files extraction.
)

cd "..\..\..\.."
echo.

echo Python environment deployed successfully.
pause
exit /b 0
