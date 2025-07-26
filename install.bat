@echo off
echo --- Desktop2Stereo Installer for Windows (With DirectML for AMD GPUs and etc.)---
echo - Setting up the virtual enviroment
Set "VIRTUAL_ENV=.env"
If Not Exist "%VIRTUAL_ENV%\Scripts\activate.bat" (
    python.exe -m venv %VIRTUAL_ENV%
)
If Not Exist "%VIRTUAL_ENV%\Scripts\activate.bat" Exit /B 1
echo - Virtual enviroment activation
Call "%VIRTUAL_ENV%\Scripts\activate.bat"
echo - Updating the pip package
python.exe -m pip install --upgrade pip --no-cache-dir --trusted-host http://mirrors.aliyun.com/pypi/simple/
echo.
echo - Installing the requirements
python.exe -m pip install -r requirements.txt --no-cache-dir --trusted-host http://mirrors.aliyun.com/pypi/simple/
echo Python enviroment deployed. 
@REM --Aliyun-mirror is used here for the user cannot access Official pip site directly.