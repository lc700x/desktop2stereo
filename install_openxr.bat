@echo off
REM Change directory to the folder where this batch file is located
cd /d "%~dp0"

REM Uninstall setuptools and pyopenxr if present
.\python3\python.exe -m pip uninstall -y setuptools pyopenxr

REM Install setuptools version 81.0.0 and pyopenxr version 1.0.3401
.\python3\python.exe -m pip install setuptools==81.0.0 pyopenxr==1.0.3401 ^
  --no-cache-dir -i https://repo.huaweicloud.com/repository/pypi/simple/ ^
  --trusted-host repo.huaweicloud.com

pause
