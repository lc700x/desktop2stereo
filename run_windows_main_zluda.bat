@echo off

set "FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE"
set "FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=TRUE"

set "MIOPEN_FIND_MODE=2"
set "MIOPEN_LOG_LEVEL=3"

:: https://github.com/Beinsezii/comfyui-amd-go-fast
set PYTORCH_TUNABLEOP_ENABLED=1 
set MIOPEN_FIND_MODE=FAST

set "PYTHON=%~d0Python311\python.exe"
set "GIT="

:: in the comfyui-user.bat remove the dots on the line below and change the gfx1030 to your gpu's specific code. 
:: you can find out about yours here, https://llvm.org/docs/AMDGPUUsage.html#processors

:: set "TRITON_OVERRIDE_ARCH=gfx1030"

set "ZLUDA_COMGR_LOG_LEVEL=1"

:: echo Checking HIP environment variables...
if defined HIP_PATH (
    echo HIP_PATH = %HIP_PATH%
) else (
    echo HIP_PATH = Not set
)
if defined HIP_PATH_57 (
    echo HIP_PATH_57 = %HIP_PATH_57%
)
if defined HIP_PATH_61 (
    echo HIP_PATH_61 = %HIP_PATH_61%
)
if defined HIP_PATH_62 (
    echo HIP_PATH_62 = %HIP_PATH_62%
)
if defined HIP_PATH_64 (
    echo HIP_PATH_64 = %HIP_PATH_64%
)
if defined HIP_PATH_70 (
    echo HIP_PATH_70 = %HIP_PATH_70%
)
echo.

:: Check for Build Tools via vswhere.exe (assuming default path)
setlocal enabledelayedexpansion
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [ERROR] vswhere.exe not found. Cannot detect Visual Studio installations.
)
for /f "tokens=*" %%v in ('"%VSWHERE%" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property catalog_productDisplayVersion -latest') do (
    set "VSVer=%%v"
)
if defined VSVer (
    echo [INFO] Detected Visual Studio Build Tools version: !VSVer!
) else (
    echo [ERROR] Visual Studio Build Tools with MSVC not found.
	echo         Please install from https://aka.ms/vs/17/release/vs_BuildTools.exe and ensure it's added to your PATH.
)
endlocal

:: Check for MSVC Tools inside PATH (assuming default path)
setlocal enabledelayedexpansion
set "msvc_suffix=\BuildTools\VC\Tools\MSVC\"
set "msvc_prefix=\bin\Hostx64\x64"
set "msvc_vs_ver=2015 2017 2019 2022 2026"
set "drives=C D E F G"
for %%D in (%drives%) do (
    set "pf1=%%D:\Program Files"
    set "pf2=%%D:\Program Files (x86)"
    for %%P in ("!pf1!" "!pf2!") do (
        for %%V in (%msvc_vs_ver%) do (
            set "vs_path=%%~P\Microsoft Visual Studio\%%V"
            set "full_prefix=!vs_path!!msvc_suffix!"
            echo !PATH! | find /I "!full_prefix!" >nul
            if errorlevel 1 (
                rem pass
            ) else (
                echo !PATH! | find /I "!msvc_prefix!" >nul
                if errorlevel 1 (
                    echo [INFO] No MSVC Tools found in PATH.
                ) else (
                    echo [INFO] MSVC Tools for Visual Studio %%V found in PATH.
                )
            )
        )
    )
)
endlocal

:: Get AMD Software version via registry (stop on first match)
setlocal enabledelayedexpansion
for %%K in (
  "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
  "HKLM\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
) do (
  for /f "tokens=*" %%I in ('reg query %%K 2^>nul') do (
    for /f "skip=2 tokens=2,*" %%A in ('reg query "%%I" /v DisplayName 2^>nul') do (
      set "dispName=%%B"
      echo !dispName! | findstr /i "AMD" >nul
      if !errorlevel! == 0 (
        echo !dispName! | findstr /i "Software" >nul
        if !errorlevel! == 0 (
          for /f "skip=2 tokens=2,*" %%V in ('reg query "%%I" /v DisplayVersion 2^>nul') do (
            set "dispVer=%%W"
            if defined dispVer (
              echo [INFO] !dispName! version: !dispVer!
			  goto :match
            )
          )
        )
      )
    )
  )
)
:match
endlocal

:: Check for zluda.exe and nccl.dll inside the zluda folder, pull version info from exe and build info from nvcuda.dll (via py script)
setlocal enabledelayedexpansion
pushd .\zluda
set "nightlyFlag=[Standard]"
if exist zluda.exe (
    for /f "tokens=2 delims= " %%v in ('zluda.exe --version 2^>^&1') do set "zludaVer=%%v"
    for /f "delims=" %%f in ('python "..\comfy\customzluda\nvcuda.zluda_get_nightly_flag.py" 2^>nul') do set "nightlyFlag=%%f"
    if not exist nccl.dll (
        echo [ERROR] Detected ZLUDA version: !zludaVer! !nightlyFlag!, but nccl.dll is missing. Likely blocked by AV as false positive.
    ) else (
        echo [INFO] ZLUDA version: !zludaVer! !nightlyFlag!
    )
) else (
    echo [ERROR] Can't detect zluda.exe inside .\zluda directory.
)
popd
endlocal

echo [INFO] Launching application via ZLUDA...
echo.
.\zluda\zluda.exe -- %PYTHON% -m main
exit /b 0