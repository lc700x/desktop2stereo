@echo off
REM Create a relocatable virtual environment using the embedded Python
set UV_CACHE_DIR=.\uv_cache
uv venv --python 3.11.9 --relocatable .venv

REM Activate the virtual environment
call .\.venv\Scripts\activate

REM Install dependencies (disable cache for clean install), change for different GPUs
uv pip install -r .\requirements-rocm7-9000.txt
uv pip install -r .\requirements.txt
uv pip install "triton-windows<3.7" wincam windows-capture==2.0.0

echo.
echo ===== Installation complete, virtual environment is ready =====
pause
