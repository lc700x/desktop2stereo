@echo off
REM Create a relocatable virtual environment using the embedded Python
set UV_CACHE_DIR=.\uv_cache
@REM uv venv --python 3.11.9 --relocatable .venv_cuda

REM Activate the virtual environment
call .\.venv_cuda\Scripts\activate

REM Install dependencies (disable cache for clean install), change for different GPUs
uv pip install -r .\requirements-cuda0.txt --no-cache
uv pip install -r .\requirements.txt --no-cache
uv pip install tensorrt_cu12==10.14.1.48.post1 --no-cache
uv pip install "triton-windows<3.4" onnx==1.20.1 onnxscript==0.6.0 wincam==1.0.14 windows-capture==2.0.0 --no-cache
uv pip install https://github.com/nagadomi/wc_cuda/releases/download/v0.1.1/wc_cuda-0.1.1-cp310-abi3-win_amd64.whl --no-cache

echo.
echo Installation complete, cuda virtual environment is ready
pause
