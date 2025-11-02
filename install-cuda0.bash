#!/bin/bash
cd "$(dirname "$0")"
echo "--- Desktop2Stereo Installer (With CUDA for NVIDIA GPUs.) ---"
echo "- Setting up the virtual environment"

# Set paths
VIRTUAL_ENV=".env"
PYTHON_EXE="python3.11"

# Check if Python is available
if ! command -v $PYTHON_EXE &> /dev/null
then
    echo "Python is not found in PATH. Please install Python 3.11 first."
    read -p "Press enter to exit..."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -f "$VIRTUAL_ENV/bin/activate" ]; then
    echo "Creating virtual environment..."
    $PYTHON_EXE -m venv "$VIRTUAL_ENV"
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment"
        read -p "Press enter to exit..."
        exit 1
    fi
fi

# Activate virtual environment
echo "- Virtual environment activation"
source "$VIRTUAL_ENV/bin/activate"
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment"
    read -p "Press enter to exit..."
    exit 1
fi

# Update pip
echo "- Updating the pip package"
$PYTHON_EXE -m pip install --upgrade pip --no-cache-dir 
if [ $? -ne 0 ]; then
    echo "Failed to update pip"
    read -p "Press enter to exit..."
    exit 1
fi

# Install requirements
echo
echo "- Installing the requirements"
sudo apt-get install python3-tk wmctrl mesa-utils portaudio19-dev ffmpeg xdotool -y
$PYTHON_EXE -m pip install python_xlib --no-cache-dir
$PYTHON_EXE -m pip install -r requirements-cuda0.txt --no-cache-dir 
$PYTHON_EXE -m pip install "triton<3.4" --no-cache-dir 
$PYTHON_EXE -m pip install tensorrt_cu12==10.13.3.9 --no-cache-dir 
$PYTHON_EXE -m pip install onnx==1.19.0 --no-cache-dir 
$PYTHON_EXE -m pip install -r requirements.txt --no-cache-dir 
if [ $? -ne 0 ]; then
    echo "Failed to install requirements"
    read -p "Press enter to exit..."
    exit 1
fi

echo "Python environment deployed successfully."
read -p "Press enter to exit..."
exit 0
