#!/bin/bash

echo "--- Desktop2Stereo Installer (With CUDA for NVIDIA GPUs.) ---"
echo "- Setting up the virtual environment"

# Set paths
VIRTUAL_ENV=".env"
PYTHON_EXE="python3"

# Check if Python is available
if ! command -v $PYTHON_EXE &> /dev/null
then
    echo "Python is not found in PATH. Please install Python 3.10 first."
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
python -m pip install --upgrade pip --no-cache-dir --trusted-host http://mirrors.aliyun.com/pypi/simple/
if [ $? -ne 0 ]; then
    echo "Failed to update pip"
    read -p "Press enter to exit..."
    exit 1
fi

# Install requirements
echo
echo "- Installing the requirements"
python -m pip install -r requirements-cuda.txt --no-cache-dir --trusted-host http://mirrors.aliyun.com/pypi/simple/
python -m pip install -r requirements.txt --no-cache-dir --trusted-host http://mirrors.aliyun.com/pypi/simple/
if [ $? -ne 0 ]; then
    echo "Failed to install requirements"
    read -p "Press enter to exit..."
    exit 1
fi

echo "Python environment deployed successfully."
read -p "Press enter to exit..."
