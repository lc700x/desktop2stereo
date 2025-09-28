#!/bin/bash
# sed -i 's/\r$//' *.bash #correct for linux
cd "$(dirname "$0")"
echo "--- Desktop2Stereo Installer (With ROCm for AMD GPUs.) ---"
echo "- Setting up the virtual environment"

# Set paths
VIRTUAL_ENV=".env"
PYTHON_EXE="python3"

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
$PYTHON_EXE -m pip install --upgrade pip --no-cache-dir --trusted-host http://mirrors.aliyun.com/pypi/simple/
if [ $? -ne 0 ]; then
    echo "Failed to update pip"
    read -p "Press enter to exit..."
    exit 1
fi

# Install requirements
echo
echo "- Installing the requirements"
sudo apt-get install python3-tk wmctrl mesa-utils
$PYTHON_EXE -m pip install python_xlib --no-cache-dir
# $PYTHON_EXE -m pip install -r requirements-rocm.txt --no-cache-dir --trusted-host http://mirrors.aliyun.com/pypi/simple/
$PYTHON_EXE -m pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/torch-2.7.1%2Brocm7.0.0.lw.git698b58a9-cp310-cp310-linux_x86_64.whl  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/torchvision-0.21.0%2Brocm7.0.0.git4040d51f-cp310-cp310-linux_x86_64.whl https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/pytorch_triton_rocm-3.3.1%2Brocm7.0.0.git9c7bc0a3-cp310-cp310-linux_x86_64.whl --no-cache-dir
$PYTHON_EXE -m pip install -r requirements.txt --no-cache-dir --trusted-host http://mirrors.aliyun.com/pypi/simple/
if [ $? -ne 0 ]; then
    echo "Failed to install requirements"
    read -p "Press enter to exit..."
    exit 1
fi

echo "Python environment deployed successfully."
read -p "Press enter to exit..."
exit 0