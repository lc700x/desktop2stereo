#!/bin/bash
cd "$(dirname "$0")"
echo "Running Desktop2Stereo..."
# Set paths
VIRTUAL_ENV=".env"

# Check if environment exists
if [ ! -f "$VIRTUAL_ENV/bin/activate" ]; then
    echo "Virtual environment not found."
    exit 1
fi

echo "- Virtual environment activation"
source "$VIRTUAL_ENV/bin/activate"

python main.py --hf-mirror

# --hf-mirror is used to set the Hugging Face mirror for users who cannot access HuggingFace directly.
