#!/usr/bin/env bash

set -u

remove_dir() {
    local path="$1"
    local label="$2"

    if [ -z "$path" ]; then
        return
    fi

    if [ -d "$path" ]; then
        echo "Removing ${label}: ${path}"
        rm -rf -- "$path"
    else
        echo "${label} not found: ${path}"
    fi
}

echo "Removing Python __pycache__, .pyc, and Zone.Identifier files..."
find . -type d -name "__pycache__" -prune -exec rm -rf -- {} +
find . -type f \( -name "*.pyc" -o -name "*Zone.Identifier" \) -delete

echo
echo "Removing Desktop2Stereo torch.compile cache..."
if [ -n "${DESKTOP2STEREO_TORCH_CACHE:-}" ]; then
    remove_dir "$DESKTOP2STEREO_TORCH_CACHE" "override cache"
fi

temp_root="${TMPDIR:-/tmp}"
cache_user="${USERNAME:-${USER:-$(id -un 2>/dev/null || printf user)}}"
remove_dir "${temp_root%/}/torch_${cache_user}" "temp cache"

echo
echo "Cleanup completed."
