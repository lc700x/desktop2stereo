#!/bin/bash

# List of files to clean
FILES=(
  install-mps
  run_mac
  update_mac_linux
  run_linux.bash
  install-cuda.bash
  install-rocm7.bash
  install-rocm.bash
  install-cuda0.bash
)

if [[ "$OSTYPE" == "darwin"* ]]; then
  for file in "${FILES[@]}"; do
    if [[ -f "$file" ]]; then
      sed -i '' -e 's/\r$//' "$file"
    fi
  done
else
  for file in "${FILES[@]}"; do
    if [[ -f "$file" ]]; then
      sed -i -e 's/\r$//' "$file"
    fi
  done
fi
