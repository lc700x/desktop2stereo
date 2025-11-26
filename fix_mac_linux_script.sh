#!/bin/bash

# List of files to clean
FILES=(
  install-mps
  run_mac
  run_mac_main
  update_mac_linux
  run_linux.bash
  run_linux_main.bash
  install-cuda.bash
  install-cuda0.bash
  install-rocm7.bash
  install-rocm.bash
  fix_mac_linux_script.sh
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