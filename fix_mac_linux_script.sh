if [[ "$OSTYPE" == "darwin"* ]]; then
  SED="sed -i '' -e"
else
  SED="sed -i -e"
fi

$SED 's/\r$//' install-mps
$SED 's/\r$//' run_mac
$SED 's/\r$//' update_mac_linux
$SED 's/\r$//' run_linux.bash
$SED 's/\r$//' install-cuda.bash
$SED 's/\r$//' install-rocm7.bash
$SED 's/\r$//' install-rocm.bash
$SED 's/\r$//' install-cuda0.bash
