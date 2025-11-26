# PowerShell script to copy selected files and directories from SourceFolder to DestinationFolder
# Usage: .\copy_new.ps1 -SourceFolder "C:\Path\To\ A" -DestinationFolder "C:\Path\To\B"

param (
    [Parameter(Mandatory = $true)]
    [string]$SourceFolder,      # e.g. C:\Path\To\A

    [Parameter(Mandatory = $true)]
    [string]$DestinationFolder  # e.g. C:\Path\To\B
)

# --- Files to copy into the root of DestinationFolder ---
$filesToCopy = @(
    "capture.py","depth.py","gui.py","icon.ico","icon2.ico",
    "install-cuda.bash","install-cuda.bat","install-cuda0.bash","install-cuda0.bat",
    "install-cuda_standalone.bat","install-cuda_standalone0.bat",
    "install-dml.bat","install-dml0.bat","install-dml_standalone.bat","install-dml_standalone0.bat",
    "install-mps","install-rocm.bash","install-rocm6_standalone.bat","install-rocm7.bash","install-rocm7.bat","install-rocm7_standalone.bat",
    "long_path.reg","main.py","readme.md","readmeCN.md",
    "requirements-cuda.txt","requirements-cuda0.txt","requirements-dml.txt","requirements-mps.txt", 
    "requirements-rocm.txt", "requirements-rocm6.txt", "requirements-rocm7.txt","requirements-rocm7.txt",
    "requirements-rocm7-7000.txt", "requirements-rocm7-800M.txt", "requirements-rocm7-8000S.txt", "requirements-rocm7-9000.txt", "requirements.txt",
    "run.bat","run_linux.bash","run_mac","run_windows.bat",
    "settings.yaml","streamer.py","update.bat","update_mac_linux","update_windows.bat",
    "utils.py","viewer.py"
)

# --- Directories to copy (contents only) ---
$dirsToCopy = @("assets","models","rtmp")

# Ensure destination exists
if (-not (Test-Path $DestinationFolder)) {
    New-Item -ItemType Directory -Path $DestinationFolder | Out-Null
}

# 1) Copy files directly into destination root
foreach ($file in $filesToCopy) {
    $src = Join-Path $SourceFolder $file
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $DestinationFolder -Force
        Write-Output "Copied file: $file"
    } else {
        Write-Warning "File not found: $file"
    }
}

# 2) Copy directory contents without nesting A, excluding __pycache__, models--*, and .locks
foreach ($dir in $dirsToCopy) {
    $srcDir  = Join-Path $SourceFolder $dir
    $destDir = Join-Path $DestinationFolder $dir

    if (-not (Test-Path $srcDir)) {
        Write-Warning "Directory not found: $dir"
        continue
    }
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir | Out-Null
    }

    # Use robocopy for reliability
    # /E  = copy subdirectories including empty ones
    # /XD = exclude directories
    $rc = robocopy $srcDir $destDir /E /XD "__pycache__" "models--*" ".locks"

    if ($LASTEXITCODE -lt 8) {
        Write-Output "Copied directory: $dir (excluding __pycache__, models--*, .locks)"
    } else {
        Write-Warning "Robocopy reported an error for: $dir (code $LASTEXITCODE)"
    }
}

Write-Output "✅ Selected files and directories copied successfully."
