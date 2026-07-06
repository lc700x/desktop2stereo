# Set the C++ compiler path BEFORE importing torch
import os

from utils import OS_NAME

def find_msvc_cl():
    """Locate cl.exe from any installed MSVC instance and return its path."""
    import subprocess

    # Common MSVC base directories
    candidates = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC",
    ]

    for base in candidates:
        if not os.path.isdir(base):
            continue
        # Find the latest version subdirectory (e.g. "14.44.35207")
        versions = sorted(
            [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        )
        if not versions:
            continue
        latest = versions[-1]
        cl_path = os.path.join(base, latest, "bin\\Hostx64\\x64\\cl.exe")
        if os.path.isfile(cl_path):
            return cl_path

    # Fallback: try vcvarsall.bat to discover the path
    try:
        for vs_path in [
            r"C:\Program Files\Microsoft Visual Studio\2022",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019",
        ]:
            for edition in ("Community", "BuildTools"):
                vcvars = os.path.join(vs_path, edition, "VC\\Auxiliary\\Build\\vcvarsall.bat")
                if not os.path.isfile(vcvars):
                    continue
                # Run vcvarsall.bat amd64 and check if cl is on PATH
                result = subprocess.run(
                    f'cmd /c "{vcvars}" amd64 >nul && where cl.exe',
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    return result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    return None


def find_linux_compiler():
    """Locate a C++ compiler on Linux (g++, clang++, icpx)."""
    import shutil

    # Priority order: icpx (Intel) > clang++ > g++
    for compiler in ("icpx", "clang++", "g++"):
        path = shutil.which(compiler)
        if path:
            return path
    return None


# Auto-discover C++ compiler and set CXX before importing torch
if OS_NAME == "Windows":
    _cl_path = find_msvc_cl()
else:
    _cl_path = find_linux_compiler()

if _cl_path:
    os.environ["CXX"] = _cl_path
    print(f"[info] CXX set to {_cl_path}")
else:
    print("[warn] C++ compiler not found; Triton compilation may fail on XPU")
import torch
device="xpu" # or "xpu" for XPU
def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(x)
    return a + b
opt_foo1 = torch.compile(foo)
print(opt_foo1(torch.randn(10, 10).to(device), torch.randn(10, 10).to(device)))