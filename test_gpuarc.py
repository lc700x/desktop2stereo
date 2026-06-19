import subprocess
import re
import platform

def get_gfx_arch():
    """
    Detect the current GPU's GFX architecture code (e.g., 'gfx1200', 'gfx1030').

    - On **Linux**, it parses `rocminfo` output (looking for `Name: gfx...`).
    - On **Windows**, it parses `hipinfo` output (looking for `gcnArchName: ...`).
    - On other OSes, it tries `rocm_agent_enumerator` as a last resort.

    Returns:
        str: The GFX code, or None if not found / command unavailable.
    """
    system = platform.system()

    try:
        if system == "Linux":
            # Use rocminfo
            result = subprocess.run(['rocminfo'], capture_output=True, text=True, check=True)
            output = result.stdout

            # Most reliable: find a line like "Name: gfx1030"
            match = re.search(r'Name:\s*(gfx[0-9a-f]+)', output, re.IGNORECASE)
            if match:
                return match.group(1)

            # Fallback: search for any "gfxXXXX" string
            fallback = re.search(r'gfx[0-9a-f]+', output)
            if fallback:
                return fallback.group(0)

        elif system == "Windows":
            # Use hipinfo (available with ROCm on Windows)
            result = subprocess.run(['hipinfo'], capture_output=True, text=True, check=True)
            output = result.stdout
            match = re.search(r'gcnArchName:\s*(\S+)', output)
            if match:
                return match.group(1)

        # For other OSes (macOS, etc.) or if the above fail, try rocm_agent_enumerator
        result = subprocess.run(['rocm_agent_enumerator'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split()
        if lines:
            return lines[0]   # usually the first GPU

    except (subprocess.SubprocessError, FileNotFoundError):
        # Command not found or execution failed
        pass

    return None

print(get_gfx_arch())