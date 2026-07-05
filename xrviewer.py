"""Compatibility wrapper for the OpenXR viewer package.

The implementation lives in :mod:`xr_viewer.implementation`; this module keeps
legacy imports such as ``from xrviewer import OpenXRViewer`` working for
``main.py`` and existing scripts.
"""

from xr_viewer import OPENXR_AVAILABLE, OpenXRViewer, load_glb_model
from xr_viewer.implementation import _run_standalone_test

__all__ = ["OPENXR_AVAILABLE", "OpenXRViewer", "load_glb_model"]

if __name__ == "__main__":
    _run_standalone_test()
