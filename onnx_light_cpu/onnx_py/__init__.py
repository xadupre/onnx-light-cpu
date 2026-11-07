"""Python bindings subpackage for onnx-light-cpu."""

import sys

if sys.platform == "win32":
    import os
    from importlib.util import find_spec
    from pathlib import Path

    # Python's Windows extension loader does not search PATH for dependent DLLs.
    # Locate the selected package without importing its native extensions.
    _onnx_light_spec = find_spec("onnx_light")
    if _onnx_light_spec is not None and _onnx_light_spec.origin is not None:
        _onnx_light_dll_dir = Path(_onnx_light_spec.origin).resolve().parent / "onnx_py"
        if _onnx_light_dll_dir.is_dir():
            # Retain the handle so the directory remains available to later imports.
            _onnx_light_dll_handle = os.add_dll_directory(str(_onnx_light_dll_dir))
