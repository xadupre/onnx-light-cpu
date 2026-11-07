"""Python bindings subpackage for onnx-light-cpu."""

import os
import sys
from importlib.util import find_spec
from pathlib import Path

_bindings_dir = Path(__file__).resolve().parent
if any(
    (Path(entry) / "onnx_light_cpu" / "onnx_py").resolve() == _bindings_dir
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if entry
):
    # Editable-install finders can override PYTHONPATH for individual extensions.
    # Check both origins before loading either module to avoid mixing native builds.
    for _module in ("_cpukernels", "_cpuregister"):
        _name = f"{__name__}.{_module}"
        _spec = find_spec(_name)
        if (
            _spec is not None
            and _spec.origin is not None
            and Path(_spec.origin).resolve().parent != _bindings_dir
        ):
            raise ImportError(
                f"PYTHONPATH selects onnx-light-cpu from {_bindings_dir}, but "
                f"{_name} was found at {_spec.origin}; expected it in {_bindings_dir}. "
                "A conflicting installation or editable-install finder may be "
                "overriding native imports. Use an environment without the conflicting "
                "installation and build the selected checkout in place; "
                "mixing native builds is not supported."
            )

if sys.platform == "win32":
    # Python's Windows extension loader does not search PATH for dependent DLLs.
    # Locate the selected package without importing its native extensions.
    _onnx_light_spec = find_spec("onnx_light")
    if _onnx_light_spec is not None and _onnx_light_spec.origin is not None:
        _onnx_light_dll_dir = Path(_onnx_light_spec.origin).resolve().parent / "onnx_py"
        if _onnx_light_dll_dir.is_dir():
            # Retain the handle so the directory remains available to later imports.
            _onnx_light_dll_handle = os.add_dll_directory(str(_onnx_light_dll_dir))
