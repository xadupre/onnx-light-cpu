# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Native-origin preflight shared by isolated import-test subprocesses."""

from importlib.util import find_spec
from pathlib import Path


def _check_native_import_origins(root):
    bindings_dir = (Path(root) / "onnx_light_cpu" / "onnx_py").resolve()
    for module in ("_cpukernels", "_cpuregister"):
        name = f"onnx_light_cpu.onnx_py.{module}"
        spec = find_spec(name)
        if (
            spec is not None
            and spec.origin is not None
            and Path(spec.origin).resolve().parent != bindings_dir
        ):
            raise ImportError(
                f"PYTHONPATH selects onnx-light-cpu from {bindings_dir}, but "
                f"{name} was found at {spec.origin}; expected it in {bindings_dir}. "
                "A conflicting installation or editable-install finder may be "
                "overriding native imports. Use an environment without the conflicting "
                "installation and build the selected checkout in place; "
                "mixing native builds is not supported."
            )
