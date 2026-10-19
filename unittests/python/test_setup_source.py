# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Checks source-runtime selection using real package paths and symbolic links."""

import importlib.machinery
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_SETUP = Path(__file__).resolve().parents[2] / "setup.py"
_SCRIPT = """
import ast
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
tree.body = [
    node for node in tree.body
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
]
namespace = {"__file__": str(path)}
exec(compile(tree, str(path), "exec"), namespace)
info = namespace["_onnx_light_source_build_info"]()
cmake_args = namespace["_add_onnx_light_source_defines"]([], info)
print(json.dumps({"info": info, "cmake_args": cmake_args}))
"""


def _check_runtime(*, symlinks=False, foreign=False, threads=False):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        package = root / "onnx_light"
        runtime = package / "onnx_py"
        runtime.mkdir(parents=True)
        header = package / "onnx_core/runtime/kernels/kernel_dispatch_table.h"
        header.parent.mkdir(parents=True)
        header.touch()
        if threads:
            thread_header = root / "onnx_threads/include/onnx_threads/config.h"
            thread_header.parent.mkdir(parents=True)
            thread_header.touch()
        (package / "__init__.py").touch()
        (runtime / "__init__.py").touch()
        artifacts = root / "build" if symlinks or foreign else runtime
        artifacts.mkdir(exist_ok=True)
        extension_name = "_onnxpyprotoop" + importlib.machinery.EXTENSION_SUFFIXES[0]
        names = [extension_name, "liblib_onnx_core.so", "liblib_onnx_proto.so"]
        if threads:
            names.append("liblib_onnx_threads.so")
        for name in names:
            artifact = artifacts / name
            artifact.touch()
            if symlinks:
                (runtime / name).symlink_to(artifact)
        if foreign:
            (runtime / "__init__.py").write_text(
                f"__path__ = [{str(artifacts)!r}]\n", encoding="utf-8"
            )
        env = dict(os.environ, PYTHONPATH=str(root))
        result = subprocess.run(
            [sys.executable, "-c", _SCRIPT, str(_SETUP)],
            env=env,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if foreign:
            assert result.returncode != 0
            assert f"extension was found at {artifacts / extension_name}" in result.stderr
            assert f"expected it in {runtime}" in result.stderr
        else:
            assert result.returncode == 0, result.stderr
            output = json.loads(result.stdout)
            info = output["info"]
            assert info["include_dir"] == str(package)
            assert info["core_library"] == str(artifacts / "liblib_onnx_core.so")
            assert info["proto_library"] == str(artifacts / "liblib_onnx_proto.so")
            if threads:
                assert info["threads_library"] == str(artifacts / "liblib_onnx_threads.so")
                assert (
                    f"-DONNX_LIGHT_CPU_ONNX_LIGHT_THREADS_LIBRARY={info['threads_library']}"
                    in output["cmake_args"]
                )
            else:
                assert "threads_library" not in info


def test_inplace_files():
    _check_runtime()


@unittest.skipIf(sys.platform == "win32", "Symbolic links require Windows privileges.")
def test_inplace_symlinks():
    _check_runtime(symlinks=True)


def test_foreign_extension():
    _check_runtime(foreign=True)


def test_inplace_threads():
    _check_runtime(threads=True)


@unittest.skipIf(sys.platform == "win32", "Symbolic links require Windows privileges.")
def test_inplace_threads_symlinks():
    _check_runtime(symlinks=True, threads=True)


def test_cmake_source_header_layouts():
    for layout in ("legacy", "proto", "threads"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "onnx-light"
            package = source / "onnx_light"
            package.mkdir(parents=True)
            if layout != "legacy":
                proto = source / "onnx_proto/include/onnx_proto"
                helpers = source / "onnx_proto/helpers/include"
            else:
                proto = package / "onnx_proto"
                helpers = package / "onnx_helpers/include"
            proto.mkdir(parents=True)
            helpers.mkdir(parents=True)
            (proto / "onnx.h").touch()
            runtime = package / "onnx_py"
            runtime.mkdir()
            suffix = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
            libraries = {}
            components = ["core", "proto", "kernels"]
            if layout == "threads":
                thread_header = source / "onnx_threads/include/onnx_threads/config.h"
                thread_header.parent.mkdir(parents=True)
                thread_header.touch()
                components.append("threads")
            for component in components:
                library = runtime / f"liblib_onnx_{component}{suffix}"
                library.touch()
                libraries[component] = library
            command = [
                "cmake",
                "-S",
                str(_SETUP.parent),
                "-B",
                str(root / "build"),
                "-DONNX_LIGHT_CPU_BUILD_PYTHON=OFF",
                "-DONNX_LIGHT_CPU_BUILD_TESTS=OFF",
                "-DONNX_LIGHT_CPU_INSTALL=OFF",
                "-DONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON",
                f"-DONNX_LIGHT_CPU_ONNX_LIGHT_SOURCE_DIR={source}",
                f"-DONNX_LIGHT_CPU_ONNX_LIGHT_LIBRARY={libraries['core']}",
                f"-DONNX_LIGHT_CPU_ONNX_LIGHT_PROTO_LIBRARY={libraries['proto']}",
                f"-DONNX_LIGHT_CPU_ONNX_LIGHT_KERNELS_LIBRARY={libraries['kernels']}",
            ]
            if layout == "threads":
                command.append(
                    f"-DONNX_LIGHT_CPU_ONNX_LIGHT_THREADS_LIBRARY={libraries['threads']}"
                )
            if sys.platform == "win32":
                command.extend(
                    f"-DONNX_LIGHT_CPU_ONNX_LIGHT"
                    f"{'' if component == 'core' else '_' + component.upper()}_IMPLIB={library}"
                    for component, library in libraries.items()
                )
            result = subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=120
            )
            assert result.returncode == 0, result.stdout + result.stderr
            if layout == "threads":
                command.append("-DONNX_LIGHT_CPU_ONNX_LIGHT_THREADS_LIBRARY=")
                result = subprocess.run(
                    command, capture_output=True, text=True, check=False, timeout=120
                )
                assert result.returncode == 0, result.stdout + result.stderr
                libraries["threads"].unlink()
                result = subprocess.run(
                    command, capture_output=True, text=True, check=False, timeout=120
                )
                assert result.returncode != 0
                assert "threading shared library is required" in result.stderr
