# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Checks native import origins and optional runtime discovery without loading binaries."""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from textwrap import dedent
from unittest import TestCase


class TestNativePackageImport(TestCase):
    def test_editable_finder_native_origins(self):
        root = Path(__file__).resolve().parents[2]
        script = dedent("""
            import sys
            from importlib import import_module
            from importlib.machinery import EXTENSION_SUFFIXES
            from importlib.util import spec_from_file_location
            from pathlib import Path

            root = Path(sys.argv[1])
            foreign = Path(sys.argv[2])
            conflicting = sys.argv[3]
            first = sys.argv[4]
            source_selected = sys.argv[5] == "True"
            local = root / "onnx_light_cpu" / "onnx_py"
            names = ("onnx_light_cpu.onnx_py._cpukernels",
                     "onnx_light_cpu.onnx_py._cpuregister")

            class EditableFinder:
                @classmethod
                def find_spec(cls, fullname, path=None, target=None):
                    if fullname in names:
                        directory = foreign if fullname == conflicting else local
                        return spec_from_file_location(
                            fullname,
                            directory / (fullname.rsplit(".", 1)[1] + EXTENSION_SUFFIXES[0]),
                        )
                    return None

            # Match an editable finder that takes precedence over PathFinder.
            sys.meta_path.insert(0, EditableFinder)
            sys.path.insert(0, str(root))
            try:
                if conflicting and source_selected:
                    import_module(first)
                else:
                    import_module("onnx_light_cpu.onnx_py")
            except ImportError as exc:
                assert conflicting and source_selected, str(exc)
                message = str(exc)
                assert conflicting in message, message
                assert str(foreign) in message, message
                assert str(local) in message, message
                assert "editable-install finder" in message, message
                assert "mixing native builds is not supported" in message, message
            else:
                assert not (conflicting and source_selected), "Conflict was not diagnosed"
            assert all(name not in sys.modules for name in names)
            assert "onnx_light" not in sys.modules
            """)
        modules = (
            "onnx_light_cpu.onnx_py._cpukernels",
            "onnx_light_cpu.onnx_py._cpuregister",
        )
        with TemporaryDirectory() as temporary:
            foreign = Path(temporary).resolve() / "site-packages" / "onnx_light_cpu" / "onnx_py"
            for source_selected in (True, False):
                env = os.environ.copy()
                env.pop("PYTHONPATH", None)
                if source_selected:
                    env["PYTHONPATH"] = str(root)
                for conflicting in ("", *modules):
                    for cwd in (root, Path(temporary)):
                        for first in modules:
                            with self.subTest(
                                source_selected=source_selected,
                                conflicting=conflicting,
                                cwd=str(cwd),
                                first=first,
                            ):
                                result = subprocess.run(
                                    [
                                        sys.executable,
                                        "-S",
                                        "-c",
                                        script,
                                        str(root),
                                        str(foreign),
                                        conflicting,
                                        first,
                                        str(source_selected),
                                    ],
                                    cwd=cwd,
                                    env=env,
                                    capture_output=True,
                                    text=True,
                                    timeout=60,
                                    check=False,
                                )
                                self.assertEqual(
                                    result.returncode, 0, f"{result.stdout}\n{result.stderr}"
                                )

    def test_optional_runtime_discovery(self):
        root = Path(__file__).resolve().parents[2]
        script = dedent("""
            import sys
            from pathlib import Path

            # -S excludes installed packages; keep only this checkout and stdlib.
            sys.path = [sys.argv[1], *sys.path[1:]]
            runtime = Path(sys.argv[2])
            if runtime.is_dir():
                sys.path.insert(0, str(runtime))
            import onnx_light_cpu.onnx_py as bindings

            assert "onnx_light" not in sys.modules
            assert "onnx_light_cpu.onnx_py._cpukernels" not in sys.modules
            assert "onnx_light_cpu.onnx_py._cpuregister" not in sys.modules
            if sys.platform == "win32" and (runtime / "onnx_light" / "onnx_py").is_dir():
                assert bindings._onnx_light_dll_handle.path == str(
                    runtime / "onnx_light" / "onnx_py"
                )
            else:
                assert not hasattr(bindings, "_onnx_light_dll_handle")
            """)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        with TemporaryDirectory() as temporary:
            runtime = Path(temporary).resolve() / "runtime"
            for state in ("absent", "unbuilt", "built"):
                with self.subTest(state=state):
                    if state == "unbuilt":
                        package = runtime / "onnx_light"
                        package.mkdir(parents=True)
                        (package / "__init__.py").write_text(
                            'raise AssertionError("DLL discovery must not import onnx-light")\n',
                            encoding="utf-8",
                        )
                    elif state == "built":
                        (runtime / "onnx_light" / "onnx_py").mkdir()
                    result = subprocess.run(
                        [sys.executable, "-S", "-c", script, str(root), str(runtime)],
                        cwd=temporary,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=60,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
