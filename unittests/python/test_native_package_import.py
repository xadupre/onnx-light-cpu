# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Checks that DLL discovery does not initialize the optional native runtime."""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from textwrap import dedent
import unittest


class TestNativePackageImport(unittest.TestCase):
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
        env.pop("PYTHONPATH", None)
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
