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
print(json.dumps(namespace["_onnx_light_source_build_info"]()))
"""


class TestSourceRuntimeSelection(unittest.TestCase):
    def _check_runtime(self, *, symlinks=False, foreign=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            package = root / "onnx_light"
            runtime = package / "onnx_py"
            runtime.mkdir(parents=True)
            header = package / "onnx_core/runtime/kernels/kernel_dispatch_table.h"
            header.parent.mkdir(parents=True)
            header.touch()
            (package / "__init__.py").touch()
            (runtime / "__init__.py").touch()
            artifacts = root / "build" if symlinks or foreign else runtime
            artifacts.mkdir(exist_ok=True)
            extension_name = "_onnxpyprotoop" + importlib.machinery.EXTENSION_SUFFIXES[0]
            for name in (extension_name, "liblib_onnx_core.so", "liblib_onnx_proto.so"):
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
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    f"extension was found at {artifacts / extension_name}", result.stderr
                )
                self.assertIn(f"expected it in {runtime}", result.stderr)
            else:
                self.assertEqual(result.returncode, 0, result.stderr)
                info = json.loads(result.stdout)
                self.assertEqual(info["include_dir"], str(package))
                self.assertEqual(info["core_library"], str(artifacts / "liblib_onnx_core.so"))
                self.assertEqual(info["proto_library"], str(artifacts / "liblib_onnx_proto.so"))

    def test_inplace_files(self):
        self._check_runtime()

    @unittest.skipIf(sys.platform == "win32", "Symbolic links require Windows privileges.")
    def test_inplace_symlinks(self):
        self._check_runtime(symlinks=True)

    def test_foreign_extension(self):
        self._check_runtime(foreign=True)


if __name__ == "__main__":
    unittest.main()
