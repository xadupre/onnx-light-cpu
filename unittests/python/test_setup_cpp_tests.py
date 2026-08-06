# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``--cpp-tests`` option of ``setup.py``.

The option builds the C++ unit tests and then runs them with ``ctest``. These
tests exercise the dry-run wiring (which prints the commands without executing
them) so they run quickly without compiling the extension.
"""

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SETUP_PY = _ROOT / "setup.py"


def _dry_run(*extra_args):
    result = subprocess.run(
        [sys.executable, str(_SETUP_PY), "build_ext", "--inplace", *extra_args, "--dry-run"],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout + result.stderr


class TestSetupCppTests:
    def test_cpp_tests_flag_runs_ctest(self):
        output = _dry_run("--cpp-tests")
        assert "ONNX_LIGHT_CPU_BUILD_TESTS=ON" in output
        assert "ctest" in output

    def test_without_flag_skips_ctest(self):
        output = _dry_run()
        assert "ONNX_LIGHT_CPU_BUILD_TESTS=ON" not in output
        assert "ctest" not in output
