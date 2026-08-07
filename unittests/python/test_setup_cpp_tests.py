# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``build_ext`` options of ``setup.py``.

``--cpp-tests`` builds the C++ unit tests and then runs them with ``ctest``.
``--onnx-light`` enables the onnx-light kernel-registration integration against
a locally built, importable onnx-light. ``--onnx-light-source`` enables the same
integration but builds onnx-light from a local source tree (auto-discovered from
the importable onnx-light) instead of ``find_package``. These tests exercise the
dry-run wiring (which prints the commands without executing them) so they run
quickly without compiling the extension or requiring onnx-light to be installed.
"""

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SETUP_PY = _ROOT / "setup.py"


def _dry_run(*extra_args):
    result = subprocess.run(
        [
            sys.executable,
            str(_SETUP_PY),
            "build_ext",
            "--inplace",
            *extra_args,
            "--dry-run",
        ],
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


class TestSetupOnnxLight:
    def test_onnx_light_flag_enables_integration(self):
        output = _dry_run("--onnx-light")
        assert "ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON" in output

    def test_onnx_light_source_flag_enables_integration(self):
        output = _dry_run("--onnx-light-source")
        assert "ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON" in output

    def test_without_flag_skips_integration(self):
        output = _dry_run()
        assert "ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON" not in output
