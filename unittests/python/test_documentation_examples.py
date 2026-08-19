# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Run the documentation examples as unit tests.

Every ``docs/examples/**/plot_*.py`` gallery script is executed in a subprocess so
a failure (e.g. a broken import or a mismatched result) surfaces as a failing
test. Examples that need optional dependencies which are not installed are
skipped instead of failing.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _ROOT / "docs" / "examples"

# Optional third-party modules each example needs on top of ``onnx-light-cpu``
# and ``numpy``. When any of them is missing the example is skipped. Examples
# not listed here fall back to requiring ``matplotlib`` (every gallery plot
# script imports it).
_REQUIREMENTS = {
    "plot_abs_simd.py": ("matplotlib",),
    "plot_abs_benchmark.py": ("matplotlib", "onnx", "onnxruntime", "onnx_light"),
    "plot_gemm_benchmark.py": ("matplotlib", "onnx", "onnxruntime", "onnx_light"),
    "plot_gemm_dtype_benchmark.py": ("matplotlib", "onnx", "onnx_light", "ml_dtypes"),
}


def _example_files():
    return sorted(_EXAMPLES_DIR.rglob("plot_*.py"))


def _missing_modules(names):
    return [name for name in names if importlib.util.find_spec(name) is None]


@pytest.mark.parametrize("example", _example_files(), ids=lambda p: p.name)
def test_documentation_example(example):
    required = _REQUIREMENTS.get(example.name, ("matplotlib",))
    missing = _missing_modules(required)
    if missing:
        pytest.skip(f"missing modules: {', '.join(missing)}")

    env = dict(os.environ)
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(_ROOT) if not python_path else str(_ROOT) + os.pathsep + python_path
    # Use a non-interactive backend so ``plt.show()`` does not block or need a
    # display.
    env["MPLBACKEND"] = "Agg"
    # Shrink the benchmark examples (fewer/smaller sizes) so they run quickly as
    # unit tests while still exercising every code path.
    env["UNITTEST_GOING"] = "1"

    proc = subprocess.run(
        [sys.executable, "-u", str(example)],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"Example {example.name!r} failed with return code {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
