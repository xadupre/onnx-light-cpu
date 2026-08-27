# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Run the documentation examples as unit tests.

Every ``docs/examples/**/plot_*.py`` gallery script is executed in a subprocess so
a failure (e.g. a broken import or a mismatched result) surfaces as a failing
test. All of their optional third-party dependencies (matplotlib, onnxruntime,
onnx-light, ...) are part of the ``dev`` extra, so they are assumed to be
installed rather than probed for and skipped.

``docs/examples/benchmarks`` compares onnx-light-cpu kernels against onnx-light's
``ReferenceEvaluator``/onnxruntime, so those examples additionally need
onnx-light-cpu's onnx-light integration extension (``_cpuregister``), which is
only built with ``ONNX_LIGHT_CPU_WITH_ONNX_LIGHT``/``--onnx-light-source``
(e.g. the plain ``core`` CI matrix builds onnx-light-cpu's standalone kernels
without that integration). Skip that whole directory rather than each
individual example when the extension is unavailable.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from unittest import TestCase

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _ROOT / "docs" / "examples"


def _example_files():
    return sorted(_EXAMPLES_DIR.rglob("plot_*.py"))


def _requires_onnx_light_integration(example):
    return example.parent.name == "benchmarks"


def _onnx_light_integration_available():
    return importlib.util.find_spec("onnx_light_cpu.onnx_py._cpuregister") is not None


class TestDocumentationExamples(TestCase):
    def test_documentation_examples(self):
        for example in _example_files():
            with self.subTest(example=example.name):
                if (
                    _requires_onnx_light_integration(example)
                    and not _onnx_light_integration_available()
                ):
                    self.skipTest(
                        "onnx-light-cpu was built without onnx-light integration "
                        "(ONNX_LIGHT_CPU_WITH_ONNX_LIGHT/--onnx-light-source)"
                    )

                env = dict(os.environ)
                python_path = env.get("PYTHONPATH")
                env["PYTHONPATH"] = (
                    str(_ROOT) if not python_path else str(_ROOT) + os.pathsep + python_path
                )
                # Use a non-interactive backend so ``plt.show()`` does not block or need a
                # display.
                env["MPLBACKEND"] = "Agg"
                # Shrink the benchmark examples (fewer/smaller sizes) so they run quickly as
                # unit tests while still exercising every code path.
                env["UNITTEST_GOING"] = "1"

                command = [sys.executable, "-u", str(example)]
                if _requires_onnx_light_integration(example):
                    command.extend(["-r", "2", "-w", "1", "-t", "0.1"])

                proc = subprocess.run(
                    command,
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
