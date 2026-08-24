# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Ensure the processor profile remains in its dedicated gallery."""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def test_processor_performance_has_dedicated_gallery():
    conf = (_ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
    examples = (_ROOT / "docs" / "examples.rst").read_text(encoding="utf-8")

    assert '"examples/processor"' in conf
    assert '"auto_examples/processor"' in conf
    assert "auto_examples/processor/index" in examples
    assert (_ROOT / "docs/examples/processor/plot_processor_performance.py").is_file()
    assert not (_ROOT / "docs/examples/benchmarks/plot_processor_performance.py").exists()
