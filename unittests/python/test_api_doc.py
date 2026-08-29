# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Ensure ``docs/api/python`` documents the current public Python API.

The top-level ``onnx_light_cpu`` package re-exports its public helpers through
``__all__``. The Python API reference must list every one of them so the
documentation reflects the current status of the project.
"""

from pathlib import Path

import onnx_light_cpu

_PYTHON_API_DIR = Path(__file__).resolve().parents[2] / "docs" / "api" / "python"
_PYTHON_API_TOPICS = (
    "registration",
    "kernel_inventory",
    "custom_operators",
    "processor_performance",
)


class TestApiDocReflectsPublicApi:
    def test_every_public_symbol_is_documented(self):
        assert _PYTHON_API_DIR.is_dir(), "docs/api/python is missing"
        python_index_path = _PYTHON_API_DIR / "index.rst"
        assert python_index_path.is_file(), "docs/api/python/index.rst is missing"
        topic_paths = [_PYTHON_API_DIR / f"{topic}.rst" for topic in _PYTHON_API_TOPICS]
        assert all(path.is_file() for path in topic_paths), "Python API topic page is missing"
        marker = ".. py:module:: onnx_light_cpu\n"
        python_index = python_index_path.read_text(encoding="utf-8")
        assert marker in python_index, "Python API module directive is missing"
        section = "\n".join(
            [
                python_index.split(marker, 1)[1],
                *(path.read_text(encoding="utf-8") for path in topic_paths),
            ]
        )
        for name in onnx_light_cpu.__all__:
            # Functions are documented with ``.. py:function::``; classes (e.g.
            # the ``RegisteredKernel`` NamedTuple) with ``.. py:class::``.
            assert f".. py:function:: {name}" in section or f".. py:class:: {name}" in section, (
                f"docs/api/python does not document the public symbol {name!r} "
                "in the onnx_light_cpu package section"
            )
