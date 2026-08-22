# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Ensure ``docs/api/index.rst`` documents the current public Python API.

The top-level ``onnx_light_cpu`` package re-exports its public helpers through
``__all__``. The Python API reference in ``docs/api/index.rst`` must list every one of
them so the documentation reflects the current status of the project.
"""

from pathlib import Path

import onnx_light_cpu

_API_RST = Path(__file__).resolve().parents[2] / "docs" / "api" / "index.rst"


class TestApiDocReflectsPublicApi:
    def test_every_public_symbol_is_documented(self):
        text = _API_RST.read_text(encoding="utf-8")
        # Restrict the search to the top-level ``onnx_light_cpu`` package section
        # so a function documented only for one of the compiled extension
        # modules does not count as documenting the top-level re-export.
        marker = ".. py:module:: onnx_light_cpu\n"
        assert marker in text, "docs/api/index.rst is missing the onnx_light_cpu module directive"
        section = text.split(marker, 1)[1]
        for name in onnx_light_cpu.__all__:
            # Functions are documented with ``.. py:function::``; classes (e.g.
            # the ``RegisteredKernel`` NamedTuple) with ``.. py:class::``.
            assert f".. py:function:: {name}" in section or f".. py:class:: {name}" in section, (
                f"docs/api/index.rst does not document the public symbol {name!r} "
                "in the onnx_light_cpu package section"
            )
