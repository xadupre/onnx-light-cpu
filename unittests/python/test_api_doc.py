# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Ensure ``docs/api.rst`` documents the current public Python API.

The top-level ``onnx_light_cpu`` package re-exports its public helpers through
``__all__``. The Python API reference in ``docs/api.rst`` must list every one of
them so the documentation reflects the current status of the project.
"""

from pathlib import Path

import onnx_light_cpu

_API_RST = Path(__file__).resolve().parents[2] / "docs" / "api.rst"


class TestApiDocReflectsPublicApi:
    def test_every_public_symbol_is_documented(self):
        text = _API_RST.read_text(encoding="utf-8")
        for name in onnx_light_cpu.__all__:
            assert f".. py:function:: {name}" in text, (
                f"docs/api.rst does not document the public function {name!r}"
            )
