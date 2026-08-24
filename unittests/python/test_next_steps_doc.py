# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Ensure the next-steps table remains interactive."""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def test_next_steps_table_is_sortable_and_filterable():
    conf = (_ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
    next_steps = (_ROOT / "docs" / "next_steps" / "index.rst").read_text(encoding="utf-8")

    assert conf.index('"sphinxcontrib.jquery"') < conf.index('"sphinx_datatables"')
    assert ":class: sphinx-datatable" in next_steps
