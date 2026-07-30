# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Sphinx extension rendering this repository's own CPU kernels as a table.

Provides the ``registered-kernels`` directive which scans **this repository's**
C++ kernel declarations (see :mod:`kernel_scan`) and emits a table with one row
per operator, listing all supported data types together in a single column.
Only kernels provided by this repository
(``onnx-light-cpu``) are listed; the kernels shipped by onnx-light itself are
not scanned. Because the table is generated at build time, it always reflects
the kernels this repository actually provides.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from docutils import nodes
from docutils.parsers.rst import Directive, directives

from kernel_scan import (
    DEFAULT_GLOBS,
    Kernel,
    find_registered_kernels,
    group_by_operator,
    iter_source_files,
    operator_function,
)

_HEADERS = ("Operator", "Data types", "Function")


def _row(cells: Sequence[str]) -> nodes.row:
    row = nodes.row()
    for cell in cells:
        entry = nodes.entry()
        entry += nodes.paragraph(text=str(cell))
        row += entry
    return row


def _build_table(kernels: List[Kernel]) -> nodes.table:
    table = nodes.table()
    table["classes"].append("registered-kernels")
    tgroup = nodes.tgroup(cols=len(_HEADERS))
    table += tgroup
    for _ in _HEADERS:
        tgroup += nodes.colspec(colwidth=1)

    thead = nodes.thead()
    thead += _row(_HEADERS)
    tgroup += thead

    tbody = nodes.tbody()
    for operator, entries in group_by_operator(kernels).items():
        dtypes = ", ".join(dtype for dtype, _ in entries)
        function = operator_function(operator)
        tbody += _row((operator, dtypes, function))
    tgroup += tbody
    return table


class RegisteredKernelsDirective(Directive):
    """Render a table of the kernels provided by this repository."""

    has_content = False
    option_spec = {"glob": directives.unchanged}

    def run(self) -> List[nodes.Node]:
        env = self.state.document.settings.env
        root = Path(env.srcdir).parent
        globs = tuple(self.options["glob"].split()) if "glob" in self.options else DEFAULT_GLOBS

        # Rebuild the page whenever a scanned source file changes.
        for path in iter_source_files(root, globs):
            env.note_dependency(str(path))

        kernels = find_registered_kernels(root, globs)
        if not kernels:
            warning = self.state_machine.reporter.warning(
                f"registered-kernels: no kernels found under {root} for globs {globs}.",
                line=self.lineno,
            )
            paragraph = nodes.paragraph(text="No kernels are currently available.")
            return [paragraph, warning]

        return [_build_table(kernels)]


def setup(app):
    app.add_directive("registered-kernels", RegisteredKernelsDirective)
    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
