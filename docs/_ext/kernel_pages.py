# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Generate one stable documentation page per registered kernel.

Unlike the previous ``kernel_scan``/``onnx_kernels`` extension, this module
does not read any C++ source. It consumes only the public Python inventory,
``onnx_light_cpu.registered_kernels()`` (see xadupre/onnx-light-cpu#349), so
the generated pages always mirror exactly the C++ registrations the runtime
executes: adding, renaming, or removing a ``Register*Kernel[s]`` call updates
the generated index and pages without editing any documentation-side list.

Generation is deterministic: :func:`registered_kernels` already returns
records sorted by ``(domain, op_type, device, kernel_name)``, filenames are
derived from that same key, and disambiguated in that same stable order on
the rare collision, so two consecutive generations produce byte-identical
output. Stale pages -- left over from a registration that was renamed or
removed -- are deleted so the generated directory always matches the current
inventory exactly.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, NamedTuple, Sequence, Tuple

#: Name of the directory (relative to the Sphinx source directory) that holds
#: the generated index and per-kernel pages. Entirely build-generated: every
#: ``*.rst`` file under it not produced by the current inventory is removed.
GENERATED_DIR_NAME = "kernels_generated"

_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


class KernelRecord(NamedTuple):
    """Documentation-side mirror of ``onnx_light_cpu.RegisteredKernel``."""

    domain: str
    op_type: str
    device: str
    kernel_name: str
    types: Tuple[str, ...]
    since_version: int | None
    until_version: int | None


def load_registered_kernels() -> List[KernelRecord]:
    """Returns one :class:`KernelRecord` per ``onnx_light_cpu`` registration.

    Imports ``onnx_light_cpu`` lazily so importing this module (e.g. from
    tests) never requires the compiled extension to be built.
    """
    from onnx_light_cpu import registered_kernels

    return [
        KernelRecord(
            domain=record.domain,
            op_type=record.op_type,
            device=record.device,
            kernel_name=record.kernel_name,
            types=tuple(record.types),
            since_version=record.since_version,
            until_version=record.until_version,
        )
        for record in registered_kernels()
    ]


def _slugify(*parts: str) -> str:
    """Returns a lowercase, filesystem-safe stem built from ``parts``."""
    text = "_".join(parts)
    text = _SLUG_RE.sub("_", text).strip("_")
    return text.lower() or "kernel"


def stem_for_record(record: KernelRecord) -> str:
    """Returns the (possibly colliding) base filename stem for ``record``."""
    return _slugify(record.domain, record.op_type, record.device)


def assign_stems(records: Sequence[KernelRecord]) -> OrderedDict[str, KernelRecord]:
    """Returns ``{stem: record}`` with one deterministic, collision-free stem per record.

    ``records`` is assumed to already be in the deterministic order returned
    by ``onnx_light_cpu.registered_kernels()``. When two records would
    otherwise slugify to the same base stem, the later one (in that same
    stable order) is disambiguated by appending ``_2``, ``_3``, ... so no
    generated page is ever silently overwritten by another.
    """
    used: set = set()
    mapping: OrderedDict[str, KernelRecord] = OrderedDict()
    for record in records:
        base = stem_for_record(record)
        stem = base
        suffix = 2
        while stem in used:
            stem = f"{base}_{suffix}"
            suffix += 1
        used.add(stem)
        mapping[stem] = record
    return mapping


def _format_opset_bounds(record: KernelRecord) -> str:
    bounds = []
    if record.since_version is not None:
        bounds.append(f"since opset {record.since_version}")
    if record.until_version is not None:
        bounds.append(f"until opset {record.until_version}")
    return ", ".join(bounds) if bounds else "none"


def render_kernel_page(record: KernelRecord) -> str:
    """Returns the RST source of the page documenting one registration."""
    title = f"{record.op_type} ({record.device})"
    types = ", ".join(f"``{t}``" for t in record.types) if record.types else "none declared"
    lines = [
        title,
        "=" * len(title),
        "",
        f"* Domain: ``{record.domain}``",
        f"* Device: ``{record.device}``",
        f"* Kernel: ``{record.kernel_name}``",
        f"* Supported types: {types}",
        f"* Opset bounds: {_format_opset_bounds(record)}",
        "",
    ]
    return "\n".join(lines)


def render_index(stems: Iterable[str]) -> str:
    """Returns the RST source of the generated index toctree page."""
    lines = [
        "Registered kernels",
        "===================",
        "",
        (
            "This page is generated automatically from"
            " ``onnx_light_cpu.registered_kernels()``. It lists every kernel"
            " registration this repository's runtime provides, one page per"
            " registration."
        ),
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "",
    ]
    for stem in stems:
        lines.append(f"   {stem}")
    lines.append("")
    return "\n".join(lines)


def _write_if_changed(path: Path, content: str) -> None:
    """Writes ``content`` to ``path`` unless it already holds exactly that text.

    Avoids needlessly bumping the file's mtime (which Sphinx and other tools
    may treat as "changed") when the generated content is identical to what
    is already on disk -- as it is on every rebuild where the inventory
    itself did not change.
    """
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def generate_kernel_pages(records: Sequence[KernelRecord], output_dir: Path) -> None:
    """Writes the index and per-kernel pages for ``records`` under ``output_dir``.

    Also removes any ``*.rst`` file already present under ``output_dir`` that
    is not part of the current generation (a page whose registration was
    renamed or removed), so the directory always matches ``records`` exactly.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stems = assign_stems(records)

    expected = {"index.rst"}
    for stem, record in stems.items():
        filename = f"{stem}.rst"
        expected.add(filename)
        _write_if_changed(output_dir / filename, render_kernel_page(record))
    _write_if_changed(output_dir / "index.rst", render_index(stems.keys()))

    for existing in output_dir.glob("*.rst"):
        if existing.name not in expected:
            existing.unlink()


def _generate(app) -> None:
    """``builder-inited`` callback: (re)generates the kernel pages in-place."""
    records = load_registered_kernels()
    output_dir = Path(app.srcdir) / GENERATED_DIR_NAME
    generate_kernel_pages(records, output_dir)


def setup(app):
    app.connect("builder-inited", _generate)
    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
