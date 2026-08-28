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
from typing import Any, Iterable, List, NamedTuple, Sequence, Tuple

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


class SupportRecord(NamedTuple):
    """Documentation-side mirror of ``onnx_light_cpu.OperatorSupport``."""

    domain: str
    op_type: str
    shape_inference_function: str
    peak_memory_function: str
    fusion_patterns: Tuple[str, ...]
    has_gradient: bool


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


def load_operator_support() -> List[SupportRecord]:
    """Returns the custom operator support without mutating any registry."""
    from onnx_light_cpu import operator_support

    return [
        SupportRecord(
            domain=record.domain,
            op_type=record.op_type,
            shape_inference_function=record.shape_inference_function,
            peak_memory_function=record.peak_memory_function,
            fusion_patterns=tuple(record.fusion_patterns),
            has_gradient=record.has_gradient,
        )
        for record in operator_support()
    ]


def load_custom_schemas() -> List[Any]:
    """Returns the non-standard ``LightOpSchema`` records."""
    from onnx_light_cpu import custom_op_schemas

    return list(custom_op_schemas())


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


def _cpp_role(kind: str, qualified_name: str) -> str:
    return f":cpp:{kind}:`{qualified_name}`"


def _schema_type_name(value: object) -> str:
    name = getattr(value, "name", None)
    type_name = str(name or value).rsplit(".", maxsplit=1)[-1]
    if type_name.startswith("k") and len(type_name) > 1 and type_name[1].isupper():
        return f"tensor({type_name[1:].lower()})"
    return type_name


def render_light_op_schema(schema: Any) -> str:
    """Returns the RST source for one non-standard ``LightOpSchema``."""
    lines = [
        "LightOpSchema",
        "~~~~~~~~~~~~~",
        "",
        f"**Version:** {schema.since_version}",
        "",
    ]
    if schema.doc:
        lines.extend([schema.doc.strip(), ""])
    for title, values in (("Inputs", schema.inputs), ("Outputs", schema.outputs)):
        lines.extend([f"**{title}**", ""])
        for value in values:
            type_name = getattr(value, "type_str", getattr(value, "type", ""))
            lines.append(f"- **{value.name}** (``{type_name}``): {value.description}")
        lines.append("")
    if schema.attributes:
        lines.extend(["**Attributes**", ""])
        attributes = (
            schema.attributes.values()
            if hasattr(schema.attributes, "values")
            else schema.attributes
        )
        for attribute in sorted(attributes, key=lambda value: value.name):
            qualifiers = ["required" if getattr(attribute, "required", False) else "optional"]
            default_value = getattr(attribute, "default_value_repr", None)
            if default_value is not None:
                qualifiers.append(f"default: ``{default_value}``")
            lines.append(
                f"- **{attribute.name}** (``{_schema_type_name(attribute.type)}``): "
                f"{attribute.description} ({', '.join(qualifiers)})"
            )
        lines.append("")
    if schema.type_constraints:
        lines.extend(["**Type constraints**", ""])
        for constraint in schema.type_constraints:
            allowed = ", ".join(
                sorted(_schema_type_name(value) for value in constraint.allowed_type_strs)
            )
            lines.append(
                f"- **{constraint.type_param_str}**: {constraint.description} "
                f"Allowed types: {allowed}."
            )
        lines.append("")
    return "\n".join(lines)


def _operator_support_lines(support: SupportRecord | None) -> List[str]:
    lines = ["Operator support", "----------------", ""]
    if support is None:
        return [
            *lines,
            (
                "The standard ONNX schema and any standard shape inference come from "
                "onnx-light. No additional shape, peak-memory, fusion-pattern, or gradient "
                "implementation is registered by onnx-light-cpu for this operator."
            ),
            "",
        ]

    patterns = ", ".join(_cpp_role("class", pattern) for pattern in support.fusion_patterns)
    gradient = (
        _cpp_role("func", "onnx_light_cpu::RegisterCustomOperatorGradients")
        if support.has_gradient
        else "Not provided"
    )
    return [
        *lines,
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 30 70",
        "",
        "   * - Capability",
        "     - Implementation",
        "   * - Shape inference",
        f"     - {_cpp_role('func', support.shape_inference_function)}",
        "   * - Peak memory",
        f"     - {_cpp_role('func', support.peak_memory_function)}",
        "   * - Fusion patterns",
        f"     - {patterns or 'None'}",
        "   * - Gradient",
        f"     - {gradient}",
        "",
    ]


def render_kernel_page(
    record: KernelRecord,
    schemas: Iterable[Any] = (),
    support: SupportRecord | None = None,
) -> str:
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
        *_operator_support_lines(support),
    ]
    matching_schemas = [
        schema
        for schema in schemas
        if schema.domain == record.domain and schema.name == record.op_type
    ]
    for schema in sorted(matching_schemas, key=lambda value: value.since_version):
        lines.append(render_light_op_schema(schema))
    return "\n".join(lines)


def render_index(stems: OrderedDict[str, KernelRecord]) -> str:
    """Returns the RST source of the generated ByOp index."""
    lines = [
        "Kernels",
        "-------",
        "",
        (
            "This catalogue is generated from ``onnx_light_cpu.registered_kernels()``. "
            "Each row links to one kernel registration and its available operator support."
        ),
        "",
        ".. datatables-js:: table.byop-kernel-table",
        "",
        "   {",
        "     pageLength: 10,",
        "     lengthMenu: [10, 100],",
        "     initComplete: function () {",
        "       const api = this.api();",
        "       const filters = $('<div class=\"byop-kernel-filters\"></div>');",
        "       const addFilter = function (columnIndex, title, allLabel) {",
        "         const column = api.column(columnIndex);",
        "         const label = $('<label></label>').text(title + ' ');",
        "         const select = $('<select></select>')",
        "           .attr('aria-label', title)",
        "           .append($('<option></option>').attr('value', '').text(allLabel))",
        "           .on('change', function () {",
        "             const value = $.fn.dataTable.util.escapeRegex(this.value);",
        "             column.search(value ? '^' + value + '$' : '', true, false).draw();",
        "           });",
        "         column.cache('search').unique().sort().each(function (value) {",
        "           select.append($('<option></option>').attr('value', value).text(value));",
        "         });",
        "         filters.append(label.append(select));",
        "       };",
        "       addFilter(0, 'Operator', 'All operators');",
        "       addFilter(1, 'Domain', 'All domains');",
        "       $(api.table().container()).prepend(filters);",
        "     }",
        "   }",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 22 22 12 30 14",
        "   :class: byop-kernel-table",
        "",
        "   * - Operator",
        "     - Domain",
        "     - Device",
        "     - Types",
        "     - Opset",
    ]
    for stem, record in stems.items():
        types = ", ".join(record.types) if record.types else "All"
        lines.extend(
            [
                f"   * - :doc:`{record.op_type} <kernels_generated/{stem}>`",
                f"     - ``{record.domain}``",
                f"     - ``{record.device}``",
                f"     - {types}",
                f"     - {_format_opset_bounds(record)}",
            ]
        )
    lines.extend(
        [
            "",
            ".. toctree::",
            "   :hidden:",
            "   :maxdepth: 1",
            "",
        ]
    )
    for stem in stems:
        lines.append(f"   kernels_generated/{stem}")
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


def generate_kernel_pages(
    records: Sequence[KernelRecord],
    output_dir: Path,
    schemas: Iterable[Any] = (),
    support_records: Iterable[SupportRecord] = (),
) -> None:
    """Writes the index and per-kernel pages for ``records`` under ``output_dir``.

    Also removes any ``*.rst`` file already present under ``output_dir`` that
    is not part of the current generation (a page whose registration was
    renamed or removed), so the directory always matches ``records`` exactly.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stems = assign_stems(records)
    schemas = tuple(schemas)
    support_by_operator = {(record.domain, record.op_type): record for record in support_records}

    expected = {"index.rst"}
    for stem, record in stems.items():
        filename = f"{stem}.rst"
        expected.add(filename)
        _write_if_changed(
            output_dir / filename,
            render_kernel_page(
                record,
                schemas=schemas,
                support=support_by_operator.get((record.domain, record.op_type)),
            ),
        )
    _write_if_changed(output_dir / "index.rst", render_index(stems))

    for existing in output_dir.glob("*.rst"):
        if existing.name not in expected:
            existing.unlink()


def _generate(app) -> None:
    """``builder-inited`` callback: (re)generates the kernel pages in-place."""
    records = load_registered_kernels()
    output_dir = Path(app.srcdir) / GENERATED_DIR_NAME
    generate_kernel_pages(
        records,
        output_dir,
        schemas=load_custom_schemas(),
        support_records=load_operator_support(),
    )


def setup(app):
    app.connect("builder-inited", _generate)
    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
