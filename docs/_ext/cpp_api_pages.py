# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Generate the C++ API pages from the public declarations in Doxygen XML."""

from xml.etree import ElementTree


def _source_path(location):
    return location.get("file", "").replace("\\", "/").split("onnx_light_cpu/", 1)[-1]


def collect_api_symbols(xml_dir):
    """Return the public kernel types and custom-operator declarations."""
    kernels = set()
    custom = set()
    index = ElementTree.parse(xml_dir / "index.xml").getroot()
    for compound in index.findall("compound"):
        kind = compound.get("kind")
        if kind not in {"class", "struct", "namespace"}:
            continue
        definition = ElementTree.parse(xml_dir / f"{compound.get('refid')}.xml").getroot()
        if kind == "namespace":
            if compound.findtext("name") != "onnx_light_cpu":
                continue
            for member in definition.findall(".//memberdef[@kind='function']"):
                location = member.find("location")
                name = member.findtext("name")
                if location is None or not name:
                    continue
                path = _source_path(location)
                if path.startswith(("shapes/", "schemas/", "gradient/")):
                    custom.add(("doxygenfunction", f"onnx_light_cpu::{name}"))
            continue
        entry = definition.find("compounddef")
        if entry is None:
            continue
        location = entry.find("location")
        name = entry.findtext("compoundname")
        if location is None or not name or not name.startswith("onnx_light_cpu::"):
            continue
        path = _source_path(location)
        short_name = name.rsplit("::", 1)[-1]
        if path.startswith("kernels/") and short_name.endswith(("Kernel", "Result")):
            kernels.add((f"doxygen{kind}", name))
        elif path.startswith("patterns/") and short_name.endswith("FusionPattern"):
            custom.add((f"doxygen{kind}", name))
    return sorted(kernels), sorted(custom)


def _render(title, intro, symbols, notes=None):
    lines = [title, "-" * len(title), "", intro, ""]
    for directive, name in symbols:
        lines.extend([f".. {directive}:: {name}", "   :project: onnx_light_cpu"])
        if directive != "doxygenfunction":
            lines.append("   :members:")
        lines.append("")
    if notes:
        lines.extend([f".. include:: {notes}", ""])
    return "\n".join(lines)


def generate_cpp_api_pages(xml_dir, output_dir):
    """Write both pages, without touching unchanged files on incremental builds."""
    kernels, custom = collect_api_symbols(xml_dir)
    pages = {
        "kernels.rst": _render(
            "Kernel classes",
            "Public kernel implementations declared in ``onnx_light_cpu/kernels``.",
            kernels,
            "kernels_notes.rst",
        ),
        "custom_operators.rst": _render(
            "Custom operators",
            "The custom operator support inventory and its public implementations "
            "are available through these APIs.",
            custom,
        ),
    }
    for filename, text in pages.items():
        path = output_dir / filename
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
