# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

_EXT_DIR = Path(__file__).resolve().parents[2] / "docs" / "_ext"
if str(_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXT_DIR))

from cpp_api_pages import generate_cpp_api_pages  # noqa: E402


class TestCppApiPages(TestCase):
    def test_generates_from_public_declarations(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            xml = root / "xml"
            output = root / "api" / "cpp"
            xml.mkdir()
            output.mkdir(parents=True)
            (xml / "index.xml").write_text(
                '<doxygenindex><compound kind="class" refid="abs"/>'
                '<compound kind="struct" refid="result"/>'
                '<compound kind="class" refid="pattern"/>'
                '<compound kind="namespace" refid="shapes"><name>onnx_light_cpu</name></compound>'
                '<compound kind="class" refid="internal"/></doxygenindex>',
                encoding="utf-8",
            )
            for refid, kind, name, path in (
                ("abs", "class", "AbsKernel", "kernels/math/abs_kernel.h"),
                (
                    "result",
                    "struct",
                    "SimplifiedLayerNormalizationResult",
                    "kernels/math/simplified_layer_normalization_kernel.h",
                ),
                ("pattern", "class", "CDistFusionPattern", "patterns/com_microsoft/patterns.h"),
                ("internal", "class", "HiddenKernel", "impl/math/hidden.h"),
            ):
                (xml / f"{refid}.xml").write_text(
                    f'<doxygen><compounddef kind="{kind}">'
                    f"<compoundname>onnx_light_cpu::{name}</compoundname>"
                    f'<location file="../onnx_light_cpu/{path}"/>'
                    "</compounddef></doxygen>",
                    encoding="utf-8",
                )
            (xml / "shapes.xml").write_text(
                '<doxygen><compounddef kind="namespace"><sectiondef>'
                '<memberdef kind="function"><name>ComputeShapeCDist</name>'
                '<location file="../onnx_light_cpu/shapes/com_microsoft/shape_inference.h"/>'
                '</memberdef><memberdef kind="variable"><name>Ignored</name>'
                '<location file="../onnx_light_cpu/shapes/com_microsoft/shape_inference.h"/>'
                "</memberdef></sectiondef></compounddef></doxygen>",
                encoding="utf-8",
            )

            generate_cpp_api_pages(xml, output)
            kernels = (output / "kernels.rst").read_text(encoding="utf-8")
            custom = (output / "custom_operators.rst").read_text(encoding="utf-8")
            assert ".. doxygenclass:: onnx_light_cpu::AbsKernel" in kernels
            assert (
                ".. doxygenstruct:: onnx_light_cpu::SimplifiedLayerNormalizationResult" in kernels
            )
            assert "HiddenKernel" not in kernels
            assert ".. include:: kernels_notes.rst" in kernels
            assert ".. doxygenclass:: onnx_light_cpu::CDistFusionPattern" in custom
            assert ".. doxygenfunction:: onnx_light_cpu::ComputeShapeCDist" in custom
            assert "Ignored" not in custom

            old_mtime = (output / "kernels.rst").stat().st_mtime_ns
            generate_cpp_api_pages(xml, output)
            assert (output / "kernels.rst").stat().st_mtime_ns == old_mtime

            (xml / "index.xml").write_text(
                (xml / "index.xml")
                .read_text(encoding="utf-8")
                .replace(
                    "</doxygenindex>", '<compound kind="class" refid="new"/></doxygenindex>'
                ),
                encoding="utf-8",
            )
            (xml / "new.xml").write_text(
                '<doxygen><compounddef kind="class">'
                "<compoundname>onnx_light_cpu::NewKernel</compoundname>"
                '<location file="../onnx_light_cpu/kernels/math/new_kernel.h"/>'
                "</compounddef></doxygen>",
                encoding="utf-8",
            )
            generate_cpp_api_pages(xml, output)
            assert ".. doxygenclass:: onnx_light_cpu::NewKernel" in (
                output / "kernels.rst"
            ).read_text(encoding="utf-8")
