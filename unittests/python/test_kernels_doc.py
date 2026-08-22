# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

_EXT_DIR = Path(__file__).resolve().parents[2] / "docs" / "_ext"
if str(_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXT_DIR))

from kernel_pages import (  # noqa: E402
    KernelRecord,
    assign_stems,
    generate_kernel_pages,
    render_index,
    render_kernel_page,
    stem_for_record,
)


def _record(op_type, device="cpu", domain="ai.onnx", kernel_name=None, types=("FLOAT",)):
    return KernelRecord(
        domain=domain,
        op_type=op_type,
        device=device,
        kernel_name=kernel_name or f"onnx_light_cpu::{op_type}",
        types=types,
        since_version=None,
        until_version=None,
    )


class TestStemForRecord:
    def test_slugifies_domain_op_and_device(self):
        record = _record("Abs")
        assert stem_for_record(record) == "ai_onnx_abs_cpu"

    def test_is_deterministic(self):
        record = _record("Gemm")
        assert stem_for_record(record) == stem_for_record(record)


class TestAssignStems:
    def test_one_stem_per_record_preserves_order(self):
        records = [_record("Abs"), _record("Gemm"), _record("Not")]
        stems = assign_stems(records)
        assert list(stems.keys()) == ["ai_onnx_abs_cpu", "ai_onnx_gemm_cpu", "ai_onnx_not_cpu"]
        assert list(stems.values()) == records

    def test_colliding_stems_are_disambiguated_in_stable_order(self):
        # Two distinct registrations that slugify to the same base stem (here
        # via punctuation collapsing) must not silently overwrite each other.
        first = _record("Abs", domain="ai.onnx")
        second = _record("Abs", domain="ai onnx")
        stems = assign_stems([first, second])
        assert list(stems.keys()) == ["ai_onnx_abs_cpu", "ai_onnx_abs_cpu_2"]
        assert stems["ai_onnx_abs_cpu"] is first
        assert stems["ai_onnx_abs_cpu_2"] is second


class TestRenderKernelPage:
    def test_includes_metadata(self):
        record = _record("Gemm", types=("FLOAT", "DOUBLE"))
        text = render_kernel_page(record)
        assert "Gemm (cpu)" in text
        assert "``ai.onnx``" in text
        assert "onnx_light_cpu::Gemm" in text
        assert "``FLOAT``, ``DOUBLE``" in text
        assert "none" in text  # opset bounds

    def test_reports_opset_bounds(self):
        record = KernelRecord(
            domain="ai.onnx",
            op_type="Gemm",
            device="cpu",
            kernel_name="onnx_light_cpu::Gemm",
            types=("FLOAT",),
            since_version=9,
            until_version=13,
        )
        text = render_kernel_page(record)
        assert "since opset 9" in text
        assert "until opset 13" in text


class TestRenderIndex:
    def test_lists_every_stem_in_a_toctree(self):
        text = render_index(["ai_onnx_abs_cpu", "ai_onnx_gemm_cpu"])
        assert ".. toctree::" in text
        assert "   ai_onnx_abs_cpu" in text
        assert "   ai_onnx_gemm_cpu" in text


class TestGenerateKernelPages:
    def test_generation_is_byte_identical_across_two_runs(self, tmp_path):
        records = [_record("Abs"), _record("Gemm")]
        output_dir = tmp_path / "kernels_generated"

        generate_kernel_pages(records, output_dir)
        first = {p.name: p.read_bytes() for p in sorted(output_dir.glob("*.rst"))}

        generate_kernel_pages(records, output_dir)
        second = {p.name: p.read_bytes() for p in sorted(output_dir.glob("*.rst"))}

        assert first == second
        assert set(first) == {"index.rst", "ai_onnx_abs_cpu.rst", "ai_onnx_gemm_cpu.rst"}

    def test_unchanged_pages_are_not_rewritten(self, tmp_path):
        records = [_record("Abs")]
        output_dir = tmp_path / "kernels_generated"

        generate_kernel_pages(records, output_dir)
        page = output_dir / "ai_onnx_abs_cpu.rst"
        mtime_before = page.stat().st_mtime_ns

        generate_kernel_pages(records, output_dir)
        assert page.stat().st_mtime_ns == mtime_before

    def test_renaming_a_registration_removes_the_stale_page(self, tmp_path):
        output_dir = tmp_path / "kernels_generated"

        generate_kernel_pages([_record("Abs")], output_dir)
        assert (output_dir / "ai_onnx_abs_cpu.rst").exists()

        # "Abs" is renamed to "AbsV2": the old page must disappear and only
        # the new one (plus the index) must remain.
        generate_kernel_pages([_record("AbsV2")], output_dir)
        remaining = {p.name for p in output_dir.glob("*.rst")}
        assert remaining == {"index.rst", "ai_onnx_absv2_cpu.rst"}

    def test_removing_a_registration_removes_its_page(self, tmp_path):
        output_dir = tmp_path / "kernels_generated"

        generate_kernel_pages([_record("Abs"), _record("Gemm")], output_dir)
        generate_kernel_pages([_record("Gemm")], output_dir)

        remaining = {p.name for p in output_dir.glob("*.rst")}
        assert remaining == {"index.rst", "ai_onnx_gemm_cpu.rst"}

    def test_does_not_touch_unrelated_files_outside_the_output_dir(self, tmp_path):
        sibling = tmp_path / "unrelated.rst"
        sibling.write_text("keep me", encoding="utf-8")

        generate_kernel_pages([_record("Abs")], tmp_path / "kernels_generated")

        assert sibling.read_text(encoding="utf-8") == "keep me"
