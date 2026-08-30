# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end parity between the live kernel inventory and the generated docs.

``onnx_light_cpu.registered_kernels()`` (via ``docs/_ext/kernel_pages.py``'s
``load_registered_kernels``) requires the ``_cpuregister`` extension built
with ``ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON``, i.e. the onnx-light integration
build (see ``unittests/python/test_kernels_e2e.py``). It is therefore kept in
its own module, excluded from the plain ``core`` CI job the same way, and only
exercised where that build is available.
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from onnx_light.ext_test_case import ExtTestCase

_EXT_DIR = Path(__file__).resolve().parents[2] / "docs" / "_ext"
if str(_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXT_DIR))

from kernel_pages import (  # noqa: E402
    assign_stems,
    generate_kernel_pages,
    load_custom_schemas,
    load_operator_support,
    load_registered_kernels,
)


class TestGenerationParityWithLiveInventory(ExtTestCase):
    """End-to-end parity between the runtime C++ registrations (through
    ``onnx_light_cpu.registered_kernels()``) and what ``kernel_pages`` would
    generate for the real Sphinx build: exactly one page per record, an
    index entry for every one of them, and page contents that mirror the
    record they were generated from. Unlike the ``TestGenerateKernelPages``
    tests in ``test_kernels_doc.py`` (which use small, fabricated
    ``KernelRecord`` fixtures), this exercises the actual inventory the
    compiled extension reports.
    """

    def setUp(self):
        super().setUp()
        self._temporary_directory = TemporaryDirectory()
        self._tmp_path = Path(self._temporary_directory.name)

    def tearDown(self):
        self._temporary_directory.cleanup()
        super().tearDown()

    def test_generated_pages_and_index_match_registered_kernels_exactly(self):
        records = load_registered_kernels()
        assert records, "expected at least one live registered kernel record"

        output_dir = self._tmp_path / "kernels_generated"
        generate_kernel_pages(
            records,
            output_dir,
            schemas=load_custom_schemas(),
            support_records=load_operator_support(),
        )

        stems = assign_stems(records)
        # One generated page per record (no collisions silently dropped),
        # plus exactly one index.
        expected_files = {"index.rst"} | {f"{stem}.rst" for stem in stems}
        actual_files = {p.name for p in output_dir.glob("*.rst")}
        assert actual_files == expected_files
        assert len(stems) == len(records)

        index_text = (output_dir / "index.rst").read_text(encoding="utf-8")
        for stem, record in stems.items():
            # The index toctree references every generated page...
            assert f"   kernels_generated/{stem}" in index_text
            # ...and each page's content matches the record it documents.
            page_text = (output_dir / f"{stem}.rst").read_text(encoding="utf-8")
            assert f"{record.op_type} ({record.device})" in page_text
            assert f"``{record.domain}``" in page_text
            assert f"``{record.kernel_name}``" in page_text
            assert "Operator support" in page_text
            if record.domain == "com.microsoft":
                assert "LightOpSchema" in page_text

    def test_regenerating_from_the_live_inventory_is_byte_identical(self):
        records = load_registered_kernels()
        output_dir = self._tmp_path / "kernels_generated"

        generate_kernel_pages(
            records,
            output_dir,
            schemas=load_custom_schemas(),
            support_records=load_operator_support(),
        )
        first = {p.name: p.read_bytes() for p in sorted(output_dir.glob("*.rst"))}

        generate_kernel_pages(
            records,
            output_dir,
            schemas=load_custom_schemas(),
            support_records=load_operator_support(),
        )
        second = {p.name: p.read_bytes() for p in sorted(output_dir.glob("*.rst"))}

        assert first == second

    def test_group_query_attention_page_links_to_all_registered_support(self):
        records = load_registered_kernels()
        output_dir = self._tmp_path / "kernels_generated"
        generate_kernel_pages(
            records,
            output_dir,
            schemas=load_custom_schemas(),
            support_records=load_operator_support(),
        )

        group_query_attention = next(
            record
            for record in records
            if record.domain == "com.microsoft" and record.op_type == "GroupQueryAttention"
        )
        stem = next(
            stem
            for stem, record in assign_stems(records).items()
            if record == group_query_attention
        )
        page = output_dir / f"{stem}.rst"
        text = page.read_text(encoding="utf-8")
        assert ":cpp:class:`onnx_light_cpu::GroupQueryAttentionFusionPattern`" in text
        assert ":cpp:func:`onnx_light_cpu::ComputeShapeGroupQueryAttention`" in text
        assert ":cpp:func:`onnx_light_cpu::ComputePeakMemoryGroupQueryAttention`" in text
