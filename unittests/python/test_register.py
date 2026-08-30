# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`onnx_light_cpu.register_kernels`.

``register_kernels`` installs the onnx-light-cpu kernel classes into
onnx-light's shared C++ ``KernelDispatchTable`` by calling the
``register_all_kernels`` binding. The binding links onnx-light and is only
present in builds compiled with ``ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON``, so the
tests patch it to exercise the wrapper without requiring that build.
"""

import sys
from types import ModuleType
from unittest import mock

from onnx_light.ext_test_case import ExtTestCase
import onnx_light_cpu._register as reg
from onnx_light_cpu import (
    OperatorSupport,
    RegisteredKernel,
    operator_support,
    has_backend_test_cases,
    register_operator_support,
    register_kernels,
    registered_kernel_names,
    registered_kernels,
)


class TestRegisterKernels(ExtTestCase):
    def test_loads_runtime_before_registration_extension(self):
        extension = ModuleType("onnx_light_cpu.onnx_py._cpuregister")
        extension.register_all_kernels = mock.Mock()

        with (
            mock.patch.object(reg, "import_module") as import_runtime,
            mock.patch.dict(
                sys.modules,
                {"onnx_light_cpu.onnx_py._cpuregister": extension},
            ),
        ):
            reg._register_all_kernels()

        assert import_runtime.call_args_list == [
            mock.call("onnx_light.onnx.reference"),
            mock.call("onnx_light.onnx_op"),
        ]
        extension.register_all_kernels.assert_called_once_with()

    def test_calls_register_all_kernels(self):
        with mock.patch.object(reg, "_register_all_kernels") as m:
            register_kernels()
            m.assert_called_once_with()

    def test_returns_sess_unchanged(self):
        sentinel = object()
        with mock.patch.object(reg, "_register_all_kernels") as m:
            result = register_kernels(sentinel)
            m.assert_called_once_with()
            assert result is sentinel

    def test_returns_none_without_sess(self):
        with mock.patch.object(reg, "_register_all_kernels"):
            assert register_kernels() is None


class TestBackendTestRegistration(ExtTestCase):
    def test_reports_backend_test_binding_availability(self):
        extension = ModuleType("onnx_light_cpu.onnx_py._cpuregister")
        extension.has_backend_test_cases = False
        with mock.patch.dict(sys.modules, {"onnx_light_cpu.onnx_py._cpuregister": extension}):
            assert has_backend_test_cases() is False

        extension.has_backend_test_cases = True
        with mock.patch.dict(sys.modules, {"onnx_light_cpu.onnx_py._cpuregister": extension}):
            assert has_backend_test_cases() is True


class TestRegisteredKernels(ExtTestCase):
    """Tests for :func:`onnx_light_cpu.registered_kernels`.

    The underlying binding (``_cpuregister.registered_kernels``) requires the
    onnx-light integration build, so these tests patch it with plain tuples
    (mirroring what the C++ binding returns) to exercise the Python-side
    wrapping into immutable :class:`onnx_light_cpu.RegisteredKernel` records.
    """

    def _patch_binding(self, raw_records):
        extension = ModuleType("onnx_light_cpu.onnx_py._cpuregister")
        extension.registered_kernels = mock.Mock(return_value=raw_records)
        return mock.patch.dict(sys.modules, {"onnx_light_cpu.onnx_py._cpuregister": extension})

    def test_wraps_raw_tuples_into_registered_kernel_records(self):
        raw_records = [
            ("ai.onnx", "Abs", "CPU", "onnx_light_cpu::Abs", ["FLOAT", "DOUBLE"], None, None),
            ("ai.onnx", "Gemm", "CPU", "onnx_light_cpu::Gemm", ["FLOAT"], 7, None),
        ]
        with self._patch_binding(raw_records):
            records = registered_kernels()

        assert records == (
            RegisteredKernel(
                "ai.onnx", "Abs", "CPU", "onnx_light_cpu::Abs", ("FLOAT", "DOUBLE"), None, None
            ),
            RegisteredKernel(
                "ai.onnx", "Gemm", "CPU", "onnx_light_cpu::Gemm", ("FLOAT",), 7, None
            ),
        )
        assert isinstance(records, tuple)
        assert all(isinstance(record.types, tuple) for record in records)

    def test_registered_kernel_names_is_derived_from_registered_kernels(self):
        raw_records = [
            ("ai.onnx", "Abs", "CPU", "onnx_light_cpu::Abs", ["FLOAT"], None, None),
            ("ai.onnx", "Not", "CPU", "onnx_light_cpu::Not", ["BOOL"], None, None),
        ]
        with self._patch_binding(raw_records):
            assert registered_kernel_names() == {
                "Abs": "onnx_light_cpu::Abs",
                "Not": "onnx_light_cpu::Not",
            }

    def test_records_are_immutable(self):
        record = RegisteredKernel(
            "ai.onnx", "Abs", "CPU", "onnx_light_cpu::Abs", ("FLOAT",), None, None
        )
        with self.assertRaises(AttributeError):
            record.op_type = "Other"


class TestOperatorSupport(ExtTestCase):
    def test_wraps_raw_tuples(self):
        extension = ModuleType("onnx_light_cpu.onnx_py._cpuregister")
        extension.operator_support = mock.Mock(
            return_value=[
                (
                    "com.microsoft",
                    "CDist",
                    "onnx_light_cpu::ComputeShapeCDist",
                    "onnx_light_cpu::ComputePeakMemoryCDist",
                    ["onnx_light_cpu::CDistFusionPattern"],
                    True,
                )
            ]
        )
        with mock.patch.dict(sys.modules, {"onnx_light_cpu.onnx_py._cpuregister": extension}):
            assert operator_support() == (
                OperatorSupport(
                    "com.microsoft",
                    "CDist",
                    "onnx_light_cpu::ComputeShapeCDist",
                    "onnx_light_cpu::ComputePeakMemoryCDist",
                    ("onnx_light_cpu::CDistFusionPattern",),
                    True,
                ),
            )

    def test_registers_operator_support(self):
        extension = ModuleType("onnx_light_cpu.onnx_py._cpuregister")
        extension.register_custom_operator_support = mock.Mock()
        with (
            mock.patch.object(reg, "import_module") as import_runtime,
            mock.patch.dict(sys.modules, {"onnx_light_cpu.onnx_py._cpuregister": extension}),
        ):
            register_operator_support()

        import_runtime.assert_called_once_with("onnx_light.onnx_op")
        extension.register_custom_operator_support.assert_called_once_with()
