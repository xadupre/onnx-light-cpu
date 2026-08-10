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

import onnx_light_cpu._register as reg
from onnx_light_cpu import register_kernels


class TestRegisterKernels:
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

        import_runtime.assert_called_once_with("onnx_light.onnx.reference")
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
