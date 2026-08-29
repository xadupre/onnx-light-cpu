# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests the public SIMD inspection API."""

import onnx_light_cpu
from onnx_light_cpu import SimdLevel, detect_simd_level, has_cpu_kernels


class TestDetection:
    def test_has_cpu_kernels(self):
        assert has_cpu_kernels() is True

    def test_detect_simd_level(self):
        level = detect_simd_level()
        assert isinstance(level, SimdLevel)
        assert level in SimdLevel

    def test_public_api_exports_simd_helpers(self):
        assert "SimdLevel" in onnx_light_cpu.__all__
        assert "detect_simd_level" in onnx_light_cpu.__all__
        assert "has_cpu_kernels" in onnx_light_cpu.__all__
