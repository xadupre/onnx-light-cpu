# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from onnx_light_cpu.onnx_py._cpukernels import logical_not


class TestNotBool:
    def test_basic(self):
        inp = np.array([True, False, True, False, False, True], dtype=np.bool_)
        out = logical_not(inp)
        assert out.dtype == np.bool_
        np.testing.assert_array_equal(out, np.logical_not(inp))

    def test_empty(self):
        inp = np.array([], dtype=np.bool_)
        out = logical_not(inp)
        assert out.dtype == np.bool_
        assert out.shape == (0,)

    @pytest.mark.parametrize("size", [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 1000])
    def test_various_sizes(self, size):
        inp = (np.arange(size) % 2 == 0).astype(np.bool_)
        out = logical_not(inp)
        np.testing.assert_array_equal(out, np.logical_not(inp))

    def test_all_true(self):
        inp = np.ones(10, dtype=np.bool_)
        out = logical_not(inp)
        np.testing.assert_array_equal(out, np.zeros(10, dtype=np.bool_))

    def test_all_false(self):
        inp = np.zeros(10, dtype=np.bool_)
        out = logical_not(inp)
        np.testing.assert_array_equal(out, np.ones(10, dtype=np.bool_))


class TestNotDispatch:
    def test_unsupported_dtype(self):
        inp = np.array([0, 1, 2], dtype=np.int32)
        with pytest.raises(ValueError):
            logical_not(inp)
