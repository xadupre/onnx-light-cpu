# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise the installed release wheel's session-local native registration."""

import importlib.machinery
from importlib.util import find_spec

import numpy as np

from onnx_light.onnx import TensorProto, helper
from onnx_light.onnx.reference import ReferenceEvaluator

from onnx_light_cpu import (
    register_kernels_for_session,
    registered_kernels,
    set_kernel_usage_recording,
    used_kernel_names,
)

extension = find_spec("onnx_light_cpu.onnx_py._cpuregister")
assert extension is not None and extension.origin is not None
assert any(extension.origin.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
assert any(record.op_type == "Abs" for record in registered_kernels())

model = helper.make_model(
    helper.make_graph(
        [helper.make_node("Abs", ["X"], ["Y"])],
        "wheel_registration",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
    ),
    opset_imports=[helper.make_opsetid("", 18)],
)
session = ReferenceEvaluator(model)
assert register_kernels_for_session(session) > 0
set_kernel_usage_recording(session, True)
(result,) = session.run(None, {"X": np.array([-1.0, 2.0], dtype=np.float32)})
np.testing.assert_array_equal(result, [1.0, 2.0])
assert used_kernel_names(session) == ["onnx_light_cpu::Abs"]
