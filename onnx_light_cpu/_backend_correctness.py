# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Run the onnx-light backend correctness corpus with CPU kernels installed."""

from __future__ import annotations

from typing import Any, NamedTuple

from ._register import (
    MicrosoftKernelImplementation,
    clear_used_kernel_names,
    register_backend_test_cases,
    register_kernels,
    registered_kernels,
    used_kernel_names,
)


class BackendCaseResult(NamedTuple):
    """One skipped, failed, or successfully executed backend test case."""

    op_type: str
    case_name: str
    reason: str


class BackendCorrectnessReport(NamedTuple):
    """Result of :func:`run_backend_correctness_tests`."""

    executed: int
    passed: int
    skipped: tuple[BackendCaseResult, ...]
    failed: tuple[BackendCaseResult, ...]


def _to_numpy(tensor: Any, tensor_proto: Any, tensor_dtypes: dict[int, Any]):
    import numpy as np

    if int(tensor.data_type) == int(tensor_proto.STRING):
        return np.asarray(tensor.string_data()).reshape(tuple(int(d) for d in tensor.shape))
    return np.frombuffer(tensor.raw_data(), dtype=tensor_dtypes[int(tensor.data_type)]).reshape(
        tuple(int(d) for d in tensor.shape)
    )


def _model_opset(model, domain):
    for opset in model.opset_import:
        if (opset.domain or "ai.onnx") == domain:
            return int(opset.version)
    return None


def _case_is_applicable(case: Any, kernel: Any, tensor_proto: Any):
    matching_nodes = [
        node
        for node in case.model.graph.node
        if (node.domain or "ai.onnx") == kernel.domain and node.op_type == kernel.op_type
    ]
    if not matching_nodes:
        return False, "model has no matching node"
    opset = _model_opset(case.model, kernel.domain)
    if opset is None:
        return False, f"model has no {kernel.domain} opset"
    if kernel.since_version is not None and opset < kernel.since_version:
        return False, f"opset {opset} is below supported version {kernel.since_version}"
    if kernel.until_version is not None and opset > kernel.until_version:
        return False, f"opset {opset} is above supported version {kernel.until_version}"
    input_types = {
        tensor_proto.DataType.Name(value.type.tensor_type.elem_type)
        for value in case.model.graph.input
        if value.type.HasField("tensor_type")
    }
    if not input_types & set(kernel.types):
        return False, f"input types {sorted(input_types)} are unsupported"
    return True, ""


def _assert_outputs(actual, expected, rtol, atol):
    import numpy as np

    if len(actual) != len(expected):
        raise AssertionError(
            f"output count mismatch: got {len(actual)}, expected {len(expected)}"
        )
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        if got.shape != want.shape:
            raise AssertionError(
                f"output {index} shape mismatch: got {got.shape}, expected {want.shape}"
            )
        if got.dtype != want.dtype:
            raise AssertionError(
                f"output {index} dtype mismatch: got {got.dtype}, expected {want.dtype}"
            )
        if want.dtype.kind in "biuUS":
            np.testing.assert_array_equal(got, want, err_msg=f"output {index}")
        else:
            np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=f"output {index}")


def run_backend_correctness_tests(
    microsoft_implementation: MicrosoftKernelImplementation = (
        MicrosoftKernelImplementation.OPTIMIZED
    ),
) -> BackendCorrectnessReport:
    """Runs all applicable ``TestMode.TEST`` backend cases for registered CPU kernels.

    The report records unsupported cases as skips and execution or comparison errors as
    failures. A kernel without an applicable correctness case is reported as a failure.
    """
    import ml_dtypes
    import numpy as np

    from onnx_light.onnx import TensorProto
    from onnx_light.onnx.backend import TestMode, collect_test_cases
    from onnx_light.onnx.reference import ReferenceEvaluator

    tensor_dtypes = {
        int(TensorProto.FLOAT): np.float32,
        int(TensorProto.DOUBLE): np.float64,
        int(TensorProto.INT8): np.int8,
        int(TensorProto.INT16): np.int16,
        int(TensorProto.INT32): np.int32,
        int(TensorProto.INT64): np.int64,
        int(TensorProto.UINT8): np.uint8,
        int(TensorProto.UINT16): np.uint16,
        int(TensorProto.UINT32): np.uint32,
        int(TensorProto.UINT64): np.uint64,
        int(TensorProto.BOOL): np.bool_,
        int(TensorProto.FLOAT16): np.float16,
        int(TensorProto.BFLOAT16): ml_dtypes.bfloat16,
        int(TensorProto.COMPLEX64): np.complex64,
        int(TensorProto.COMPLEX128): np.complex128,
    }
    register_backend_test_cases()
    register_kernels(microsoft_implementation=microsoft_implementation)
    skipped = []
    failed = []
    executed = passed = 0
    covered = set()
    seen_cases = set()
    for kernel in registered_kernels(microsoft_implementation):
        for case in collect_test_cases(kernel.op_type, include_big=False, mode=TestMode.TEST):
            key = (kernel.domain, case.name)
            if key in seen_cases:
                continue
            applicable, reason = _case_is_applicable(case, kernel, TensorProto)
            if not applicable:
                skipped.append(BackendCaseResult(kernel.op_type, case.name, reason))
                continue
            seen_cases.add(key)
            try:
                input_names = [value.name for value in case.model.graph.input]
                session = ReferenceEvaluator(case.model)
                for data_set in case.data_sets:
                    if not data_set.expected_outputs_generated:
                        raise AssertionError("TEST case omitted expected outputs")
                    feeds = {
                        name: _to_numpy(tensor, TensorProto, tensor_dtypes)
                        for name, tensor in zip(input_names, data_set.inputs, strict=True)
                    }
                    clear_used_kernel_names()
                    actual = session.run(None, feeds)
                    _assert_outputs(
                        actual,
                        [
                            _to_numpy(tensor, TensorProto, tensor_dtypes)
                            for tensor in data_set.outputs
                        ],
                        case.rtol,
                        case.atol,
                    )
                    if kernel.kernel_name not in used_kernel_names():
                        raise AssertionError(f"expected kernel {kernel.kernel_name} did not run")
                executed += 1
                passed += 1
                covered.add(kernel)
            except Exception as exc:
                failed.append(BackendCaseResult(kernel.op_type, case.name, str(exc)))
            finally:
                case.unload()
    for kernel in registered_kernels(microsoft_implementation):
        if kernel not in covered:
            failed.append(
                BackendCaseResult(
                    kernel.op_type, "", "no applicable TEST backend correctness case"
                )
            )
    return BackendCorrectnessReport(executed, passed, tuple(skipped), tuple(failed))
