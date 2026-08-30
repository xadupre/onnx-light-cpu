# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Run the onnx-light backend correctness corpus with CPU kernels installed."""

from __future__ import annotations

import re
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
    """One skipped or failed backend test case."""

    op_type: str
    case_name: str
    reason: str


class BackendCorrectnessReport(NamedTuple):
    """Result of :func:`run_backend_correctness_tests`."""

    executed: int
    passed: int
    skipped: tuple[BackendCaseResult, ...]
    failed: tuple[BackendCaseResult, ...]


def _model_opset(model: Any, domain: str) -> int | None:
    for opset in model.opset_import:
        if (opset.domain or "ai.onnx") == domain:
            return int(opset.version)
    return None


def _case_is_applicable(case: Any, kernel: Any, tensor_proto: Any) -> tuple[bool, str]:
    if not any(
        (node.domain or "ai.onnx") == kernel.domain and node.op_type == kernel.op_type
        for node in case.model.graph.node
    ):
        return False, "model has no matching node"
    opset = _model_opset(case.model, kernel.domain)
    if opset is None:
        return False, f"model has no {kernel.domain} opset"
    if kernel.since_version is not None and opset < kernel.since_version:
        return False, f"opset {opset} is below supported version {kernel.since_version}"
    if kernel.until_version is not None and opset > kernel.until_version:
        return False, f"opset {opset} is above supported version {kernel.until_version}"
    input_types = {
        value.type.tensor_type.elem_type
        for value in case.model.graph.input
        if value.type.HasField("tensor_type")
    }
    supported_types = {getattr(tensor_proto, name) for name in kernel.types}
    if not input_types & supported_types:
        return False, f"input types {sorted(input_types)} are unsupported"
    return True, ""


def _run_case(kernel_name: str):
    from onnx_light.onnx.reference import ReferenceEvaluator  # pyrefly: ignore[missing-import]

    def run(model, *inputs):
        session = ReferenceEvaluator(model)
        clear_used_kernel_names()
        outputs = session.run(None, dict(zip(session.input_names, inputs, strict=True)))
        if kernel_name not in used_kernel_names():
            raise AssertionError(f"expected kernel {kernel_name} did not run")
        return outputs

    return run


def run_backend_correctness_tests(
    microsoft_implementation: MicrosoftKernelImplementation = (
        MicrosoftKernelImplementation.OPTIMIZED
    ),
) -> BackendCorrectnessReport:
    """Runs all applicable ``TestMode.TEST`` backend cases for registered CPU kernels.

    The report records unsupported cases as skips and execution or comparison errors as
    failures. A kernel without an applicable correctness case is reported as a failure.
    """
    from onnx_light.onnx import TensorProto  # pyrefly: ignore[missing-import]
    from onnx_light.onnx.backend import (  # pyrefly: ignore[missing-import]
        TestMode,
        collect_test_cases,
        make_test_class,
    )

    register_backend_test_cases()
    register_kernels(microsoft_implementation=microsoft_implementation)
    skipped = []
    failed = []
    executed = passed = 0
    covered_kernel_names = set()
    seen_cases = set()
    kernels = registered_kernels(microsoft_implementation)
    for kernel in kernels:
        case_names = []
        for case in collect_test_cases(kernel.op_type, include_big=False, mode=TestMode.TEST):
            key = (kernel.domain, case.name)
            if key in seen_cases:
                continue
            applicable, reason = _case_is_applicable(case, kernel, TensorProto)
            if not applicable:
                skipped.append(BackendCaseResult(kernel.op_type, case.name, reason))
                continue
            seen_cases.add(key)
            case_names.append(case.name)
        if not case_names:
            continue
        case_class = make_test_class(
            _run_case(kernel.kernel_name),
            include_regex=[f"^(?:{'|'.join(re.escape(name) for name in case_names)})$"],
        )
        for case_name in case_names:
            case_test = case_class(f"test_{case_name}")
            executed += 1
            try:
                case_test.debug()
                passed += 1
                covered_kernel_names.add(kernel.kernel_name)
            except Exception as exc:
                failed.append(BackendCaseResult(kernel.op_type, case_name, str(exc)))
    for kernel in kernels:
        if kernel.kernel_name not in covered_kernel_names:
            failed.append(
                BackendCaseResult(
                    kernel.op_type, "", "no applicable TEST backend correctness case"
                )
            )
    return BackendCorrectnessReport(executed, passed, tuple(skipped), tuple(failed))
