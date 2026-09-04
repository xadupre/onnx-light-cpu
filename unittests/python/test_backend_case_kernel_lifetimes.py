# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_CASES = _ROOT / "onnx_light_cpu" / "backend_test" / "cases"
_LAMBDA_HEADER = re.compile(
    r"\[[^\]]*\]\s*(?:\([^{};]*\))?\s*(?:mutable\s*)?(?:noexcept\s*)?(?:->[^{]+)?$"
)
_MATERIALIZATION_CALLS = {"Expect", "RegisterUnaryBenchmark"}


def _strip_comments_and_literals(source: str) -> str:
    pattern = re.compile(
        r"//[^\n]*|/\*.*?\*/|R\"([^(\\s]*)\\(.*?\\)\\1\"|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
        re.DOTALL,
    )
    return pattern.sub(lambda match: "\n" * match.group(0).count("\n"), source)


def _function_range(source: str, name: str) -> tuple[int, int]:
    signature = re.search(rf"\b{name}\s*\([^;]*?\)\s*\{{", source, re.DOTALL)
    if signature is None:
        raise AssertionError(f"unable to find {name}")
    opening = source.find("{", signature.start())
    depth = 1
    position = opening + 1
    while depth:
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
        position += 1
    return opening, position


def _contexts_outside_lambdas(source: str) -> list[int]:
    brace_stack: list[bool] = []
    call_stack: list[str | None] = []
    failures: list[int] = []
    boundary = 0
    position = 0
    while position < len(source):
        if source.startswith("KernelContext", position):
            following = source[position + len("KernelContext") :].lstrip()
            constructs_context = following.startswith(("{", "(")) or re.match(
                r"[A-Za-z_]\w*\s*[{(]", following
            )
            if constructs_context and not any(brace_stack):
                failures.append(position)
            position += len("KernelContext")
            continue
        character = source[position]
        if character == "(":
            identifier = re.search(r"([A-Za-z_]\w*)\s*$", source[:position])
            call_stack.append(identifier.group(1) if identifier else None)
        elif character == ")":
            call_stack.pop()
        elif character == "{":
            header = source[boundary:position].rstrip()
            is_lambda = bool(_LAMBDA_HEADER.search(header))
            assigned_builder = bool(re.search(r"\b(?:auto\s+)?build(?:_data)?\s*=", header))
            called_builder = any(name in _MATERIALIZATION_CALLS for name in call_stack)
            brace_stack.append(is_lambda and (assigned_builder or called_builder))
            boundary = position + 1
        elif character == "}":
            brace_stack.pop()
            boundary = position + 1
        elif character == ";":
            boundary = position + 1
        position += 1
    return failures


def test_kernel_context_scope_parser_rejects_eager_construction():
    eager = """
    void Register() {
      KernelContext named{opset};
      auto kernel = [] { return FooKernel{KernelContext{opset}}; }();
    }
    """
    assert len(_contexts_outside_lambdas(eager)) == 2


def test_kernel_context_scope_parser_accepts_materialization_builders():
    lazy = """
    void Register() {
      Expect(registry, [] { return FooKernel{KernelContext{opset}}; });
      auto build = [] { return FooKernel{KernelContext{opset}}; };
    }
    """
    assert not _contexts_outside_lambdas(lazy)


def test_backend_case_kernel_contexts_are_created_only_during_materialization():
    failures = []
    for path in sorted(_CASES.rglob("*")):
        if path.suffix not in {".cc", ".h", ".hpp"}:
            continue
        source = _strip_comments_and_literals(path.read_text(encoding="utf-8"))
        allowed_ranges = []
        if path.name == "cases_binary.cc":
            # This pure helper is invoked only by the lazy builders in this file.
            allowed_ranges.append(_function_range(source, "MakeBinaryIoData"))
        if path.name == "linear_attention_cases.cc":
            # These helpers are invoked only by the lazy builder passed to Expect in this file.
            allowed_ranges.append(_function_range(source, "OnnxReference"))
            allowed_ranges.append(_function_range(source, "MakeLinearAttentionData"))
        for position in _contexts_outside_lambdas(source):
            if any(start <= position < end for start, end in allowed_ranges):
                continue
            line = source.count("\n", 0, position) + 1
            failures.append(f"{path.relative_to(_ROOT)}:{line}")
    assert not failures, (
        "KernelContext must be constructed inside a backend-case materialization lambda; "
        f"found eager construction at {', '.join(failures)}"
    )
