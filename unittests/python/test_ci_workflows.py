# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests the cross-repository CI ownership contract."""

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_CORE_WORKFLOW = (_ROOT / ".github" / "workflows" / "ci_core.yml").read_text(encoding="utf-8")
_DOCS_WORKFLOW = (_ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
_CODECOV_CONFIG = (_ROOT / ".codecov.yml").read_text(encoding="utf-8")


def test_documentation_build_is_linux_only():
    assert "runs-on: ubuntu-latest" in _DOCS_WORKFLOW
    assert "matrix:" not in _DOCS_WORKFLOW
    assert "sphinx-build -W -b html docs dist/html" in _DOCS_WORKFLOW
    assert "PYTHONPATH: ${{ github.workspace }}" in _DOCS_WORKFLOW


def test_documentation_does_not_replace_onnx_light_main():
    assert "git clone --depth 1 --branch main" in _DOCS_WORKFLOW
    assert ".[docs,dev]" not in _DOCS_WORKFLOW
    assert "--onnx-light-source" in _DOCS_WORKFLOW
    assert "--cpp-tests" not in _DOCS_WORKFLOW
    assert "python -m pytest" not in _DOCS_WORKFLOW


def test_onnx_light_main_integration_runs_on_every_supported_os():
    source_job = _CORE_WORKFLOW.split("  setup_onnx_light_source:", 1)[1].split(
        "  arm_gemm_native:", 1
    )[0]
    assert 'os: ["ubuntu-latest", "windows-latest", "macos-latest"]' in source_job
    assert "git clone --depth 1 --branch main" in source_job
    assert "scikit-build-core setuptools" in source_job
    assert "ONNX_LIGHT_CPU_ONNX_LIGHT_IMPLIB_DIR=" in source_job
    assert "--cpp-tests --onnx-light-source" in source_job
    assert "-DCMAKE_DISABLE_FIND_PACKAGE_OpenSSL=ON" in source_job
    assert "0.1.19" not in source_job


def test_cpp_coverage_is_carried_forward_between_weekly_runs():
    assert "cpp:\n    carryforward: true" in _CODECOV_CONFIG


def test_python_test_classes_inherit_from_ext_test_case():
    missing_bases = []
    for path in (_ROOT / "unittests" / "python").rglob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                if not any(
                    (isinstance(base, ast.Name) and base.id == "ExtTestCase")
                    or (isinstance(base, ast.Attribute) and base.attr == "ExtTestCase")
                    for base in node.bases
                ):
                    missing_bases.append(f"{path.relative_to(_ROOT)}:{node.name}")
    assert not missing_bases, f"Test classes must inherit from ExtTestCase: {missing_bases}"
