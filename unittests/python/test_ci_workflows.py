# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests the cross-repository CI ownership contract."""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_CORE_WORKFLOW = (_ROOT / ".github" / "workflows" / "ci_core.yml").read_text(encoding="utf-8")
_DOCS_WORKFLOW = (_ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")


def test_documentation_build_is_linux_only():
    assert "runs-on: ubuntu-latest" in _DOCS_WORKFLOW
    assert "matrix:" not in _DOCS_WORKFLOW
    assert "sphinx-build -b html docs dist/html" in _DOCS_WORKFLOW


def test_documentation_does_not_replace_onnx_light_main():
    assert "git clone --depth 1 --branch main" in _DOCS_WORKFLOW
    assert ".[docs,dev]" not in _DOCS_WORKFLOW
    assert "--onnx-light-source" in _DOCS_WORKFLOW


def test_onnx_light_main_integration_runs_on_every_supported_os():
    source_job = _CORE_WORKFLOW.split("  setup_onnx_light_source:", 1)[1].split(
        "  arm_gemm_native:", 1
    )[0]
    assert 'os: ["ubuntu-latest", "windows-latest", "macos-latest"]' in source_job
    assert "git clone --depth 1 --branch main" in source_job
    assert "--cpp-tests --onnx-light-source" in source_job
    assert "0.1.19" not in source_job
