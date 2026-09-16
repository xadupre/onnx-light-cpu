# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests the project dependency metadata."""

import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def test_onnx_light_dev_dependency_version():
    metadata = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["optional-dependencies"]["dev"]
    onnx_light = [
        dependency for dependency in dependencies if dependency.startswith("onnx-light @")
    ]

    assert len(onnx_light) == 3
    assert all("/0.1.27/onnx_light-0.1.27-" in dependency for dependency in onnx_light)


def test_avx2_parity_uses_onnx_light_dev_dependency():
    metadata = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["optional-dependencies"]["dev"]
    linux_dependency = next(
        dependency.split(" ; ", 1)[0]
        for dependency in dependencies
        if dependency.startswith("onnx-light @") and "sys_platform == 'linux'" in dependency
    )
    workflow = (_ROOT / ".github" / "workflows" / "avx2_parity.yml").read_text(encoding="utf-8")

    assert f'"{linux_dependency}"' in workflow
