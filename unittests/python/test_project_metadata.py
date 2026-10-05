# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests the project dependency metadata."""

import re
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def test_project_version_consistency():
    metadata = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = metadata["project"]["version"]
    assert version == "0.1.20"

    for path, pattern in (
        ("setup.py", r'version="([^"]+)"'),
        ("CMakeLists.txt", r"project\(onnx_light_cpu VERSION ([\d.]+)"),
        ("onnx_light_cpu/__init__.py", r'__version__ = "([^"]+)"'),
    ):
        text = (_ROOT / path).read_text(encoding="utf-8")
        match = re.search(pattern, text)
        assert match is not None, path
        assert match.group(1) == version, path


def test_changelog_release_sections():
    changelog = (_ROOT / "CHANGELOGS.md").read_text(encoding="utf-8")
    sections = dict(
        re.findall(r"^## \[(\d+\.\d+\.\d+)\](.*?)(?=^## \[|\Z)", changelog, re.M | re.S)
    )
    assert "Unreleased" in sections["0.1.20"]
    assert all(f"#{number}" in sections["0.1.20"] for number in (853, 859, 860))
    assert "2026-10-05" in sections["0.1.19"]
    assert "#854" in sections["0.1.19"]
    assert "2026-10-05" in sections["0.1.18"]
    assert "#846" in sections["0.1.18"]
    assert "2026-10-02" in sections["0.1.17"]
    assert "2026-10-01" in sections["0.1.16"]


def test_onnx_light_dev_dependency_version():
    metadata = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["optional-dependencies"]["dev"]
    onnx_light = [
        dependency for dependency in dependencies if dependency.startswith("onnx-light @")
    ]

    assert len(onnx_light) == 3
    assert all("/0.1.30/onnx_light-0.1.30-" in dependency for dependency in onnx_light)


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
