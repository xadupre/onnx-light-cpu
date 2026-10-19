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
    assert version == "0.2.0"

    for path, pattern in (
        ("setup.py", r'version="([^"]+)"'),
        ("CMakeLists.txt", r"project\(onnx_light_cpu VERSION ([\d.]+)"),
        (
            "onnx_light_cpu_compute/CMakeLists.txt",
            r"project\(onnx_light_cpu_compute VERSION ([\d.]+)",
        ),
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
    assert "Unreleased" in sections["0.2.0"]
    assert "2026-10-05" in sections["0.1.20"]
    assert all(f"#{number}" in sections["0.1.20"] for number in (853, 859, 860))
    assert "2026-10-05" in sections["0.1.19"]
    assert "#854" in sections["0.1.19"]
    assert "2026-10-05" in sections["0.1.18"]
    assert "#846" in sections["0.1.18"]
    assert "2026-10-02" in sections["0.1.17"]
    assert "2026-10-01" in sections["0.1.16"]


def test_dev_dependencies_exclude_onnx_light_runtime():
    metadata = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["optional-dependencies"]["dev"]
    onnx_light = [
        dependency for dependency in dependencies if dependency.startswith("onnx-light")
    ]

    assert not onnx_light
