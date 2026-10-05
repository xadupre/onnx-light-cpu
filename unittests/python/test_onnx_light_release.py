import re
import tomllib
from pathlib import Path

import pytest

from tools.onnx_light_release import (
    ONNX_LIGHT_VERSION,
    parse_dumpbin_exports,
    wheel_name,
    wheel_url,
)

_ROOT = Path(__file__).resolve().parents[2]


def test_release_version_matches_project_dependencies():
    metadata = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["optional-dependencies"]["dev"]
    versions = {
        match.group(1)
        for dependency in dependencies
        if dependency.startswith("onnx-light @")
        if (match := re.search(r"/releases/download/([^/]+)/", dependency))
    }
    assert versions == {ONNX_LIGHT_VERSION}


def test_release_wheel_names():
    assert wheel_name("Linux", "x86_64", (3, 12)) == (
        "onnx_light-0.1.30-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    )
    assert wheel_name("Linux", "aarch64", (3, 13)) == (
        "onnx_light-0.1.30-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
    )
    assert wheel_name("Darwin", "arm64", (3, 14)) == (
        "onnx_light-0.1.30-cp314-cp314-macosx_13_0_universal2.whl"
    )
    assert wheel_name("Windows", "AMD64", (3, 12)) == (
        "onnx_light-0.1.30-cp312-cp312-win_amd64.whl"
    )
    assert wheel_url("Windows", "ARM64", (3, 13)).endswith(
        "/0.1.30/onnx_light-0.1.30-cp313-cp313-win_arm64.whl"
    )


def test_unsupported_release_platform():
    with pytest.raises(ValueError, match="Linux ppc64le"):
        wheel_name("Linux", "ppc64le", (3, 12))


def test_parse_dumpbin_exports():
    output = """
      ordinal hint RVA      name
            1    0 00011000 ?CreateKernel@@YAPEAXXZ
            2    1 00012000 PlainExport
    """
    assert parse_dumpbin_exports(output) == ["?CreateKernel@@YAPEAXXZ", "PlainExport"]
