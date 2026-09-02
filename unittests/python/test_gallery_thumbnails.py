# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for explicit Sphinx-Gallery thumbnails."""

import pathlib
import re
import struct

_ROOT = pathlib.Path(__file__).parents[2]
_DOCS = _ROOT / "docs"
_THUMBNAIL_RE = re.compile(r'^# sphinx_gallery_thumbnail_path = "([^"]+)"$', re.MULTILINE)
_THUMBNAIL_BY_EXAMPLE = {
    "kernels/plot_custom_operators.py": "custom_operators.png",
}


def test_explicit_gallery_thumbnail_selection():
    """Checks that every non-plotting example selects its thumbnail."""
    expected = {
        example: f"_static/gallery_thumbnails/{thumbnail}"
        for example, thumbnail in _THUMBNAIL_BY_EXAMPLE.items()
    }
    selected = {}
    for example_path in (_DOCS / "examples").glob("*/plot_*.py"):
        matches = _THUMBNAIL_RE.findall(example_path.read_text(encoding="utf-8"))
        assert len(matches) <= 1, example_path
        if matches:
            selected[example_path.relative_to(_DOCS / "examples").as_posix()] = matches[0]

    assert selected == expected


def test_gallery_thumbnail_images():
    """Checks that every selected thumbnail is a 640 by 480 PNG."""
    for thumbnail in _THUMBNAIL_BY_EXAMPLE.values():
        path = _DOCS / "_static" / "gallery_thumbnails" / thumbnail
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n", path
        width, height = struct.unpack(">II", data[16:24])
        assert (width, height) == (640, 480), path
