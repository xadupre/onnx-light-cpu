# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Guard against short section underlines that fail the strict Sphinx build."""

import re
from pathlib import Path

_DOCS = Path(__file__).resolve().parents[2] / "docs"


def test_heading_underlines_cover_titles():
    for path in sorted(_DOCS.rglob("*.rst")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index in range(1, len(lines)):
            underline = lines[index]
            title = lines[index - 1]
            if (
                re.fullmatch(r"""([=\-~^"+#*])\1{2,}""", underline)
                and title
                and not title[0].isspace()
                and (index < 2 or not lines[index - 2].strip())
            ):
                assert len(underline) >= len(title), (
                    f"{path.relative_to(_DOCS)}:{index + 1}: "
                    f"underline has {len(underline)} chars for {len(title)}-char title"
                )
