# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Keep documentation examples independent of machine-specific directories."""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def test_documentation_paths_are_portable():
    docs = _ROOT / "docs"
    paths = [
        _ROOT / "README.md",
        *docs.glob("*.rst"),
        *(docs / "design").rglob("*.rst"),
        *(docs / "next_steps").rglob("*.rst"),
        docs / "api" / "cpp" / "kernels_notes.rst",
    ]
    for path in paths:
        content = path.read_text(encoding="utf-8")
        assert not re.search(r"/(?:home|Users|tmp)/", content), (
            f"{path.relative_to(_ROOT)} contains a machine-specific absolute path; "
            "use a relative path or derive it from the environment"
        )
