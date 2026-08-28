# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Ensure the next-steps table remains interactive."""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def test_next_steps_table_is_sortable_and_filterable():
    conf = (_ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
    next_steps = (_ROOT / "docs" / "next_steps" / "index.rst").read_text(encoding="utf-8")

    assert '"sphinxcontrib.jquery"' in conf
    assert '"sphinx_datatables"' in conf
    assert conf.index('"sphinxcontrib.jquery"') < conf.index('"sphinx_datatables"')
    assert ":class: sphinx-datatable" in next_steps


def test_next_steps_index_matches_roadmap_statuses():
    next_steps_dir = _ROOT / "docs" / "next_steps"
    next_steps = (next_steps_dir / "index.rst").read_text(encoding="utf-8")
    roadmap_paths = re.findall(r"^    (2026/\S+)$", next_steps, flags=re.MULTILINE)
    section_matches = list(
        re.finditer(
            r"^(Started|Planned|Completed|Discussion)\n-+\n",
            next_steps,
            flags=re.MULTILINE,
        )
    )
    sections = {
        match.group(1): next_steps[
            match.end() : (
                section_matches[index + 1].start()
                if index + 1 < len(section_matches)
                else len(next_steps)
            )
        ]
        for index, match in enumerate(section_matches)
    }
    expected_sections = {
        "in progress": "Started",
        "planned": "Planned",
        "complete": "Completed",
        "discussion": "Discussion",
    }

    for roadmap_path in roadmap_paths:
        roadmap = (next_steps_dir / f"{roadmap_path}.rst").read_text(encoding="utf-8")
        status_match = re.search(
            r"^\*\*(complete|in progress|planned|discussion)\*\*",
            roadmap,
            flags=re.MULTILINE,
        )
        assert status_match is not None, f"{roadmap_path} has no canonical status"
        matching_sections = [
            section for section, body in sections.items() if f"<{roadmap_path}>" in body
        ]
        assert matching_sections == [expected_sections[status_match.group(1)]], roadmap_path
