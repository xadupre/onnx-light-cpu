# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Ensure the next-steps tables remain interactive and match roadmap statuses."""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def test_next_steps_table_is_sortable_and_filterable():
    conf = (_ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
    assert '"sphinxcontrib.jquery"' in conf
    assert '"sphinx_datatables"' in conf
    assert conf.index('"sphinxcontrib.jquery"') < conf.index('"sphinx_datatables"')
    for index in ("index.rst", "2026/index.rst"):
        next_steps = (_ROOT / "docs" / "next_steps" / index).read_text(encoding="utf-8")
        assert ":class: sphinx-datatable" in next_steps


def test_next_steps_index_matches_roadmap_statuses():
    next_steps_dir = _ROOT / "docs" / "next_steps"
    next_steps = (next_steps_dir / "index.rst").read_text(encoding="utf-8")
    archive = (next_steps_dir / "2026" / "index.rst").read_text(encoding="utf-8")
    assert "    2026/index\n" in next_steps
    assert "<2026/index>" in next_steps
    assert "Completed\n" not in next_steps
    active_paths = re.findall(r"^    (2026/2026_\S+)$", next_steps, flags=re.MULTILINE)
    archived_paths = re.findall(r"^    (2026_\S+)$", archive, flags=re.MULTILINE)
    roadmap_paths = active_paths + [f"2026/{path}" for path in archived_paths]
    assert len(roadmap_paths) == len(set(roadmap_paths))
    assert set(roadmap_paths) == {
        path.relative_to(next_steps_dir).with_suffix("").as_posix()
        for path in (next_steps_dir / "2026").glob("2026_*.rst")
    }
    section_matches = list(
        re.finditer(
            r"^(Started|Planned|Completed|Discussion)\n-+\n",
            next_steps,
            flags=re.MULTILINE,
        )
    )
    assert [match.group(1) for match in section_matches] == [
        "Started",
        "Planned",
        "Discussion",
    ]
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
    sections["Completed"] = archive.split("Completed\n---------\n", maxsplit=1)[1]
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
            section
            for section, body in sections.items()
            if f"<{Path(roadmap_path).name if section == 'Completed' else roadmap_path}>" in body
        ]
        assert matching_sections == [expected_sections[status_match.group(1)]], roadmap_path
        assert (roadmap_path in active_paths) == (status_match.group(1) != "complete")
