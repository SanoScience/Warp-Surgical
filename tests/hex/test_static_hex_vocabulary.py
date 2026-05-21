# SPDX-License-Identifier: Apache-2.0
"""Static guards for the hex particle-grid cleanup."""

from __future__ import annotations

from pathlib import Path


def test_no_deleted_corner_module_references():
    root = Path(__file__).resolve().parents[2]
    legacy = "corner"
    banned = (
        legacy + "_grid",
        legacy + "_delete",
        legacy + "_heat",
        legacy + "_solver",
        "_legacy_" + legacy + "_app",
        "kernels." + legacy,
        "build_" + legacy,
        "make_" + legacy,
        "Corner" + "ParticleGrid",
        "Corner" + "GridAuxState",
        "Corner" + "DeletionState",
        "Corner" + "Heat",
        "Solver" + "Corner",
        "to_" + legacy + "_atlas",
    )
    paths = list((root / "omnisurg").rglob("*.py")) + list((root / "tests").rglob("*.py"))
    offenders: list[str] = []
    for path in paths:
        if "__pycache__" in path.parts or path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned if term in text]
        if hits:
            offenders.append(f"{path.relative_to(root)}: {', '.join(hits)}")

    assert not offenders, "\n".join(offenders)
