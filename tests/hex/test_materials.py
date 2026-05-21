# SPDX-License-Identifier: Apache-2.0
"""Sanity checks for the Material palette and Digimouse remap table."""

from __future__ import annotations

import numpy as np

from omnisurg.hex.materials import (
    BONE,
    DEFAULT_MATERIALS,
    MaterialTable,
    Phase,
    digimouse_material_table,
)


def test_material_table_ids_are_contiguous():
    table = MaterialTable(DEFAULT_MATERIALS)
    assert list(range(len(table))) == [m.id for m in table.materials]


def test_resistance_conductivity_match_reference():
    # Ported verbatim from springGrid.h:118-146.
    table = MaterialTable(DEFAULT_MATERIALS)
    r = {m.name: m.resistance for m in table.materials}
    c = {m.name: m.conductivity for m in table.materials}
    assert r["fat"] == 40.0 and c["fat"] == 1.0
    assert r["muscle"] == 80.0 and c["muscle"] == 0.9
    assert r["liver"] == 80.0 and c["liver"] == 0.9
    assert r["bone"] == 100.0 and c["bone"] == 0.2
    assert r["artery"] == 60.0 and c["artery"] == 1.0


def test_bone_is_rigid_phase():
    assert BONE.phase is Phase.RIGID


def test_digimouse_remap_covers_22_labels():
    table, remap = digimouse_material_table()
    assert remap.shape == (22,)
    assert remap.dtype == np.uint8
    assert (remap < len(table)).all(), "all remapped labels must index the palette"
    # Label 2 (atlas 'skeleton') must map to a rigid-phase material.
    assert table[remap[2]].phase is Phase.RIGID
    # Label 0 (atlas 'background') must map to the background material (id 0).
    assert remap[0] == 0
