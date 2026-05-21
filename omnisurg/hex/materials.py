# SPDX-License-Identifier: Apache-2.0
"""Tissue material definitions for the PBD cutting simulation.

Numbers for resistance / conductivity are ported verbatim from the reference
implementation at ``EfficientPBDCutting/ufrgs/springGrid.h`` (Particle ctor,
lines 118-146). ``resistance`` is the heat threshold above which a particle
is removed; ``conductivity`` is the inter-particle heat diffusion coefficient
used in the paper's Eq. 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Phase(IntEnum):
    """Dynamic phase assigned to a particle."""

    SOFT = 0       # deformable tissue, springs coupled to neighbors
    RIGID = 1      # part of a rigid cluster (bone)
    FLUID = 2      # SPH-style fluid (blood)


@dataclass(frozen=True)
class Material:
    """Per-tissue physical parameters.

    Attributes:
        id: integer label as stored in the voxel atlas (0 reserved for background).
        name: human-readable tissue name.
        phase: dynamic class (soft / rigid / fluid).
        mass: per-particle mass, relative units (matches FleX "invMass" scaling).
        resistance: heat threshold above which particle is removed. Paper §3.3.
        conductivity: inter-particle heat diffusion coefficient. Paper Eq. 1.
        color: sRGB triple for rendering.
    """

    id: int
    name: str
    phase: Phase
    mass: float
    resistance: float
    conductivity: float
    color: tuple[float, float, float]


# Canonical materials ported from springGrid.h:118-146.
# IDs match the order of the original Material enum and are used internally;
# dataset loaders map atlas labels onto these.
BACKGROUND = Material(0, "background", Phase.SOFT, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0))
MUSCLE = Material(1, "muscle", Phase.SOFT, 1.0, 80.0, 0.9, (254 / 254, 0, 0))
FAT = Material(2, "fat", Phase.SOFT, 0.8, 40.0, 1.0, (254 / 254, 254 / 254, 254 / 254))
BONE = Material(3, "bone", Phase.RIGID, 1.5, 100.0, 0.2, (0.25, 0.22, 0.20))
SKIN = Material(4, "skin", Phase.SOFT, 1.0, 80.0, 0.9, (0.90, 0.72, 0.62))
ARTERY = Material(5, "artery", Phase.SOFT, 1.0, 60.0, 1.0, (0, 0, 254 / 254))
BRAIN = Material(6, "brain", Phase.SOFT, 1.0, 80.0, 0.9, (230 / 254, 190 / 254, 190 / 254))
HEART = Material(7, "heart", Phase.SOFT, 1.0, 80.0, 0.9, (152 / 254, 102 / 254, 153 / 254))
LUNG = Material(8, "lung", Phase.SOFT, 0.6, 80.0, 0.9, (255 / 254, 255 / 254, 0))
LIVER = Material(9, "liver", Phase.SOFT, 1.0, 80.0, 0.9, (0, 255 / 254, 0))
STOMACH = Material(10, "stomach", Phase.SOFT, 1.0, 80.0, 0.9, (51 / 254, 102 / 254, 103 / 254))
ORGAN = Material(11, "organ", Phase.SOFT, 1.0, 80.0, 0.9, (200 / 254, 110 / 254, 90 / 254))
BLOOD = Material(12, "blood", Phase.FLUID, 0.5, 80.0, 1.0, (200 / 254, 20 / 254, 20 / 254))


class MaterialTable:
    """Contiguous array-of-structs representation of the material palette.

    Kernels want flat GPU-friendly arrays rather than a dict; this class owns
    the canonical materials list and exposes columns as numpy arrays ready to
    be uploaded as ``wp.array``\\ s.
    """

    def __init__(self, materials: list[Material]):
        if materials[0].id != 0:
            raise ValueError("MaterialTable[0] must be the background/empty material")
        ids = [m.id for m in materials]
        if ids != list(range(len(materials))):
            raise ValueError(f"Material ids must be 0..N-1 in order, got {ids}")
        self.materials = tuple(materials)

    def __len__(self) -> int:
        return len(self.materials)

    def __getitem__(self, label: int) -> Material:
        return self.materials[label]

    @property
    def resistance(self) -> np.ndarray:
        return np.asarray([m.resistance for m in self.materials], dtype=np.float32)

    @property
    def conductivity(self) -> np.ndarray:
        return np.asarray([m.conductivity for m in self.materials], dtype=np.float32)

    @property
    def mass(self) -> np.ndarray:
        return np.asarray([m.mass for m in self.materials], dtype=np.float32)

    @property
    def phase(self) -> np.ndarray:
        return np.asarray([int(m.phase) for m in self.materials], dtype=np.int32)

    @property
    def color(self) -> np.ndarray:
        return np.asarray([m.color for m in self.materials], dtype=np.float32)


DEFAULT_MATERIALS: list[Material] = [
    BACKGROUND, MUSCLE, FAT, BONE, SKIN, ARTERY,
    BRAIN, HEART, LUNG, LIVER, STOMACH, ORGAN, BLOOD,
]


def digimouse_material_table() -> tuple[MaterialTable, np.ndarray]:
    """Return the canonical material palette plus a Digimouse label-remap table.

    The Digimouse atlas stores 22 labels (0..21); the returned ``remap`` array
    converts an atlas label into an index into the ``MaterialTable``. See
    ``Digimouse/atlas/atlas/atlas_380x992x208.txt`` for the label list.

    Returns:
        table: the canonical MaterialTable.
        remap: ``uint8[22]`` where ``remap[atlas_label]`` is the material id.
    """

    table = MaterialTable(DEFAULT_MATERIALS)
    remap = np.zeros(22, dtype=np.uint8)
    remap[0] = BACKGROUND.id
    remap[1] = SKIN.id                # skin
    remap[2] = BONE.id                # skeleton
    remap[3] = ORGAN.id               # eye
    remap[4] = BRAIN.id               # medulla
    remap[5] = BRAIN.id               # cerebellum
    remap[6] = BRAIN.id               # olfactory bulbs
    remap[7] = BRAIN.id               # external cerebrum
    remap[8] = BRAIN.id               # striatum
    remap[9] = HEART.id               # heart
    remap[10] = BRAIN.id              # rest of the brain
    remap[11] = MUSCLE.id             # masseter muscles
    remap[12] = ORGAN.id              # lachrymal glands
    remap[13] = ORGAN.id              # bladder
    remap[14] = ORGAN.id              # testis
    remap[15] = STOMACH.id            # stomach
    remap[16] = ORGAN.id              # spleen
    remap[17] = ORGAN.id              # pancreas
    remap[18] = LIVER.id              # liver
    remap[19] = ORGAN.id              # kidneys
    remap[20] = ORGAN.id              # adrenal glands
    remap[21] = LUNG.id               # lungs
    return table, remap
