# SPDX-License-Identifier: Apache-2.0
"""Surgical tool model: a list of capsule segments with per-segment roles.

The reference ``MedicalEquipment`` (medicalEquipment.h) carries three parallel
segment lists per instrument - haptic probes, electrode segments, and mass
collision segments. For v1 we only need the electrode role (heat injection)
and optionally a haptic role for force feedback later; both reduce to
``(p0, p1, radius)`` capsules in world space.

This module is plain Python data (no kernels). Kernels in :mod:`.kernels.heat`
consume raw ``wp.array``\\ s of the segment fields so they can be updated from
either a mouse-driven handle (examples 1-2) or the Phantom Omni driver
(Phase 7) without changing the downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class ToolSegments:
    """GPU arrays describing a tool's capsule segments in world space.

    All arrays are length ``num_segments``. ``p0`` and ``p1`` are the capsule
    endpoints, ``radius`` is the capsule radius. ``role_electrode`` is a 0/1
    mask: only segments flagged as electrodes inject heat into nearby tissue.
    """

    p0: wp.array  # vec3[num_segments]
    p1: wp.array  # vec3[num_segments]
    radius: wp.array  # float32[num_segments]
    role_electrode: wp.array  # int32[num_segments]
    num_segments: int


class Tool:
    """Host-side bookkeeping; pushes segment data to the GPU every frame."""

    def __init__(self, segments: list[tuple[tuple[float, float, float], tuple[float, float, float], float, bool]], device: wp.context.Device):
        """Create a tool from a list of ``(p0, p1, radius, is_electrode)`` tuples."""
        if not segments:
            raise ValueError("tool needs at least one segment")
        self.device = device
        self._p0 = np.asarray([s[0] for s in segments], dtype=np.float32)
        self._p1 = np.asarray([s[1] for s in segments], dtype=np.float32)
        self._radius = np.asarray([s[2] for s in segments], dtype=np.float32)
        self._role_electrode = np.asarray([int(s[3]) for s in segments], dtype=np.int32)
        self._power: float = 220.0
        self._active: bool = False

        self.segments = ToolSegments(
            p0=wp.array(self._p0, dtype=wp.vec3, device=device),
            p1=wp.array(self._p1, dtype=wp.vec3, device=device),
            radius=wp.array(self._radius, dtype=wp.float32, device=device),
            role_electrode=wp.array(self._role_electrode, dtype=wp.int32, device=device),
            num_segments=len(segments),
        )

    @property
    def power(self) -> float:
        return self._power

    @power.setter
    def power(self, value: float) -> None:
        self._power = float(value)

    @property
    def active(self) -> bool:
        """True when the electrocautery pedal is pressed."""
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        self._active = bool(value)

    def set_pose(self, p0: np.ndarray, p1: np.ndarray) -> None:
        """Replace the tool's segment endpoints in one shot.

        ``p0`` and ``p1`` are ``(num_segments, 3)`` arrays. The capsule radii
        and electrode roles stay fixed at construction; only position changes
        each frame.
        """
        p0 = np.ascontiguousarray(p0, dtype=np.float32)
        p1 = np.ascontiguousarray(p1, dtype=np.float32)
        if p0.shape != self._p0.shape or p1.shape != self._p1.shape:
            raise ValueError(f"pose shape mismatch: expected {self._p0.shape}")
        self._p0 = p0
        self._p1 = p1
        self.segments.p0.assign(p0)
        self.segments.p1.assign(p1)

    @classmethod
    def cautery_probe(cls, length: float, radius: float, device: wp.context.Device) -> "Tool":
        """A single-segment needle-tip cautery (paper Figure 7a-d)."""
        return cls(
            segments=[
                ((0.0, 0.0, 0.0), (0.0, 0.0, length), radius, True),
            ],
            device=device,
        )
