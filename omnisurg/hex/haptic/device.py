# SPDX-License-Identifier: Apache-2.0
"""Hex haptic pose data types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InputPose:
    """Latest snapshot of the device.

    Attributes:
        position: tool tip position in device-space millimetres (OpenHaptics
            returns mm). Callers apply whatever world-space scaling they want.
        quaternion: tool orientation as ``(x, y, z, w)``.
        button1: first front button state.
        button2: second front button state (None if the device only has one).
        tool_pos: MiniMou tool-position scalar used as the cut trigger.
        grip: normalized handle/tool closure in [0, 1], where 1 is closed.
        handle_pos: raw MiniMou handle opening value when available.
        handle_active: MiniMou handle activity bit when available.
        valid: False until the scheduler has produced at least one update.
    """

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    button1: bool = False
    button2: bool = False
    tool_pos: float = 0.0
    grip: float = 0.0
    handle_pos: float = 0.0
    handle_active: bool = False
    valid: bool = False
