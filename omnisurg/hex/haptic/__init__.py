# SPDX-License-Identifier: Apache-2.0
"""Hex haptic pose data types and coordinate-frame helpers.

Live, replay, and fallback input construction is owned by
``omnisurg.input.factory``. This package keeps the hex-specific pose shape and
frame conversion math used by the runtime.
"""

from __future__ import annotations

from .device import InputPose
from .frames import (
    FALLBACK_PROFILE,
    MINIMOU_PITCH_FIX_DEGREES,
    MINIMOU_PROFILE,
    OPENHAPTICS_PROFILE,
    HapticFrameConfig,
    HapticFrameProfile,
    WorldPose,
    axis_degrees_to_quaternion,
    axis_radians_to_quaternion,
    axis_vector,
    matrix_to_quaternion,
    minimou_angles_to_quaternion,
    minimou_orientation_to_quaternion,
    minimou_position_to_adapter,
    minimou_reflect_x_handedness,
    openhaptics_calibrated_quaternion,
    orientation_basis_matrix,
    parse_axis,
    pose_to_world,
    quat_between_vectors,
    quat_conjugate,
    quat_mul,
    quat_normalize,
    quat_rotate,
    quat_to_matrix,
    quat_to_xyz_degrees,
    scene_basis,
    world_position_from_device,
    world_quaternion_from_device,
    y_up_pitch_yaw_correction,
)

__all__ = [
    "FALLBACK_PROFILE",
    "HapticFrameConfig",
    "HapticFrameProfile",
    "InputPose",
    "MINIMOU_PROFILE",
    "MINIMOU_PITCH_FIX_DEGREES",
    "OPENHAPTICS_PROFILE",
    "WorldPose",
    "axis_degrees_to_quaternion",
    "axis_radians_to_quaternion",
    "axis_vector",
    "matrix_to_quaternion",
    "minimou_angles_to_quaternion",
    "minimou_orientation_to_quaternion",
    "minimou_position_to_adapter",
    "minimou_reflect_x_handedness",
    "openhaptics_calibrated_quaternion",
    "orientation_basis_matrix",
    "parse_axis",
    "pose_to_world",
    "quat_between_vectors",
    "quat_conjugate",
    "quat_mul",
    "quat_normalize",
    "quat_rotate",
    "quat_to_matrix",
    "quat_to_xyz_degrees",
    "scene_basis",
    "world_position_from_device",
    "world_quaternion_from_device",
    "y_up_pitch_yaw_correction",
]

_REMOVED_BACKEND_API = {
    "FallbackInput",
    "HapticInput",
    "HapticUnavailable",
    "MiniMouInput",
    "open_haptic_inputs",
    "open_minimou_inputs",
}


def __getattr__(name: str):
    if name in _REMOVED_BACKEND_API:
        raise AttributeError(
            f"omnisurg.hex.haptic.{name} was removed; construct runtime inputs through "
            "omnisurg.input.factory.open_input_sources() or "
            "omnisurg.hex.runtime_resources.open_hex_instrument_inputs()."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
