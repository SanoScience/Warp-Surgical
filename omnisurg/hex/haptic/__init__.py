# SPDX-License-Identifier: Apache-2.0
"""6-DOF haptic input for the cutting simulation.

Wraps the copied-in ``pyopenhaptics`` ctypes bindings so the simulation can
read tool pose + button state from a Phantom Omni. No force feedback is
exposed; that was explicitly out of scope for v1.

If the OpenHaptics SDK (``HD.dll`` / ``libHD.so``) is not installed the
:class:`HapticInput` constructor raises ``HapticUnavailable`` and the caller
falls back to :class:`FallbackInput`, a keyboard-driven stub.
"""

from __future__ import annotations

from .device import (
    FallbackInput,
    HapticInput,
    HapticUnavailable,
    InputPose,
    MiniMouInput,
    open_haptic_inputs,
    open_minimou_inputs,
)
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
    "FallbackInput",
    "HapticInput",
    "HapticFrameConfig",
    "HapticFrameProfile",
    "HapticUnavailable",
    "InputPose",
    "MINIMOU_PROFILE",
    "MINIMOU_PITCH_FIX_DEGREES",
    "MiniMouInput",
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
    "open_haptic_inputs",
    "open_minimou_inputs",
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
