# SPDX-License-Identifier: Apache-2.0
"""Coordinate frame conversion for haptic pose sources."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


class PoseLike(Protocol):
    position: Sequence[float]
    quaternion: Sequence[float]


@dataclass(frozen=True)
class HapticFrameProfile:
    """Device-specific calibration layered on top of shared basis math."""

    name: str
    orientation_calibration: str = "none"
    rotate_orientation_basis: bool = False
    correct_y_up_pitch_yaw: bool = False


OPENHAPTICS_PROFILE = HapticFrameProfile(
    name="openhaptics",
    orientation_calibration="openhaptics",
    rotate_orientation_basis=True,
    correct_y_up_pitch_yaw=True,
)
MINIMOU_PROFILE = HapticFrameProfile(name="minimou", correct_y_up_pitch_yaw=True)
FALLBACK_PROFILE = HapticFrameProfile(name="fallback")
MINIMOU_PITCH_FIX_DEGREES = 0.0


@dataclass(frozen=True)
class HapticFrameConfig:
    profile: HapticFrameProfile = OPENHAPTICS_PROFILE
    position_scale: float = 0.001
    center: Vector3 = (0.0, 0.0, 0.0)
    device_offset: Vector3 = (0.0, 0.0, 0.0)
    up_axis: str = "Z"


@dataclass(frozen=True)
class WorldPose:
    position: np.ndarray
    quaternion: Quaternion


def parse_axis(axis: str) -> str:
    axis = axis.upper()
    if axis not in {"X", "Y", "Z"}:
        raise ValueError(f"invalid axis {axis!r}")
    return axis


def axis_vector(axis: str) -> np.ndarray:
    axis = parse_axis(axis)
    if axis == "X":
        return np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    if axis == "Y":
        return np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
    return np.asarray((0.0, 0.0, 1.0), dtype=np.float32)


def scene_basis(up_axis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return world right/depth/up axes for the selected Newton up axis."""
    up_axis = parse_axis(up_axis)
    if up_axis == "Z":
        right = axis_vector("X")
        depth = axis_vector("Y")
    elif up_axis == "Y":
        right = axis_vector("X")
        depth = -axis_vector("Z")
    else:
        # Current convention for X-up scenes: right=Y, depth=Z, up=X.
        right = axis_vector("Y")
        depth = axis_vector("Z")
    return right, depth, axis_vector(up_axis)


def orientation_basis_matrix(up_axis: str) -> np.ndarray:
    """Return the orthonormal basis matrix used for orientation remapping."""
    right, depth, up = scene_basis(up_axis)
    return np.column_stack((right, up, depth)).astype(np.float32)


def quat_normalize(q: Sequence[float]) -> Quaternion:
    n = math.sqrt(float(q[0]) ** 2 + float(q[1]) ** 2 + float(q[2]) ** 2 + float(q[3]) ** 2)
    if n <= 1.0e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (float(q[0]) / n, float(q[1]) / n, float(q[2]) / n, float(q[3]) / n)


def quat_conjugate(q: Sequence[float]) -> Quaternion:
    x, y, z, w = quat_normalize(q)
    return (-x, -y, -z, w)


def quat_mul(a: Sequence[float], b: Sequence[float]) -> Quaternion:
    ax, ay, az, aw = quat_normalize(a)
    bx, by, bz, bw = quat_normalize(b)
    return quat_normalize(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )


def axis_degrees_to_quaternion(ax: float, ay: float, az: float, angle_degrees: float) -> Quaternion:
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm <= 1.0e-12:
        return (0.0, 0.0, 0.0, 1.0)
    half_angle = 0.5 * math.radians(angle_degrees)
    s = math.sin(half_angle) / norm
    return (ax * s, ay * s, az * s, math.cos(half_angle))


def axis_radians_to_quaternion(ax: float, ay: float, az: float, angle_radians: float) -> Quaternion:
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm <= 1.0e-12:
        return (0.0, 0.0, 0.0, 1.0)
    half_angle = 0.5 * angle_radians
    s = math.sin(half_angle) / norm
    return quat_normalize((ax * s, ay * s, az * s, math.cos(half_angle)))


def matrix_to_quaternion(matrix: Sequence[Sequence[float]] | np.ndarray) -> Quaternion:
    """Shepperd's algorithm, numerically stable across all orientations."""
    m = np.asarray(matrix, dtype=np.float64)
    m00, m01, m02 = float(m[0, 0]), float(m[0, 1]), float(m[0, 2])
    m10, m11, m12 = float(m[1, 0]), float(m[1, 1]), float(m[1, 2])
    m20, m21, m22 = float(m[2, 0]), float(m[2, 1]), float(m[2, 2])
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = ((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        q = (0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        q = ((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        q = ((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)
    return quat_normalize(q)


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        (
            (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
            (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
            (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
        ),
        dtype=np.float32,
    )


def quat_between_vectors(a: Sequence[float], b: Sequence[float]) -> Quaternion:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    va = va / np.linalg.norm(va)
    vb = vb / np.linalg.norm(vb)
    dot = float(np.dot(va, vb))
    if dot > 1.0 - 1.0e-8:
        return (0.0, 0.0, 0.0, 1.0)
    if dot < -1.0 + 1.0e-8:
        axis = np.cross(va, axis_vector("X"))
        if np.linalg.norm(axis) <= 1.0e-8:
            axis = np.cross(va, axis_vector("Y"))
        axis = axis / np.linalg.norm(axis)
        return (float(axis[0]), float(axis[1]), float(axis[2]), 0.0)
    xyz = np.cross(va, vb)
    return quat_normalize((float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0 + dot))


def quat_rotate(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    qx, qy, qz, qw = quat_normalize(q)
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return np.asarray(
        (
            vx + qw * tx + qy * tz - qz * ty,
            vy + qw * ty + qz * tx - qx * tz,
            vz + qw * tz + qx * ty - qy * tx,
        ),
        dtype=np.float32,
    )


def quat_to_xyz_degrees(q: Sequence[float]) -> tuple[float, float, float]:
    x, y, z, w = quat_normalize(q)
    sin_x = 2.0 * (w * x + y * z)
    cos_x = 1.0 - 2.0 * (x * x + y * y)
    angle_x = math.atan2(sin_x, cos_x)

    sin_y = 2.0 * (w * y - z * x)
    angle_y = math.asin(max(-1.0, min(1.0, sin_y)))

    sin_z = 2.0 * (w * z + x * y)
    cos_z = 1.0 - 2.0 * (y * y + z * z)
    angle_z = math.atan2(sin_z, cos_z)
    return (math.degrees(angle_x), math.degrees(angle_y), math.degrees(angle_z))


def world_position_from_device(
    position: Sequence[float],
    scale: float,
    center: Sequence[float],
    device_offset: Sequence[float],
    up_axis: str,
) -> np.ndarray:
    """Map device coordinates as ``x * right - z * depth + y * up``."""
    x, y, z = float(position[0]), float(position[1]), float(position[2])
    right, depth, up = scene_basis(up_axis)
    center_v = np.asarray(center, dtype=np.float32)
    offset_v = np.asarray(device_offset, dtype=np.float32)
    return center_v + offset_v + float(scale) * (x * right - z * depth + y * up)


def openhaptics_calibrated_quaternion(q: Sequence[float]) -> Quaternion:
    """Apply the empirical OpenHaptics orientation calibration.

    The conjugate and X/Y component flip are device calibration steps. The
    following basis matrix multiplication is the separate change-of-basis step.
    """
    qx, qy, qz, qw = quat_conjugate(q)
    return (-qx, -qy, qz, qw)




def minimou_position_to_adapter(position: Sequence[float]) -> Vector3:
    """Map MiniMou hardware position into the shared adapter pose frame."""
    return (-float(position[0]), float(position[1]), float(position[2]))


def minimou_reflect_x_handedness(q: Sequence[float]) -> Quaternion:
    """Reflect a quaternion through the MiniMou adapter's X-axis handedness flip."""
    reflection = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
    reflected = reflection @ quat_to_matrix(q) @ reflection
    return matrix_to_quaternion(reflected)


def minimou_orientation_to_quaternion(orientation: Sequence[float]) -> Quaternion:
    """Convert a MiniMou axis-angle orientation into the shared adapter frame."""
    if len(orientation) < 4:
        return (0.0, 0.0, 0.0, 1.0)
    ax = -float(orientation[0])
    ay = float(orientation[1])
    az = float(orientation[2])
    angle = float(orientation[3])
    return axis_radians_to_quaternion(ax, ay, az, angle)


def minimou_angles_to_quaternion(rot_degrees: float, pitch_degrees: float, yaw_degrees: float) -> Quaternion:
    """Convert MiniMou joint angles to the adapter-frame quaternion used by tests and replay."""
    pitch_fix = axis_degrees_to_quaternion(1.0, 0.0, 0.0, MINIMOU_PITCH_FIX_DEGREES)
    rot = axis_degrees_to_quaternion(0.0, 1.0, 0.0, float(rot_degrees))
    pitch = axis_degrees_to_quaternion(1.0, 0.0, 0.0, -float(pitch_degrees))
    yaw = axis_degrees_to_quaternion(0.0, 0.0, 1.0, float(yaw_degrees))
    return quat_mul(yaw, quat_mul(pitch, quat_mul(rot, pitch_fix)))




def y_up_pitch_yaw_correction(q: Sequence[float]) -> Quaternion:
    """Undo the Y-up pitch/yaw mirror while preserving roll/twist around Z."""
    qx, qy, qz, qw = quat_normalize(q)
    return (-qx, -qy, qz, qw)



def world_quaternion_from_device(q: Sequence[float], profile: HapticFrameProfile, up_axis: str) -> Quaternion:
    if profile.orientation_calibration == "none":
        world_q = quat_normalize(q)
    elif profile.orientation_calibration == "openhaptics":
        world_q = openhaptics_calibrated_quaternion(q)
    else:
        raise ValueError(f"unknown orientation calibration {profile.orientation_calibration!r}")

    if profile.rotate_orientation_basis:
        c_rot = orientation_basis_matrix(up_axis)
        world_q = matrix_to_quaternion(c_rot @ quat_to_matrix(world_q) @ c_rot.T)
    if parse_axis(up_axis) == "Y" and profile.correct_y_up_pitch_yaw:
        world_q = y_up_pitch_yaw_correction(world_q)
    return quat_normalize(world_q)


def pose_to_world(pose: PoseLike, config: HapticFrameConfig) -> WorldPose:
    position = world_position_from_device(
        pose.position,
        config.position_scale,
        config.center,
        config.device_offset,
        config.up_axis,
    )
    quaternion = world_quaternion_from_device(pose.quaternion, config.profile, config.up_axis)
    return WorldPose(position=position, quaternion=quaternion)


__all__ = [
    "FALLBACK_PROFILE",
    "MINIMOU_PROFILE",
    "MINIMOU_PITCH_FIX_DEGREES",
    "OPENHAPTICS_PROFILE",
    "HapticFrameConfig",
    "HapticFrameProfile",
    "Quaternion",
    "Vector3",
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
