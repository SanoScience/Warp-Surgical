# SPDX-License-Identifier: Apache-2.0
"""Viewer interaction math for the OmniSurg Hex runtime."""

from __future__ import annotations

import numpy as np
from newton._src.geometry.flags import ParticleFlags

from .haptic import matrix_to_quaternion, quat_to_matrix

ACTIVE_BIT = int(ParticleFlags.ACTIVE)


def _mouse_world_ray(viewer, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(viewer, "screen_to_world_ray"):
        origin, direction = viewer.screen_to_world_ray(x, y)
        origin = np.asarray(origin, dtype=np.float32).reshape(3)
        direction = np.asarray(direction, dtype=np.float32).reshape(3)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 1.0e-8:
            direction /= direction_norm
        return origin, direction

    fb_x, fb_y = viewer._to_framebuffer_coords(x, y)  # noqa: SLF001
    ray_start, ray_dir = viewer.camera.get_world_ray(fb_x, fb_y)
    origin = np.asarray(ray_start, dtype=np.float32).reshape(3)
    direction = np.asarray((ray_dir.x, ray_dir.y, ray_dir.z), dtype=np.float32)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm > 1.0e-8:
        direction /= direction_norm
    return origin, direction


def _vec3_array(value) -> np.ndarray:
    try:
        return np.asarray((float(value.x), float(value.y), float(value.z)), dtype=np.float32)
    except Exception:
        return np.asarray(value, dtype=np.float32).reshape(3)


def _normalize_or(value: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm > 1.0e-8:
        return (vec / norm).astype(np.float32, copy=False)
    return np.asarray(fallback, dtype=np.float32)


def _viewer_camera_frame(viewer) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if viewer is None:
        return None

    if all(hasattr(viewer, name) for name in ("_camera_pos", "_camera_right", "_camera_up", "_camera_forward")):
        try:
            pos = _vec3_array(getattr(viewer, "_camera_pos"))
            right = _normalize_or(_vec3_array(getattr(viewer, "_camera_right")), (1.0, 0.0, 0.0))
            up = _normalize_or(_vec3_array(getattr(viewer, "_camera_up")), (0.0, 0.0, 1.0))
            forward = _normalize_or(_vec3_array(getattr(viewer, "_camera_forward")), (0.0, 1.0, 0.0))
            return pos, right, up, forward
        except Exception:
            pass

    camera = getattr(viewer, "camera", None)
    if camera is None:
        return None
    try:
        pos = _vec3_array(camera.pos)
        forward = _normalize_or(_vec3_array(camera.get_front()), (0.0, 1.0, 0.0))
        right = _normalize_or(_vec3_array(camera.get_right()), (1.0, 0.0, 0.0))
        up = _normalize_or(_vec3_array(camera.get_up()), (0.0, 0.0, 1.0))
    except Exception:
        return None
    return pos, right, up, forward


def _camera_local_offsets(
    points: np.ndarray,
    camera_frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    pos, right, up, forward = camera_frame
    rel = np.asarray(points, dtype=np.float32).reshape(-1, 3) - np.asarray(pos, dtype=np.float32).reshape(1, 3)
    basis = np.stack((right, up, forward), axis=1).astype(np.float32, copy=False)
    return np.ascontiguousarray(rel @ basis, dtype=np.float32)


def _camera_points_from_local_offsets(
    offsets: np.ndarray,
    camera_frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    pos, right, up, forward = camera_frame
    local = np.asarray(offsets, dtype=np.float32).reshape(-1, 3)
    return np.ascontiguousarray(
        np.asarray(pos, dtype=np.float32).reshape(1, 3)
        + local[:, 0:1] * right.reshape(1, 3)
        + local[:, 1:2] * up.reshape(1, 3)
        + local[:, 2:3] * forward.reshape(1, 3),
        dtype=np.float32,
    )


def _camera_basis_matrix(camera_frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    _pos, right, up, forward = camera_frame
    return np.column_stack((right, up, forward)).astype(np.float32, copy=False)


def _camera_transform_points_between_frames(
    points: np.ndarray,
    reference_frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    current_frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    return _camera_points_from_local_offsets(_camera_local_offsets(points, reference_frame), current_frame)


def _camera_transform_quaternion_between_frames(
    quaternion: tuple[float, float, float, float] | np.ndarray,
    reference_frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    current_frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float, float, float]:
    delta = _camera_basis_matrix(current_frame) @ _camera_basis_matrix(reference_frame).T
    return matrix_to_quaternion(delta @ quat_to_matrix(quaternion))


def _pick_particle_from_ray(
    particle_q: np.ndarray,
    particle_flags: np.ndarray,
    particle_inv_mass: np.ndarray,
    particle_radius: np.ndarray,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    base_pick_radius: float,
) -> tuple[int, np.ndarray | None]:
    active_mask = ((particle_flags & ACTIVE_BIT) != 0) & (particle_inv_mass > 0.0)
    if not np.any(active_mask):
        return -1, None

    active_indices = np.nonzero(active_mask)[0]
    rel = particle_q[active_indices] - ray_origin[None, :]
    t = rel @ ray_direction
    forward_mask = t >= 0.0
    if not np.any(forward_mask):
        return -1, None

    active_indices = active_indices[forward_mask]
    rel = rel[forward_mask]
    t = t[forward_mask]
    closest = rel - t[:, None] * ray_direction[None, :]
    dist_sq = np.einsum("ij,ij->i", closest, closest)
    pick_radius = np.maximum(particle_radius[active_indices] * 4.0, base_pick_radius)
    hit_mask = dist_sq <= pick_radius * pick_radius
    if not np.any(hit_mask):
        return -1, None

    hit_indices = active_indices[hit_mask]
    hit_t = t[hit_mask]
    best = int(np.argmin(hit_t))
    particle = int(hit_indices[best])
    hit_point = ray_origin + hit_t[best] * ray_direction
    return particle, hit_point.astype(np.float32)


def _select_drag_particles(
    particle_q: np.ndarray,
    particle_flags: np.ndarray,
    particle_inv_mass: np.ndarray,
    seed_particle: int,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    if seed_particle < 0:
        return np.zeros(0, dtype=np.int32), np.zeros((0, 3), dtype=np.float32)

    active_mask = ((particle_flags & ACTIVE_BIT) != 0) & (particle_inv_mass > 0.0)
    if not active_mask[seed_particle]:
        return np.zeros(0, dtype=np.int32), np.zeros((0, 3), dtype=np.float32)

    seed_pos = particle_q[seed_particle].astype(np.float32, copy=False)
    delta = particle_q - seed_pos[None, :]
    radius_sq = max(float(radius), 0.0) ** 2
    selected_mask = active_mask & (np.einsum("ij,ij->i", delta, delta, optimize=True) <= radius_sq)
    selected = np.nonzero(selected_mask)[0].astype(np.int32)
    offsets = (particle_q[selected] - seed_pos[None, :]).astype(np.float32, copy=False)
    return selected, offsets


def _select_sphere_drag_particles(
    particle_q: np.ndarray,
    particle_flags: np.ndarray,
    particle_inv_mass: np.ndarray,
    particle_radius: np.ndarray,
    sphere_center: np.ndarray,
    sphere_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    if sphere_radius <= 0.0:
        return np.zeros(0, dtype=np.int32), np.zeros((0, 3), dtype=np.float32)

    active_mask = ((particle_flags & ACTIVE_BIT) != 0) & (particle_inv_mass > 0.0)
    if not np.any(active_mask):
        return np.zeros(0, dtype=np.int32), np.zeros((0, 3), dtype=np.float32)

    center = np.asarray(sphere_center, dtype=np.float32).reshape(3)
    q = np.asarray(particle_q, dtype=np.float32)
    radii = np.maximum(np.asarray(particle_radius, dtype=np.float32), 0.0)
    delta = q - center[None, :]
    dist_sq = np.einsum("ij,ij->i", delta, delta, optimize=True)
    contact_radius = np.maximum(float(sphere_radius), 0.0) + radii
    selected_mask = active_mask & (dist_sq <= contact_radius * contact_radius)
    selected = np.nonzero(selected_mask)[0].astype(np.int32)
    offsets = delta[selected].astype(np.float32, copy=False)
    return selected, offsets


def _intersect_ray_plane(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray | None:
    denom = float(np.dot(ray_direction, plane_normal))
    if abs(denom) < 1.0e-6:
        return None
    t = float(np.dot(plane_origin - ray_origin, plane_normal) / denom)
    return (ray_origin + t * ray_direction).astype(np.float32)


__all__ = [
    "_camera_basis_matrix",
    "_camera_local_offsets",
    "_camera_points_from_local_offsets",
    "_camera_transform_points_between_frames",
    "_camera_transform_quaternion_between_frames",
    "_intersect_ray_plane",
    "_mouse_world_ray",
    "_normalize_or",
    "_pick_particle_from_ray",
    "_select_drag_particles",
    "_select_sphere_drag_particles",
    "_vec3_array",
    "_viewer_camera_frame",
]
