# SPDX-License-Identifier: Apache-2.0
"""Digimouse hex-grid viewer with shape matching, MC surface, and mouse drag.

Combines the full Digimouse marching-cubes viewer from example 08 with the
local particle-only shape-matching solver from example 09.

Features:

* Digimouse by default, synthetic block fallback via ``--size``
* GPU marching-cubes surface and optional cryo texturing
* left-click cell deletion
* right-mouse particle drag for active dynamic nodes
* material stiffness plus shape-matching controls
* optional CUDA graph replay when no live drag is active
* CUDA/OpenGL interop enabled by default so ViewerGL mesh uploads stay GPU-side
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import newton
import numpy as np
import warp as wp
from newton._src.geometry.flags import ParticleFlags

from omnisurg.rendering.bridge import RenderBridge  # noqa: E402
from omnisurg.hex.deletion import DeviceDeletionResult, make_hex_deletion_state  # noqa: E402
from omnisurg.hex.heat import make_hex_heat_state  # noqa: E402
from omnisurg.hex.hex_grid import (  # noqa: E402
    build_hex_particle_grid,
    build_hierarchical_shape_matching_clusters,
    build_shape_matching_clusters,
)
from omnisurg.hex.shape_matching_solver import (  # noqa: E402
    HIERARCHICAL_SHAPE_MATCHING_FULL27,
    HIERARCHICAL_SHAPE_MATCHING_LABELS,
    HIERARCHICAL_SHAPE_MATCHING_OFF,
    HIERARCHICAL_SHAPE_MATCHING_OUTER8,
    L2_HIERARCHICAL_SHAPE_MATCHING_LABELS,
    SHAPE_MATCHING_GS_WEIGHT_AVERAGED,
    SHAPE_MATCHING_GS_WEIGHT_ALPHAS,
    SHAPE_MATCHING_GS_WEIGHT_FULL,
    SHAPE_MATCHING_GS_WEIGHT_LABELS,
    SHAPE_MATCHING_GS_WEIGHT_SQRT,
    SHAPE_MATCHING_SOLVE_COLORED_GS,
    SHAPE_MATCHING_SOLVE_GATHER,
    SHAPE_MATCHING_SOLVE_LABELS,
    SHAPE_MATCHING_SOLVE_SCATTER,
    HexShapeMatchingSolver,
)
from omnisurg.hex.io.cryo import load_cryo_texture  # noqa: E402
from omnisurg.hex.io.digimouse import DigimouseAtlas, load_digimouse  # noqa: E402
from omnisurg.hex.kernels.cell_render import HoverPicker, update_cell_render_state  # noqa: E402
from omnisurg.hex.kernels.grab import project_grab_distance_constraints  # noqa: E402
from omnisurg.hex.kernels.marching_cubes import (  # noqa: E402
    allocate_mc_buffers,
    bake_vertex_uv3,
    compute_mc_topology,
    compute_mc_vertex_positions,
    upload_mc_tables,
)
from omnisurg.hex.materials import DEFAULT_MATERIALS, SKIN, MaterialTable  # noqa: E402
from omnisurg.hex.render import (  # noqa: E402
    build_segmentation_color_texture,
    ColoredParticleOverlay,
    CryoMeshVertexOverlay,
    CryoTextureAtlas,
    GrabConstraintOverlay,
    ShapeMatchingClusterOverlay,
    SurfaceRenderer,
    install_cryo_volume_shader_patch,
    make_scoped_timer,
    set_scoped_timer_dict,
)
from omnisurg.hex.slang_viewer import (  # noqa: E402
    PROCEDURAL_MATERIAL_PARAM_NAMES,
    SLANG_RENDER_BACKENDS,
    SLANG_SURFACE_DEBUG_VIEW_HEIGHT,
    SLANG_SURFACE_DEBUG_VIEW_LABELS,
    SLANG_SURFACE_DEBUG_VIEW_OFF,
    SlangHexViewer,
    build_procedural_uv3_noise_scale,
    clamp_material_maker_params,
    clamp_procedural_material_params,
    clamp_slang_surface_debug_view,
    is_slang_backend,
    make_default_material_maker_params,
    make_default_procedural_materials,
    material_maker_parameter_row,
    material_maker_setting_value,
)
from omnisurg.hex.haptic import (  # noqa: E402
    FallbackInput,
    HapticFrameConfig,
    HapticUnavailable,
    InputPose,
    MINIMOU_PROFILE,
    matrix_to_quaternion,
    OPENHAPTICS_PROFILE,
    open_haptic_inputs,
    open_minimou_inputs,
    pose_to_world,
    quat_rotate,
    quat_to_matrix,
)
from omnisurg.hex.data.crop import (  # noqa: E402
    VisibleClassCrop,
    crop_aligned_texture_rgb,
    crop_labels_to_visible_classes,
)

ACTIVE_BIT = int(ParticleFlags.ACTIVE)
HDRI_MAP_EXTENSIONS = (".hdr", ".exr")
HDRI_NONE_LABEL = "None"
HIERARCHICAL_MODE_BY_NAME = {
    "off": HIERARCHICAL_SHAPE_MATCHING_OFF,
    "outer8": HIERARCHICAL_SHAPE_MATCHING_OUTER8,
    "full27": HIERARCHICAL_SHAPE_MATCHING_FULL27,
}
L2_HIERARCHICAL_MODE_BY_NAME = {
    "off": HIERARCHICAL_SHAPE_MATCHING_OFF,
    "outer8": HIERARCHICAL_SHAPE_MATCHING_OUTER8,
    "full125": HIERARCHICAL_SHAPE_MATCHING_FULL27,
}
L0_SHAPE_MATCHING_MODE_BY_NAME = {
    "scatter": SHAPE_MATCHING_SOLVE_SCATTER,
    "gather": SHAPE_MATCHING_SOLVE_GATHER,
    "gs": SHAPE_MATCHING_SOLVE_COLORED_GS,
}
GS_WEIGHTING_BY_NAME = {
    "averaged": SHAPE_MATCHING_GS_WEIGHT_AVERAGED,
    "sqrt": SHAPE_MATCHING_GS_WEIGHT_SQRT,
    "full": SHAPE_MATCHING_GS_WEIGHT_FULL,
}

_OMNISURG_PREPARED_VOLUME = None
_OMNISURG_PREPARED_TEXTURE_RGB = None
_INSTRUMENT_COUNT = 2
_INSTRUMENT_TRIGGER_THRESHOLD = 0.1
_MINIMOU_CUT_THRESHOLD = _INSTRUMENT_TRIGGER_THRESHOLD
_INSTRUMENT_TOOL_MODES = ("diathermy", "grasper", "scissors", "bipolar")
_INSTRUMENT_TOOL_MODE_ALIASES = {"cutting": "diathermy"}


def _normalize_instrument_tool_mode(mode: str) -> str:
    normalized = _INSTRUMENT_TOOL_MODE_ALIASES.get(str(mode).strip().lower(), str(mode).strip().lower())
    if normalized not in _INSTRUMENT_TOOL_MODES:
        choices = ", ".join((*_INSTRUMENT_TOOL_MODES, *sorted(_INSTRUMENT_TOOL_MODE_ALIASES)))
        raise ValueError(f"unknown instrument tool mode {mode!r}; expected one of: {choices}")
    return normalized


def _should_capture_instrument_grasp(
    *,
    is_grasper: bool,
    trigger_down: bool,
    trigger_was_down: bool,
    grasp_count: int,
) -> bool:
    return bool(is_grasper and trigger_down and ((not trigger_was_down) or int(grasp_count) <= 0))


@dataclass(frozen=True)
class TimerPanelRow:
    name: str
    calls: int
    avg_ms: float
    total_ms: float
    gpu_avg_ms: float | None = None


@dataclass(frozen=True)
class TimerPanelSnapshot:
    window_secs: float
    window_frames: int
    fps: float
    frame_count: int
    triangle_count: int
    rows: tuple[TimerPanelRow, ...]


def build_timer_panel_snapshot(
    timer_stats: dict[str, list[float]],
    *,
    window_secs: float,
    window_frames: int,
    frame_count: int,
    triangle_count: int,
    gpu_stats: dict[str, list[float]] | None = None,
    max_rows: int = 14,
) -> TimerPanelSnapshot:
    """Build an immutable Slang timing-panel snapshot from accumulated timer samples."""
    rows: list[TimerPanelRow] = []
    for name, samples in timer_stats.items():
        if not samples:
            continue
        total_ms = float(sum(samples))
        calls = len(samples)
        gpu_avg_ms = None
        if gpu_stats is not None:
            gpu_samples = gpu_stats.get(name)
            if gpu_samples:
                gpu_avg_ms = float(sum(gpu_samples)) / len(gpu_samples)
        rows.append(
            TimerPanelRow(
                name=str(name),
                calls=calls,
                avg_ms=total_ms / calls,
                total_ms=total_ms,
                gpu_avg_ms=gpu_avg_ms,
            )
        )

    rows.sort(key=lambda row: (row.avg_ms, row.total_ms, row.name), reverse=True)
    limit = max(0, int(max_rows))
    if limit:
        rows = rows[:limit]
    else:
        rows = []
    fps = float(window_frames) / float(window_secs) if window_secs > 0.0 and window_frames > 0 else 0.0
    return TimerPanelSnapshot(
        window_secs=float(window_secs),
        window_frames=int(window_frames),
        fps=fps,
        frame_count=int(frame_count),
        triangle_count=int(triangle_count),
        rows=tuple(rows),
    )


def _effective_gs_support_alpha(weighting: int, alpha: float) -> float:
    if alpha < 0.0:
        return float(SHAPE_MATCHING_GS_WEIGHT_ALPHAS[int(weighting)])
    return min(1.0, max(0.0, float(alpha)))


def _make_block_atlas(size: int, voxel: float) -> DigimouseAtlas:
    """Pad the synthetic block so MC sees an exterior boundary."""
    labels = np.zeros((size + 2, size + 2, size + 2), dtype=np.uint8)
    labels[1:-1, 1:-1, 1:-1] = 1
    return DigimouseAtlas(labels=labels, voxel_size=voxel, materials=MaterialTable(DEFAULT_MATERIALS))


def _upload_rgb_texture(texture_rgb: np.ndarray, device):
    host = np.ascontiguousarray(texture_rgb)
    if host.ndim != 4 or host.shape[-1] != 3 or host.dtype != np.uint8:
        raise ValueError(f"expected uint8 (nx, ny, nz, 3) texture, got {host.shape} {host.dtype}")
    as_float = host.astype(np.float32) / 255.0
    return host, wp.array(as_float, dtype=wp.vec3, device=device)


def _atlas_class_map(atlas: DigimouseAtlas) -> dict[int, str]:
    return {int(idx): str(material.name) for idx, material in enumerate(atlas.materials.materials)}


def _crop_hex_atlas_to_visible_classes(
    atlas: DigimouseAtlas,
    settings_path: str | Path | None,
    margin_voxels: int,
) -> tuple[DigimouseAtlas, VisibleClassCrop | None]:
    if settings_path is None:
        return atlas, None
    crop = crop_labels_to_visible_classes(
        atlas.labels,
        _atlas_class_map(atlas),
        settings_path,
        margin_voxels=margin_voxels,
    )
    origin = tuple(float(atlas.origin[i]) + float(crop.crop_min[i]) * float(atlas.voxel_size) for i in range(3))
    metadata = dict(getattr(atlas, "metadata", {}) or {})
    metadata["visible_class_crop"] = dict(crop.metadata)
    return dataclasses.replace(atlas, labels=crop.labels, origin=origin, metadata=metadata), crop


def _load_cryo_texture_for_crop(path: Path, crop: VisibleClassCrop | None, device):
    if crop is None:
        return load_cryo_texture(path, device=device)
    host = np.load(path, mmap_mode="r")
    cropped = crop_aligned_texture_rgb(host, crop, texture_name="cryo texture")
    return _upload_rgb_texture(cropped, device)


def _discover_hdri_maps(folder: str | Path) -> tuple[Path, ...]:
    root = Path(folder).expanduser()
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            (
                path
                for path in root.iterdir()
                if path.is_file() and path.suffix.lower() in HDRI_MAP_EXTENSIONS
            ),
            key=lambda path: path.name.lower(),
        )
    )


def _hdri_map_key(path: str | Path | None) -> str:
    if not path:
        return ""
    return str(Path(path).expanduser().resolve(strict=False))


def _hdri_map_choice_paths(folder: str | Path, current_path: str | Path | None = None) -> tuple[Path | None, ...]:
    choices: list[Path | None] = [None]
    seen = {""}
    for path in _discover_hdri_maps(folder):
        key = _hdri_map_key(path)
        choices.append(path)
        seen.add(key)
    if current_path and _hdri_map_key(current_path) not in seen:
        choices.append(Path(current_path).expanduser())
    return tuple(choices)


def _hdri_map_choice_labels(paths: tuple[Path | None, ...]) -> tuple[str, ...]:
    return tuple(HDRI_NONE_LABEL if path is None else path.name for path in paths)


def _hdri_map_choice_index(paths: tuple[Path | None, ...], current_path: str | Path | None) -> int:
    current_key = _hdri_map_key(current_path)
    for idx, path in enumerate(paths):
        if _hdri_map_key(path) == current_key:
            return idx
    return 0


def _apply_gravity(model, enabled: bool, gravity_on: np.ndarray, gravity_off: np.ndarray) -> None:
    model.gravity.assign(gravity_on if enabled else gravity_off)


def _viewer_supports_mouse_interaction(viewer) -> bool:
    return isinstance(viewer, newton.viewer.ViewerGL) or bool(getattr(viewer, "supports_mouse_interaction", False))


class _HeadlessHexViewer:
    supports_mouse_interaction = False
    supports_implot = False

    def __init__(self):
        self.renderer = self
        self.show_particles = False
        self.show_ui = False
        self._paused = False
        self._key_handler = {}
        self.ui = SimpleNamespace(
            is_available=False,
            io=SimpleNamespace(display_size=(0.0, 0.0)),
            is_capturing=lambda: False,
        )

    def set_model(self, _model) -> None:
        return None

    def set_camera(self, *args, **kwargs) -> None:
        return None

    def begin_frame(self, _time: float) -> None:
        return None

    def end_frame(self) -> None:
        return None

    def log_state(self, _state) -> None:
        return None

    def log_mesh(self, *args, **kwargs) -> None:
        return None

    def log_points(self, *args, **kwargs) -> None:
        return None

    def register_ui_callback(self, *args, **kwargs) -> None:
        return None

    def is_paused(self) -> bool:
        return bool(self._paused)

    def is_running(self) -> bool:
        return False

    def close(self) -> None:
        return None


def _viewer_ui_capturing(viewer) -> bool:
    ui = getattr(viewer, "ui", None)
    if ui is None:
        return False
    try:
        return bool(ui.is_capturing())
    except Exception:
        return False


def _delete_stats_dirty(delete_state) -> bool:
    dirty_domains = getattr(delete_state, "_dirty_domains", None)
    if dirty_domains is None:
        return True
    return "stats" in dirty_domains


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


def _active_cells_for_material(
    cell_material: np.ndarray,
    cell_active: np.ndarray,
    material_idx: int,
) -> np.ndarray:
    active = (cell_active != 0) & (cell_material == int(material_idx))
    return np.nonzero(active)[0].astype(np.int32, copy=False)


def _active_cells_outside_cluster_coverage(
    cell_active: np.ndarray,
    cell_to_cluster: np.ndarray,
) -> np.ndarray:
    outside = (cell_active != 0) & (cell_to_cluster < 0)
    return np.nonzero(outside)[0].astype(np.int32, copy=False)


def _outer_layer_cells(
    cell_grid_xyz: np.ndarray,
    cell_active: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> np.ndarray:
    active_indices = np.nonzero(cell_active != 0)[0].astype(np.int32, copy=False)
    if active_indices.size == 0:
        return np.empty(0, dtype=np.int32)

    coords = cell_grid_xyz[active_indices].astype(np.int32, copy=False)
    nx, ny, nz = (int(v) for v in grid_shape)
    occupied = np.zeros((nx + 2, ny + 2, nz + 2), dtype=bool)
    x = coords[:, 0] + 1
    y = coords[:, 1] + 1
    z = coords[:, 2] + 1
    occupied[x, y, z] = True

    interior = (
        occupied[x - 1, y, z]
        & occupied[x + 1, y, z]
        & occupied[x, y - 1, z]
        & occupied[x, y + 1, z]
        & occupied[x, y, z - 1]
        & occupied[x, y, z + 1]
    )
    return active_indices[~interior].astype(np.int32, copy=False)


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


def _make_ground_plane_mesh(
    particle_q: np.ndarray,
    *,
    ground_height: float,
    min_margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a quad that covers the body's XY footprint at the solver ground height."""
    points = np.asarray(particle_q, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"expected particle positions with shape (N, 3), got {points.shape}")

    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    footprint = max(float(bounds_max[0] - bounds_min[0]), float(bounds_max[1] - bounds_min[1]))
    margin = max(float(min_margin), 0.5 * footprint)
    z = float(ground_height)
    mesh_points = np.asarray(
        [
            [bounds_min[0] - margin, bounds_min[1] - margin, z],
            [bounds_max[0] + margin, bounds_min[1] - margin, z],
            [bounds_max[0] + margin, bounds_max[1] + margin, z],
            [bounds_min[0] - margin, bounds_max[1] + margin, z],
        ],
        dtype=np.float32,
    )
    mesh_indices = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.int32)
    return mesh_points, mesh_indices


def _initial_instrument_center(particle_q: np.ndarray) -> tuple[float, float, float]:
    points = np.asarray(particle_q, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return (0.0, 0.0, 0.0)
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    center = 0.5 * (bounds_min + bounds_max)
    return (float(center[0]), float(center[1]), float(center[2]))


def _fallback_instrument_inputs() -> list[FallbackInput]:
    return [
        FallbackInput(InputPose(position=(0.0, 0.0, 0.0), valid=True)),
        FallbackInput(InputPose(position=(0.0, 0.0, 0.0), valid=True)),
    ]


def _open_instrument_inputs(
    backend: str,
    device_name: str,
    left_device_name: str,
) -> list:
    if backend == "off":
        return []
    if backend == "fallback":
        return _fallback_instrument_inputs()
    if backend == "minimou":
        return open_minimou_inputs(count=_INSTRUMENT_COUNT)
    if backend == "openhaptics":
        return open_haptic_inputs([device_name, left_device_name])
    raise ValueError(f"unknown input backend {backend!r}")


def _widen_viewer_left_panel(viewer, width: float = 600.0) -> None:
    """Patch this Newton ViewerGL instance to use a wider built-in side panel."""
    if not isinstance(viewer, newton.viewer.ViewerGL):
        return

    original_render_left_panel = viewer._render_left_panel

    def _render_left_panel_with_width():
        ui = getattr(viewer, "ui", None)
        if ui is None or not ui.is_available:
            return original_render_left_panel()

        imgui = ui.imgui
        original_set_next_window_size = imgui.set_next_window_size
        patched_first_size = False

        def _set_next_window_size(size, *args, **kwargs):
            nonlocal patched_first_size
            if not patched_first_size:
                patched_first_size = True
                panel_width = min(float(width), max(300.0, float(ui.io.display_size[0]) - 20.0))
                size = imgui.ImVec2(panel_width, size.y)
            return original_set_next_window_size(size, *args, **kwargs)

        imgui.set_next_window_size = _set_next_window_size
        try:
            return original_render_left_panel()
        finally:
            imgui.set_next_window_size = original_set_next_window_size

    viewer._render_left_panel = _render_left_panel_with_width


def _segmentation_panel_class_indices(material_names: list[str]) -> range:
    """Return the material indices controlled by the visible class panel."""
    return range(1, len(material_names)) if len(material_names) > 1 else range(len(material_names))


def _clamp_unit_float(value: float, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(default)
    if not np.isfinite(result):
        result = float(default)
    return float(np.clip(result, 0.0, 1.0))


def _clamp_rgb_tuple(value: object, default: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> tuple[float, float, float]:
    try:
        rgb = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return default
    if rgb.size < 3:
        return default
    rgb = np.where(np.isfinite(rgb[:3]), rgb[:3], np.asarray(default, dtype=np.float64))
    clamped = np.clip(rgb, 0.0, 1.0)
    return (float(clamped[0]), float(clamped[1]), float(clamped[2]))


def _clamp_combo_index(value: int, item_count: int) -> int:
    if item_count <= 0:
        return 0
    return max(0, min(int(value), item_count - 1))


def _imgui_combo_index(imgui, label: str, value: int, items: tuple[str, ...]) -> tuple[bool, int]:
    if not items:
        return False, 0
    current = _clamp_combo_index(value, len(items))
    if hasattr(imgui, "combo"):
        try:
            changed, new_value = imgui.combo(label, current, items)
            return bool(changed), _clamp_combo_index(new_value, len(items))
        except TypeError:
            pass
    changed, new_value = imgui.slider_int(label, current, 0, len(items) - 1)
    return bool(changed), _clamp_combo_index(new_value, len(items))


def _imgui_combo_int(imgui, label: str, value: int, items: tuple[str, ...]) -> tuple[bool, int]:
    changed, new_value = _imgui_combo_index(imgui, label, clamp_slang_surface_debug_view(value), items)
    return changed, clamp_slang_surface_debug_view(new_value)


def _panel_settings_material_item(
    idx: int,
    name: str,
    by_id: dict[int, dict],
    by_name: dict[str, dict],
) -> dict | None:
    item = by_name.get(name)
    if item is not None:
        return item
    item = by_id.get(idx)
    if item is None:
        return None
    saved_name = item.get("name")
    if isinstance(saved_name, str) and saved_name and saved_name != name:
        return None
    return item


def _save_segmentation_panel_settings(path: str | Path, ui: UiState) -> None:
    assert ui.material_names is not None
    assert ui.material_stiffness_scale is not None
    assert ui.material_visible is not None
    assert ui.material_cuttable is not None
    assert ui.material_locked is not None
    procedural = (
        ui.material_procedural
        if ui.material_procedural is not None and len(ui.material_procedural) == len(ui.material_names)
        else make_default_procedural_materials(len(ui.material_names))
    )
    material_maker = (
        ui.material_maker_params
        if ui.material_maker_params is not None and len(ui.material_maker_params) == len(ui.material_names)
        else make_default_material_maker_params(len(ui.material_names), ui.material_maker_parameter_specs)
    )

    payload = {
        "version": 2,
        "slang_surface": {
            "procedural_enabled": bool(ui.slang_procedural_surface),
            "procedural_world_space": bool(ui.slang_procedural_world_space),
            "lighting_enabled": bool(ui.slang_surface_lighting),
            "key_light_enabled": bool(ui.slang_key_light),
            "fill_light_enabled": bool(ui.slang_fill_light),
            "ambient_light_enabled": bool(ui.slang_ambient_light),
            "environment_lighting_enabled": bool(ui.slang_environment_lighting),
            "debug_view": clamp_slang_surface_debug_view(ui.slang_debug_view),
            "cryo_mix": _clamp_unit_float(ui.slang_cryo_mix, 0.0),
            "state_overlay_strength": _clamp_unit_float(ui.slang_state_overlay_strength, 1.0),
            "environment_background": bool(ui.slang_environment_background),
            "environment_map": str(ui.slang_environment_map),
            "environment_intensity": max(float(ui.slang_environment_intensity), 0.0),
            "environment_rotation_degrees": float(ui.slang_environment_rotation_degrees),
            "environment_pitch_degrees": float(ui.slang_environment_pitch_degrees),
        },
        "materials": [
            {
                "id": int(idx),
                "name": str(name),
                "visible": bool(ui.material_visible[idx]),
                "cuttable": bool(ui.material_cuttable[idx]),
                "locked": bool(ui.material_locked[idx]),
                "stiffness_scale": float(ui.material_stiffness_scale[idx]),
                "color": (
                    list(_clamp_rgb_tuple(ui.material_colors[idx]))
                    if ui.material_colors is not None and idx < len(ui.material_colors)
                    else [1.0, 1.0, 1.0]
                ),
                "procedural": clamp_procedural_material_params(procedural[idx]),
                "material_maker": clamp_material_maker_params(
                    material_maker[idx] if idx < len(material_maker) else None,
                    ui.material_maker_parameter_specs,
                ),
            }
            for idx, name in enumerate(ui.material_names)
        ],
    }
    settings_path = Path(path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_segmentation_panel_settings(path: str | Path, ui: UiState) -> int:
    assert ui.material_names is not None
    assert ui.material_stiffness_scale is not None
    assert ui.material_visible is not None
    assert ui.material_cuttable is not None
    assert ui.material_locked is not None

    settings_path = Path(path)
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    materials = payload.get("materials") if isinstance(payload, dict) else None
    if not isinstance(materials, list):
        raise ValueError(f"{settings_path} does not contain a material settings list")

    slang_surface = payload.get("slang_surface") if isinstance(payload, dict) else None
    if isinstance(slang_surface, dict):
        if "procedural_enabled" in slang_surface:
            ui.slang_procedural_surface = bool(slang_surface["procedural_enabled"])
        if "procedural_world_space" in slang_surface:
            ui.slang_procedural_world_space = bool(slang_surface["procedural_world_space"])
        if "lighting_enabled" in slang_surface:
            ui.slang_surface_lighting = bool(slang_surface["lighting_enabled"])
        if "key_light_enabled" in slang_surface:
            ui.slang_key_light = bool(slang_surface["key_light_enabled"])
        if "fill_light_enabled" in slang_surface:
            ui.slang_fill_light = bool(slang_surface["fill_light_enabled"])
        if "ambient_light_enabled" in slang_surface:
            ui.slang_ambient_light = bool(slang_surface["ambient_light_enabled"])
        if "environment_lighting_enabled" in slang_surface:
            ui.slang_environment_lighting = bool(slang_surface["environment_lighting_enabled"])
        if "debug_view" in slang_surface:
            ui.slang_debug_view = clamp_slang_surface_debug_view(slang_surface["debug_view"])
        elif "height_debug" in slang_surface:
            ui.slang_debug_view = clamp_slang_surface_debug_view(None, height_debug=bool(slang_surface["height_debug"]))
        if "cryo_mix" in slang_surface:
            ui.slang_cryo_mix = _clamp_unit_float(slang_surface["cryo_mix"], ui.slang_cryo_mix)
        if "state_overlay_strength" in slang_surface:
            ui.slang_state_overlay_strength = _clamp_unit_float(
                slang_surface["state_overlay_strength"],
                ui.slang_state_overlay_strength,
            )
        if "environment_background" in slang_surface:
            ui.slang_environment_background = bool(slang_surface["environment_background"])
        if "environment_map" in slang_surface:
            ui.slang_environment_map = str(slang_surface["environment_map"] or "")
        if "environment_intensity" in slang_surface:
            ui.slang_environment_intensity = max(float(slang_surface["environment_intensity"]), 0.0)
        if "environment_rotation_degrees" in slang_surface:
            ui.slang_environment_rotation_degrees = float(slang_surface["environment_rotation_degrees"])
        if "environment_pitch_degrees" in slang_surface:
            ui.slang_environment_pitch_degrees = float(slang_surface["environment_pitch_degrees"])

    by_id: dict[int, dict] = {}
    by_name: dict[str, dict] = {}
    for item in materials:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("id")
        if isinstance(raw_id, int):
            by_id[raw_id] = item
        raw_name = item.get("name")
        if isinstance(raw_name, str):
            by_name[raw_name] = item

    applied = 0
    procedural_changed = False
    material_maker_changed = False
    colors_changed = False
    if ui.material_procedural is None or len(ui.material_procedural) != len(ui.material_names):
        ui.material_procedural = make_default_procedural_materials(len(ui.material_names))
    if ui.material_maker_params is None or len(ui.material_maker_params) != len(ui.material_names):
        ui.material_maker_params = make_default_material_maker_params(len(ui.material_names), ui.material_maker_parameter_specs)
    for idx, name in enumerate(ui.material_names):
        item = _panel_settings_material_item(idx, name, by_id, by_name)
        if item is None:
            continue
        if "visible" in item:
            ui.material_visible[idx] = bool(item["visible"])
        if "cuttable" in item:
            ui.material_cuttable[idx] = bool(item["cuttable"])
        if "locked" in item:
            ui.material_locked[idx] = bool(item["locked"])
        if "stiffness_scale" in item:
            scale = float(item["stiffness_scale"])
            if np.isfinite(scale):
                ui.material_stiffness_scale[idx] = float(np.clip(scale, 0.0, 5.0))
        if "color" in item and ui.material_colors is not None and idx < len(ui.material_colors):
            new_color = _clamp_rgb_tuple(item["color"], ui.material_colors[idx])
            if new_color != ui.material_colors[idx]:
                ui.material_colors[idx] = new_color
                colors_changed = True
        procedural_item = item.get("procedural")
        if isinstance(procedural_item, dict):
            ui.material_procedural[idx] = clamp_procedural_material_params(procedural_item)
            procedural_changed = True
        material_maker_item = item.get("material_maker")
        if isinstance(material_maker_item, dict):
            ui.material_maker_params[idx] = clamp_material_maker_params(
                material_maker_item,
                ui.material_maker_parameter_specs,
            )
            material_maker_changed = True
        applied += 1
    if procedural_changed:
        ui.material_procedural_revision += 1
    if material_maker_changed:
        ui.material_maker_params_revision += 1
    if colors_changed:
        ui.material_colors_revision += 1
    return applied


def _apply_grab_distance_constraints(
    state,
    particle_inv_mass: wp.array,
    particle_flags: wp.array,
    drag_indices: wp.array,
    drag_offsets: wp.array,
    drag_count: int,
    target: np.ndarray,
    stiffness: float,
    device,
) -> None:
    if drag_count <= 0:
        return
    project_grab_distance_constraints(
        state.particle_q,
        state.particle_qd,
        particle_inv_mass,
        particle_flags,
        drag_indices,
        drag_offsets,
        int(drag_count),
        target,
        float(stiffness),
        device=device,
    )


@wp.kernel(enable_backward=False)
def _enforce_locked_nodes_kernel(
    locked_indices: wp.array(dtype=wp.int32),
    locked_positions: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle_idx = locked_indices[tid]
    if particle_idx < 0:
        return

    particle_q[particle_idx] = locked_positions[tid]
    particle_qd[particle_idx] = wp.vec3(0.0, 0.0, 0.0)


def _enforce_locked_nodes(
    state,
    locked_indices: wp.array,
    locked_positions: wp.array,
    locked_count: int,
    device,
) -> None:
    if locked_count <= 0:
        return
    wp.launch(
        kernel=_enforce_locked_nodes_kernel,
        dim=int(locked_count),
        inputs=[locked_indices, locked_positions],
        outputs=[state.particle_q, state.particle_qd],
        device=device,
    )


def _merge_locked_node_positions(
    locked_indices_host: np.ndarray,
    locked_positions_host: np.ndarray,
    locked_slot_by_node: dict[int, int],
    locked_count: int,
    node_indices: np.ndarray,
    node_positions: np.ndarray,
) -> int:
    nodes = np.asarray(node_indices, dtype=np.int32).reshape(-1)
    positions = np.asarray(node_positions, dtype=np.float32)
    if positions.shape != (nodes.size, 3):
        raise ValueError(f"locked node positions shape mismatch: expected {(nodes.size, 3)}, got {positions.shape}")

    for node, position in zip(nodes.tolist(), positions, strict=True):
        if node < 0:
            continue
        slot = locked_slot_by_node.get(node)
        if slot is None:
            if locked_count >= locked_indices_host.shape[0]:
                raise RuntimeError("locked node buffer is full")
            slot = int(locked_count)
            locked_slot_by_node[node] = slot
            locked_indices_host[slot] = node
            locked_count += 1
        locked_positions_host[slot] = position
    return int(locked_count)


class _StartupPhase:
    """Context manager that records wall-clock time for one startup phase.

    When ``sync_device`` is given, ``wp.synchronize_device(device)`` is called
    on exit so the timing reflects GPU work that was queued inside the block
    (uploads, kernel launches) rather than just the host-side return.
    """

    def __init__(
        self,
        name: str,
        sink: list[tuple[str, float]],
        *,
        sync_device: wp.context.Device | None = None,
    ) -> None:
        self.name = name
        self.sink = sink
        self.sync_device = sync_device

    def __enter__(self) -> _StartupPhase:
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        if self.sync_device is not None:
            wp.synchronize_device(self.sync_device)
        self.sink.append((self.name, time.perf_counter() - self.t0))


def _print_startup_report(phases: list[tuple[str, float]], total: float) -> None:
    captured = sum(dt for _, dt in phases)
    print(f"[startup] total {total * 1000:.1f} ms (captured {captured * 1000:.1f} ms)")
    for name, dt in sorted(phases, key=lambda kv: -kv[1]):
        share = (100.0 * dt / total) if total > 0.0 else 0.0
        print(f"  {name:<32} {dt * 1000:8.1f} ms  ({share:4.1f}%)")
    leftover = max(0.0, total - captured)
    if leftover > 0.0:
        print(f"  {'(uncaptured)':<32} {leftover * 1000:8.1f} ms")


@dataclass
class UiState:
    gravity_enabled: bool = True
    viewer_log_state: bool = True
    show_mesh: bool = True
    show_ground_plane: bool = True
    show_cell_particles: bool = False
    show_nodes: bool = False
    show_mc_vertex_samples: bool = False
    show_timing_panel: bool = True
    show_lighting_panel: bool = True
    timer_panel_snapshot: TimerPanelSnapshot | None = None
    particle_particle_collisions: bool = False
    cryo_colored_cells: bool = False
    stress_colored_surface: bool = False
    stress_color_scale: float = 4.0
    cryo_scale_x: float = 1.0
    cryo_scale_y: float = 1.0
    cryo_scale_z: float = 1.0
    slang_procedural_surface: bool = True
    slang_procedural_world_space: bool = False
    slang_surface_lighting: bool = True
    slang_key_light: bool = True
    slang_fill_light: bool = True
    slang_ambient_light: bool = False
    slang_environment_lighting: bool = True
    slang_debug_view: int = SLANG_SURFACE_DEBUG_VIEW_OFF
    slang_cryo_mix: float = 0.0
    slang_state_overlay_strength: float = 1.0
    slang_procedural_material_scale: float = 1.0
    slang_environment_map: str = "environments/photo_studio_01_1k.hdr"
    slang_environment_intensity: float = 1.0
    slang_environment_background: bool = True
    slang_environment_rotation_degrees: float = 0.0
    slang_environment_pitch_degrees: float = -90.0
    material_names: list[str] | None = None
    material_stiffness_scale: list[float] | None = None
    material_visible: list[bool] | None = None
    material_cuttable: list[bool] | None = None
    material_locked: list[bool] | None = None
    material_colors: list[tuple[float, float, float]] | None = None
    material_colors_revision: int = 0
    material_colors_revision_pending: bool = False
    material_procedural: list[dict[str, float]] | None = None
    material_procedural_revision: int = 0
    material_procedural_revision_pending: bool = False
    material_maker_params: list[dict[str, Any]] | None = None
    material_maker_params_revision: int = 0
    material_maker_params_revision_pending: bool = False
    material_maker_parameter_specs: tuple[Any, ...] = ()
    material_shader_edit_index: int = 1
    material_dirty: bool = False
    material_visibility_dirty: bool = False
    material_visibility_revision: int = 0
    material_settings_status: str = ""
    smooth_mesh_normals: bool = True
    taubin_iterations: int = 2
    taubin_lambda: float = 1.00
    taubin_mu: float = -0.34
    active_cut_fast_surface: bool = True
    active_cut_smooth_mesh_normals: bool = False
    active_cut_taubin_iterations: int = 0
    enable_shape_matching: bool = True
    shape_matching_mode: int = SHAPE_MATCHING_SOLVE_SCATTER
    shape_matching_gs_weighting: int = SHAPE_MATCHING_GS_WEIGHT_AVERAGED
    shape_matching_gs_support_alpha: float = -1.0
    shape_matching_use_computed_prolongation: bool = True
    enable_volume_preservation: bool = False
    volume_preservation_stiffness: float = 0.0
    volume_preservation_passes: int = 1
    show_l0_shape_clusters: bool = False
    show_l1_shape_clusters: bool = False
    show_l2_shape_clusters: bool = False
    sleep_l0_shape_matching: bool = False
    hierarchical_shape_matching_mode: int = HIERARCHICAL_SHAPE_MATCHING_OUTER8
    l2_hierarchical_shape_matching_mode: int = HIERARCHICAL_SHAPE_MATCHING_OUTER8
    hierarchical_shape_matching_use_gs: bool = False
    l2_hierarchical_shape_matching_use_gs: bool = False
    hierarchical_shape_matching_outer8_prolongation: bool = True
    l2_hierarchical_shape_matching_outer8_prolongation: bool = True
    hierarchical_shape_matching_outer8_absolute_projection: bool = False
    shape_matching_stiffness: float = 1.0
    shape_matching_relaxation: float = 1.0
    shape_matching_passes: int = 1
    hierarchical_shape_matching_stiffness: float = 1.0
    hierarchical_shape_matching_relaxation: float = 1.0
    hierarchical_shape_matching_passes: int = 1
    l2_hierarchical_shape_matching_stiffness: float = 1.0
    l2_hierarchical_shape_matching_relaxation: float = 1.0
    l2_hierarchical_shape_matching_passes: int = 1
    substeps: int = 8
    iterations: int = 8
    frame: int = 0
    active_cells: int = 0
    deleted_total: int = 0
    tri_count: int = 0
    last_pick_cell: int = -1
    last_deleted_cell: int = -1
    pending_delete_material: int = -1
    pending_toggle_material_lock: int = -1
    pending_peel_outer_layer: bool = False
    pending_delete_outside_l1_clusters: bool = False
    pending_delete_outside_l2_clusters: bool = False
    ray_cut_depth_scale: float = 8.0
    last_ray_deleted_count: int = 0
    plane_cut_depth_scale: float = 8.0
    last_plane_deleted_count: int = 0
    pending_reset_simulation: bool = False
    drag_particle: int = -1
    drag_count: int = 0
    drag_radius_scale: float = 8.0
    drag_pull_stiffness: float = 1.0
    show_grab_constraints: bool = False
    show_instruments: bool = True
    instrument_follow_camera: bool = False
    instrument_collision_enabled: bool = True
    instrument_collision_use_mc_triangles: bool = False
    instrument_radius_scale: float = 5.0
    instrument_collision_relaxation: float = 0.9
    instrument_contact_iterations: int = 1
    instrument_max_correction_scale: float = 1.0
    instrument_tool_modes: list[str] = field(default_factory=lambda: ["diathermy"] * _INSTRUMENT_COUNT)
    instrument_grasp_counts: list[int] = field(default_factory=lambda: [0] * _INSTRUMENT_COUNT)
    show_heat_overlay: bool = False
    diathermy_power: float = 400.0
    heat_diffusion: float = 0.25
    heat_cooling: float = 0.10
    heat_substeps: int = 1
    heat_min: float = 0.0
    heat_max: float = 0.0
    blade_length_scale: float = 8.0
    blade_radius_scale: float = 0.75

    @property
    def slang_height_debug(self) -> bool:
        return self.slang_debug_view == SLANG_SURFACE_DEBUG_VIEW_HEIGHT

    @slang_height_debug.setter
    def slang_height_debug(self, enabled: bool) -> None:
        self.slang_debug_view = SLANG_SURFACE_DEBUG_VIEW_HEIGHT if enabled else SLANG_SURFACE_DEBUG_VIEW_OFF


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=str, default="Digimouse/atlas/atlas")
    parser.add_argument(
        "--downsample",
        type=int,
        default=16,
        help="Digimouse downsample factor. Use 16 for responsive picking; 8 is dense.",
    )
    parser.add_argument("--size", type=int, default=0, help="If > 0, use a synthetic NxNxN block instead of Digimouse.")
    parser.add_argument("--voxel", type=float, default=0.005, help="Block mode voxel size (ignored for Digimouse).")
    parser.add_argument(
        "--atlas-pad",
        type=int,
        default=1,
        help="Empty-voxel margin padded around the Digimouse atlas so MC closes caps where the mouse touches a grid edge. Default 1.",
    )
    parser.add_argument(
        "--crop-visible-classes",
        type=str,
        default=None,
        help="Segmentation panel settings JSON used to crop around visible classes before building the grid.",
    )
    parser.add_argument(
        "--crop-visible-margin-voxels",
        type=int,
        default=4,
        help="Voxel margin around visible crop seed classes when --crop-visible-classes is set.",
    )
    parser.add_argument(
        "--digimouse-cache",
        type=str,
        default=None,
        help="Override the Digimouse .npz cache path. Default: <atlas>/.cache/digimouse_ds<ds>_remap<hash>.npz.",
    )
    parser.add_argument(
        "--digimouse-use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read/write the Digimouse cache (default ON).",
    )
    parser.add_argument(
        "--digimouse-rebuild-cache",
        action="store_true",
        help="Force-reparse the .hdr/.img and overwrite the cache.",
    )
    parser.add_argument(
        "--viewer",
        choices=("gl", "headless", *sorted(SLANG_RENDER_BACKENDS)),
        default="gl",
        help=(
            "Renderer backend. 'gl' keeps ViewerGL; 'headless' creates no window; "
            "'slang' selects Vulkan on Linux and D3D12 on Windows."
        ),
    )
    parser.add_argument("--usd", type=str, default=None)
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Number of frames to run. Defaults to infinite for ViewerGL; required with --usd.",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--timer-report-secs",
        type=float,
        default=0.0,
        help="print aggregated ScopedTimer summaries every N wall-clock seconds (0 disables; default).",
    )
    parser.add_argument(
        "--timer-sync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="synchronize selected hot-path timers so their wall times include queued GPU work.",
    )
    parser.add_argument(
        "--timer-gpu-activities",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="collect CUDA kernel/memcpy/memset activity totals for selected hot-path timers.",
    )
    parser.add_argument(
        "--cut-debug-validate",
        action="store_true",
        help="run synchronizing GPU deletion-state validation after cut attempts.",
    )
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--drop-height", type=float, default=0.04)
    parser.add_argument("--ground-height", type=float, default=0.0)
    parser.add_argument(
        "--global-scale",
        type=float,
        default=1.0,
        help=(
            "Uniformly scales spatial quantities (voxel size, origin, drop height, particle radius, render extents). "
            "Stiffness, damping, mass, and friction are untouched."
        ),
    )
    parser.add_argument("--gravity-on", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dynamic-bones",
        action="store_true",
        help="Compatibility no-op: OmniSurg Hex always gives bone materials dynamic mass.",
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture the per-frame substep loop into a CUDA graph and replay it.",
    )
    parser.add_argument(
        "--particle-particle-collisions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable particle-particle self-collision in the local solver (default OFF).",
    )
    parser.add_argument(
        "--gl-interop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable Newton's CUDA/OpenGL VBO interop path for ViewerGL (default ON; use --no-gl-interop to force CPU uploads).",
    )
    parser.add_argument(
        "--viewer-log-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="call Newton's viewer.log_state() each frame (default ON).",
    )
    parser.add_argument(
        "--input-backend",
        choices=("fallback", "openhaptics", "minimou", "off"),
        default="fallback",
        help="pose source for the two kinematic instrument spheres.",
    )
    parser.add_argument("--device-name", default="Default Device", help="first OpenHaptics device name")
    parser.add_argument("--left-device-name", default="Left Device", help="second OpenHaptics device name")
    parser.add_argument(
        "--position-scale",
        type=float,
        default=0.001,
        help="world units per input-device millimetre; 0.001 maps mm to metres.",
    )
    parser.add_argument(
        "--device-x-offsets",
        type=float,
        nargs=2,
        default=(-0.17, 0.17),
        metavar=("FIRST_X", "SECOND_X"),
        help="world-space X offsets for first/right and second/left instrument bases.",
    )
    parser.add_argument(
        "--instrument-radius",
        "--instrument-radius-scale",
        dest="instrument_radius_scale",
        type=float,
        default=5.0,
        help="kinematic instrument sphere radius in voxel widths.",
    )
    parser.add_argument(
        "--instrument-collision-relaxation",
        type=float,
        default=0.9,
        help="sphere-vs-particle contact correction blend in [0, 1].",
    )
    parser.add_argument(
        "--instrument-contact-iterations",
        type=int,
        default=1,
        help="sphere-vs-particle contact projection passes per solver constraint iteration.",
    )
    parser.add_argument(
        "--instrument-max-correction-scale",
        type=float,
        default=1.0,
        help="maximum instrument contact position correction per pass in voxel widths; 0 disables the cap.",
    )
    parser.add_argument(
        "--show-instruments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="render the two kinematic instrument spheres.",
    )
    parser.add_argument(
        "--instrument-follow-camera",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "move the instrument spheres in the camera frame. "
            "Default: ON for fallback input, OFF for hardware input."
        ),
    )
    parser.add_argument(
        "--instrument-tool-modes",
        nargs=_INSTRUMENT_COUNT,
        choices=(*_INSTRUMENT_TOOL_MODES, *sorted(_INSTRUMENT_TOOL_MODE_ALIASES)),
        default=None,
        metavar=("FIRST", "SECOND"),
        help=(
            "per-instrument mode for the two input devices; defaults to grasper for MiniMou "
            "and diathermy for other backends. Deprecated cutting aliases to diathermy."
        ),
    )
    parser.add_argument("--show-heat-overlay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diathermy-power", type=float, default=400.0)
    parser.add_argument("--heat-diffusion", type=float, default=0.25)
    parser.add_argument("--heat-cooling", type=float, default=0.10)
    parser.add_argument("--heat-substeps", type=int, default=1)
    parser.add_argument("--blade-length", "--blade-length-scale", dest="blade_length_scale", type=float, default=8.0)
    parser.add_argument("--blade-radius", "--blade-radius-scale", dest="blade_radius_scale", type=float, default=0.75)
    parser.add_argument("--shape-matching-stiffness", type=float, default=1.0)
    parser.add_argument("--shape-matching-relaxation", type=float, default=1.0)
    parser.add_argument("--shape-matching-passes", type=int, default=1)
    parser.add_argument(
        "--shape-matching-mode",
        choices=tuple(L0_SHAPE_MATCHING_MODE_BY_NAME),
        default=None,
        help="L0 shape-matching solve mode: scatter, gather, or gs.",
    )
    parser.add_argument(
        "--shape-matching-gs-weighting",
        choices=tuple(GS_WEIGHTING_BY_NAME),
        default="averaged",
        help="Colored-GS support weighting: averaged, sqrt, or full.",
    )
    parser.add_argument(
        "--shape-matching-gs-support-alpha",
        type=float,
        default=-1.0,
        help="Colored-GS support exponent in [0, 1]; negative uses the selected weighting preset.",
    )
    parser.add_argument(
        "--shape-matching-gather",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use the experimental particle-gather shape-matching apply path (default uses cluster scatter)",
    )
    parser.add_argument(
        "--shape-matching-computed-prolongation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compute hierarchy/sleep prolongation parents from grid coordinates instead of reading precomputed tables",
    )
    parser.add_argument(
        "--volume-preservation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable the optional L0 hexahedral volume-preservation pass",
    )
    parser.add_argument("--volume-preservation-stiffness", type=float, default=0.0)
    parser.add_argument("--volume-preservation-passes", type=int, default=1)
    parser.add_argument(
        "--sleep-l0-shape-matching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="skip L0 shape matching inside intact active outer8 hierarchy blocks and trilinearly project children",
    )
    parser.add_argument("--sleep-l0-wake-halo-blocks", type=int, default=1)
    parser.add_argument(
        "--hierarchical-shape-matching",
        choices=tuple(HIERARCHICAL_MODE_BY_NAME),
        default="outer8",
        help="L1 prepass mode: off, outer8 with prolongation, or full27 direct.",
    )
    parser.add_argument("--hierarchical-shape-matching-stiffness", type=float, default=1.0)
    parser.add_argument("--hierarchical-shape-matching-relaxation", type=float, default=1.0)
    parser.add_argument("--hierarchical-shape-matching-passes", type=int, default=1)
    parser.add_argument(
        "--hierarchical-shape-matching-gs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run the L1 hierarchy prepass as colored Gauss-Seidel instead of scatter Jacobi",
    )
    parser.add_argument(
        "--hierarchical-shape-matching-outer8-prolongation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply L1 Outer8 coarse-vertex corrections to skipped fine nodes by trilinear prolongation",
    )
    parser.add_argument(
        "--l2-hierarchical-shape-matching",
        choices=tuple(L2_HIERARCHICAL_MODE_BY_NAME),
        default="outer8",
        help="L2 prepass mode: off, outer8 with optional prolongation, or full125 direct over 4x4x4 blocks.",
    )
    parser.add_argument("--l2-hierarchical-shape-matching-stiffness", type=float, default=1.0)
    parser.add_argument("--l2-hierarchical-shape-matching-relaxation", type=float, default=1.0)
    parser.add_argument("--l2-hierarchical-shape-matching-passes", type=int, default=1)
    parser.add_argument(
        "--l2-hierarchical-shape-matching-gs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run the L2 hierarchy prepass as colored Gauss-Seidel instead of scatter Jacobi",
    )
    parser.add_argument(
        "--l2-hierarchical-shape-matching-outer8-prolongation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply L2 Outer8 coarse-vertex corrections to skipped fine nodes by trilinear prolongation",
    )
    parser.add_argument(
        "--hierarchical-shape-matching-outer8-absolute-projection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="set L1/L2 Outer8 prolonged children to absolute trilinear vertex positions instead of applying vertex deltas",
    )
    parser.add_argument("--particle-radius-scale", type=float, default=0.18)
    parser.add_argument("--node-render-radius-scale", type=float, default=0.22)
    parser.add_argument("--cell-render-radius-scale", type=float, default=0.5)
    parser.add_argument(
        "--drag-radius-scale",
        type=float,
        default=8.0,
        help="RMB drag selection radius in voxel widths.",
    )
    parser.add_argument(
        "--drag-pull-stiffness",
        type=float,
        default=1.0,
        help="RMB/instrument grab distance-constraint blend per solver iteration, in [0, 1].",
    )
    parser.add_argument(
        "--show-grab-constraints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show GL pull-point lines and particles for mouse and grasper grabs.",
    )
    parser.add_argument(
        "--plane-cut-depth-scale",
        type=float,
        default=8.0,
        help="Left Shift + MMB plane-cut ray length in voxel widths.",
    )
    parser.add_argument(
        "--ray-cut-depth-scale",
        type=float,
        default=8.0,
        help="Left Alt ray-cut depth in voxel widths.",
    )
    parser.add_argument(
        "--render-particles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show the cell-centred particle overlay (default OFF).",
    )
    parser.add_argument("--overlay-line-width", type=float, default=0.00035)
    parser.add_argument(
        "--cryo-texture",
        type=str,
        default="Digimouse/cryo_texture.npy",
        help="Packed 3D RGB volume from tools/build_cryo_texture.py. Pass empty string to disable texturing.",
    )
    parser.add_argument(
        "--cryo-renderer",
        choices=("volume", "atlas", "off"),
        default="volume",
        help="MC surface cryo texturing path: direct GL 3D volume, baked 2D atlas, or disabled.",
    )
    parser.add_argument(
        "--slang-procedural-material",
        type=str,
        default="",
        help="Material Maker 3D Slang export to hot-load for the Slang procedural surface.",
    )
    parser.add_argument(
        "--slang-procedural-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to UV3/procedural coordinates before sampling the Material Maker shader.",
    )
    parser.add_argument(
        "--slang-procedural-hot-reload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Watch and hot-reload the external Material Maker Slang shader and sibling textures.",
    )
    parser.add_argument(
        "--slang-environment-map",
        type=str,
        default="environments/photo_studio_01_1k.hdr",
        help="HDRI/equirectangular environment map sampled by the Slang surface shader. Pass empty string to disable.",
    )
    parser.add_argument(
        "--slang-environment-intensity",
        type=float,
        default=1.0,
        help="Brightness multiplier for the Slang HDRI environment map.",
    )
    parser.add_argument(
        "--slang-environment-background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the Slang HDRI environment map behind the scene.",
    )
    parser.add_argument(
        "--slang-environment-rotation-deg",
        type=float,
        default=0.0,
        help="Yaw rotation applied to Slang HDRI sampling and background, in degrees.",
    )
    parser.add_argument(
        "--slang-environment-pitch-deg",
        type=float,
        default=-90.0,
        help="Pitch rotation applied to Slang HDRI sampling and background, in degrees.",
    )
    parser.add_argument(
        "--color-by-segmentation-map",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Color the MC surface by Digimouse/material segmentation labels; takes precedence over cryo texturing.",
    )
    parser.add_argument(
        "--color-by-stress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Color the MC surface by per-cell stretch using a cold-to-warm ramp.",
    )
    parser.add_argument(
        "--stress-color-scale",
        type=float,
        default=30.0,
        help="Multiplier applied to per-cell strain before mapping stress colours.",
    )
    parser.add_argument(
        "--active-cut-fast-surface",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use cheaper surface shading while continuous/ray cutting is active.",
    )
    parser.add_argument(
        "--active-cut-smooth-normals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep smooth MC normals enabled during active continuous/ray cutting.",
    )
    parser.add_argument(
        "--active-cut-taubin-iterations",
        type=int,
        default=0,
        help="Taubin smoothing iterations while continuous/ray cutting is active.",
    )
    parser.add_argument("--atlas-tile-size", type=int, default=8)
    parser.add_argument(
        "--segmentation-panel-settings",
        type=str,
        default="segmentation_panel_settings.json",
        help="JSON path used by the Tissue Classes panel Load/Save buttons.",
    )
    parser.add_argument(
        "--exit-after-init",
        action="store_true",
        help="Print the startup phase breakdown and exit before entering the frame loop.",
    )
    args = parser.parse_args(argv)
    if args.slang_procedural_material and args.viewer == "gl":
        args.viewer = "slang"
    slang_viewer_requested = is_slang_backend(args.viewer)
    if args.instrument_follow_camera is None:
        args.instrument_follow_camera = str(args.input_backend) == "fallback"
    environment_map_folder = (
        Path(args.slang_environment_map).expanduser().parent if args.slang_environment_map else Path("environments")
    )
    if args.instrument_tool_modes is None:
        default_mode = "grasper" if str(args.input_backend) == "minimou" else "diathermy"
        args.instrument_tool_modes = (default_mode,) * _INSTRUMENT_COUNT
    args.instrument_tool_modes = tuple(_normalize_instrument_tool_mode(mode) for mode in args.instrument_tool_modes)

    if args.usd is not None and args.frames is None:
        parser.error("--frames is required when using --usd")
    if args.usd is not None and slang_viewer_requested:
        parser.error("--viewer slang cannot be combined with --usd")
    if int(args.crop_visible_margin_voxels) < 0:
        parser.error("--crop-visible-margin-voxels must be >= 0")
    if float(args.instrument_radius_scale) < 0.0:
        parser.error("--instrument-radius-scale must be >= 0")
    if int(args.instrument_contact_iterations) < 1:
        parser.error("--instrument-contact-iterations must be >= 1")
    if float(args.instrument_max_correction_scale) < 0.0:
        parser.error("--instrument-max-correction-scale must be >= 0")

    startup_phases: list[tuple[str, float]] = []
    startup_t0 = time.perf_counter()
    visible_class_crop: VisibleClassCrop | None = None

    with _StartupPhase("atlas_load", startup_phases):
        if _OMNISURG_PREPARED_VOLUME is not None:
            atlas = _OMNISURG_PREPARED_VOLUME.to_hex_atlas()
            base_origin = getattr(_OMNISURG_PREPARED_VOLUME, "origin", (0.0, 0.0, 0.0))
            origin = (
                float(base_origin[0]),
                float(base_origin[1]),
                float(base_origin[2]) + float(args.drop_height),
            )
            scene_label = str(getattr(_OMNISURG_PREPARED_VOLUME, "metadata", {}).get("scene_label", "OmniSurg Hex"))
        elif args.size > 0:
            atlas = _make_block_atlas(args.size, args.voxel)
            block_extent = args.size * args.voxel
            origin = (-0.5 * block_extent - args.voxel, -0.5 * block_extent - args.voxel, args.drop_height - args.voxel)
            scene_label = f"block {args.size}^3"
        else:
            atlas = load_digimouse(
                args.atlas,
                downsample=args.downsample,
                pad=int(args.atlas_pad),
                cache_path=args.digimouse_cache,
                use_cache=bool(args.digimouse_use_cache),
                rebuild_cache=bool(args.digimouse_rebuild_cache),
            )
            origin = (0.0, 0.0, args.drop_height)
            scene_label = f"Digimouse --downsample {args.downsample}"

        if _OMNISURG_PREPARED_VOLUME is None and args.crop_visible_classes is not None:
            atlas, visible_class_crop = _crop_hex_atlas_to_visible_classes(
                atlas,
                args.crop_visible_classes,
                int(args.crop_visible_margin_voxels),
            )
            origin = tuple(float(origin[i]) + float(atlas.origin[i]) for i in range(3))
            scene_label = f"{scene_label}  (visible crop)"

        if args.global_scale != 1.0:
            scale = float(args.global_scale)
            atlas = dataclasses.replace(atlas, voxel_size=atlas.voxel_size * scale)
            origin = tuple(c * scale for c in origin)
            scene_label = f"{scene_label}  (global_scale={scale:g})"

    with _StartupPhase("build_hex_particle_grid", startup_phases):
        pg = build_hex_particle_grid(
            atlas,
            origin=origin,
            particle_radius=args.particle_radius_scale * atlas.voxel_size,
            kinematic_bones=False,
        )
    model = pg.model
    dev = model.device
    n_nodes = pg.aux.num_nodes
    n_cells = pg.aux.num_cells

    with _StartupPhase("make_hex_deletion_state", startup_phases, sync_device=dev):
        delete_state = make_hex_deletion_state(model, pg.aux)
    with _StartupPhase("make_hex_heat_state", startup_phases, sync_device=dev):
        heat_state = make_hex_heat_state(model, pg.aux)
    with _StartupPhase("clusters_l0", startup_phases, sync_device=dev):
        clusters = build_shape_matching_clusters(pg)
    with _StartupPhase("clusters_l1_hierarchy", startup_phases, sync_device=dev):
        hierarchy = build_hierarchical_shape_matching_clusters(pg)
    hierarchical_mode = HIERARCHICAL_MODE_BY_NAME[str(args.hierarchical_shape_matching)]
    l2_hierarchical_mode = L2_HIERARCHICAL_MODE_BY_NAME[str(args.l2_hierarchical_shape_matching)]
    if args.shape_matching_mode is None:
        shape_matching_mode = SHAPE_MATCHING_SOLVE_GATHER if bool(args.shape_matching_gather) else SHAPE_MATCHING_SOLVE_SCATTER
    else:
        shape_matching_mode = L0_SHAPE_MATCHING_MODE_BY_NAME[str(args.shape_matching_mode)]
    gs_weighting = GS_WEIGHTING_BY_NAME[str(args.shape_matching_gs_weighting)]
    gs_support_alpha = float(args.shape_matching_gs_support_alpha)
    with _StartupPhase("solver_ctor", startup_phases, sync_device=dev):
        solver = HexShapeMatchingSolver(
            model,
            clusters,
            iterations=args.iterations,
            enable_shape_matching=True,
            enable_self_collisions=bool(args.particle_particle_collisions),
            enable_ground_plane=True,
            shape_matching_stiffness=float(args.shape_matching_stiffness),
            shape_matching_relaxation=float(args.shape_matching_relaxation),
            shape_matching_passes=int(args.shape_matching_passes),
            shape_matching_mode=shape_matching_mode,
            shape_matching_gs_weighting=gs_weighting,
            shape_matching_gs_support_alpha=gs_support_alpha,
            shape_matching_use_computed_prolongation=bool(args.shape_matching_computed_prolongation),
            enable_volume_preservation=bool(args.volume_preservation),
            volume_preservation_stiffness=float(args.volume_preservation_stiffness),
            volume_preservation_passes=int(args.volume_preservation_passes),
            hierarchy=hierarchy,
            hierarchical_shape_matching_mode=hierarchical_mode,
            hierarchical_shape_matching_stiffness=float(args.hierarchical_shape_matching_stiffness),
            hierarchical_shape_matching_relaxation=float(args.hierarchical_shape_matching_relaxation),
            hierarchical_shape_matching_passes=int(args.hierarchical_shape_matching_passes),
            hierarchical_shape_matching_use_gs=bool(args.hierarchical_shape_matching_gs),
            hierarchical_shape_matching_outer8_prolongation=bool(args.hierarchical_shape_matching_outer8_prolongation),
            hierarchical_shape_matching_outer8_absolute_projection=bool(
                args.hierarchical_shape_matching_outer8_absolute_projection
            ),
            l2_hierarchical_shape_matching_mode=l2_hierarchical_mode,
            l2_hierarchical_shape_matching_stiffness=float(args.l2_hierarchical_shape_matching_stiffness),
            l2_hierarchical_shape_matching_relaxation=float(args.l2_hierarchical_shape_matching_relaxation),
            l2_hierarchical_shape_matching_passes=int(args.l2_hierarchical_shape_matching_passes),
            l2_hierarchical_shape_matching_use_gs=bool(args.l2_hierarchical_shape_matching_gs),
            l2_hierarchical_shape_matching_outer8_prolongation=bool(args.l2_hierarchical_shape_matching_outer8_prolongation),
            l2_hierarchical_shape_matching_outer8_absolute_projection=bool(
                args.hierarchical_shape_matching_outer8_absolute_projection
            ),
            sleep_l0_shape_matching=bool(args.sleep_l0_shape_matching),
            sleep_l0_wake_halo_blocks=int(args.sleep_l0_wake_halo_blocks),
            ground_height=float(args.ground_height),
        )

    ui = UiState(
        gravity_enabled=bool(args.gravity_on),
        viewer_log_state=bool(args.viewer_log_state),
        show_cell_particles=bool(args.render_particles),
        particle_particle_collisions=bool(args.particle_particle_collisions),
        shape_matching_stiffness=float(args.shape_matching_stiffness),
        shape_matching_relaxation=float(args.shape_matching_relaxation),
        shape_matching_passes=int(args.shape_matching_passes),
        shape_matching_mode=shape_matching_mode,
        shape_matching_gs_weighting=gs_weighting,
        shape_matching_gs_support_alpha=gs_support_alpha,
        shape_matching_use_computed_prolongation=bool(args.shape_matching_computed_prolongation),
        enable_volume_preservation=bool(args.volume_preservation),
        volume_preservation_stiffness=float(args.volume_preservation_stiffness),
        volume_preservation_passes=int(args.volume_preservation_passes),
        sleep_l0_shape_matching=bool(args.sleep_l0_shape_matching),
        hierarchical_shape_matching_mode=hierarchical_mode,
        hierarchical_shape_matching_stiffness=float(args.hierarchical_shape_matching_stiffness),
        hierarchical_shape_matching_relaxation=float(args.hierarchical_shape_matching_relaxation),
        hierarchical_shape_matching_passes=int(args.hierarchical_shape_matching_passes),
        hierarchical_shape_matching_use_gs=bool(args.hierarchical_shape_matching_gs),
        hierarchical_shape_matching_outer8_prolongation=bool(args.hierarchical_shape_matching_outer8_prolongation),
        hierarchical_shape_matching_outer8_absolute_projection=bool(
            args.hierarchical_shape_matching_outer8_absolute_projection
        ),
        l2_hierarchical_shape_matching_mode=l2_hierarchical_mode,
        l2_hierarchical_shape_matching_stiffness=float(args.l2_hierarchical_shape_matching_stiffness),
        l2_hierarchical_shape_matching_relaxation=float(args.l2_hierarchical_shape_matching_relaxation),
        l2_hierarchical_shape_matching_passes=int(args.l2_hierarchical_shape_matching_passes),
        l2_hierarchical_shape_matching_use_gs=bool(args.l2_hierarchical_shape_matching_gs),
        l2_hierarchical_shape_matching_outer8_prolongation=bool(args.l2_hierarchical_shape_matching_outer8_prolongation),
        substeps=int(args.substeps),
        iterations=int(args.iterations),
        ray_cut_depth_scale=float(args.ray_cut_depth_scale),
        plane_cut_depth_scale=float(args.plane_cut_depth_scale),
        drag_radius_scale=float(args.drag_radius_scale),
        drag_pull_stiffness=float(args.drag_pull_stiffness),
        show_grab_constraints=bool(args.show_grab_constraints),
        show_instruments=bool(args.show_instruments),
        instrument_follow_camera=bool(args.instrument_follow_camera),
        instrument_collision_enabled=str(args.input_backend) != "off",
        instrument_radius_scale=float(args.instrument_radius_scale),
        instrument_collision_relaxation=float(np.clip(float(args.instrument_collision_relaxation), 0.0, 1.0)),
        instrument_contact_iterations=max(1, int(args.instrument_contact_iterations)),
        instrument_max_correction_scale=float(args.instrument_max_correction_scale),
        instrument_tool_modes=[_normalize_instrument_tool_mode(mode) for mode in args.instrument_tool_modes],
        show_heat_overlay=bool(args.show_heat_overlay),
        diathermy_power=max(0.0, float(args.diathermy_power)),
        heat_diffusion=max(0.0, float(args.heat_diffusion)),
        heat_cooling=max(0.0, float(args.heat_cooling)),
        heat_substeps=max(1, int(args.heat_substeps)),
        blade_length_scale=max(0.0, float(args.blade_length_scale)),
        blade_radius_scale=max(0.0, float(args.blade_radius_scale)),
        active_cut_fast_surface=bool(args.active_cut_fast_surface),
        active_cut_smooth_mesh_normals=bool(args.active_cut_smooth_normals),
        active_cut_taubin_iterations=max(0, int(args.active_cut_taubin_iterations)),
        stress_colored_surface=bool(args.color_by_stress),
        stress_color_scale=max(0.0, float(args.stress_color_scale)),
        slang_procedural_material_scale=max(float(args.slang_procedural_scale), 0.000001),
        slang_environment_map=str(args.slang_environment_map or ""),
        slang_environment_intensity=max(float(args.slang_environment_intensity), 0.0),
        slang_environment_background=bool(args.slang_environment_background),
        slang_environment_rotation_degrees=float(args.slang_environment_rotation_deg),
        slang_environment_pitch_degrees=float(args.slang_environment_pitch_deg),
        active_cells=n_cells - int(delete_state.deleted_total),
        deleted_total=int(delete_state.deleted_total),
    )

    mats = pg.aux.materials
    atlas_metadata = getattr(atlas, "metadata", {}) or {}
    ui.material_names = [m.name for m in mats.materials]
    ui.material_stiffness_scale = list(atlas_metadata.get("material_stiffness_scale", [1.0] * len(mats)))
    ui.material_visible = list(atlas_metadata.get("material_visible", [True] * len(mats)))
    ui.material_cuttable = list(atlas_metadata.get("material_cuttable", [m.id == SKIN.id for m in mats.materials]))
    if len(ui.material_stiffness_scale) != len(mats):
        ui.material_stiffness_scale = [1.0] * len(mats)
    if len(ui.material_visible) != len(mats):
        ui.material_visible = [True] * len(mats)
    if len(ui.material_cuttable) != len(mats):
        ui.material_cuttable = [m.id == SKIN.id for m in mats.materials]
    ui.material_locked = [False] * len(mats)
    ui.material_colors = [tuple(m.color) for m in mats.materials]
    ui.material_procedural = make_default_procedural_materials(len(mats))
    material_visible_wp = wp.array(
        np.asarray([1 if v else 0 for v in ui.material_visible], dtype=np.int32),
        dtype=wp.int32,
        device=dev,
    )
    material_cuttable_wp = wp.array(
        np.asarray([1 if v else 0 for v in ui.material_cuttable], dtype=np.int32),
        dtype=wp.int32,
        device=dev,
    )
    material_colors_wp = wp.array(mats.color, dtype=wp.vec3, device=dev)
    delete_state.set_material_stiffness_scale(np.asarray(ui.material_stiffness_scale, dtype=np.float32))
    cell_material_host = pg.aux.cell_material.numpy().astype(np.int32, copy=False)
    cell_grid_xyz_host = pg.aux.cell_grid_xyz.numpy().astype(np.int32, copy=False)
    material_locked_nodes = [np.empty(0, dtype=np.int32) for _ in range(len(mats))]

    gravity_on_vec = model.gravity.numpy().copy()
    gravity_off_vec = np.zeros_like(gravity_on_vec)
    _apply_gravity(model, ui.gravity_enabled, gravity_on_vec, gravity_off_vec)

    timer_report_secs = max(0.0, float(args.timer_report_secs))
    timer_stats: dict[str, list[float]] | None = {} if timer_report_secs > 0.0 else None
    timer_gpu_stats: dict[str, list[float]] | None = {} if (timer_report_secs > 0.0 and args.timer_gpu_activities) else None
    timer_gpu_breakdown: dict[str, dict[str, list[float]]] | None = (
        {} if (timer_report_secs > 0.0 and args.timer_gpu_activities) else None
    )
    gpu_timer_scopes = {
        "physics",
        "graph.capture",
        "log_state",
        "surface.update",
        "mc.log_mesh",
        "shape_cluster_overlay",
        "end_frame",
    }
    gpu_cuda_filter = (
        wp.TIMING_KERNEL | wp.TIMING_MEMCPY | wp.TIMING_MEMSET | wp.TIMING_GRAPH
        if args.timer_gpu_activities
        else 0
    )
    active_gpu_timer_scopes = gpu_timer_scopes if (args.timer_sync or args.timer_gpu_activities) else set()
    set_scoped_timer_dict(
        timer_stats,
        gpu_dict=timer_gpu_stats,
        gpu_breakdown=timer_gpu_breakdown,
        sync_names=active_gpu_timer_scopes,
        sync_enabled=bool(args.timer_sync),
        cuda_filter=gpu_cuda_filter,
    )

    def _scoped_timer(name: str):
        return make_scoped_timer(name)

    reported_frame_count = 0
    last_timer_report_time = time.perf_counter()

    def _flush_timer_report(completed_frames: int, force: bool = False) -> None:
        nonlocal last_timer_report_time, reported_frame_count
        if timer_stats is None:
            return

        def _shorten_label(label: str, max_len: int = 56) -> str:
            if len(label) <= max_len:
                return label
            return label[: max_len - 3] + "..."

        now = time.perf_counter()
        window = now - last_timer_report_time
        if not force and window < timer_report_secs:
            return

        frames_in_window = completed_frames - reported_frame_count
        items: list[tuple[float, str, int, float]] = []
        for name, samples in timer_stats.items():
            if not samples:
                continue
            total_ms = float(sum(samples))
            calls = len(samples)
            avg_ms = total_ms / calls
            items.append((total_ms, name, calls, avg_ms))
        if not items and not force:
            return

        items.sort(reverse=True)
        fps = (frames_in_window / window) if window > 0.0 and frames_in_window > 0 else 0.0
        ui.timer_panel_snapshot = build_timer_panel_snapshot(
            timer_stats,
            window_secs=window,
            window_frames=frames_in_window,
            frame_count=completed_frames,
            triangle_count=ui.tri_count,
            gpu_stats=timer_gpu_stats,
            max_rows=14,
        )
        print(f"[timers {window:.1f}s] frames={frames_in_window} fps={fps:.1f}")
        for total_ms, name, calls, avg_ms in items[:14]:
            line = f"  {name:<20} avg={avg_ms:7.3f} ms  calls={calls:4d} total={total_ms:8.1f} ms"
            if timer_gpu_stats is not None and name in timer_gpu_stats and timer_gpu_stats[name]:
                gpu_samples = timer_gpu_stats[name]
                gpu_avg_ms = float(sum(gpu_samples)) / len(gpu_samples)
                line += f"  gpu={gpu_avg_ms:7.3f} ms"
            print(line)
        if len(items) > 14:
            print(f"  ... {len(items) - 14} more timers")

        if timer_gpu_breakdown:
            gpu_scope_items: list[tuple[float, str]] = []
            for scope_name, samples in timer_gpu_stats.items():
                if samples:
                    gpu_scope_items.append((float(sum(samples)), scope_name))
            if gpu_scope_items:
                gpu_scope_items.sort(reverse=True)
                print("[cuda activities]")
                for _, scope_name in gpu_scope_items[:5]:
                    activity_items: list[tuple[float, str, int, float, float]] = []
                    for activity_name, samples in timer_gpu_breakdown.get(scope_name, {}).items():
                        if samples:
                            total_ms = float(sum(samples))
                            calls = len(samples)
                            avg_call_ms = total_ms / calls
                            avg_frame_ms = total_ms / frames_in_window if frames_in_window > 0 else total_ms
                            activity_items.append((total_ms, activity_name, calls, avg_call_ms, avg_frame_ms))
                    if not activity_items:
                        continue
                    activity_items.sort(reverse=True)
                    print(f"  {scope_name:<20}")
                    for _, activity_name, calls, avg_call_ms, avg_frame_ms in activity_items[:4]:
                        label = _shorten_label(activity_name)
                        print(
                            f"    {label:<56} frame={avg_frame_ms:7.3f} ms  calls={calls:4d}  call={avg_call_ms:7.3f} ms"
                        )

        for samples in timer_stats.values():
            samples.clear()
        if timer_gpu_stats is not None:
            for samples in timer_gpu_stats.values():
                samples.clear()
        if timer_gpu_breakdown is not None:
            for activities in timer_gpu_breakdown.values():
                for samples in activities.values():
                    samples.clear()
        reported_frame_count = completed_frames
        last_timer_report_time = now

    with _StartupPhase("state_and_drag_buffers", startup_phases, sync_device=dev):
        state_0 = pg.state
        state_1 = model.state()
        state_reset = model.state()
        state_reset.assign(state_0)
        ground_plane_points_host, ground_plane_indices_host = _make_ground_plane_mesh(
            state_0.particle_q.numpy(),
            ground_height=float(args.ground_height),
            min_margin=max(float(atlas.voxel_size) * 4.0, 1.0e-3),
        )
        ground_plane_points = wp.array(ground_plane_points_host, dtype=wp.vec3, device=dev)
        ground_plane_indices = wp.array(ground_plane_indices_host, dtype=wp.int32, device=dev)

        drag_indices_host = np.full(n_nodes, -1, dtype=np.int32)
        drag_offsets_host = np.zeros((n_nodes, 3), dtype=np.float32)
        drag_indices = wp.array(drag_indices_host, dtype=wp.int32, device=dev)
        drag_offsets = wp.array(drag_offsets_host, dtype=wp.vec3, device=dev)
        instrument_grasp_indices_host = [np.full(n_nodes, -1, dtype=np.int32) for _ in range(_INSTRUMENT_COUNT)]
        instrument_grasp_offsets_host = [
            np.zeros((n_nodes, 3), dtype=np.float32) for _ in range(_INSTRUMENT_COUNT)
        ]
        instrument_grasp_indices = [
            wp.array(indices, dtype=wp.int32, device=dev) for indices in instrument_grasp_indices_host
        ]
        instrument_grasp_offsets = [
            wp.array(offsets, dtype=wp.vec3, device=dev) for offsets in instrument_grasp_offsets_host
        ]
        locked_indices_host = np.full(n_nodes, -1, dtype=np.int32)
        locked_positions_host = np.zeros((n_nodes, 3), dtype=np.float32)
        locked_indices = wp.array(locked_indices_host, dtype=wp.int32, device=dev)
        locked_positions = wp.array(locked_positions_host, dtype=wp.vec3, device=dev)
        locked_slot_by_node: dict[int, int] = {}
        locked_count = 0

    input_devices = []
    instrument_backend = str(args.input_backend)
    instrument_q_host = np.zeros((_INSTRUMENT_COUNT, 3), dtype=np.float32)
    instrument_q_prev_host = np.zeros((_INSTRUMENT_COUNT, 3), dtype=np.float32)
    instrument_quat_host = np.zeros((_INSTRUMENT_COUNT, 4), dtype=np.float32)
    instrument_quat_host[:, 3] = 1.0
    instrument_tool_pos_host = np.zeros((_INSTRUMENT_COUNT,), dtype=np.float32)
    instrument_handle_pos_host = np.zeros((_INSTRUMENT_COUNT,), dtype=np.float32)
    instrument_grip_host = np.zeros((_INSTRUMENT_COUNT,), dtype=np.float32)
    instrument_handle_active_host = np.zeros((_INSTRUMENT_COUNT,), dtype=np.int32)
    instrument_pose_valid_host = np.zeros((_INSTRUMENT_COUNT,), dtype=np.int32)
    instrument_trigger_down_host = np.zeros((_INSTRUMENT_COUNT,), dtype=np.int32)
    instrument_trigger_prev_host = np.zeros((_INSTRUMENT_COUNT,), dtype=np.int32)
    instrument_cut_enabled_host = np.zeros((_INSTRUMENT_COUNT,), dtype=np.int32)
    instrument_diathermy_power_host = np.zeros((_INSTRUMENT_COUNT,), dtype=np.float32)
    instrument_grasp_count_host = np.zeros((_INSTRUMENT_COUNT,), dtype=np.int32)
    instrument_last_grasp_miss_print_time = np.zeros((_INSTRUMENT_COUNT,), dtype=np.float64)
    instrument_q = wp.array(instrument_q_host, dtype=wp.vec3, device=dev)
    instrument_q_prev = wp.array(instrument_q_prev_host, dtype=wp.vec3, device=dev)
    instrument_cut_enabled = wp.array(instrument_cut_enabled_host, dtype=wp.int32, device=dev)
    instrument_diathermy_power = wp.array(instrument_diathermy_power_host, dtype=wp.float32, device=dev)
    instrument_center = _initial_instrument_center(state_0.particle_q.numpy())
    instrument_device_offsets = [
        (float(args.device_x_offsets[0]), 0.0, 0.0),
        (float(args.device_x_offsets[1]), 0.0, 0.0),
    ]
    instrument_profile = MINIMOU_PROFILE if instrument_backend == "minimou" else OPENHAPTICS_PROFILE
    instrument_frame_configs = [
        HapticFrameConfig(
            profile=instrument_profile,
            position_scale=float(args.position_scale),
            center=instrument_center,
            device_offset=instrument_device_offsets[idx],
            up_axis="Z",
        )
        for idx in range(_INSTRUMENT_COUNT)
    ]
    for idx, frame_config in enumerate(instrument_frame_configs):
        world_pose = pose_to_world(InputPose(valid=True), frame_config)
        instrument_q_host[idx] = world_pose.position
        instrument_quat_host[idx] = np.asarray(world_pose.quaternion, dtype=np.float32)
    instrument_device_q_host = instrument_q_host.copy()
    instrument_device_quat_host = instrument_quat_host.copy()
    instrument_q_prev_host[:] = instrument_q_host
    instrument_q.assign(instrument_q_host)
    instrument_q_prev.assign(instrument_q_prev_host)
    viewer = None
    instrument_camera_follow_reference_frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    if instrument_backend != "off":
        try:
            input_devices = _open_instrument_inputs(
                instrument_backend,
                str(args.device_name),
                str(args.left_device_name),
            )
        except HapticUnavailable as exc:
            print(f"Instrument input unavailable: {exc}")
            return 2
        if len(input_devices) != _INSTRUMENT_COUNT:
            for input_device in reversed(input_devices):
                input_device.close()
            print(f"Instrument input unavailable: expected {_INSTRUMENT_COUNT} devices, got {len(input_devices)}")
            return 2
        if instrument_backend == "fallback":
            print("instrument input: static fallback poses")
        elif instrument_backend == "minimou":
            print("instrument input: Follou MiniMou devices connected")
        else:
            print(f'instrument input: OpenHaptics devices connected: "{args.device_name}", "{args.left_device_name}"')

    def _sync_instrument_camera_follow_reference_frame() -> bool:
        nonlocal instrument_camera_follow_reference_frame
        frame = _viewer_camera_frame(viewer)
        if frame is None:
            return False
        instrument_camera_follow_reference_frame = frame
        return True

    def _apply_instrument_camera_follow() -> bool:
        nonlocal instrument_camera_follow_reference_frame
        if not bool(ui.instrument_follow_camera):
            return False
        current_frame = _viewer_camera_frame(viewer)
        if current_frame is None:
            return False
        if instrument_camera_follow_reference_frame is None:
            instrument_camera_follow_reference_frame = current_frame
        reference_frame = instrument_camera_follow_reference_frame
        instrument_q_host[:] = _camera_transform_points_between_frames(
            instrument_device_q_host,
            reference_frame,
            current_frame,
        )
        for idx in range(_INSTRUMENT_COUNT):
            instrument_quat_host[idx] = _camera_transform_quaternion_between_frames(
                instrument_device_quat_host[idx],
                reference_frame,
                current_frame,
            )
        instrument_pose_valid_host[:] = 1
        return True

    def _set_instrument_follow_camera(enabled: bool) -> None:
        nonlocal instrument_camera_follow_reference_frame
        ui.instrument_follow_camera = bool(enabled)
        if ui.instrument_follow_camera:
            _sync_instrument_camera_follow_reference_frame()
            if _apply_instrument_camera_follow():
                instrument_q_prev_host[:] = instrument_q_host
                instrument_q_prev.assign(instrument_q_prev_host)
                instrument_q.assign(instrument_q_host)
            print("[instrument] camera follow ON")
        else:
            instrument_camera_follow_reference_frame = None
            print("[instrument] camera follow off")

    def _poll_instrument_positions(*, initial: bool = False) -> None:
        if not input_devices:
            return
        if not initial:
            instrument_q_prev_host[:] = instrument_q_host
        for idx, input_device in enumerate(input_devices):
            pose = input_device.poll()
            instrument_pose_valid_host[idx] = 1 if pose.valid else 0
            instrument_tool_pos_host[idx] = float(getattr(pose, "tool_pos", 0.0)) if pose.valid else 0.0
            instrument_handle_pos_host[idx] = float(getattr(pose, "handle_pos", 0.0)) if pose.valid else 0.0
            instrument_grip_host[idx] = float(getattr(pose, "grip", 0.0)) if pose.valid else 0.0
            instrument_handle_active_host[idx] = 1 if (pose.valid and bool(getattr(pose, "handle_active", False))) else 0
            trigger_down = False
            if pose.valid:
                world_pose = pose_to_world(pose, instrument_frame_configs[idx])
                instrument_device_q_host[idx] = world_pose.position
                instrument_device_quat_host[idx] = np.asarray(world_pose.quaternion, dtype=np.float32)
                if instrument_backend == "minimou":
                    trigger_down = (
                        bool(getattr(pose, "button1", False))
                        or bool(getattr(pose, "handle_active", False))
                        or float(getattr(pose, "grip", 0.0)) >= 0.5
                    )
                else:
                    trigger_down = bool(getattr(pose, "button1", False))
            instrument_trigger_down_host[idx] = 1 if trigger_down else 0
        instrument_q_host[:] = instrument_device_q_host
        instrument_quat_host[:] = instrument_device_quat_host
        _apply_instrument_camera_follow()
        if initial:
            instrument_q_prev_host[:] = instrument_q_host
        instrument_q_prev.assign(instrument_q_prev_host)
        instrument_q.assign(instrument_q_host)

    _poll_instrument_positions(initial=True)
    # Grasper capture is edge-triggered. Keep the first live frame eligible so
    # devices that boot with an active trigger/tool value can still grasp.
    instrument_trigger_prev_host[:] = 0
    last_tool_pos_print_time = 0.0

    def _print_instrument_tool_positions() -> None:
        nonlocal last_tool_pos_print_time
        if not input_devices:
            return
        now = time.perf_counter()
        if now - last_tool_pos_print_time < 0.25:
            return
        last_tool_pos_print_time = now
        if instrument_backend == "minimou":
            values = " ".join(
                f"device{idx}=tool:{float(instrument_tool_pos_host[idx]):.3f}"
                f"/handle:{float(instrument_handle_pos_host[idx]):.3f}"
                f"/grip:{float(instrument_grip_host[idx]):.2f}"
                f"/active:{int(instrument_handle_active_host[idx])}"
                f"/trigger:{'1' if instrument_trigger_down_host[idx] else '0'}"
                for idx in range(_INSTRUMENT_COUNT)
            )
        else:
            values = " ".join(
                f"device{idx}={float(instrument_tool_pos_host[idx]):.3f}" for idx in range(_INSTRUMENT_COUNT)
            )
        print(f"[instrument] {values}", flush=True)

    instrument_base_colors_host = np.asarray(
        [[1.0, 0.22, 0.08], [0.05, 0.62, 1.0]],
        dtype=np.float32,
    )
    instrument_colors_host = instrument_base_colors_host.copy()
    instrument_colors = wp.array(instrument_colors_host, dtype=wp.vec3, device=dev)
    instrument_radii_host = np.ones((_INSTRUMENT_COUNT,), dtype=np.float32)
    instrument_radii = wp.array(instrument_radii_host, dtype=wp.float32, device=dev)

    def _update_instrument_colors() -> None:
        if instrument_backend == "minimou":
            tool_pos = np.clip(instrument_tool_pos_host.reshape(_INSTRUMENT_COUNT, 1), 0.0, 1.0)
            instrument_colors_host[:] = 1.0 + (instrument_base_colors_host - 1.0) * tool_pos
        else:
            instrument_colors_host[:] = instrument_base_colors_host
        instrument_colors.assign(instrument_colors_host)

    def _instrument_radius() -> float:
        return (
            max(0.0, float(ui.instrument_radius_scale))
            * float(atlas.voxel_size)
        )

    def _instrument_max_correction() -> float:
        scale = max(0.0, float(ui.instrument_max_correction_scale))
        if scale == 0.0:
            return 0.0
        return scale * float(atlas.voxel_size)

    def _instrument_collision_active() -> bool:
        return bool(input_devices and ui.instrument_collision_enabled and _instrument_radius() > 0.0)

    def _instrument_mc_triangle_collision_active() -> bool:
        return bool(_instrument_collision_active() and ui.instrument_collision_use_mc_triangles)

    def _instrument_action_active() -> bool:
        return bool(input_devices and _instrument_radius() > 0.0)

    def _instrument_mode(idx: int) -> str:
        if 0 <= idx < len(ui.instrument_tool_modes):
            return _normalize_instrument_tool_mode(ui.instrument_tool_modes[idx])
        return "diathermy"

    def _instrument_is_grasper(idx: int) -> bool:
        return _instrument_mode(idx) == "grasper"

    def _clear_instrument_grasp(idx: int) -> None:
        if not (0 <= idx < _INSTRUMENT_COUNT):
            return
        if int(instrument_grasp_count_host[idx]) <= 0 and int(ui.instrument_grasp_counts[idx]) <= 0:
            return
        instrument_grasp_indices_host[idx].fill(-1)
        instrument_grasp_offsets_host[idx].fill(0.0)
        instrument_grasp_indices[idx].assign(instrument_grasp_indices_host[idx])
        instrument_grasp_offsets[idx].assign(instrument_grasp_offsets_host[idx])
        instrument_grasp_count_host[idx] = 0
        ui.instrument_grasp_counts[idx] = 0

    def _clear_all_instrument_grasps() -> None:
        for idx in range(_INSTRUMENT_COUNT):
            _clear_instrument_grasp(idx)

    def _capture_instrument_grasp(idx: int) -> None:
        if not _instrument_collision_active():
            _clear_instrument_grasp(idx)
            return
        particle_q_np = state_0.particle_q.numpy()
        selected, offsets = _select_sphere_drag_particles(
            particle_q=particle_q_np,
            particle_flags=model.particle_flags.numpy(),
            particle_inv_mass=model.particle_inv_mass.numpy(),
            particle_radius=model.particle_radius.numpy(),
            sphere_center=instrument_q_host[idx],
            sphere_radius=_instrument_radius(),
        )
        grasp_count = int(selected.shape[0])
        instrument_grasp_indices_host[idx].fill(-1)
        instrument_grasp_offsets_host[idx].fill(0.0)
        if grasp_count > 0:
            instrument_grasp_indices_host[idx][:grasp_count] = selected
            instrument_grasp_offsets_host[idx][:grasp_count] = offsets
        instrument_grasp_indices[idx].assign(instrument_grasp_indices_host[idx])
        instrument_grasp_offsets[idx].assign(instrument_grasp_offsets_host[idx])
        instrument_grasp_count_host[idx] = grasp_count
        ui.instrument_grasp_counts[idx] = grasp_count
        if grasp_count > 0:
            print(f"[instrument {idx + 1}] grasped {grasp_count} particles")
        else:
            now = time.perf_counter()
            if now - float(instrument_last_grasp_miss_print_time[idx]) >= 1.0:
                instrument_last_grasp_miss_print_time[idx] = now
                active_mask = ((model.particle_flags.numpy() & ACTIVE_BIT) != 0) & (model.particle_inv_mass.numpy() > 0.0)
                nearest_gap = float("nan")
                if np.any(active_mask):
                    distances = np.linalg.norm(particle_q_np[active_mask] - instrument_q_host[idx][None, :], axis=1)
                    active_radii = model.particle_radius.numpy()[active_mask]
                    nearest_gap = float(np.min(distances - (active_radii + _instrument_radius())))
                print(
                    f"[instrument {idx + 1}] grasp found 0 particles "
                    f"(trigger={'ON' if instrument_trigger_down_host[idx] else 'off'}, "
                    f"center={instrument_q_host[idx].tolist()}, r={_instrument_radius():.4g}, "
                    f"nearest_gap={nearest_gap:.4g})",
                    flush=True,
                )

    def _sync_instrument_grasps() -> None:
        if not _instrument_collision_active():
            _clear_all_instrument_grasps()
            instrument_trigger_prev_host[:] = instrument_trigger_down_host
            return

        for idx in range(_INSTRUMENT_COUNT):
            trigger_down = bool(instrument_trigger_down_host[idx])
            trigger_went_down = trigger_down and not bool(instrument_trigger_prev_host[idx])
            if not _instrument_is_grasper(idx) or not trigger_down:
                if int(instrument_grasp_count_host[idx]) > 0:
                    _clear_instrument_grasp(idx)
                instrument_trigger_prev_host[idx] = 1 if trigger_down else 0
                continue
            if _should_capture_instrument_grasp(
                is_grasper=True,
                trigger_down=trigger_down,
                trigger_was_down=not trigger_went_down,
                grasp_count=int(instrument_grasp_count_host[idx]),
            ):
                _capture_instrument_grasp(idx)
            instrument_trigger_prev_host[idx] = 1

    def _instrument_grasp_active() -> bool:
        for idx in range(_INSTRUMENT_COUNT):
            if (
                _instrument_is_grasper(idx)
                and bool(instrument_trigger_down_host[idx])
                and int(instrument_grasp_count_host[idx]) > 0
            ):
                return True
        return False

    def _instrument_grasp_target(idx: int, step_idx: int, substeps: int) -> np.ndarray:
        alpha = (float(step_idx) + 1.0) / float(max(1, substeps))
        return (instrument_q_prev_host[idx] + (instrument_q_host[idx] - instrument_q_prev_host[idx]) * alpha).astype(
            np.float32,
            copy=False,
        )

    def _apply_instrument_grasps(state, step_idx: int, substeps: int) -> bool:
        moved = False
        for idx in range(_INSTRUMENT_COUNT):
            grasp_count = int(instrument_grasp_count_host[idx])
            if grasp_count <= 0 or not _instrument_is_grasper(idx) or not bool(instrument_trigger_down_host[idx]):
                continue
            _apply_grab_distance_constraints(
                state,
                model.particle_inv_mass,
                model.particle_flags,
                instrument_grasp_indices[idx],
                instrument_grasp_offsets[idx],
                grasp_count,
                _instrument_grasp_target(idx, step_idx, substeps),
                ui.drag_pull_stiffness,
                dev,
            )
            moved = True
        return moved

    def _configure_solver_instrument_contacts(step_idx: int, substeps: int, mc_triangle_count: int = 0) -> None:
        if not _instrument_collision_active():
            solver.clear_kinematic_sphere_contacts()
            return
        alpha = (float(step_idx) + 1.0) / float(substeps)
        if ui.instrument_collision_use_mc_triangles:
            if mc_triangle_count <= 0:
                solver.clear_kinematic_sphere_contacts()
                return
            solver.set_kinematic_sphere_mc_triangle_contacts(
                instrument_q_prev,
                instrument_q,
                sphere_count=_INSTRUMENT_COUNT,
                sphere_radius=_instrument_radius(),
                interpolation_alpha=alpha,
                relaxation=ui.instrument_collision_relaxation,
                cell_nodes=pg.aux.cell_nodes,
                cell_active=pg.aux.cell_active,
                vertex_pos=instrument_mc_buffers.vertex_pos,
                tri_indices=instrument_mc_buffers.tri_indices,
                triangle_count=int(mc_triangle_count),
                iterations=ui.instrument_contact_iterations,
                max_correction=_instrument_max_correction(),
            )
            return
        solver.set_kinematic_sphere_contacts(
            instrument_q_prev,
            instrument_q,
            sphere_count=_INSTRUMENT_COUNT,
            sphere_radius=_instrument_radius(),
            interpolation_alpha=alpha,
            relaxation=ui.instrument_collision_relaxation,
            iterations=ui.instrument_contact_iterations,
            max_correction=_instrument_max_correction(),
        )

    particle_contact_grid = model.particle_grid
    particle_contact_max_radius = float(model.particle_max_radius)
    use_graph = bool(args.cuda_graph) and dev.is_cuda
    graph = None
    graph_key: tuple | None = None
    graph_invalidation_reason: str | None = None
    graph_capture_count = 0

    def _invalidate_graph_capture(reason: str) -> None:
        nonlocal graph, graph_key, graph_invalidation_reason
        had_capture = graph is not None or graph_key is not None or graph_capture_count > 0
        graph = None
        graph_key = None
        if had_capture:
            graph_invalidation_reason = str(reason)

    def _set_particle_particle_collisions(enabled: bool, force: bool = False) -> None:
        enabled = bool(enabled)
        if not force and enabled == ui.particle_particle_collisions:
            return
        ui.particle_particle_collisions = enabled
        if enabled:
            model.particle_grid = particle_contact_grid
            model.particle_max_radius = particle_contact_max_radius
        else:
            model.particle_grid = None
            model.particle_max_radius = 0.0
        _invalidate_graph_capture("particle-collisions")

    def _active_material_nodes(material_idx: int) -> np.ndarray:
        material_cells = _active_cells_for_material(
            cell_material_host,
            delete_state.cell_active,
            material_idx,
        )
        if material_cells.size == 0:
            return np.empty(0, dtype=np.int32)
        return np.unique(pg.aux.cell_nodes_host[material_cells].reshape(-1)).astype(np.int32, copy=False)

    def _manual_locked_nodes() -> np.ndarray:
        if not locked_slot_by_node:
            return np.empty(0, dtype=np.int32)
        return np.asarray(list(locked_slot_by_node.keys()), dtype=np.int32)

    def _sync_material_locks() -> int:
        assert ui.material_locked is not None
        locked_sources = [
            material_locked_nodes[idx]
            for idx, locked in enumerate(ui.material_locked)
            if locked and material_locked_nodes[idx].size > 0
        ]
        manual_nodes = _manual_locked_nodes()
        if manual_nodes.size > 0:
            locked_sources.append(manual_nodes)
        if locked_sources:
            locked_nodes = np.unique(np.concatenate(locked_sources)).astype(np.int32, copy=False)
        else:
            locked_nodes = np.empty(0, dtype=np.int32)
        total_locked_count = delete_state.set_locked_nodes(locked_nodes)
        _invalidate_graph_capture("class-lock")
        return int(total_locked_count)

    def _refresh_material_locks_after_topology_change() -> None:
        assert ui.material_locked is not None
        if not any(ui.material_locked):
            return
        delete_state.sync_host_mirrors({"cells", "nodes", "mass"})
        for idx, locked in enumerate(ui.material_locked):
            if locked:
                material_locked_nodes[idx] = _active_material_nodes(idx)
        _sync_material_locks()

    def _toggle_material_lock(material_idx: int) -> None:
        assert ui.material_locked is not None
        if not (0 <= material_idx < len(ui.material_locked)):
            return
        material_name = ui.material_names[material_idx] if ui.material_names is not None else str(material_idx)
        if ui.material_locked[material_idx]:
            ui.material_locked[material_idx] = False
            material_locked_nodes[material_idx] = np.empty(0, dtype=np.int32)
            total_locked_count = _sync_material_locks()
            print(
                f"[class unlock] unlocked {material_name} ({material_idx}); "
                f"{total_locked_count} locked nodes remain"
            )
            return

        delete_state.sync_host_mirrors({"cells", "nodes", "mass"})
        nodes = _active_material_nodes(material_idx)
        if nodes.size == 0:
            print(f"[class lock] no active cells for {material_name} ({material_idx})")
            return
        ui.material_locked[material_idx] = True
        material_locked_nodes[material_idx] = nodes
        total_locked_count = _sync_material_locks()
        print(
            f"[class lock] locked {nodes.size} nodes from {material_name} ({material_idx}); "
            f"{total_locked_count} locked nodes total"
        )

    def _apply_slang_environment_to_viewer() -> None:
        if not slang_viewer_requested:
            return
        render_bridge.set_environment_path(ui.slang_environment_map or None)
        render_bridge.set_environment_background_enabled(ui.slang_environment_background)
        render_bridge.set_environment_intensity(ui.slang_environment_intensity)
        render_bridge.set_environment_rotation_degrees(ui.slang_environment_rotation_degrees)
        render_bridge.set_environment_pitch_degrees(ui.slang_environment_pitch_degrees)

    def _apply_material_panel_settings_to_runtime() -> None:
        assert ui.material_visible is not None
        assert ui.material_cuttable is not None
        assert ui.material_locked is not None
        assert ui.material_stiffness_scale is not None

        material_visible_wp.assign(np.asarray([1 if v else 0 for v in ui.material_visible], dtype=np.int32))
        material_cuttable_wp.assign(np.asarray([1 if v else 0 for v in ui.material_cuttable], dtype=np.int32))
        ui.material_dirty = True
        ui.material_visibility_dirty = True

        delete_state.sync_host_mirrors({"cells", "nodes", "mass"})
        for idx, locked in enumerate(ui.material_locked):
            material_locked_nodes[idx] = _active_material_nodes(idx) if locked else np.empty(0, dtype=np.int32)
        total_locked_count = _sync_material_locks()
        print(f"[class settings] applied panel settings; {total_locked_count} locked nodes")
        _apply_slang_environment_to_viewer()

    def _commit_pending_material_shader_edits(*, force: bool = False) -> None:
        if not slang_viewer_requested:
            return
        capturing = _viewer_ui_capturing(viewer)
        if not force and capturing:
            return
        if ui.material_colors_revision_pending:
            assert ui.material_colors is not None
            material_colors_wp.assign(np.asarray(ui.material_colors, dtype=np.float32))
            ui.material_colors_revision += 1
            ui.material_colors_revision_pending = False
        if ui.material_procedural_revision_pending:
            ui.material_procedural_revision += 1
            ui.material_procedural_revision_pending = False
        if ui.material_maker_params_revision_pending:
            ui.material_maker_params_revision += 1
            ui.material_maker_params_revision_pending = False

    def _current_material_maker_parameter_specs() -> tuple[Any, ...]:
        if not slang_viewer_requested or not hasattr(viewer, "material_maker_parameter_specs"):
            return ()
        return tuple(viewer.material_maker_parameter_specs())

    def _current_material_maker_fields() -> tuple[str, ...]:
        if not slang_viewer_requested or not hasattr(viewer, "material_maker_fields"):
            return ()
        return tuple(viewer.material_maker_fields())

    def _sync_material_maker_parameter_specs(material_count: int) -> tuple[Any, ...]:
        specs = _current_material_maker_parameter_specs()
        if specs != ui.material_maker_parameter_specs:
            ui.material_maker_parameter_specs = specs
            existing = ui.material_maker_params or []
            ui.material_maker_params = [
                clamp_material_maker_params(
                    existing[idx] if idx < len(existing) else None,
                    specs,
                )
                for idx in range(material_count)
            ]
            ui.material_maker_params_revision += 1
        elif ui.material_maker_params is None or len(ui.material_maker_params) != material_count:
            existing = ui.material_maker_params or []
            ui.material_maker_params = [
                clamp_material_maker_params(
                    existing[idx] if idx < len(existing) else None,
                    specs,
                )
                for idx in range(material_count)
            ]
        return specs

    def _prepare_slang_material_shader_resources() -> None:
        if not slang_viewer_requested or not hasattr(viewer, "prepare_cryo_material_resources"):
            return
        if ui.material_colors is None:
            return
        if ui.material_procedural is None or len(ui.material_procedural) != len(ui.material_colors):
            ui.material_procedural = make_default_procedural_materials(len(ui.material_colors))
        _sync_material_maker_parameter_specs(len(ui.material_colors))
        viewer.prepare_cryo_material_resources(
            material_colors=ui.material_colors,
            material_colors_revision=ui.material_colors_revision,
            procedural_params=ui.material_procedural,
            procedural_params_revision=ui.material_procedural_revision,
            material_maker_params=ui.material_maker_params,
            material_maker_params_revision=ui.material_maker_params_revision,
        )

    def _active_grab_constraint_specs(
        drag_count: int,
        drag_target: np.ndarray | None,
        step_idx: int,
        substeps: int,
    ) -> list[tuple[wp.array, wp.array, int, np.ndarray, float]]:
        specs: list[tuple[wp.array, wp.array, int, np.ndarray, float]] = []
        if drag_target is not None and drag_count > 0:
            specs.append((drag_indices, drag_offsets, int(drag_count), drag_target, ui.drag_pull_stiffness))
        for idx in range(_INSTRUMENT_COUNT):
            grasp_count = int(instrument_grasp_count_host[idx])
            if grasp_count <= 0 or not _instrument_is_grasper(idx) or not bool(instrument_trigger_down_host[idx]):
                continue
            specs.append(
                (
                    instrument_grasp_indices[idx],
                    instrument_grasp_offsets[idx],
                    grasp_count,
                    _instrument_grasp_target(idx, step_idx, substeps),
                    ui.drag_pull_stiffness,
                )
            )
        return specs

    def _run_substeps(substeps: int, sub_dt: float, drag_count: int, drag_target: np.ndarray | None) -> None:
        odd = (substeps % 2) == 1
        s0 = state_0
        s1 = state_1
        _enforce_locked_nodes(s0, locked_indices, locked_positions, locked_count, dev)
        _enforce_locked_nodes(s1, locked_indices, locked_positions, locked_count, dev)
        try:
            for step_idx in range(substeps):
                solver.set_grab_distance_constraints(
                    _active_grab_constraint_specs(drag_count, drag_target, step_idx, substeps)
                )
                _enforce_locked_nodes(s0, locked_indices, locked_positions, locked_count, dev)
                s0.clear_forces()
                mc_triangle_count = (
                    _refresh_instrument_mc_collision_mesh(s0)
                    if _instrument_mc_triangle_collision_active()
                    else 0
                )
                _configure_solver_instrument_contacts(step_idx, substeps, mc_triangle_count)
                solver.step(s0, s1, None, None, sub_dt)
                _enforce_locked_nodes(s1, locked_indices, locked_positions, locked_count, dev)
                if odd and step_idx == substeps - 1:
                    s0.assign(s1)
                    _enforce_locked_nodes(s0, locked_indices, locked_positions, locked_count, dev)
                else:
                    s0, s1 = s1, s0
        finally:
            solver.clear_grab_distance_constraints()

    def _build_substep_graph(substeps: int, iterations: int, sub_dt: float):
        solver.iterations = int(iterations)
        solver.clear_grab_distance_constraints()
        wp.synchronize_device(dev)
        substeps = int(substeps)
        odd = (substeps % 2) == 1
        with wp.ScopedCapture(dev) as cap:
            s0 = state_0
            s1 = state_1
            _enforce_locked_nodes(s0, locked_indices, locked_positions, locked_count, dev)
            _enforce_locked_nodes(s1, locked_indices, locked_positions, locked_count, dev)
            for step_idx in range(substeps):
                _enforce_locked_nodes(s0, locked_indices, locked_positions, locked_count, dev)
                s0.clear_forces()
                _configure_solver_instrument_contacts(step_idx, substeps)
                solver.step(s0, s1, None, None, sub_dt)
                _enforce_locked_nodes(s1, locked_indices, locked_positions, locked_count, dev)
                if odd and step_idx == substeps - 1:
                    s0.assign(s1)
                    _enforce_locked_nodes(s0, locked_indices, locked_positions, locked_count, dev)
                else:
                    s0, s1 = s1, s0
        return cap.graph

    with _StartupPhase("mc_tables_and_buffers", startup_phases, sync_device=dev):
        tables = upload_mc_tables(device=dev)
        mc_buffers = allocate_mc_buffers(pg.aux.grid_shape, n_cells, device=dev)
        instrument_mc_buffers = allocate_mc_buffers(pg.aux.grid_shape, n_cells, device=dev)
        mc_factor = atlas.voxel_size * 0.5
        bake_vertex_uv3(mc_buffers, pg.aux.cell_grid_xyz, pg.aux.grid_shape, device=dev)
        procedural_uv3_noise_scale = build_procedural_uv3_noise_scale(pg.aux.grid_shape)

    render_aux = SimpleNamespace(
        particle_material=pg.aux.cell_material,
        particle_grid_xyz=pg.aux.cell_grid_xyz,
        grid_to_particle=pg.aux.grid_to_cell,
        grid_shape=pg.aux.grid_shape,
    )
    instrument_mc_collision_topology_revision = -1
    instrument_mc_collision_triangle_count = 0

    def _refresh_instrument_mc_collision_mesh(state) -> int:
        nonlocal instrument_mc_collision_topology_revision, instrument_mc_collision_triangle_count
        update_cell_render_state(pg.aux, state.particle_q, device=dev, compute_stretch=False)
        compute_mc_vertex_positions(
            buffers=instrument_mc_buffers,
            particle_q=pg.aux.cell_center_q,
            particle_orientation=pg.aux.cell_orientation,
            mc_factor=mc_factor,
            device=dev,
        )
        topology_revision = int(delete_state.topology_revision)
        if instrument_mc_collision_topology_revision != topology_revision:
            instrument_mc_collision_triangle_count = compute_mc_topology(
                buffers=instrument_mc_buffers,
                tables=tables,
                particle_flags=pg.aux.cell_render_flags,
                grid_to_particle=pg.aux.grid_to_cell,
                device=dev,
            )
            instrument_mc_collision_topology_revision = topology_revision
        return int(instrument_mc_collision_triangle_count)

    with _StartupPhase("overlays_and_picker", startup_phases, sync_device=dev):
        surface = SurfaceRenderer("hex_shape_matching_surface", n_cells, device=dev, max_triangles=mc_buffers.max_triangles)
        cell_overlay = ColoredParticleOverlay(n_cells, pg.aux.materials, device=dev, name="/model/particles")
        node_overlay = ColoredParticleOverlay(n_nodes, pg.aux.materials, device=dev, name="/hex_grid/nodes")
        mouse_grab_overlay = GrabConstraintOverlay(n_nodes, device=dev, name="/grab/mouse")
        instrument_grab_overlays = [
            GrabConstraintOverlay(n_nodes, device=dev, name=f"/grab/instrument_{idx + 1}")
            for idx in range(_INSTRUMENT_COUNT)
        ]
        l0_shape_overlay = ShapeMatchingClusterOverlay(device=dev, name="/shape_matching/l0_clusters")
        l1_shape_overlay = ShapeMatchingClusterOverlay(device=dev, name="/shape_matching/l1_clusters")
        l2_shape_overlay = ShapeMatchingClusterOverlay(device=dev, name="/shape_matching/l2_clusters")
        mc_vertex_overlay = CryoMeshVertexOverlay(mc_buffers.max_triangles, device=dev)
        hover_picker = HoverPicker(n_cells, device=dev)

    surface_color_by_segmentation = bool(args.color_by_segmentation_map)
    cryo_surface_renderer = str(args.cryo_renderer)
    if surface_color_by_segmentation and cryo_surface_renderer != "off":
        print("surface segmentation colors enabled; disabling cryo surface renderer.")
        cryo_surface_renderer = "off"
    if slang_viewer_requested and cryo_surface_renderer == "atlas":
        print("Slang viewer samples the cryo texture as a 3D volume; ignoring --cryo-renderer atlas.")
        cryo_surface_renderer = "volume"
    if cryo_surface_renderer == "volume" and args.usd is not None:
        print("cryo renderer 'volume' is ViewerGL-only; using atlas for USD output.")
        cryo_surface_renderer = "atlas"

    segmentation_surface_atlas: CryoTextureAtlas | None = None
    segmentation_texture_host: np.ndarray | None = None
    segmentation_texture_3d = None
    segmentation_volume_direct = False
    with _StartupPhase("surface_segmentation_texture", startup_phases, sync_device=dev):
        if surface_color_by_segmentation:
            if args.usd is None and install_cryo_volume_shader_patch():
                segmentation_texture_host, segmentation_texture_3d = build_segmentation_color_texture(
                    atlas.labels,
                    mats.color,
                    device=dev,
                )
                segmentation_volume_direct = True
                print(f"surface colors: segmentation map {segmentation_texture_3d.shape}  renderer: direct GL 3D volume")
            else:
                if args.usd is None:
                    print("surface segmentation volume renderer unavailable; falling back to atlas.")
                segmentation_surface_atlas = CryoTextureAtlas(
                    max_triangles=max(1024, min(mc_buffers.max_triangles, n_cells)),
                    device=dev,
                    tile_size=max(1, args.atlas_tile_size),
                )
                print(
                    "surface colors: segmentation map atlas(init) "
                    f"{segmentation_surface_atlas.atlas_width}x{segmentation_surface_atlas.atlas_height} px "
                    f"(tile {segmentation_surface_atlas.tile_size}, max_tris {segmentation_surface_atlas.max_triangles})"
                )

    stress_surface_atlas: CryoTextureAtlas | None = None

    def _ensure_stress_surface_atlas() -> CryoTextureAtlas:
        nonlocal stress_surface_atlas
        if stress_surface_atlas is None:
            stress_surface_atlas = CryoTextureAtlas(
                max_triangles=max(1024, min(mc_buffers.max_triangles, n_cells)),
                device=dev,
                tile_size=2,
            )
            print(
                "surface colors: cell stretch "
                f"{stress_surface_atlas.atlas_width}x{stress_surface_atlas.atlas_height} px "
                f"(tile {stress_surface_atlas.tile_size}, max_tris {stress_surface_atlas.max_triangles})"
            )
        return stress_surface_atlas

    cryo_atlas: CryoTextureAtlas | None = None
    cryo_texture_host: np.ndarray | None = None
    cryo_texture_3d = None
    cryo_volume_direct = False
    with _StartupPhase("cryo_texture_load", startup_phases, sync_device=dev):
        if cryo_surface_renderer != "off" and _OMNISURG_PREPARED_TEXTURE_RGB is not None:
            cryo_texture_host, cryo_texture_3d = _upload_rgb_texture(_OMNISURG_PREPARED_TEXTURE_RGB, dev)
            if cryo_surface_renderer == "volume":
                if slang_viewer_requested:
                    cryo_volume_direct = True
                    print(f"cryo texture: {cryo_texture_3d.shape}  renderer: Slang 3D volume")
                elif install_cryo_volume_shader_patch():
                    cryo_volume_direct = True
                    print(f"cryo texture: {cryo_texture_3d.shape}  renderer: direct GL 3D volume")
                else:
                    print("cryo renderer 'volume' unavailable with this Newton shader; falling back to atlas.")
                    cryo_surface_renderer = "atlas"
            if cryo_surface_renderer == "atlas":
                cryo_atlas = CryoTextureAtlas(
                    max_triangles=max(1024, min(mc_buffers.max_triangles, n_cells)),
                    device=dev,
                    tile_size=max(1, args.atlas_tile_size),
                )
                print(
                    f"cryo texture: {cryo_texture_3d.shape}  renderer: atlas(init) "
                    f"{cryo_atlas.atlas_width}x{cryo_atlas.atlas_height} px "
                    f"(tile {cryo_atlas.tile_size}, max_tris {cryo_atlas.max_triangles})"
                )
        elif cryo_surface_renderer != "off" and args.cryo_texture:
            cryo_path = Path(args.cryo_texture)
            if cryo_path.exists():
                cryo_texture_host, cryo_texture_3d = _load_cryo_texture_for_crop(
                    cryo_path,
                    visible_class_crop,
                    dev,
                )
                if cryo_surface_renderer == "volume":
                    if slang_viewer_requested:
                        cryo_volume_direct = True
                        print(f"cryo texture: {cryo_texture_3d.shape}  renderer: Slang 3D volume")
                    elif install_cryo_volume_shader_patch():
                        cryo_volume_direct = True
                        print(f"cryo texture: {cryo_texture_3d.shape}  renderer: direct GL 3D volume")
                    else:
                        print("cryo renderer 'volume' unavailable with this Newton shader; falling back to atlas.")
                        cryo_surface_renderer = "atlas"
                if cryo_surface_renderer == "atlas":
                    cryo_atlas = CryoTextureAtlas(
                        max_triangles=max(1024, min(mc_buffers.max_triangles, n_cells)),
                        device=dev,
                        tile_size=max(1, args.atlas_tile_size),
                    )
                    print(
                        f"cryo texture: {cryo_texture_3d.shape}  renderer: atlas(init) "
                        f"{cryo_atlas.atlas_width}x{cryo_atlas.atlas_height} px "
                        f"(tile {cryo_atlas.tile_size}, max_tris {cryo_atlas.max_triangles})"
                    )
            else:
                print(
                    f"cryo texture {cryo_path} not found; run `python tools/build_cryo_texture.py` to create it. "
                    "Rendering untextured."
                )

    _set_particle_particle_collisions(ui.particle_particle_collisions, force=True)

    try:
        from newton._src.viewer.gl import opengl as newton_gl_opengl  # noqa: PLC0415
    except Exception:
        newton_gl_opengl = None

    gl_interop_enabled = (
        bool(args.gl_interop)
        and dev.is_cuda
        and args.usd is None
        and args.viewer == "gl"
        and not slang_viewer_requested
        and newton_gl_opengl is not None
    )
    if newton_gl_opengl is not None:
        newton_gl_opengl.ENABLE_CUDA_INTEROP = gl_interop_enabled
    if gl_interop_enabled:
        print("[gl-interop] enabled for ViewerGL VBO updates")
    elif args.gl_interop and args.usd is None and args.viewer == "gl" and not slang_viewer_requested:
        print("[gl-interop] requested but unavailable (requires CUDA + ViewerGL and Newton GL interop support)")
    elif not args.gl_interop and args.usd is None and dev.is_cuda:
        print("[gl-interop] disabled; ViewerGL mesh buffers will use CPU staging uploads")

    with _StartupPhase("viewer_create", startup_phases):
        if args.usd is not None:
            viewer = newton.viewer.ViewerUSD(args.usd, num_frames=args.frames)
        elif slang_viewer_requested:
            camera_pos = (0.12, -0.18, 0.12) if args.size > 0 else (0.32, 0.04, 0.08)
            try:
                viewer = SlangHexViewer(
                    backend=str(args.viewer),
                    device=dev,
                    camera_pos=camera_pos,
                    vsync=True,
                    procedural_material_path=args.slang_procedural_material or None,
                    procedural_material_scale=ui.slang_procedural_material_scale,
                    procedural_material_hot_reload=bool(args.slang_procedural_hot_reload),
                    environment_path=ui.slang_environment_map or None,
                    environment_intensity=ui.slang_environment_intensity,
                    environment_background=ui.slang_environment_background,
                    environment_rotation_degrees=ui.slang_environment_rotation_degrees,
                    environment_pitch_degrees=ui.slang_environment_pitch_degrees,
                )
            except RuntimeError as exc:
                print(f"Slang viewer unavailable: {exc}", file=sys.stderr)
                return 1
        elif args.viewer == "headless":
            viewer = _HeadlessHexViewer()
        else:
            viewer = newton.viewer.ViewerGL()
            _widen_viewer_left_panel(viewer)

    with _StartupPhase("viewer_set_model", startup_phases, sync_device=dev):
        viewer.set_model(model)
        viewer.show_particles = ui.show_cell_particles
        viewer._log_particles = lambda _state: None  # noqa: SLF001
        if hasattr(viewer, "set_camera"):
            if args.size > 0:
                viewer.set_camera(pos=wp.vec3(0.12, -0.18, 0.12), pitch=-18.0, yaw=30.0)
            else:
                viewer.set_camera(pos=wp.vec3(0.32, 0.04, 0.08), pitch=-15.0, yaw=180.0)
        if hasattr(viewer, "_cam_speed"):
            viewer._cam_speed = viewer._cam_speed / 3.0
        if ui.instrument_follow_camera:
            _sync_instrument_camera_follow_reference_frame()
            _apply_instrument_camera_follow()
            instrument_q_prev_host[:] = instrument_q_host
            instrument_q_prev.assign(instrument_q_prev_host)
            instrument_q.assign(instrument_q_host)

    render_bridge = RenderBridge.wrap_existing(viewer, device=dev)

    mouse_state = {
        "continuous_pick_xy": None,
        "cursor_xy": None,
        "cut_button_down": False,
        "left_ctrl_down": False,
        "left_alt_down": False,
        "left_shift_down": False,
        "plane_cut_start_xy": None,
        "plane_cut_origin": None,
        "plane_cut_direction": None,
        "plane_cut_depth": 0.0,
        "plane_cut_active": False,
        "pending_plane_cut": None,
        "drag_start_xy": None,
        "drag_xy": None,
        "drag_particle": -1,
        "drag_count": 0,
        "drag_plane_origin": None,
        "drag_plane_normal": None,
        "drag_offset": None,
        "pending_lock_dragged": False,
        "pending_unlock_all": False,
    }

    def _cut_modifier_down() -> bool:
        return bool(mouse_state["left_ctrl_down"])

    def _ray_cut_modifier_down() -> bool:
        return bool(mouse_state["left_alt_down"])

    def _plane_cut_modifier_down() -> bool:
        return bool(mouse_state["left_shift_down"])

    def _clear_drag_selection() -> None:
        drag_indices_host.fill(-1)
        drag_offsets_host.fill(0.0)
        drag_indices.assign(drag_indices_host)
        drag_offsets.assign(drag_offsets_host)
        mouse_state["drag_start_xy"] = None
        mouse_state["drag_xy"] = None
        mouse_state["drag_particle"] = -1
        mouse_state["drag_count"] = 0
        mouse_state["drag_plane_origin"] = None
        mouse_state["drag_plane_normal"] = None
        mouse_state["drag_offset"] = None
        ui.drag_particle = -1
        ui.drag_count = 0

    def _lock_dragged_nodes() -> int:
        nonlocal locked_count
        drag_count = int(mouse_state["drag_count"])
        if drag_count <= 0:
            return 0
        selected = np.asarray(drag_indices_host[:drag_count], dtype=np.int32)
        selected = np.unique(selected[selected >= 0])
        selected = selected[selected < n_nodes]
        delete_state.sync_host_mirrors({"nodes"})
        selected = selected[delete_state.node_support_count[selected] > 0]
        if selected.size == 0:
            return 0
        locked_positions_np = state_0.particle_q.numpy()[selected].astype(np.float32, copy=True)
        newly_locked = delete_state.lock_nodes(selected)
        locked_count = _merge_locked_node_positions(
            locked_indices_host,
            locked_positions_host,
            locked_slot_by_node,
            locked_count,
            selected,
            locked_positions_np,
        )
        locked_indices.assign(locked_indices_host)
        locked_positions.assign(locked_positions_host)
        _enforce_locked_nodes(state_0, locked_indices, locked_positions, locked_count, dev)
        _enforce_locked_nodes(state_1, locked_indices, locked_positions, locked_count, dev)
        _clear_drag_selection()
        _invalidate_graph_capture("lock-nodes")
        print(f"[L] locked {selected.size} grabbed nodes ({newly_locked} newly locked, {locked_count} total)")
        return int(selected.size)

    def _unlock_all_nodes() -> int:
        nonlocal locked_count
        unlocked_count = delete_state.unlock_all_nodes()
        if ui.material_locked is not None:
            ui.material_locked[:] = [False] * len(ui.material_locked)
        for idx in range(len(material_locked_nodes)):
            material_locked_nodes[idx] = np.empty(0, dtype=np.int32)
        locked_indices_host.fill(-1)
        locked_positions_host.fill(0.0)
        locked_slot_by_node.clear()
        locked_count = 0
        locked_indices.assign(locked_indices_host)
        locked_positions.assign(locked_positions_host)
        _invalidate_graph_capture("unlock-nodes")
        print(f"[U] unlocked {unlocked_count} nodes")
        return int(unlocked_count)

    if _viewer_supports_mouse_interaction(viewer):
        viewer._paused = True
        viewer.show_ui = True

        try:
            import pyglet
        except ImportError:
            pyglet = None

        if pyglet is not None:
            def _store_drag_xy(x: float, y: float) -> None:
                mouse_state["drag_xy"] = (float(x), float(y))

            def _store_cut_xy(x: float, y: float) -> None:
                mouse_state["cursor_xy"] = (float(x), float(y))
                mouse_state["continuous_pick_xy"] = (float(x), float(y))

            def _store_cursor_xy(x: float, y: float) -> None:
                mouse_state["cursor_xy"] = (float(x), float(y))

            def _cut_modifier_down() -> bool:
                key_handler = getattr(viewer.renderer, "_key_handler", None)
                if key_handler is not None:
                    try:
                        return bool(key_handler[pyglet.window.key.LCTRL])
                    except Exception:
                        pass
                return bool(mouse_state["left_ctrl_down"])

            def _ray_cut_modifier_down() -> bool:
                key_handler = getattr(viewer.renderer, "_key_handler", None)
                if key_handler is not None:
                    try:
                        return bool(key_handler[pyglet.window.key.LALT])
                    except Exception:
                        pass
                return bool(mouse_state["left_alt_down"])

            def _plane_cut_modifier_down() -> bool:
                key_handler = getattr(viewer.renderer, "_key_handler", None)
                if key_handler is not None:
                    try:
                        return bool(key_handler[pyglet.window.key.LSHIFT])
                    except Exception:
                        pass
                return bool(mouse_state["left_shift_down"])

            def _refresh_key_cut_xy() -> None:
                cursor_xy = mouse_state["cursor_xy"]
                if cursor_xy is not None and (_cut_modifier_down() or _ray_cut_modifier_down()):
                    mouse_state["cut_button_down"] = True
                    mouse_state["continuous_pick_xy"] = cursor_xy

            def _on_mouse_motion(x, y, dx, dy):
                if _viewer_ui_capturing(viewer):
                    return
                _store_cursor_xy(x, y)
                _refresh_key_cut_xy()

            def _on_mouse_press(x, y, button, modifiers):
                if _viewer_ui_capturing(viewer):
                    return
                _store_cursor_xy(x, y)
                _refresh_key_cut_xy()
                if button == pyglet.window.mouse.MIDDLE:
                    if _plane_cut_modifier_down() or bool(modifiers & pyglet.window.key.MOD_SHIFT):
                        ray_origin, ray_direction = _mouse_world_ray(render_bridge, x, y)
                        mouse_state["plane_cut_start_xy"] = (float(x), float(y))
                        mouse_state["plane_cut_origin"] = ray_origin
                        mouse_state["plane_cut_direction"] = ray_direction
                        mouse_state["plane_cut_depth"] = max(0.0, float(ui.plane_cut_depth_scale)) * float(atlas.voxel_size)
                        mouse_state["plane_cut_active"] = True
                        return
                    return
                if button == pyglet.window.mouse.RIGHT:
                    _store_drag_xy(x, y)
                    mouse_state["drag_start_xy"] = (float(x), float(y))

            def _on_mouse_release(x, y, button, modifiers):
                if _viewer_ui_capturing(viewer):
                    return
                _store_cursor_xy(x, y)
                _refresh_key_cut_xy()
                if button == pyglet.window.mouse.RIGHT:
                    _clear_drag_selection()
                    return
                if button == pyglet.window.mouse.MIDDLE:
                    if bool(mouse_state["plane_cut_active"]):
                        start_origin = mouse_state["plane_cut_origin"]
                        start_direction = mouse_state["plane_cut_direction"]
                        depth = float(mouse_state["plane_cut_depth"])
                        if start_origin is not None and start_direction is not None and depth > 0.0:
                            end_origin, end_direction = _mouse_world_ray(render_bridge, x, y)
                            mouse_state["pending_plane_cut"] = (
                                start_origin,
                                start_direction,
                                end_origin,
                                end_direction,
                                depth,
                            )
                        mouse_state["plane_cut_start_xy"] = None
                        mouse_state["plane_cut_origin"] = None
                        mouse_state["plane_cut_direction"] = None
                        mouse_state["plane_cut_depth"] = 0.0
                        mouse_state["plane_cut_active"] = False
                        return
                    return

            def _on_mouse_drag(x, y, dx, dy, buttons, modifiers):
                if _viewer_ui_capturing(viewer):
                    return
                _store_cursor_xy(x, y)
                _refresh_key_cut_xy()
                if buttons & pyglet.window.mouse.MIDDLE:
                    if not bool(mouse_state["plane_cut_active"]):
                        return
                if buttons & pyglet.window.mouse.RIGHT:
                    _store_drag_xy(x, y)

            def _on_key_press(symbol, modifiers):
                if symbol == pyglet.window.key.LCTRL:
                    mouse_state["left_ctrl_down"] = True
                    mouse_state["cut_button_down"] = True
                    if mouse_state["cursor_xy"] is not None:
                        _store_cut_xy(*mouse_state["cursor_xy"])
                    return
                if symbol == pyglet.window.key.LALT:
                    mouse_state["left_alt_down"] = True
                    mouse_state["cut_button_down"] = True
                    if mouse_state["cursor_xy"] is not None:
                        _store_cut_xy(*mouse_state["cursor_xy"])
                    return
                if symbol == pyglet.window.key.LSHIFT:
                    mouse_state["left_shift_down"] = True
                    return
                if symbol == pyglet.window.key.R:
                    ui.pending_reset_simulation = True
                    return
                if symbol == pyglet.window.key.G:
                    ui.gravity_enabled = not ui.gravity_enabled
                    _apply_gravity(model, ui.gravity_enabled, gravity_on_vec, gravity_off_vec)
                    print(f"[G] gravity {'ON' if ui.gravity_enabled else 'OFF'}")
                    return
                if symbol == pyglet.window.key.L:
                    mouse_state["pending_lock_dragged"] = True
                    return
                if symbol == pyglet.window.key.U:
                    mouse_state["pending_unlock_all"] = True
                    return
                if symbol == pyglet.window.key.I:
                    _set_instrument_follow_camera(not ui.instrument_follow_camera)
                    return

            def _on_key_release(symbol, modifiers):
                if symbol == pyglet.window.key.LCTRL:
                    mouse_state["left_ctrl_down"] = False
                    if not _ray_cut_modifier_down():
                        mouse_state["cut_button_down"] = False
                        mouse_state["continuous_pick_xy"] = None
                if symbol == pyglet.window.key.LALT:
                    mouse_state["left_alt_down"] = False
                    if not _cut_modifier_down():
                        mouse_state["cut_button_down"] = False
                        mouse_state["continuous_pick_xy"] = None
                if symbol == pyglet.window.key.LSHIFT:
                    mouse_state["left_shift_down"] = False

            render_bridge.set_input_callbacks(_on_key_press, _on_key_release)
            render_bridge.set_mouse_callbacks(
                on_motion=_on_mouse_motion,
                on_press=_on_mouse_press,
                on_drag=_on_mouse_drag,
                on_release=_on_mouse_release,
            )

        def render_ui(imgui):
            imgui.text(scene_label)
            imgui.text(f"Frame:          {ui.frame}")
            imgui.text(f"Active cells:   {ui.active_cells}")
            imgui.text(f"Deleted total:  {ui.deleted_total}")
            imgui.text(f"Triangles:      {ui.tri_count}")
            imgui.text(f"Ground plane z: {args.ground_height:.3f}")
            imgui.text(f"Last pick:      {ui.last_pick_cell if ui.last_pick_cell >= 0 else 'none'}")
            imgui.text(f"Last deleted:   {ui.last_deleted_cell if ui.last_deleted_cell >= 0 else 'none'}")
            imgui.text(f"Ray deleted:    {ui.last_ray_deleted_count}")
            imgui.text(f"Plane deleted:  {ui.last_plane_deleted_count}")
            imgui.text(f"Dragged node:   {ui.drag_particle if ui.drag_particle >= 0 else 'none'}")
            imgui.text(f"Dragged count:  {ui.drag_count}")
            imgui.text(f"Simulation:     {'Paused' if viewer.is_paused() else 'Running'}")
            imgui.separator()
            imgui.text("Space: start / pause")
            imgui.text("Reset: R")
            imgui.text("Delete: hold Left Ctrl")
            imgui.text("Ray delete: hold Left Alt")
            imgui.text("Plane cut: Left Shift + MMB drag, release to cut")
            imgui.text("Drag particle: hold RMB")
            imgui.text("Instrument camera follow: I")
            imgui.separator()

            if imgui.button("Reset simulation (R)"):
                ui.pending_reset_simulation = True

            changed, new_gravity = imgui.checkbox("Gravity (G)", ui.gravity_enabled)
            if changed:
                ui.gravity_enabled = bool(new_gravity)
                _apply_gravity(model, ui.gravity_enabled, gravity_on_vec, gravity_off_vec)

            changed, new_show_particles = imgui.checkbox("Show cell particles", ui.show_cell_particles)
            if changed:
                ui.show_cell_particles = bool(new_show_particles)
                viewer.show_particles = ui.show_cell_particles

            _, ui.viewer_log_state = imgui.checkbox("Viewer log_state", ui.viewer_log_state)
            _, ui.show_mesh = imgui.checkbox("Show MC surface", ui.show_mesh)
            _, ui.show_ground_plane = imgui.checkbox("Show ground plane", ui.show_ground_plane)
            _, ui.show_nodes = imgui.checkbox("Show grid nodes", ui.show_nodes)
            if slang_viewer_requested:
                _, ui.show_timing_panel = imgui.checkbox("Show timing panel", ui.show_timing_panel)
                changed, new_procedural_surface = imgui.checkbox(
                    "Procedural surface",
                    ui.slang_procedural_surface,
                )
                if changed:
                    ui.slang_procedural_surface = bool(new_procedural_surface)
                changed, new_world_space_noise = imgui.checkbox(
                    "World-space noise",
                    ui.slang_procedural_world_space,
                )
                if changed:
                    ui.slang_procedural_world_space = bool(new_world_space_noise)
                if args.slang_procedural_material:
                    imgui.text(f"MM shader: {Path(args.slang_procedural_material).name}")
                    changed, new_mm_scale = imgui.slider_float(
                        "MM coord scale",
                        ui.slang_procedural_material_scale,
                        0.001,
                        100.0,
                    )
                    if changed:
                        ui.slang_procedural_material_scale = max(float(new_mm_scale), 0.000001)
                        if hasattr(viewer, "set_procedural_material_scale"):
                            viewer.set_procedural_material_scale(ui.slang_procedural_material_scale)
                    if imgui.button("Reload MM shader"):
                        if hasattr(viewer, "request_external_material_reload"):
                            viewer.request_external_material_reload(force=True)
                        elif hasattr(viewer, "reload_external_material"):
                            viewer.reload_external_material(force=True, raise_on_error=False)
                    if hasattr(viewer, "external_material_reload_status"):
                        reload_status = viewer.external_material_reload_status()
                        if reload_status:
                            imgui.text(reload_status)
                _, ui.show_lighting_panel = imgui.checkbox("Show lighting panel", ui.show_lighting_panel)
                changed, new_debug_view = _imgui_combo_int(
                    imgui,
                    "Debug view",
                    ui.slang_debug_view,
                    SLANG_SURFACE_DEBUG_VIEW_LABELS,
                )
                if changed:
                    ui.slang_debug_view = new_debug_view
                changed, new_cryo_mix = imgui.slider_float("Cryo mix", ui.slang_cryo_mix, 0.0, 1.0)
                if changed:
                    ui.slang_cryo_mix = float(new_cryo_mix)
                changed, new_state_overlay = imgui.slider_float(
                    "State overlay",
                    ui.slang_state_overlay_strength,
                    0.0,
                    1.0,
                )
                if changed:
                    ui.slang_state_overlay_strength = float(new_state_overlay)

            changed, new_pp_collisions = imgui.checkbox("Particle-particle collisions", ui.particle_particle_collisions)
            if changed:
                _set_particle_particle_collisions(bool(new_pp_collisions))

            _, ui.show_mc_vertex_samples = imgui.checkbox("Show MC cryo vertices (diag)", ui.show_mc_vertex_samples)
            _, ui.cryo_colored_cells = imgui.checkbox("Cryo cell colour", ui.cryo_colored_cells)
            _, ui.stress_colored_surface = imgui.checkbox("Stress colours", ui.stress_colored_surface)
            changed, new_stress_scale = imgui.slider_float(
                "Stress colour multiplier",
                ui.stress_color_scale,
                0.0,
                100.0,
            )
            if changed:
                ui.stress_color_scale = float(new_stress_scale)
            _, ui.smooth_mesh_normals = imgui.checkbox("Smooth MC normals", ui.smooth_mesh_normals)
            _, ui.active_cut_fast_surface = imgui.checkbox("Fast surface while cutting", ui.active_cut_fast_surface)

            changed, new_iters = imgui.slider_int("MC smooth iters", ui.taubin_iterations, 0, 10)
            if changed:
                ui.taubin_iterations = int(new_iters)
            changed, new_cut_iters = imgui.slider_int("Cut smooth iters", ui.active_cut_taubin_iterations, 0, 10)
            if changed:
                ui.active_cut_taubin_iterations = int(new_cut_iters)
            changed, new_lambda = imgui.slider_float("MC lambda", ui.taubin_lambda, 0.0, 1.0)
            if changed:
                ui.taubin_lambda = float(new_lambda)
            changed, new_mu = imgui.slider_float("MC mu", ui.taubin_mu, -1.0, 0.0)
            if changed:
                ui.taubin_mu = float(new_mu)

            changed, new_sx = imgui.slider_float("Cryo scale X", ui.cryo_scale_x, 0.25, 4.0)
            if changed:
                ui.cryo_scale_x = float(new_sx)
            changed, new_sy = imgui.slider_float("Cryo scale Y", ui.cryo_scale_y, 0.25, 4.0)
            if changed:
                ui.cryo_scale_y = float(new_sy)
            changed, new_sz = imgui.slider_float("Cryo scale Z", ui.cryo_scale_z, 0.25, 4.0)
            if changed:
                ui.cryo_scale_z = float(new_sz)

            imgui.separator()
            imgui.text("Solver")
            changed, new_substeps = imgui.slider_int("Substeps", ui.substeps, 1, 32)
            if changed:
                ui.substeps = int(new_substeps)
            changed, new_iterations = imgui.slider_int("Constraint iterations", ui.iterations, 1, 32)
            if changed:
                ui.iterations = int(new_iterations)

            imgui.separator()
            imgui.text("Mouse Drag")
            changed, new_drag_radius = imgui.slider_float("Drag radius (voxels)", ui.drag_radius_scale, 0.0, 12.0)
            if changed:
                ui.drag_radius_scale = float(new_drag_radius)
            changed, new_drag_stiffness = imgui.slider_float(
                "Drag pull stiffness",
                ui.drag_pull_stiffness,
                0.0,
                1.0,
            )
            if changed:
                ui.drag_pull_stiffness = float(new_drag_stiffness)
            if not slang_viewer_requested:
                _, ui.show_grab_constraints = imgui.checkbox("Show grabbed particles", ui.show_grab_constraints)

            imgui.separator()
            imgui.text("Instruments")
            _, ui.show_instruments = imgui.checkbox("Show instrument spheres", ui.show_instruments)
            changed, new_follow_camera = imgui.checkbox(
                "Move instruments with camera",
                ui.instrument_follow_camera,
            )
            if changed:
                _set_instrument_follow_camera(bool(new_follow_camera))
            _, ui.instrument_collision_enabled = imgui.checkbox(
                "Instrument collisions",
                ui.instrument_collision_enabled,
            )
            changed, new_mc_collision = imgui.checkbox(
                "Use MC triangle collisions",
                ui.instrument_collision_use_mc_triangles,
            )
            if changed:
                ui.instrument_collision_use_mc_triangles = bool(new_mc_collision)
                _invalidate_graph_capture("instrument-collision-mode")
            if instrument_backend == "minimou":
                imgui.text(
                    "MiniMou handle: "
                    f"pos {instrument_handle_pos_host[0]:.2f}, {instrument_handle_pos_host[1]:.2f}; "
                    f"grip {instrument_grip_host[0]:.2f}, {instrument_grip_host[1]:.2f}; "
                    f"tool {instrument_tool_pos_host[0]:.2f}, {instrument_tool_pos_host[1]:.2f}"
                )
            imgui.text(
                "Tool triggers: "
                f"{'ON' if instrument_trigger_down_host[0] else 'off'}, "
                f"{'ON' if instrument_trigger_down_host[1] else 'off'}"
            )
            for idx in range(_INSTRUMENT_COUNT):
                current_mode = _instrument_mode(idx)
                current_mode_idx = _INSTRUMENT_TOOL_MODES.index(current_mode)
                changed, new_mode_idx = imgui.combo(
                    f"Device {idx + 1} mode",
                    current_mode_idx,
                    list(_INSTRUMENT_TOOL_MODES),
                )
                if changed:
                    ui.instrument_tool_modes[idx] = _INSTRUMENT_TOOL_MODES[int(new_mode_idx)]
                    _clear_instrument_grasp(idx)
                    instrument_trigger_prev_host[idx] = 0
                imgui.same_line()
                imgui.text(
                    f"trigger: {'ON' if instrument_trigger_down_host[idx] else 'off'}, "
                    f"power: {instrument_diathermy_power_host[idx]:.0f}, "
                    f"held: {ui.instrument_grasp_counts[idx]}"
                )
            _, ui.show_heat_overlay = imgui.checkbox("Show heat overlay", ui.show_heat_overlay)
            changed, new_power = imgui.slider_float("Diathermy power", ui.diathermy_power, 0.0, 2000.0)
            if changed:
                ui.diathermy_power = float(new_power)
            changed, new_diffusion = imgui.slider_float("Heat diffusion", ui.heat_diffusion, 0.0, 5.0)
            if changed:
                ui.heat_diffusion = float(new_diffusion)
            changed, new_cooling = imgui.slider_float("Heat cooling", ui.heat_cooling, 0.0, 5.0)
            if changed:
                ui.heat_cooling = float(new_cooling)
            changed, new_heat_substeps = imgui.slider_int("Heat substeps", ui.heat_substeps, 1, 16)
            if changed:
                ui.heat_substeps = int(new_heat_substeps)
            imgui.text(f"Heat min/max: {ui.heat_min:.2f} / {ui.heat_max:.2f}")
            changed, new_blade_length = imgui.slider_float(
                "Blade length (voxels)",
                ui.blade_length_scale,
                0.0,
                32.0,
            )
            if changed:
                ui.blade_length_scale = float(new_blade_length)
            changed, new_blade_radius = imgui.slider_float(
                "Blade radius (voxels)",
                ui.blade_radius_scale,
                0.0,
                4.0,
            )
            if changed:
                ui.blade_radius_scale = float(new_blade_radius)
            changed, new_instrument_radius = imgui.slider_float(
                "Instrument radius (voxels)",
                ui.instrument_radius_scale,
                0.0,
                16.0,
            )
            if changed:
                ui.instrument_radius_scale = float(new_instrument_radius)
            changed, new_instrument_relaxation = imgui.slider_float(
                "Instrument collision relaxation",
                ui.instrument_collision_relaxation,
                0.0,
                1.0,
            )
            if changed:
                ui.instrument_collision_relaxation = float(new_instrument_relaxation)
            changed, new_instrument_contact_iterations = imgui.slider_int(
                "Instrument contact passes/iter",
                ui.instrument_contact_iterations,
                1,
                8,
            )
            if changed:
                ui.instrument_contact_iterations = int(new_instrument_contact_iterations)
            changed, new_instrument_max_correction = imgui.slider_float(
                "Instrument max push/pass",
                ui.instrument_max_correction_scale,
                0.0,
                4.0,
            )
            if changed:
                ui.instrument_max_correction_scale = float(new_instrument_max_correction)

            imgui.separator()
            imgui.text("Ray Cut")
            changed, new_ray_depth = imgui.slider_float(
                "Alt ray depth (voxels)",
                ui.ray_cut_depth_scale,
                0.5,
                200.0,
            )
            if changed:
                ui.ray_cut_depth_scale = float(new_ray_depth)

            imgui.separator()
            imgui.text("Plane Cut")
            changed, new_plane_depth = imgui.slider_float(
                "Ray depth (voxels)",
                ui.plane_cut_depth_scale,
                0.5,
                200.0,
            )
            if changed:
                ui.plane_cut_depth_scale = float(new_plane_depth)

        render_bridge.register_ui_callback(render_ui, position="side")

        def render_shape_matching_panel(imgui):
            if not viewer.ui.is_available:
                return
            io = viewer.ui.io
            width = 320
            height = 680
            imgui.set_next_window_pos(
                imgui.ImVec2(10, max(10, io.display_size[1] - height - 10)),
                imgui.Cond_.appearing.value,
            )
            imgui.set_next_window_size(
                imgui.ImVec2(width, height),
                imgui.Cond_.appearing.value,
            )
            flags = imgui.WindowFlags_.no_collapse.value
            if not imgui.begin("Shape Matching", flags=flags):
                imgui.end()
                return

            _, ui.enable_shape_matching = imgui.checkbox("Enable shape matching", ui.enable_shape_matching)
            imgui.text(f"L0 mode: {SHAPE_MATCHING_SOLVE_LABELS[int(ui.shape_matching_mode)]}")
            changed, new_l0_mode = imgui.slider_int(
                "L0 mode index",
                int(ui.shape_matching_mode),
                SHAPE_MATCHING_SOLVE_SCATTER,
                SHAPE_MATCHING_SOLVE_COLORED_GS,
            )
            if changed:
                ui.shape_matching_mode = int(new_l0_mode)
            imgui.text(f"GS weighting: {SHAPE_MATCHING_GS_WEIGHT_LABELS[int(ui.shape_matching_gs_weighting)]}")
            changed, new_gs_weighting = imgui.slider_int(
                "GS weighting index",
                int(ui.shape_matching_gs_weighting),
                SHAPE_MATCHING_GS_WEIGHT_AVERAGED,
                SHAPE_MATCHING_GS_WEIGHT_FULL,
            )
            if changed:
                ui.shape_matching_gs_weighting = int(new_gs_weighting)
                ui.shape_matching_gs_support_alpha = -1.0
            gs_alpha = _effective_gs_support_alpha(
                int(ui.shape_matching_gs_weighting),
                float(ui.shape_matching_gs_support_alpha),
            )
            changed, new_gs_alpha = imgui.slider_float("GS support alpha", gs_alpha, 0.0, 1.0)
            if changed:
                ui.shape_matching_gs_support_alpha = float(new_gs_alpha)
            _, ui.sleep_l0_shape_matching = imgui.checkbox("Sleep L0", ui.sleep_l0_shape_matching)
            _, ui.shape_matching_use_computed_prolongation = imgui.checkbox(
                "Computed prolongation",
                ui.shape_matching_use_computed_prolongation,
            )
            _, ui.hierarchical_shape_matching_outer8_absolute_projection = imgui.checkbox(
                "Absolute Outer8 projection",
                ui.hierarchical_shape_matching_outer8_absolute_projection,
            )
            _, ui.enable_volume_preservation = imgui.checkbox("Volume preservation", ui.enable_volume_preservation)
            changed, new_volume_stiffness = imgui.slider_float(
                "Volume stiffness",
                ui.volume_preservation_stiffness,
                0.0,
                1.0,
            )
            if changed:
                ui.volume_preservation_stiffness = float(new_volume_stiffness)
            changed, new_volume_passes = imgui.slider_int(
                "Volume passes",
                ui.volume_preservation_passes,
                0,
                8,
            )
            if changed:
                ui.volume_preservation_passes = int(new_volume_passes)
            imgui.text(f"Sleeping L0 clusters: {solver.sleeping_l0_cluster_count}")

            imgui.separator()
            _, ui.show_l0_shape_clusters = imgui.checkbox("Show L0 clusters", ui.show_l0_shape_clusters)
            _, ui.show_l1_shape_clusters = imgui.checkbox("Show L1 clusters", ui.show_l1_shape_clusters)
            _, ui.show_l2_shape_clusters = imgui.checkbox("Show L2 clusters", ui.show_l2_shape_clusters)

            imgui.separator()
            changed, new_shape = imgui.slider_float("L0 stiffness", ui.shape_matching_stiffness, 0.0, 1.0)
            if changed:
                ui.shape_matching_stiffness = float(new_shape)
            changed, new_relaxation = imgui.slider_float("L0 relaxation", ui.shape_matching_relaxation, 0.0, 8.0)
            if changed:
                ui.shape_matching_relaxation = float(new_relaxation)
            changed, new_l0_passes = imgui.slider_int(
                "L0 passes",
                ui.shape_matching_passes,
                0,
                8,
            )
            if changed:
                ui.shape_matching_passes = int(new_l0_passes)

            imgui.text(f"L1 mode: {HIERARCHICAL_SHAPE_MATCHING_LABELS[int(ui.hierarchical_shape_matching_mode)]}")
            changed, new_hierarchy_mode = imgui.slider_int(
                "L1 mode index",
                int(ui.hierarchical_shape_matching_mode),
                HIERARCHICAL_SHAPE_MATCHING_OFF,
                HIERARCHICAL_SHAPE_MATCHING_FULL27,
            )
            if changed:
                ui.hierarchical_shape_matching_mode = int(new_hierarchy_mode)
            _, ui.hierarchical_shape_matching_use_gs = imgui.checkbox(
                "L1 colored GS",
                ui.hierarchical_shape_matching_use_gs,
            )
            _, ui.hierarchical_shape_matching_outer8_prolongation = imgui.checkbox(
                "L1 Outer8 prolongation",
                ui.hierarchical_shape_matching_outer8_prolongation,
            )
            changed, new_hierarchy_shape = imgui.slider_float(
                "L1 stiffness",
                ui.hierarchical_shape_matching_stiffness,
                0.0,
                1.0,
            )
            if changed:
                ui.hierarchical_shape_matching_stiffness = float(new_hierarchy_shape)
            changed, new_hierarchy_relaxation = imgui.slider_float(
                "L1 relaxation",
                ui.hierarchical_shape_matching_relaxation,
                0.0,
                8.0,
            )
            if changed:
                ui.hierarchical_shape_matching_relaxation = float(new_hierarchy_relaxation)
            changed, new_hierarchy_passes = imgui.slider_int(
                "L1 passes",
                ui.hierarchical_shape_matching_passes,
                0,
                8,
            )
            if changed:
                ui.hierarchical_shape_matching_passes = int(new_hierarchy_passes)

            imgui.separator()
            imgui.text(f"L2 mode: {L2_HIERARCHICAL_SHAPE_MATCHING_LABELS[int(ui.l2_hierarchical_shape_matching_mode)]}")
            changed, new_l2_hierarchy_mode = imgui.slider_int(
                "L2 mode index",
                int(ui.l2_hierarchical_shape_matching_mode),
                HIERARCHICAL_SHAPE_MATCHING_OFF,
                HIERARCHICAL_SHAPE_MATCHING_FULL27,
            )
            if changed:
                ui.l2_hierarchical_shape_matching_mode = int(new_l2_hierarchy_mode)
            _, ui.l2_hierarchical_shape_matching_use_gs = imgui.checkbox(
                "L2 colored GS",
                ui.l2_hierarchical_shape_matching_use_gs,
            )
            _, ui.l2_hierarchical_shape_matching_outer8_prolongation = imgui.checkbox(
                "L2 Outer8 prolongation",
                ui.l2_hierarchical_shape_matching_outer8_prolongation,
            )
            changed, new_l2_hierarchy_shape = imgui.slider_float(
                "L2 stiffness",
                ui.l2_hierarchical_shape_matching_stiffness,
                0.0,
                1.0,
            )
            if changed:
                ui.l2_hierarchical_shape_matching_stiffness = float(new_l2_hierarchy_shape)
            changed, new_l2_hierarchy_relaxation = imgui.slider_float(
                "L2 relaxation",
                ui.l2_hierarchical_shape_matching_relaxation,
                0.0,
                8.0,
            )
            if changed:
                ui.l2_hierarchical_shape_matching_relaxation = float(new_l2_hierarchy_relaxation)
            changed, new_l2_hierarchy_passes = imgui.slider_int(
                "L2 passes",
                ui.l2_hierarchical_shape_matching_passes,
                0,
                8,
            )
            if changed:
                ui.l2_hierarchical_shape_matching_passes = int(new_l2_hierarchy_passes)

            imgui.end()

        render_bridge.register_ui_callback(render_shape_matching_panel, position="free")

        def render_materials_panel(imgui):
            if not viewer.ui.is_available or ui.material_names is None:
                return
            io = viewer.ui.io
            width = min(500, max(300, int(io.display_size[0]) - 20))
            row_height = 82
            shader_detail_height = 250 if slang_viewer_requested else 0
            height = max(230, 92 + shader_detail_height + len(ui.material_names) * row_height)
            height = min(height, io.display_size[1] - 20)
            imgui.set_next_window_pos(
                imgui.ImVec2(io.display_size[0] - width - 10, 10),
                imgui.Cond_.appearing.value,
            )
            imgui.set_next_window_size(
                imgui.ImVec2(width, height),
                imgui.Cond_.appearing.value,
            )
            flags = imgui.WindowFlags_.no_collapse.value
            if not imgui.begin("Tissue Classes", flags=flags):
                imgui.end()
                return

            imgui.text("Per-material stiffness, MC visibility, cutting")
            imgui.separator()
            if imgui.button("Remove outer layer"):
                ui.pending_peel_outer_layer = True
            imgui.same_line()
            if imgui.button("Delete outside L1"):
                ui.pending_delete_outside_l1_clusters = True
            imgui.same_line()
            if imgui.button("Delete outside L2"):
                ui.pending_delete_outside_l2_clusters = True
            settings_path = Path(args.segmentation_panel_settings)
            if imgui.button("Save"):
                try:
                    _save_segmentation_panel_settings(settings_path, ui)
                    ui.material_settings_status = f"Saved {settings_path}"
                    print(f"[class settings] saved {settings_path}")
                except Exception as exc:
                    ui.material_settings_status = f"Save failed: {exc}"
                    print(f"[class settings] save failed: {exc}")
            imgui.same_line()
            if imgui.button("Load"):
                try:
                    applied = _load_segmentation_panel_settings(settings_path, ui)
                    _apply_material_panel_settings_to_runtime()
                    ui.material_settings_status = f"Loaded {applied} classes from {settings_path}"
                    print(f"[class settings] loaded {applied} classes from {settings_path}")
                except Exception as exc:
                    ui.material_settings_status = f"Load failed: {exc}"
                    print(f"[class settings] load failed: {exc}")
            imgui.separator()

            assert ui.material_stiffness_scale is not None
            assert ui.material_visible is not None
            assert ui.material_cuttable is not None
            assert ui.material_locked is not None
            assert ui.material_colors is not None
            if ui.material_procedural is None or len(ui.material_procedural) != len(ui.material_names):
                ui.material_procedural = make_default_procedural_materials(len(ui.material_names))
            class_indices = list(_segmentation_panel_class_indices(ui.material_names))
            all_visible = all(ui.material_visible[idx] for idx in class_indices)
            visibility_label = "Hide All Classes" if all_visible else "Show All Classes"
            if imgui.button(visibility_label):
                new_visible = not all_visible
                for idx in class_indices:
                    ui.material_visible[idx] = new_visible
                ui.material_visibility_dirty = True
            if ui.material_settings_status:
                imgui.text(ui.material_settings_status)
            imgui.separator()

            shader_edit_idx = None
            if slang_viewer_requested and class_indices:
                min_class_idx = int(class_indices[0])
                max_class_idx = int(class_indices[-1])
                selected_idx = int(np.clip(ui.material_shader_edit_index, min_class_idx, max_class_idx))
                if selected_idx not in class_indices:
                    selected_idx = min(class_indices, key=lambda candidate: abs(candidate - selected_idx))
                changed, new_selected_idx = imgui.slider_int(
                    "Shader params class",
                    selected_idx,
                    min_class_idx,
                    max_class_idx,
                )
                if changed:
                    selected_idx = int(np.clip(new_selected_idx, min_class_idx, max_class_idx))
                    if selected_idx not in class_indices:
                        selected_idx = min(class_indices, key=lambda candidate: abs(candidate - selected_idx))
                ui.material_shader_edit_index = selected_idx
                shader_edit_idx = selected_idx
                imgui.text(f"Shader params: {ui.material_names[shader_edit_idx]}")
                if hasattr(imgui, "color_edit3"):
                    changed, new_color = imgui.color_edit3("shader color", *ui.material_colors[shader_edit_idx])
                    if changed:
                        ui.material_colors[shader_edit_idx] = _clamp_rgb_tuple(
                            new_color,
                            ui.material_colors[shader_edit_idx],
                        )
                        ui.material_colors_revision_pending = True
                params = clamp_procedural_material_params(ui.material_procedural[shader_edit_idx])
                mm_fields = _current_material_maker_fields()
                using_mm_material = bool(args.slang_procedural_material)
                mm_has_normal = "normal" in mm_fields
                for param_name in PROCEDURAL_MATERIAL_PARAM_NAMES:
                    if using_mm_material and param_name == "roughness":
                        continue
                    if using_mm_material and mm_has_normal and param_name == "normal":
                        continue
                    lo, hi = (1.0, 500.0) if param_name == "scale" else (0.0, 1.0)
                    changed, new_param = imgui.slider_float(param_name, params[param_name], lo, hi)
                    if changed and float(new_param) != params[param_name]:
                        params[param_name] = float(new_param)
                        ui.material_procedural[shader_edit_idx] = clamp_procedural_material_params(params)
                        ui.material_procedural_revision_pending = True
                mm_specs = _sync_material_maker_parameter_specs(len(ui.material_names))
                if mm_specs:
                    assert ui.material_maker_params is not None
                    mm_params = clamp_material_maker_params(ui.material_maker_params[shader_edit_idx], mm_specs)
                    imgui.text("Material Maker")
                    for group_name, group_label in (
                        ("material", "Material controls"),
                        ("procedural", "Procedural controls"),
                    ):
                        group_specs = [spec for spec in mm_specs if spec.group == group_name]
                        if not group_specs:
                            continue
                        imgui.text(group_label)
                        for spec in group_specs:
                            current_value = mm_params.get(spec.raw_name)
                            row = material_maker_parameter_row(spec, current_value)
                            if spec.value_type == "vec4":
                                labels = ("R", "G", "B", "A") if "color" in spec.raw_name else ("X", "Y", "Z", "W")
                                new_row = list(row)
                                changed_any = False
                                for component_idx, component_label in enumerate(labels):
                                    changed, new_component = imgui.slider_float(
                                        f"{spec.label} {component_label}",
                                        row[component_idx],
                                        spec.min_value,
                                        spec.max_value,
                                    )
                                    if changed:
                                        new_row[component_idx] = float(new_component)
                                        changed_any = True
                                if changed_any:
                                    mm_params[spec.raw_name] = material_maker_setting_value(spec, tuple(new_row))
                                    ui.material_maker_params[shader_edit_idx] = clamp_material_maker_params(mm_params, mm_specs)
                                    ui.material_maker_params_revision_pending = True
                            else:
                                changed, new_param = imgui.slider_float(
                                    spec.label,
                                    row[0],
                                    spec.min_value,
                                    spec.max_value,
                                )
                                if changed and float(new_param) != row[0]:
                                    mm_params[spec.raw_name] = float(new_param)
                                    ui.material_maker_params[shader_edit_idx] = clamp_material_maker_params(mm_params, mm_specs)
                                    ui.material_maker_params_revision_pending = True
                imgui.separator()

            for idx, name in enumerate(ui.material_names):
                if idx == 0:
                    continue
                imgui.push_id(idx)
                r, g, b = ui.material_colors[idx]
                prefix = "> " if idx == shader_edit_idx else "  "
                imgui.text_colored(imgui.ImVec4(r, g, b, 1.0), f"{prefix}{name}")
                changed, new_visible = imgui.checkbox("Visible", ui.material_visible[idx])
                if changed:
                    ui.material_visible[idx] = bool(new_visible)
                    ui.material_visibility_dirty = True
                changed, new_cuttable = imgui.checkbox("Cuttable", ui.material_cuttable[idx])
                if changed:
                    ui.material_cuttable[idx] = bool(new_cuttable)
                    material_cuttable_wp.assign(
                        np.asarray([1 if v else 0 for v in ui.material_cuttable], dtype=np.int32)
                    )
                imgui.same_line()
                if imgui.button("Delete voxels"):
                    ui.pending_delete_material = idx
                imgui.same_line()
                lock_label = "Unlock" if ui.material_locked[idx] else "Lock"
                if imgui.button(lock_label):
                    ui.pending_toggle_material_lock = idx
                _, new_scale = imgui.slider_float("stiffness", ui.material_stiffness_scale[idx], 0.0, 5.0)
                if new_scale != ui.material_stiffness_scale[idx]:
                    ui.material_stiffness_scale[idx] = float(new_scale)
                    ui.material_dirty = True
                imgui.separator()
                imgui.pop_id()

            imgui.end()

        render_bridge.register_ui_callback(render_materials_panel, position="free")

        if slang_viewer_requested:

            def render_lighting_panel(imgui):
                if not viewer.ui.is_available or not ui.show_lighting_panel:
                    return
                io = viewer.ui.io
                width = min(360, max(260, int(io.display_size[0]) - 20))
                height = min(380, max(260, int(io.display_size[1]) - 20))
                imgui.set_next_window_pos(
                    imgui.ImVec2(min(450, max(10, io.display_size[0] - width - 10)), 10),
                    imgui.Cond_.appearing.value,
                )
                imgui.set_next_window_size(
                    imgui.ImVec2(width, height),
                    imgui.Cond_.appearing.value,
                )
                flags = imgui.WindowFlags_.no_collapse.value
                if not imgui.begin("Lighting / HDRI", flags=flags):
                    imgui.end()
                    return

                hdri_paths = _hdri_map_choice_paths(environment_map_folder, ui.slang_environment_map)
                hdri_labels = _hdri_map_choice_labels(hdri_paths)
                current_hdri_idx = _hdri_map_choice_index(hdri_paths, ui.slang_environment_map)
                changed, new_hdri_idx = _imgui_combo_index(imgui, "HDRI map", current_hdri_idx, hdri_labels)
                if changed:
                    selected_hdri_path = hdri_paths[new_hdri_idx]
                    ui.slang_environment_map = "" if selected_hdri_path is None else str(selected_hdri_path)
                    _apply_slang_environment_to_viewer()
                imgui.separator()

                changed, new_surface_lighting = imgui.checkbox("Surface lighting", ui.slang_surface_lighting)
                if changed:
                    ui.slang_surface_lighting = bool(new_surface_lighting)
                changed, new_key_light = imgui.checkbox("Key light", ui.slang_key_light)
                if changed:
                    ui.slang_key_light = bool(new_key_light)
                changed, new_fill_light = imgui.checkbox("Fill light", ui.slang_fill_light)
                if changed:
                    ui.slang_fill_light = bool(new_fill_light)
                changed, new_ambient_light = imgui.checkbox("Ambient light", ui.slang_ambient_light)
                if changed:
                    ui.slang_ambient_light = bool(new_ambient_light)
                changed, new_environment_lighting = imgui.checkbox("HDRI lighting", ui.slang_environment_lighting)
                if changed:
                    ui.slang_environment_lighting = bool(new_environment_lighting)
                changed, new_environment_background = imgui.checkbox("HDRI background", ui.slang_environment_background)
                if changed:
                    ui.slang_environment_background = bool(new_environment_background)
                    render_bridge.set_environment_background_enabled(ui.slang_environment_background)
                changed, new_environment_intensity = imgui.slider_float(
                    "HDRI intensity",
                    ui.slang_environment_intensity,
                    0.0,
                    5.0,
                )
                if changed:
                    ui.slang_environment_intensity = max(float(new_environment_intensity), 0.0)
                    render_bridge.set_environment_intensity(ui.slang_environment_intensity)
                changed, new_environment_rotation = imgui.slider_float(
                    "HDRI rotation",
                    ui.slang_environment_rotation_degrees,
                    -180.0,
                    180.0,
                )
                if changed:
                    ui.slang_environment_rotation_degrees = float(new_environment_rotation)
                    render_bridge.set_environment_rotation_degrees(ui.slang_environment_rotation_degrees)
                changed, new_environment_pitch = imgui.slider_float(
                    "HDRI pitch",
                    ui.slang_environment_pitch_degrees,
                    -90.0,
                    90.0,
                )
                if changed:
                    ui.slang_environment_pitch_degrees = float(new_environment_pitch)
                    render_bridge.set_environment_pitch_degrees(ui.slang_environment_pitch_degrees)

                imgui.end()

            render_bridge.register_ui_callback(render_lighting_panel, position="free")

            def render_timing_panel(imgui):
                if not viewer.ui.is_available or not ui.show_timing_panel:
                    return
                io = viewer.ui.io
                width = min(380, max(260, int(io.display_size[0]) - 20))
                height = min(430, max(220, int(io.display_size[1]) - 20))
                imgui.set_next_window_pos(
                    imgui.ImVec2(max(10, io.display_size[0] - width - 10), max(10, io.display_size[1] - height - 10)),
                    imgui.Cond_.appearing.value,
                )
                imgui.set_next_window_size(
                    imgui.ImVec2(width, height),
                    imgui.Cond_.appearing.value,
                )
                flags = imgui.WindowFlags_.no_collapse.value
                if not imgui.begin("Timing", flags=flags):
                    imgui.end()
                    return

                snapshot = ui.timer_panel_snapshot
                if snapshot is None:
                    imgui.text("Timing disabled" if timer_stats is None else "collecting...")
                    imgui.end()
                    return

                imgui.text(f"FPS ({snapshot.window_secs:.1f}s): {snapshot.fps:.1f}")
                imgui.text(f"Frame: {snapshot.frame_count}")
                imgui.text(f"Window frames: {snapshot.window_frames}")
                imgui.text(f"Triangles: {snapshot.triangle_count}")
                imgui.separator()
                if not snapshot.rows:
                    imgui.text("No timer samples")
                for row in snapshot.rows:
                    name = row.name if len(row.name) <= 24 else row.name[:21] + "..."
                    line = f"{name:<24} {row.avg_ms:7.3f} ms"
                    if row.gpu_avg_ms is not None:
                        line += f"  gpu {row.gpu_avg_ms:7.3f}"
                    imgui.text(line)
                imgui.end()

            render_bridge.register_ui_callback(render_timing_panel, position="free")

    print(
        f"{scene_label}: nodes={n_nodes}, cells={n_cells}, "
        f"clusters={clusters.num_clusters}, "
        f"l1_outer8={0 if hierarchy.outer8 is None else hierarchy.outer8.num_clusters}, "
        f"l1_full27={0 if hierarchy.full27 is None else hierarchy.full27.num_clusters}, "
        f"l2_outer8={0 if hierarchy.l2_outer8 is None else hierarchy.l2_outer8.num_clusters}, "
        f"l2_full125={0 if hierarchy.l2_full125 is None else hierarchy.l2_full125.num_clusters}, "
        f"ground_z={args.ground_height:.3f}, "
        f"l0_mode={SHAPE_MATCHING_SOLVE_LABELS[int(ui.shape_matching_mode)]}, "
        f"gs_weight={SHAPE_MATCHING_GS_WEIGHT_LABELS[int(ui.shape_matching_gs_weighting)]}, "
        f"gs_alpha={_effective_gs_support_alpha(int(ui.shape_matching_gs_weighting), float(ui.shape_matching_gs_support_alpha)):g}, "
        f"prolongation={'computed' if ui.shape_matching_use_computed_prolongation else 'table'}, "
        f"l0_relax={ui.shape_matching_relaxation:g}, "
        f"l0_passes={ui.shape_matching_passes}, "
        f"l1_relax={ui.hierarchical_shape_matching_relaxation:g}, "
        f"l2_relax={ui.l2_hierarchical_shape_matching_relaxation:g}, "
        f"volume={'ON' if ui.enable_volume_preservation else 'OFF'} "
        f"(k={ui.volume_preservation_stiffness:g}, passes={ui.volume_preservation_passes}), "
        f"sleep_l0={solver.sleeping_l0_cluster_count if ui.sleep_l0_shape_matching else 0}, "
        f"l1={HIERARCHICAL_SHAPE_MATCHING_LABELS[int(ui.hierarchical_shape_matching_mode)]}"
        f"/{'GS' if ui.hierarchical_shape_matching_use_gs else 'Jacobi'}"
        f"/prolong={'ON' if ui.hierarchical_shape_matching_outer8_prolongation else 'OFF'}"
        f"/abs={'ON' if ui.hierarchical_shape_matching_outer8_absolute_projection else 'OFF'}, "
        f"l2={L2_HIERARCHICAL_SHAPE_MATCHING_LABELS[int(ui.l2_hierarchical_shape_matching_mode)]}"
        f"/{'GS' if ui.l2_hierarchical_shape_matching_use_gs else 'Jacobi'}"
        f"/prolong={'ON' if ui.l2_hierarchical_shape_matching_outer8_prolongation else 'OFF'}"
        f"/abs={'ON' if ui.hierarchical_shape_matching_outer8_absolute_projection else 'OFF'}, "
        f"surface_color={'stress' if ui.stress_colored_surface else ('segmentation' if surface_color_by_segmentation else cryo_surface_renderer)}, "
        f"instruments={instrument_backend}"
        f"/{'camera-follow' if ui.instrument_follow_camera else 'device'}"
        f"/{'collide' if _instrument_collision_active() else 'no-collide'} "
        f"(modes={','.join(_instrument_mode(idx) for idx in range(_INSTRUMENT_COUNT))}, "
        f"r={_instrument_radius():g}, iters={ui.instrument_contact_iterations}, "
        f"max_push={_instrument_max_correction():g}), "
        f"gravity {'ON' if ui.gravity_enabled else 'OFF'}"
    )

    frame_dt = 1.0 / args.fps
    particle_pick_radius = max(float(atlas.voxel_size) * 0.4, float(model.particle_radius.numpy().max()) * 4.0)

    _print_startup_report(startup_phases, time.perf_counter() - startup_t0)

    if args.exit_after_init:
        render_bridge.close()
        for input_device in reversed(input_devices):
            input_device.close()
        return 0

    t_start = time.time()
    completed_frames = 0
    frame = 0
    last_cut_material_visibility_revision = -1

    last_async_delete_result: DeviceDeletionResult | None = None
    picker_aabbs_valid = False

    def _sync_delete_ui_from_host() -> None:
        with _scoped_timer("cut.ui_sync"):
            delete_state.sync_host_mirrors({"stats", "last-delete"})
        if delete_state.last_deleted_cells_host.size > 0:
            ui.last_deleted_cell = int(delete_state.last_deleted_cells_host[-1])
        ui.active_cells = n_cells - int(delete_state.deleted_total)
        ui.deleted_total = int(delete_state.deleted_total)

    def _commit_cell_deletion(deleted_now: int) -> None:
        nonlocal last_async_delete_result, last_cut_material_visibility_revision
        if deleted_now <= 0:
            return
        last_async_delete_result = delete_state.last_deletion_result
        if delete_state.last_deletion_result is not None:
            graph_shape_changed = solver.update_l0_sleep_after_deletion_device(
                delete_state.last_deletion_result.deleted_cells_device,
                delete_state.last_deletion_result.deleted_count_device,
                delete_state.last_deletion_result.candidate_capacity,
                sync_host=True,
            )
        else:
            graph_shape_changed = solver.update_l0_sleep_after_deletion(delete_state.last_deleted_cells_host)
        if graph_shape_changed:
            _invalidate_graph_capture("cutting-l0-fast-uniform8-transition")
        last_cut_material_visibility_revision = int(ui.material_visibility_revision)
        if args.cut_debug_validate:
            delete_state.validate_device_state()
        _sync_delete_ui_from_host()
        _refresh_material_locks_after_topology_change()

    def _delete_cells_outside_hierarchy_coverage(level_label: str, hierarchy_clusters) -> None:
        if hierarchy_clusters is None:
            print(f"[hierarchy prune] {level_label} coverage unavailable; no cells deleted")
            return
        delete_state.sync_host_mirrors({"cells"})
        outside_cells = _active_cells_outside_cluster_coverage(
            delete_state.cell_active,
            hierarchy_clusters.cell_to_cluster_host,
        )
        deleted_now = delete_state.delete_cells(outside_cells)
        if deleted_now > 0:
            _commit_cell_deletion(deleted_now)
        print(f"[hierarchy prune] deleted {deleted_now} cells outside {level_label} coverage")

    def _commit_cell_deletion_async(
        result: DeviceDeletionResult | None,
        *,
        sync_for_ui: bool,
    ) -> int | None:
        nonlocal last_async_delete_result, last_cut_material_visibility_revision
        if result is None or result.deleted_cells_device is None:
            return 0
        last_async_delete_result = result
        graph_shape_changed = solver.update_l0_sleep_after_deletion_device(
            result.deleted_cells_device,
            result.deleted_count_device,
            result.candidate_capacity,
            sync_host=False,
        )
        if graph_shape_changed:
            _invalidate_graph_capture("cutting-l0-fast-uniform8-transition")
        last_cut_material_visibility_revision = int(ui.material_visibility_revision)
        if args.cut_debug_validate:
            delete_state.validate_device_state()
        if not sync_for_ui:
            return None
        with _scoped_timer("cut.ui_sync"):
            deleted_now = delete_state.sync_last_deleted()
            delete_state.sync_host_mirrors({"stats"})
        if deleted_now > 0:
            if delete_state.last_deleted_cells_host.size > 0:
                ui.last_deleted_cell = int(delete_state.last_deleted_cells_host[-1])
            ui.active_cells = n_cells - int(delete_state.deleted_total)
            ui.deleted_total = int(delete_state.deleted_total)
            _refresh_material_locks_after_topology_change()
        return int(deleted_now)

    def _instrument_blade_segment(idx: int) -> tuple[np.ndarray, np.ndarray]:
        p0 = instrument_q_host[idx].astype(np.float32, copy=True)
        direction = quat_rotate(instrument_quat_host[idx], (0.0, 0.0, 1.0)).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-8:
            direction = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
        else:
            direction = direction / norm
        length = max(0.0, float(ui.blade_length_scale)) * float(atlas.voxel_size)
        return p0, (p0 + direction * length).astype(np.float32, copy=False)

    def _blade_radius() -> float:
        return max(0.0, float(ui.blade_radius_scale)) * float(atlas.voxel_size)

    def _diathermy_power_fraction(idx: int) -> float:
        if not (0 <= idx < _INSTRUMENT_COUNT):
            return 0.0
        if _instrument_mode(idx) != "diathermy":
            return 0.0
        if instrument_backend == "minimou":
            if int(instrument_pose_valid_host[idx]) == 0:
                return 0.0
            return float(np.clip(float(instrument_grip_host[idx]), 0.0, 1.0))
        return 1.0 if bool(instrument_trigger_down_host[idx]) else 0.0

    def _apply_instrument_tool_actions() -> None:
        instrument_radius = _instrument_radius()
        for idx in range(_INSTRUMENT_COUNT):
            power = max(0.0, float(ui.diathermy_power)) * _diathermy_power_fraction(idx)
            instrument_diathermy_power_host[idx] = power
            instrument_cut_enabled_host[idx] = (
                1
                if power > 0.0 and instrument_radius > 0.0
                else 0
            )

        if input_devices:
            instrument_cut_enabled.assign(instrument_cut_enabled_host)
            instrument_diathermy_power.assign(instrument_diathermy_power_host)
            with _scoped_timer("instrument_diathermy"):
                result = heat_state.step_diathermy_async(
                    delete_state,
                    state_0.particle_q,
                    instrument_q,
                    instrument_cut_enabled,
                    _INSTRUMENT_COUNT,
                    instrument_radius,
                    dt=frame_dt,
                    power=ui.diathermy_power,
                    diffusion=ui.heat_diffusion,
                    cooling=ui.heat_cooling,
                    substeps=ui.heat_substeps,
                    sphere_power=instrument_diathermy_power,
                    material_cuttable=material_cuttable_wp,
                )
                _commit_cell_deletion_async(result, sync_for_ui=False)

        if not _instrument_action_active():
            return

        blade_radius = _blade_radius()
        if blade_radius > 0.0 and ui.blade_length_scale > 0.0:
            for idx in range(_INSTRUMENT_COUNT):
                if not bool(instrument_trigger_down_host[idx]) or _instrument_mode(idx) != "scissors":
                    continue
                p0, p1 = _instrument_blade_segment(idx)
                with _scoped_timer("instrument_scissors_cut"):
                    result = heat_state.delete_cells_by_capsule_async(
                        delete_state,
                        state_0.particle_q,
                        p0,
                        p1,
                        blade_radius,
                        material_cuttable=material_cuttable_wp,
                    )
                    _commit_cell_deletion_async(result, sync_for_ui=False)

        bipolar_active = any(
            _instrument_mode(idx) == "bipolar" and bool(instrument_trigger_down_host[idx])
            for idx in range(_INSTRUMENT_COUNT)
        )
        if bipolar_active and blade_radius > 0.0:
            with _scoped_timer("instrument_bipolar_cut"):
                result = heat_state.delete_cells_by_capsule_async(
                    delete_state,
                    state_0.particle_q,
                    instrument_q_host[0],
                    instrument_q_host[1],
                    blade_radius,
                    material_cuttable=material_cuttable_wp,
                )
                _commit_cell_deletion_async(result, sync_for_ui=False)

    def _reset_simulation() -> None:
        nonlocal completed_frames, frame, last_async_delete_result, last_cut_material_visibility_revision
        nonlocal last_timer_report_time, locked_count, picker_aabbs_valid, reported_frame_count, t_start
        nonlocal instrument_mc_collision_topology_revision, instrument_mc_collision_triangle_count

        if ui.material_dirty:
            assert ui.material_stiffness_scale is not None
            delete_state.set_material_stiffness_scale(np.asarray(ui.material_stiffness_scale, dtype=np.float32))
            ui.material_dirty = False
        if ui.material_visibility_dirty:
            assert ui.material_visible is not None
            material_visible_wp.assign(
                np.asarray([1 if v else 0 for v in ui.material_visible], dtype=np.int32)
            )
            ui.material_visibility_revision += 1
            ui.material_visibility_dirty = False

        state_0.assign(state_reset)
        state_1.assign(state_reset)
        delete_state.reset()
        heat_state.reset()
        solver.reset_runtime_state()
        _apply_gravity(model, ui.gravity_enabled, gravity_on_vec, gravity_off_vec)

        locked_indices_host.fill(-1)
        locked_positions_host.fill(0.0)
        locked_indices.assign(locked_indices_host)
        locked_positions.assign(locked_positions_host)
        locked_slot_by_node.clear()
        locked_count = 0
        _clear_drag_selection()
        _clear_all_instrument_grasps()
        instrument_trigger_prev_host[:] = instrument_trigger_down_host

        mouse_state["continuous_pick_xy"] = None
        mouse_state["cursor_xy"] = None
        mouse_state["cut_button_down"] = False
        mouse_state["left_ctrl_down"] = False
        mouse_state["left_alt_down"] = False
        mouse_state["left_shift_down"] = False
        mouse_state["plane_cut_start_xy"] = None
        mouse_state["plane_cut_origin"] = None
        mouse_state["plane_cut_direction"] = None
        mouse_state["plane_cut_depth"] = 0.0
        mouse_state["plane_cut_active"] = False
        mouse_state["pending_plane_cut"] = None
        mouse_state["pending_lock_dragged"] = False
        mouse_state["pending_unlock_all"] = False

        ui.pending_reset_simulation = False
        ui.frame = 0
        ui.active_cells = n_cells - int(delete_state.deleted_total)
        ui.deleted_total = int(delete_state.deleted_total)
        ui.last_pick_cell = -1
        ui.last_deleted_cell = -1
        ui.last_ray_deleted_count = 0
        ui.last_plane_deleted_count = 0
        ui.drag_particle = -1
        ui.drag_count = 0
        ui.pending_toggle_material_lock = -1
        if ui.material_locked is not None:
            ui.material_locked[:] = [False] * len(ui.material_locked)
        for idx in range(len(material_locked_nodes)):
            material_locked_nodes[idx] = np.empty(0, dtype=np.int32)

        last_async_delete_result = None
        last_cut_material_visibility_revision = -1
        instrument_mc_collision_topology_revision = -1
        instrument_mc_collision_triangle_count = 0
        picker_aabbs_valid = False
        completed_frames = 0
        reported_frame_count = 0
        frame = 0
        t_start = time.time()
        last_timer_report_time = time.perf_counter()
        _invalidate_graph_capture("simulation-reset")
        print("[reset] simulation reset; current UI params kept")

    while args.frames is None or frame < args.frames:
        with _scoped_timer("frame"):
            picker_aabbs_wanted_after_physics = False
            physics_advanced = False
            active_cut_surface_fast = False
            if _viewer_supports_mouse_interaction(viewer):
                viewer.show_ui = True

            if ui.pending_reset_simulation:
                _reset_simulation()

            _poll_instrument_positions()
            _print_instrument_tool_positions()
            _sync_instrument_grasps()
            _apply_instrument_tool_actions()

            drag_start_xy = mouse_state["drag_start_xy"]
            if drag_start_xy is not None and _viewer_supports_mouse_interaction(viewer):
                ray_origin, ray_direction = _mouse_world_ray(render_bridge, drag_start_xy[0], drag_start_xy[1])
                particle_q_np = state_0.particle_q.numpy()
                particle, hit_point = _pick_particle_from_ray(
                    particle_q=particle_q_np,
                    particle_flags=model.particle_flags.numpy(),
                    particle_inv_mass=model.particle_inv_mass.numpy(),
                    particle_radius=model.particle_radius.numpy(),
                    ray_origin=ray_origin,
                    ray_direction=ray_direction,
                    base_pick_radius=particle_pick_radius,
                )
                if particle >= 0 and hit_point is not None:
                    particle_pos = particle_q_np[particle].astype(np.float32)
                    selected, offsets = _select_drag_particles(
                        particle_q=particle_q_np,
                        particle_flags=model.particle_flags.numpy(),
                        particle_inv_mass=model.particle_inv_mass.numpy(),
                        seed_particle=int(particle),
                        radius=max(0.0, float(ui.drag_radius_scale)) * float(atlas.voxel_size),
                    )
                    drag_count = int(selected.shape[0])
                    drag_indices_host.fill(-1)
                    drag_offsets_host.fill(0.0)
                    if drag_count > 0:
                        drag_indices_host[:drag_count] = selected
                        drag_offsets_host[:drag_count] = offsets
                    drag_indices.assign(drag_indices_host)
                    drag_offsets.assign(drag_offsets_host)
                    mouse_state["drag_particle"] = int(particle)
                    mouse_state["drag_count"] = drag_count
                    mouse_state["drag_plane_origin"] = particle_pos
                    mouse_state["drag_plane_normal"] = ray_direction.astype(np.float32)
                    mouse_state["drag_offset"] = particle_pos - hit_point
                    ui.drag_particle = int(particle)
                    ui.drag_count = drag_count
                else:
                    mouse_state["drag_particle"] = -1
                    mouse_state["drag_count"] = 0
                    mouse_state["drag_plane_origin"] = None
                    mouse_state["drag_plane_normal"] = None
                    mouse_state["drag_offset"] = None
                    ui.drag_particle = -1
                    ui.drag_count = 0
                mouse_state["drag_start_xy"] = None

            drag_target = None
            drag_count = int(mouse_state["drag_count"])
            drag_xy = mouse_state["drag_xy"]
            if drag_count > 0 and _viewer_supports_mouse_interaction(viewer):
                if drag_xy is not None:
                    ray_origin, ray_direction = _mouse_world_ray(render_bridge, drag_xy[0], drag_xy[1])
                    plane_hit = _intersect_ray_plane(
                        ray_origin=ray_origin,
                        ray_direction=ray_direction,
                        plane_origin=mouse_state["drag_plane_origin"],
                        plane_normal=mouse_state["drag_plane_normal"],
                    )
                    if plane_hit is not None:
                        drag_target = plane_hit + mouse_state["drag_offset"]
                elif mouse_state["drag_plane_origin"] is not None:
                    drag_target = np.asarray(mouse_state["drag_plane_origin"], dtype=np.float32)

            if bool(mouse_state["pending_unlock_all"]):
                mouse_state["pending_unlock_all"] = False
                _unlock_all_nodes()

            if bool(mouse_state["pending_lock_dragged"]):
                mouse_state["pending_lock_dragged"] = False
                if drag_target is not None and viewer.is_paused():
                    _apply_grab_distance_constraints(
                        state_0,
                        model.particle_inv_mass,
                        model.particle_flags,
                        drag_indices,
                        drag_offsets,
                        drag_count,
                        drag_target,
                        ui.drag_pull_stiffness,
                        dev,
                    )
                    picker_aabbs_valid = False
                locked_now = _lock_dragged_nodes()
                if locked_now > 0:
                    drag_count = 0
                    drag_target = None
                else:
                    print("[L] no grabbed nodes to lock")

            if drag_target is not None and viewer.is_paused():
                _apply_grab_distance_constraints(
                    state_0,
                    model.particle_inv_mass,
                    model.particle_flags,
                    drag_indices,
                    drag_offsets,
                    drag_count,
                    drag_target,
                    ui.drag_pull_stiffness,
                    dev,
                )
                picker_aabbs_valid = False

            if viewer.is_paused() and _apply_instrument_grasps(state_0, 0, 1):
                picker_aabbs_valid = False

            pending_plane_cut = mouse_state["pending_plane_cut"]
            if pending_plane_cut is not None:
                deleted_now = 0
                with _scoped_timer("plane_cut_delete"):
                    ray0_origin, ray0_direction, ray1_origin, ray1_direction, depth = pending_plane_cut
                    result = delete_state.delete_cells_by_ray_surface_async(
                        state_0.particle_q,
                        ray0_origin=ray0_origin,
                        ray0_direction=ray0_direction,
                        ray1_origin=ray1_origin,
                        ray1_direction=ray1_direction,
                        depth=float(depth),
                        material_cuttable=material_cuttable_wp,
                    )
                    committed = _commit_cell_deletion_async(result, sync_for_ui=True)
                    deleted_now = 0 if committed is None else int(committed)
                ui.last_plane_deleted_count = int(deleted_now)
                if deleted_now > 0:
                    print(f"[plane] deleted {deleted_now} cells")
                mouse_state["pending_plane_cut"] = None

            if _viewer_supports_mouse_interaction(viewer):
                key_cut_active = (
                    (_cut_modifier_down() or _ray_cut_modifier_down())
                    and not _viewer_ui_capturing(viewer)
                )
                cursor_xy = mouse_state["cursor_xy"]
                if key_cut_active and cursor_xy is not None:
                    mouse_state["cut_button_down"] = True
                    mouse_state["continuous_pick_xy"] = cursor_xy
                elif not key_cut_active:
                    mouse_state["cut_button_down"] = False
                    mouse_state["continuous_pick_xy"] = None

            cut_xy = None
            ray_cut = (
                _viewer_supports_mouse_interaction(viewer)
                and bool(mouse_state["cut_button_down"])
                and _ray_cut_modifier_down()
                and mouse_state["continuous_pick_xy"] is not None
            )
            continuous_cut = (
                _viewer_supports_mouse_interaction(viewer)
                and bool(mouse_state["cut_button_down"])
                and not ray_cut
                and _cut_modifier_down()
                and mouse_state["continuous_pick_xy"] is not None
            )
            if ray_cut:
                cut_xy = mouse_state["continuous_pick_xy"]
            elif continuous_cut:
                cut_xy = mouse_state["continuous_pick_xy"]
            picker_aabbs_wanted_after_physics = bool(continuous_cut or ray_cut)
            active_cut_surface_fast = bool(ui.active_cut_fast_surface and (continuous_cut or ray_cut))
            if cut_xy is not None and _viewer_supports_mouse_interaction(viewer):
                picked_cell = -1
                deleted_now = 0
                timer_name = (
                    "ray_cut_pick_delete"
                    if ray_cut
                    else "continuous_cut_pick_delete"
                    if continuous_cut
                    else "click_pick_delete"
                )
                with _scoped_timer(timer_name):
                    with _scoped_timer("cut.ray_build"):
                        ray_origin_np, ray_direction_np = _mouse_world_ray(render_bridge, cut_xy[0], cut_xy[1])
                        ray_origin = (float(ray_origin_np[0]), float(ray_origin_np[1]), float(ray_origin_np[2]))
                        ray_direction = (float(ray_direction_np[0]), float(ray_direction_np[1]), float(ray_direction_np[2]))
                    if not picker_aabbs_valid:
                        with _scoped_timer("pick.refresh_aabbs"):
                            hover_picker.refresh_aabbs(pg.aux, state_0.particle_q)
                        picker_aabbs_valid = True
                    with _scoped_timer("pick.device"):
                        hit_cell_device = hover_picker.pick_device(
                            pg.aux,
                            ray_origin,
                            ray_direction,
                            material_cuttable=material_cuttable_wp,
                        )
                    with _scoped_timer("cut.delete_launch"):
                        if ray_cut:
                            depth = max(0.0, float(ui.ray_cut_depth_scale)) * float(atlas.voxel_size)
                            result = delete_state.delete_cells_by_ray_segment_from_cell_async(
                                state_0.particle_q,
                                hit_cell_device,
                                ray_direction=ray_direction,
                                depth=depth,
                                material_cuttable=material_cuttable_wp,
                            )
                        else:
                            result = delete_state.delete_device_cells_async(hit_cell_device, 1)
                    sync_cut_ui = not continuous_cut and not ray_cut
                    with _scoped_timer("cut.commit"):
                        committed = _commit_cell_deletion_async(result, sync_for_ui=sync_cut_ui)
                    if committed is not None:
                        deleted_now = int(committed)
                    if sync_cut_ui:
                        picked_cell_raw = int(hit_cell_device.numpy()[0])
                        if 0 <= picked_cell_raw < n_cells:
                            picked_cell = picked_cell_raw
                if sync_cut_ui:
                    ui.last_pick_cell = int(picked_cell)
                if ray_cut and sync_cut_ui:
                    ui.last_ray_deleted_count = int(deleted_now)
                if deleted_now > 0:
                    if ray_cut:
                        print(
                            f"[ray] deleted {deleted_now} cells from {picked_cell} "
                            f"(depth {ui.ray_cut_depth_scale:g} voxels)"
                        )
                    elif sync_cut_ui:
                        ui.last_deleted_cell = int(picked_cell)
                        print(f"[{'continuous' if continuous_cut else 'click'}] deleted cell {picked_cell}")
            if ui.pending_delete_material >= 0:
                material_idx = int(ui.pending_delete_material)
                ui.pending_delete_material = -1
                if 0 <= material_idx < len(mats):
                    delete_state.sync_host_mirrors({"cells"})
                    material_cells = _active_cells_for_material(
                        cell_material_host,
                        delete_state.cell_active,
                        material_idx,
                    )
                    deleted_now = delete_state.delete_cells(material_cells)
                    if deleted_now > 0:
                        _commit_cell_deletion(deleted_now)
                        material_name = ui.material_names[material_idx] if ui.material_names is not None else str(material_idx)
                        print(f"[class delete] deleted {deleted_now} cells from {material_name} ({material_idx})")
                    else:
                        material_name = ui.material_names[material_idx] if ui.material_names is not None else str(material_idx)
                        print(f"[class delete] no active cells for {material_name} ({material_idx})")

            if ui.pending_toggle_material_lock >= 0:
                material_idx = int(ui.pending_toggle_material_lock)
                ui.pending_toggle_material_lock = -1
                _toggle_material_lock(material_idx)

            if ui.pending_peel_outer_layer:
                ui.pending_peel_outer_layer = False
                delete_state.sync_host_mirrors({"cells"})
                outer_cells = _outer_layer_cells(
                    cell_grid_xyz_host,
                    delete_state.cell_active,
                    pg.aux.grid_shape,
                )
                deleted_now = delete_state.delete_cells(outer_cells)
                if deleted_now > 0:
                    _commit_cell_deletion(deleted_now)
                    print(f"[peel] deleted outer layer: {deleted_now} cells")
                else:
                    print("[peel] no active cells remaining")

            if ui.pending_delete_outside_l1_clusters:
                ui.pending_delete_outside_l1_clusters = False
                _delete_cells_outside_hierarchy_coverage("L1", hierarchy.outer8)

            if ui.pending_delete_outside_l2_clusters:
                ui.pending_delete_outside_l2_clusters = False
                _delete_cells_outside_hierarchy_coverage("L2", hierarchy.l2_outer8)

            if ui.material_dirty:
                assert ui.material_stiffness_scale is not None
                delete_state.set_material_stiffness_scale(
                    np.asarray(ui.material_stiffness_scale, dtype=np.float32)
                )
                ui.material_dirty = False

            if ui.material_visibility_dirty:
                assert ui.material_visible is not None
                material_visible_wp.assign(
                    np.asarray([1 if v else 0 for v in ui.material_visible], dtype=np.int32)
                )
                ui.material_visibility_revision += 1
                ui.material_visibility_dirty = False

            _commit_pending_material_shader_edits()
            _prepare_slang_material_shader_resources()

            solver.iterations = int(ui.iterations)
            solver.enable_shape_matching = bool(ui.enable_shape_matching)
            solver.enable_self_collisions = bool(ui.particle_particle_collisions)
            l0_sleep_gate_changed = (
                (float(ui.shape_matching_stiffness) > 0.0) != (float(solver.shape_matching_stiffness) > 0.0)
                or (int(ui.shape_matching_passes) > 0) != (int(solver.shape_matching_passes) > 0)
            )
            sleep_state_changed = bool(ui.sleep_l0_shape_matching) != bool(solver.sleep_l0_shape_matching)
            hierarchy_sleep_state_changed = (
                int(ui.hierarchical_shape_matching_mode) != int(solver.hierarchical_shape_matching_mode)
                or int(ui.l2_hierarchical_shape_matching_mode) != int(solver.l2_hierarchical_shape_matching_mode)
                or float(ui.hierarchical_shape_matching_stiffness) != float(solver.hierarchical_shape_matching_stiffness)
                or float(ui.hierarchical_shape_matching_relaxation) != float(solver.hierarchical_shape_matching_relaxation)
                or int(ui.hierarchical_shape_matching_passes) != int(solver.hierarchical_shape_matching_passes)
                or bool(ui.hierarchical_shape_matching_outer8_prolongation)
                != bool(solver.hierarchical_shape_matching_outer8_prolongation)
                or float(ui.l2_hierarchical_shape_matching_stiffness) != float(solver.l2_hierarchical_shape_matching_stiffness)
                or float(ui.l2_hierarchical_shape_matching_relaxation) != float(solver.l2_hierarchical_shape_matching_relaxation)
                or int(ui.l2_hierarchical_shape_matching_passes) != int(solver.l2_hierarchical_shape_matching_passes)
                or bool(ui.l2_hierarchical_shape_matching_outer8_prolongation)
                != bool(solver.l2_hierarchical_shape_matching_outer8_prolongation)
            )
            outer8_projection_mode_changed = (
                bool(ui.hierarchical_shape_matching_outer8_absolute_projection)
                != bool(solver.hierarchical_shape_matching_outer8_absolute_projection)
                or bool(ui.hierarchical_shape_matching_outer8_absolute_projection)
                != bool(solver.l2_hierarchical_shape_matching_outer8_absolute_projection)
            )
            solver.shape_matching_stiffness = float(ui.shape_matching_stiffness)
            solver.shape_matching_relaxation = float(ui.shape_matching_relaxation)
            solver.shape_matching_passes = int(ui.shape_matching_passes)
            solver.shape_matching_mode = int(ui.shape_matching_mode)
            solver.shape_matching_gs_weighting = int(ui.shape_matching_gs_weighting)
            solver.shape_matching_gs_support_alpha = float(ui.shape_matching_gs_support_alpha)
            solver.shape_matching_use_computed_prolongation = bool(ui.shape_matching_use_computed_prolongation)
            solver.enable_volume_preservation = bool(ui.enable_volume_preservation)
            solver.volume_preservation_stiffness = float(ui.volume_preservation_stiffness)
            solver.volume_preservation_passes = int(ui.volume_preservation_passes)
            solver.sleep_l0_shape_matching = bool(ui.sleep_l0_shape_matching)
            solver.hierarchical_shape_matching_mode = int(ui.hierarchical_shape_matching_mode)
            solver.hierarchical_shape_matching_stiffness = float(ui.hierarchical_shape_matching_stiffness)
            solver.hierarchical_shape_matching_relaxation = float(ui.hierarchical_shape_matching_relaxation)
            solver.hierarchical_shape_matching_passes = int(ui.hierarchical_shape_matching_passes)
            solver.hierarchical_shape_matching_use_gs = bool(ui.hierarchical_shape_matching_use_gs)
            solver.hierarchical_shape_matching_outer8_prolongation = bool(ui.hierarchical_shape_matching_outer8_prolongation)
            solver.hierarchical_shape_matching_outer8_absolute_projection = bool(
                ui.hierarchical_shape_matching_outer8_absolute_projection
            )
            solver.l2_hierarchical_shape_matching_mode = int(ui.l2_hierarchical_shape_matching_mode)
            solver.l2_hierarchical_shape_matching_stiffness = float(ui.l2_hierarchical_shape_matching_stiffness)
            solver.l2_hierarchical_shape_matching_relaxation = float(ui.l2_hierarchical_shape_matching_relaxation)
            solver.l2_hierarchical_shape_matching_passes = int(ui.l2_hierarchical_shape_matching_passes)
            solver.l2_hierarchical_shape_matching_use_gs = bool(ui.l2_hierarchical_shape_matching_use_gs)
            solver.l2_hierarchical_shape_matching_outer8_prolongation = bool(
                ui.l2_hierarchical_shape_matching_outer8_prolongation
            )
            solver.l2_hierarchical_shape_matching_outer8_absolute_projection = bool(
                ui.hierarchical_shape_matching_outer8_absolute_projection
            )
            if sleep_state_changed or hierarchy_sleep_state_changed or l0_sleep_gate_changed:
                solver.refresh_l0_sleep_state()
                _invalidate_graph_capture("sleep-or-hierarchy-settings")
            if outer8_projection_mode_changed:
                _invalidate_graph_capture("outer8-projection-mode")

            substeps = max(1, int(ui.substeps))
            sub_dt = frame_dt / substeps
            graph_capture_key = (
                substeps,
                int(ui.iterations),
                sub_dt,
                bool(ui.enable_shape_matching),
                int(ui.shape_matching_mode),
                int(ui.shape_matching_gs_weighting),
                float(ui.shape_matching_gs_support_alpha),
                bool(ui.shape_matching_use_computed_prolongation),
                bool(ui.enable_volume_preservation),
                float(ui.volume_preservation_stiffness),
                int(ui.volume_preservation_passes),
                bool(ui.sleep_l0_shape_matching),
                bool(ui.particle_particle_collisions),
                float(ui.shape_matching_stiffness),
                float(ui.shape_matching_relaxation),
                int(ui.shape_matching_passes),
                int(ui.hierarchical_shape_matching_mode),
                float(ui.hierarchical_shape_matching_stiffness),
                float(ui.hierarchical_shape_matching_relaxation),
                int(ui.hierarchical_shape_matching_passes),
                bool(ui.hierarchical_shape_matching_use_gs),
                bool(ui.hierarchical_shape_matching_outer8_prolongation),
                bool(ui.hierarchical_shape_matching_outer8_absolute_projection),
                int(ui.l2_hierarchical_shape_matching_mode),
                float(ui.l2_hierarchical_shape_matching_stiffness),
                float(ui.l2_hierarchical_shape_matching_relaxation),
                int(ui.l2_hierarchical_shape_matching_passes),
                bool(ui.l2_hierarchical_shape_matching_use_gs),
                bool(ui.l2_hierarchical_shape_matching_outer8_prolongation),
                int(locked_count),
                bool(_instrument_collision_active()),
                bool(ui.instrument_collision_use_mc_triangles),
                float(ui.instrument_radius_scale),
                float(ui.instrument_collision_relaxation),
                int(ui.instrument_contact_iterations),
                float(ui.instrument_max_correction_scale),
            )
            drag_active = drag_count > 0 or _instrument_grasp_active()

            with _scoped_timer("physics"):
                if not viewer.is_paused():
                    physics_advanced = True
                    if use_graph and not drag_active and not _instrument_mc_triangle_collision_active():
                        if graph_capture_key != graph_key:
                            capture_reason = (
                                graph_invalidation_reason
                                or ("initial" if graph_capture_count == 0 else "settings-key-change")
                            )
                            with _scoped_timer("graph.capture"):
                                graph = _build_substep_graph(substeps, int(ui.iterations), sub_dt)
                            graph_key = graph_capture_key
                            graph_capture_count += 1
                            graph_invalidation_reason = None
                            print(
                                "[graph] captured "
                                f"count={graph_capture_count} "
                                f"reason={capture_reason} "
                                f"substeps={substeps} iterations={ui.iterations} "
                                f"shape={'ON' if ui.enable_shape_matching else 'OFF'} "
                                f"l0_mode={SHAPE_MATCHING_SOLVE_LABELS[int(ui.shape_matching_mode)]} "
                                f"gs_weight={SHAPE_MATCHING_GS_WEIGHT_LABELS[int(ui.shape_matching_gs_weighting)]} "
                                f"gs_alpha={_effective_gs_support_alpha(int(ui.shape_matching_gs_weighting), float(ui.shape_matching_gs_support_alpha)):g} "
                                f"prolongation={'computed' if ui.shape_matching_use_computed_prolongation else 'table'} "
                                f"l0_relax={ui.shape_matching_relaxation:g} "
                                f"l0_passes={ui.shape_matching_passes} "
                                f"l1_relax={ui.hierarchical_shape_matching_relaxation:g} "
                                f"l2_relax={ui.l2_hierarchical_shape_matching_relaxation:g} "
                                f"volume={'ON' if ui.enable_volume_preservation else 'OFF'} "
                                f"volume_k={ui.volume_preservation_stiffness:g} "
                                f"volume_passes={ui.volume_preservation_passes} "
                                f"sleep_l0={solver.sleeping_l0_cluster_count if ui.sleep_l0_shape_matching else 0} "
                                f"l1={HIERARCHICAL_SHAPE_MATCHING_LABELS[int(ui.hierarchical_shape_matching_mode)]}"
                                f"/{'GS' if ui.hierarchical_shape_matching_use_gs else 'Jacobi'}"
                                f"/prolong={'ON' if ui.hierarchical_shape_matching_outer8_prolongation else 'OFF'}"
                                f"/abs={'ON' if ui.hierarchical_shape_matching_outer8_absolute_projection else 'OFF'} "
                                f"l2={L2_HIERARCHICAL_SHAPE_MATCHING_LABELS[int(ui.l2_hierarchical_shape_matching_mode)]}"
                                f"/{'GS' if ui.l2_hierarchical_shape_matching_use_gs else 'Jacobi'}"
                                f"/prolong={'ON' if ui.l2_hierarchical_shape_matching_outer8_prolongation else 'OFF'}"
                                f"/abs={'ON' if ui.hierarchical_shape_matching_outer8_absolute_projection else 'OFF'} "
                                f"particle_particle={'ON' if ui.particle_particle_collisions else 'OFF'} "
                                f"instruments={'MC' if _instrument_mc_triangle_collision_active() else ('ON' if _instrument_collision_active() else 'OFF')} "
                                f"locked={locked_count}"
                            )
                        assert graph is not None
                        wp.capture_launch(graph)
                    else:
                        _run_substeps(substeps, sub_dt, drag_count, drag_target)
            _enforce_locked_nodes(state_0, locked_indices, locked_positions, locked_count, dev)
            with _scoped_timer("cell_render_state"):
                if picker_aabbs_wanted_after_physics:
                    with _scoped_timer("pick.refresh_aabbs"):
                        hover_picker.refresh_render_state_and_aabbs(
                            pg.aux,
                            state_0.particle_q,
                            compute_stretch=ui.stress_colored_surface,
                        )
                    picker_aabbs_valid = True
                else:
                    update_cell_render_state(
                        pg.aux,
                        state_0.particle_q,
                        device=dev,
                        compute_stretch=ui.stress_colored_surface,
                    )
                    if physics_advanced:
                        picker_aabbs_valid = False
            render_bridge.begin_frame(frame * frame_dt)
            with _scoped_timer("log_state"):
                if ui.viewer_log_state:
                    viewer.log_state(state_0)
            viewer.log_mesh(
                name="/scene/ground_plane",
                points=ground_plane_points,
                indices=ground_plane_indices,
                hidden=not ui.show_ground_plane,
                backface_culling=False,
            )
            instrument_radius = _instrument_radius()
            instrument_radii_host.fill(max(instrument_radius, 1.0e-8))
            instrument_radii.assign(instrument_radii_host)
            _update_instrument_colors()
            viewer.log_points(
                name="/instruments/spheres",
                points=instrument_q,
                radii=instrument_radii,
                colors=instrument_colors,
                hidden=not bool(input_devices and ui.show_instruments and instrument_radius > 0.0),
            )
            if isinstance(viewer, newton.viewer.ViewerGL):
                with _scoped_timer("grab_overlay"):
                    show_grab_overlay = bool(ui.show_grab_constraints)
                    grab_point_radius = args.node_render_radius_scale * atlas.voxel_size * 1.25
                    grab_line_width = args.overlay_line_width * 2.5
                    mouse_grab_overlay.update(
                        viewer=viewer,
                        grab_indices=drag_indices,
                        particle_q=state_0.particle_q,
                        pull_target=drag_target,
                        grab_count=drag_count,
                        hidden=not (show_grab_overlay and drag_count > 0 and drag_target is not None),
                        color=(1.0, 0.92, 0.10),
                        width=grab_line_width,
                        point_radius=grab_point_radius,
                    )
                    for idx, overlay in enumerate(instrument_grab_overlays):
                        grasp_count = int(instrument_grasp_count_host[idx])
                        grasp_visible = bool(
                            show_grab_overlay
                            and grasp_count > 0
                            and _instrument_is_grasper(idx)
                            and bool(instrument_trigger_down_host[idx])
                        )
                        overlay.update(
                            viewer=viewer,
                            grab_indices=instrument_grasp_indices[idx],
                            particle_q=state_0.particle_q,
                            pull_target=instrument_q_host[idx],
                            grab_count=grasp_count,
                            hidden=not grasp_visible,
                            color=tuple(float(c) for c in instrument_colors_host[idx]),
                            width=grab_line_width,
                            point_radius=grab_point_radius,
                        )

            particle_hidden = not ui.show_cell_particles
            with _scoped_timer("cell_overlay"):
                if ui.stress_colored_surface:
                    cell_overlay.update_stress(
                        viewer=viewer,
                        particle_q=pg.aux.cell_center_q,
                        particle_flags=pg.aux.cell_render_flags,
                        cell_stretch=pg.aux.cell_stretch,
                        color_scale=ui.stress_color_scale,
                        radius=args.cell_render_radius_scale * atlas.voxel_size,
                        hidden=particle_hidden,
                    )
                elif ui.cryo_colored_cells and cryo_texture_3d is not None:
                    cell_overlay.update_cryo(
                        viewer=viewer,
                        particle_q=pg.aux.cell_center_q,
                        particle_flags=pg.aux.cell_render_flags,
                        particle_grid_xyz=pg.aux.cell_grid_xyz,
                        texture_3d=cryo_texture_3d,
                        grid_shape=pg.aux.grid_shape,
                        radius=args.cell_render_radius_scale * atlas.voxel_size,
                        hidden=particle_hidden,
                        scale_x=ui.cryo_scale_x,
                        scale_y=ui.cryo_scale_y,
                        scale_z=ui.cryo_scale_z,
                    )
                else:
                    cell_overlay.update(
                        viewer=viewer,
                        particle_q=pg.aux.cell_center_q,
                        particle_flags=pg.aux.cell_render_flags,
                        particle_material=pg.aux.cell_material,
                        radius=args.cell_render_radius_scale * atlas.voxel_size,
                        hidden=particle_hidden,
                    )

            with _scoped_timer("heat_overlay"):
                heat_state.update_heat_overlay(
                    viewer=viewer,
                    radius=args.cell_render_radius_scale * atlas.voxel_size * 0.75,
                    hidden=not ui.show_heat_overlay,
                )
                ui.heat_min = float(heat_state.last_heat_min)
                ui.heat_max = float(heat_state.last_heat_max)

            with _scoped_timer("node_overlay"):
                node_overlay.update(
                    viewer=viewer,
                    particle_q=state_0.particle_q,
                    particle_flags=model.particle_flags,
                    particle_material=pg.aux.node_material,
                    radius=args.node_render_radius_scale * atlas.voxel_size,
                    hidden=not ui.show_nodes,
                )
            with _scoped_timer("shape_cluster_overlay"):
                l0_active = solver.l0_runtime_active if ui.sleep_l0_shape_matching else clusters.active
                l0_shape_overlay.update(
                    viewer=viewer,
                    indices_by_slot=clusters.indices_by_slot,
                    cluster_active=l0_active,
                    num_clusters=clusters.num_clusters,
                    particle_q=state_0.particle_q,
                    particle_flags=model.particle_flags,
                    hidden=not ui.show_l0_shape_clusters,
                    color=(0.10, 0.85, 1.0),
                    width=args.overlay_line_width * 1.4,
                )
                l1_clusters = hierarchy.outer8
                l1_shape_overlay.update(
                    viewer=viewer,
                    indices_by_slot=None if l1_clusters is None else l1_clusters.indices_by_slot,
                    cluster_active=None if l1_clusters is None else l1_clusters.active,
                    num_clusters=0 if l1_clusters is None else l1_clusters.num_clusters,
                    particle_q=state_0.particle_q,
                    particle_flags=model.particle_flags,
                    hidden=not ui.show_l1_shape_clusters,
                    color=(1.0, 0.78, 0.18),
                    width=args.overlay_line_width * 2.0,
                )
                l2_clusters = hierarchy.l2_outer8
                l2_shape_overlay.update(
                    viewer=viewer,
                    indices_by_slot=None if l2_clusters is None else l2_clusters.indices_by_slot,
                    cluster_active=None if l2_clusters is None else l2_clusters.active,
                    num_clusters=0 if l2_clusters is None else l2_clusters.num_clusters,
                    particle_q=state_0.particle_q,
                    particle_flags=model.particle_flags,
                    hidden=not ui.show_l2_shape_clusters,
                    color=(1.0, 0.25, 0.90),
                    width=args.overlay_line_width * 2.6,
                )

            rev = delete_state.topology_revision * 1_000_003 + ui.material_visibility_revision
            dirty_surface_revision = None
            dirty_cell_ids = None
            dirty_cell_count = 0
            dirty_cell_count_device = None
            dirty_cell_capacity = None
            dirty_result = delete_state.last_deletion_result or last_async_delete_result
            if (
                dirty_result is not None
                and dirty_result.deleted_cells_device is not None
                and dirty_result.topology_revision == delete_state.topology_revision
                and last_cut_material_visibility_revision == ui.material_visibility_revision
            ):
                dirty_surface_revision = (
                    int(dirty_result.topology_revision) * 1_000_003
                    + int(ui.material_visibility_revision)
                )
                if dirty_surface_revision == rev:
                    dirty_cell_ids = dirty_result.deleted_cells_device
                    dirty_cell_count_device = dirty_result.deleted_count_device
                    dirty_cell_capacity = int(dirty_result.candidate_capacity)
                    if dirty_result.host_synced:
                        dirty_cell_count = int(delete_state.last_deleted_count)
            with _scoped_timer("surface.update"):
                if ui.show_mc_vertex_samples:
                    tri_count = surface.update(
                        viewer=viewer,
                        aux=render_aux,
                        particle_q=pg.aux.cell_center_q,
                        particle_flags=pg.aux.cell_render_flags,
                        orientation=pg.aux.cell_orientation,
                        tables=tables,
                        buffers=mc_buffers,
                        mc_factor=mc_factor,
                        material_visible=material_visible_wp,
                        compute_only=True,
                        topology_revision=rev,
                        dirty_particle_ids=dirty_cell_ids,
                        dirty_particle_count=dirty_cell_count,
                        dirty_particle_count_device=dirty_cell_count_device,
                        dirty_particle_capacity=dirty_cell_capacity,
                        dirty_topology_revision=dirty_surface_revision,
                    )
                    surface.log_hidden(viewer)
                elif ui.show_mesh:
                    smooth_normals = (
                        bool(ui.smooth_mesh_normals)
                        if not active_cut_surface_fast
                        else bool(ui.smooth_mesh_normals and ui.active_cut_smooth_mesh_normals)
                    )
                    taubin_iterations = (
                        int(ui.taubin_iterations)
                        if not active_cut_surface_fast
                        else min(int(ui.taubin_iterations), int(ui.active_cut_taubin_iterations))
                    )
                    slang_procedural_enabled = bool(slang_viewer_requested and ui.slang_procedural_surface)
                    slang_direct_surface = bool(
                        slang_viewer_requested
                        and hasattr(viewer, "draw_cryo_surface")
                        and (cryo_texture_host is not None or slang_procedural_enabled)
                    )
                    if slang_direct_surface:
                        surface_frame = surface.update_cryo_surface_frame(
                            viewer=viewer,
                            aux=render_aux,
                            particle_q=pg.aux.cell_center_q,
                            particle_flags=pg.aux.cell_render_flags,
                            orientation=pg.aux.cell_orientation,
                            tables=tables,
                            buffers=mc_buffers,
                            mc_factor=mc_factor,
                            material_visible=material_visible_wp,
                            smooth_normals=smooth_normals,
                            taubin_iterations=taubin_iterations,
                            taubin_lambda=ui.taubin_lambda,
                            taubin_mu=ui.taubin_mu,
                            topology_revision=rev,
                            dirty_particle_ids=dirty_cell_ids,
                            dirty_particle_count=dirty_cell_count,
                            dirty_particle_count_device=dirty_cell_count_device,
                            dirty_particle_capacity=dirty_cell_capacity,
                            dirty_topology_revision=dirty_surface_revision,
                            include_material_state=slang_procedural_enabled,
                        )
                        tri_count = 0 if surface_frame is None else int(surface_frame.triangle_count)
                        assert ui.material_colors is not None
                        if ui.material_procedural is None or len(ui.material_procedural) != len(ui.material_colors):
                            ui.material_procedural = make_default_procedural_materials(len(ui.material_colors))
                        _sync_material_maker_parameter_specs(len(ui.material_colors))
                        render_bridge.draw_cryo_surface(
                            surface_frame,
                            cryo_texture_host,
                            scale=(ui.cryo_scale_x, ui.cryo_scale_y, ui.cryo_scale_z),
                            hidden=False,
                            material_colors=ui.material_colors,
                            material_colors_revision=ui.material_colors_revision,
                            procedural_params=ui.material_procedural,
                            procedural_params_revision=ui.material_procedural_revision,
                            material_maker_params=ui.material_maker_params,
                            material_maker_params_revision=ui.material_maker_params_revision,
                            procedural_enabled=slang_procedural_enabled,
                            procedural_world_space=ui.slang_procedural_world_space,
                            procedural_uv3_noise_scale=procedural_uv3_noise_scale,
                            lighting_enabled=ui.slang_surface_lighting,
                            key_light_enabled=ui.slang_key_light,
                            fill_light_enabled=ui.slang_fill_light,
                            ambient_light_enabled=ui.slang_ambient_light,
                            environment_lighting_enabled=ui.slang_environment_lighting,
                            cryo_mix=ui.slang_cryo_mix,
                            state_overlay_strength=ui.slang_state_overlay_strength,
                            debug_view=ui.slang_debug_view,
                        )
                    else:
                        active_stress_atlas = _ensure_stress_surface_atlas() if ui.stress_colored_surface else None
                        tri_count = surface.update(
                            viewer=viewer,
                            aux=render_aux,
                            particle_q=pg.aux.cell_center_q,
                            particle_flags=pg.aux.cell_render_flags,
                            orientation=pg.aux.cell_orientation,
                            tables=tables,
                            buffers=mc_buffers,
                            mc_factor=mc_factor,
                            hidden=False,
                            material_visible=material_visible_wp,
                            segmentation_atlas=segmentation_surface_atlas,
                            segmentation_material_colors=(
                                material_colors_wp
                                if surface_color_by_segmentation and not segmentation_volume_direct
                                else None
                            ),
                            segmentation_texture_host=segmentation_texture_host if segmentation_volume_direct else None,
                            segmentation_volume_direct=segmentation_volume_direct,
                            stress_atlas=active_stress_atlas,
                            cell_stretch=pg.aux.cell_stretch if ui.stress_colored_surface else None,
                            stress_color_scale=ui.stress_color_scale,
                            cryo_atlas=cryo_atlas if cryo_texture_3d is not None else None,
                            cryo_texture_3d=cryo_texture_3d,
                            cryo_texture_host=cryo_texture_host if cryo_volume_direct and not slang_viewer_requested else None,
                            cryo_volume_direct=cryo_volume_direct and not slang_viewer_requested,
                            cryo_scale_x=ui.cryo_scale_x,
                            cryo_scale_y=ui.cryo_scale_y,
                            cryo_scale_z=ui.cryo_scale_z,
                            smooth_normals=smooth_normals,
                            taubin_iterations=taubin_iterations,
                            taubin_lambda=ui.taubin_lambda,
                            taubin_mu=ui.taubin_mu,
                            topology_revision=rev,
                            dirty_particle_ids=dirty_cell_ids,
                            dirty_particle_count=dirty_cell_count,
                            dirty_particle_count_device=dirty_cell_count_device,
                            dirty_particle_capacity=dirty_cell_capacity,
                            dirty_topology_revision=dirty_surface_revision,
                        )
                else:
                    surface.log_hidden(viewer)
                    tri_count = ui.tri_count

            if cryo_texture_3d is not None and ui.show_mc_vertex_samples and tri_count > 0:
                with _scoped_timer("mc_vertex_overlay"):
                    mc_vertex_overlay.update_cryo(
                        viewer=viewer,
                        tri_indices=mc_buffers.tri_indices,
                        vertex_pos=mc_buffers.vertex_pos,
                        vertex_uv3=mc_buffers.vertex_uv3,
                        num_triangles=tri_count,
                        texture_3d=cryo_texture_3d,
                        radius=atlas.voxel_size * 0.45,
                        hidden=False,
                        scale_x=ui.cryo_scale_x,
                        scale_y=ui.cryo_scale_y,
                        scale_z=ui.cryo_scale_z,
                    )
            else:
                viewer.log_points(name=mc_vertex_overlay.name, points=None, hidden=True)

            with _scoped_timer("end_frame"):
                render_bridge.end_frame()

            ui.frame = frame
            ui.tri_count = int(tri_count)
            if frame % 30 == 0 and _delete_stats_dirty(delete_state):
                delete_state.sync_host_mirrors({"stats"})
            ui.active_cells = n_cells - int(delete_state.deleted_total)
            ui.deleted_total = int(delete_state.deleted_total)
            completed_frames = frame + 1

            if frame % 60 == 0 or (args.frames is not None and frame == args.frames - 1):
                print(
                    f"  frame {frame:4d}: active={ui.active_cells:6d} deleted={ui.deleted_total:5d} "
                    f"tris={ui.tri_count:6d} pick={ui.last_pick_cell:6d} "
                    f"drag={ui.drag_particle:6d} drag_count={ui.drag_count:4d} "
                    f"grasp={ui.instrument_grasp_counts}"
                )

            _flush_timer_report(completed_frames)
            if not render_bridge.is_running():
                break

        frame += 1
    _flush_timer_report(completed_frames, force=True)
    set_scoped_timer_dict(None)
    _commit_pending_material_shader_edits(force=True)
    for input_device in reversed(input_devices):
        input_device.close()
    render_bridge.close()
    elapsed = time.time() - t_start
    avg_fps = (completed_frames / elapsed) if elapsed > 0.0 else 0.0
    print(f"wall: {elapsed:.1f} s  ({avg_fps:.1f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
