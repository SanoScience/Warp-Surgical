# SPDX-License-Identifier: Apache-2.0
"""ImGui state for the OmniSurg Hex runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omnisurg.rendering.slang_cryo import (
    SLANG_SURFACE_DEBUG_VIEW_HEIGHT,
    SLANG_SURFACE_DEBUG_VIEW_OFF,
)

from .shape_matching_solver import (
    HIERARCHICAL_SHAPE_MATCHING_OUTER8,
    SHAPE_MATCHING_GS_WEIGHT_AVERAGED,
    SHAPE_MATCHING_SOLVE_SCATTER,
)

_INSTRUMENT_COUNT = 2
HDRI_MAP_EXTENSIONS = (".hdr", ".exr")
HDRI_NONE_LABEL = "None"


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


def discover_hdri_maps(folder: str | Path) -> tuple[Path, ...]:
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


def hdri_map_key(path: str | Path | None) -> str:
    if not path:
        return ""
    return str(Path(path).expanduser().resolve(strict=False))


def hdri_map_choice_paths(folder: str | Path, current_path: str | Path | None = None) -> tuple[Path | None, ...]:
    choices: list[Path | None] = [None]
    seen = {""}
    for path in discover_hdri_maps(folder):
        key = hdri_map_key(path)
        choices.append(path)
        seen.add(key)
    if current_path and hdri_map_key(current_path) not in seen:
        choices.append(Path(current_path).expanduser())
    return tuple(choices)


def hdri_map_choice_labels(paths: tuple[Path | None, ...]) -> tuple[str, ...]:
    return tuple(HDRI_NONE_LABEL if path is None else path.name for path in paths)


def hdri_map_choice_index(paths: tuple[Path | None, ...], current_path: str | Path | None) -> int:
    current_key = hdri_map_key(current_path)
    for idx, path in enumerate(paths):
        if hdri_map_key(path) == current_key:
            return idx
    return 0


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


__all__ = [
    "TimerPanelRow",
    "TimerPanelSnapshot",
    "UiState",
    "build_timer_panel_snapshot",
    "discover_hdri_maps",
    "hdri_map_choice_index",
    "hdri_map_choice_labels",
    "hdri_map_choice_paths",
    "hdri_map_key",
]
