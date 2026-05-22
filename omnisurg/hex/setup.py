# SPDX-License-Identifier: Apache-2.0
"""Shared setup helpers for the internal hex runtimes."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from omnisurg.hex.data.crop import (
    VisibleClassCrop,
    crop_aligned_texture_rgb,
    crop_labels_to_visible_classes,
)
from omnisurg.hex.data.types import PreparedVolume
from omnisurg.hex.deletion import make_hex_deletion_state
from omnisurg.hex.heat import make_hex_heat_state
from omnisurg.hex.hex_grid import (
    HexParticleGrid,
    build_hex_particle_grid,
    build_hierarchical_shape_matching_clusters,
    build_shape_matching_clusters,
)
from omnisurg.hex.io.cryo import load_cryo_texture
from omnisurg.hex.io.digimouse import DigimouseAtlas, load_digimouse
from omnisurg.hex.materials import DEFAULT_MATERIALS, MaterialTable
from omnisurg.hex.shape_matching_solver import (
    HIERARCHICAL_SHAPE_MATCHING_FULL27,
    HIERARCHICAL_SHAPE_MATCHING_OFF,
    HIERARCHICAL_SHAPE_MATCHING_OUTER8,
    SHAPE_MATCHING_GS_WEIGHT_AVERAGED,
    SHAPE_MATCHING_GS_WEIGHT_FULL,
    SHAPE_MATCHING_GS_WEIGHT_SQRT,
    SHAPE_MATCHING_SOLVE_COLORED_GS,
    SHAPE_MATCHING_SOLVE_GATHER,
    SHAPE_MATCHING_SOLVE_SCATTER,
    HexShapeMatchingSolver,
)

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


@dataclass(frozen=True)
class HexAtlasSetup:
    atlas: DigimouseAtlas
    origin: tuple[float, float, float]
    scene_label: str
    visible_class_crop: VisibleClassCrop | None = None
    prepared_texture_rgb: np.ndarray | None = None


@dataclass(frozen=True)
class HexSolverModes:
    shape_matching_mode: int
    hierarchical_shape_matching_mode: int
    l2_hierarchical_shape_matching_mode: int
    shape_matching_gs_weighting: int
    shape_matching_gs_support_alpha: float


@dataclass
class HexCoreSetup:
    atlas_setup: HexAtlasSetup
    solver_modes: HexSolverModes
    particle_grid: HexParticleGrid
    model: Any
    device: Any
    n_nodes: int
    n_cells: int
    delete_state: Any
    heat_state: Any
    clusters: Any
    hierarchy: Any
    solver: HexShapeMatchingSolver


class StartupPhase:
    """Context manager that records wall-clock time for one startup phase."""

    def __init__(
        self,
        name: str,
        sink: list[tuple[str, float]],
        *,
        sync_device: wp.context.Device | None = None,
    ) -> None:
        self.name = str(name)
        self.sink = sink
        self.sync_device = sync_device
        self.t0 = 0.0

    def __enter__(self) -> "StartupPhase":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        if self.sync_device is not None:
            wp.synchronize_device(self.sync_device)
        self.sink.append((self.name, time.perf_counter() - self.t0))


def print_startup_report(phases: list[tuple[str, float]], total: float) -> None:
    captured = sum(dt for _, dt in phases)
    print(f"[startup] total {total * 1000:.1f} ms (captured {captured * 1000:.1f} ms)")
    for name, dt in sorted(phases, key=lambda kv: -kv[1]):
        share = (100.0 * dt / total) if total > 0.0 else 0.0
        print(f"  {name:<32} {dt * 1000:8.1f} ms  ({share:4.1f}%)")
    leftover = max(0.0, total - captured)
    if leftover > 0.0:
        print(f"  {'(uncaptured)':<32} {leftover * 1000:8.1f} ms")


def make_block_atlas(size: int, voxel: float) -> DigimouseAtlas:
    """Pad the synthetic block so marching cubes sees an exterior boundary."""
    labels = np.zeros((int(size) + 2, int(size) + 2, int(size) + 2), dtype=np.uint8)
    labels[1:-1, 1:-1, 1:-1] = 1
    return DigimouseAtlas(
        labels=labels,
        voxel_size=float(voxel),
        materials=MaterialTable(DEFAULT_MATERIALS),
    )


def upload_rgb_texture(texture_rgb: np.ndarray, device):
    host = np.ascontiguousarray(texture_rgb)
    if host.ndim != 4 or host.shape[-1] != 3 or host.dtype != np.uint8:
        raise ValueError(f"expected uint8 (nx, ny, nz, 3) texture, got {host.shape} {host.dtype}")
    as_float = host.astype(np.float32) / 255.0
    return host, wp.array(as_float, dtype=wp.vec3, device=device)


def atlas_class_map(atlas: DigimouseAtlas) -> dict[int, str]:
    return {int(idx): str(material.name) for idx, material in enumerate(atlas.materials.materials)}


def crop_hex_atlas_to_visible_classes(
    atlas: DigimouseAtlas,
    settings_path: str | Path | None,
    margin_voxels: int,
) -> tuple[DigimouseAtlas, VisibleClassCrop | None]:
    if settings_path is None:
        return atlas, None
    crop = crop_labels_to_visible_classes(
        atlas.labels,
        atlas_class_map(atlas),
        settings_path,
        margin_voxels=int(margin_voxels),
    )
    origin = tuple(
        float(atlas.origin[i]) + float(crop.crop_min[i]) * float(atlas.voxel_size)
        for i in range(3)
    )
    metadata = dict(getattr(atlas, "metadata", {}) or {})
    metadata["visible_class_crop"] = dict(crop.metadata)
    return dataclasses.replace(atlas, labels=crop.labels, origin=origin, metadata=metadata), crop


def load_cryo_texture_for_crop(path: Path, crop: VisibleClassCrop | None, device):
    if crop is None:
        return load_cryo_texture(path, device=device)
    host = np.load(path, mmap_mode="r")
    cropped = crop_aligned_texture_rgb(host, crop, texture_name="cryo texture")
    return upload_rgb_texture(cropped, device)


def _arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def resolve_hex_atlas_setup(
    args: Any,
    *,
    prepared_volume: PreparedVolume | None = None,
) -> HexAtlasSetup:
    drop_height = float(_arg(args, "drop_height", 0.0))

    if prepared_volume is not None:
        atlas = prepared_volume.to_hex_atlas()
        base_origin = prepared_volume.origin
        origin = (
            float(base_origin[0]),
            float(base_origin[1]),
            float(base_origin[2]) + drop_height,
        )
        scene_label = str(prepared_volume.metadata.get("scene_label", "OmniSurg Hex"))
        prepared_texture_rgb = prepared_volume.texture_rgb
        visible_class_crop = None
    elif int(_arg(args, "size", 0)) > 0:
        size = int(_arg(args, "size", 0))
        voxel = float(_arg(args, "voxel", 0.005))
        atlas = make_block_atlas(size, voxel)
        block_extent = size * voxel
        origin = (-0.5 * block_extent - voxel, -0.5 * block_extent - voxel, drop_height - voxel)
        scene_label = f"block {size}^3"
        prepared_texture_rgb = None
        visible_class_crop = None
    else:
        atlas = load_digimouse(
            _arg(args, "atlas", "Digimouse/atlas/atlas"),
            downsample=int(_arg(args, "downsample", 16)),
            pad=int(_arg(args, "atlas_pad", 1)),
            cache_path=_arg(args, "digimouse_cache", None),
            use_cache=bool(_arg(args, "digimouse_use_cache", True)),
            rebuild_cache=bool(_arg(args, "digimouse_rebuild_cache", False)),
        )
        origin = (0.0, 0.0, drop_height)
        scene_label = f"Digimouse --downsample {int(_arg(args, 'downsample', 16))}"
        prepared_texture_rgb = None
        visible_class_crop = None

    crop_settings_path = _arg(args, "crop_visible_classes", None)
    if prepared_volume is None and crop_settings_path is not None:
        atlas, visible_class_crop = crop_hex_atlas_to_visible_classes(
            atlas,
            crop_settings_path,
            int(_arg(args, "crop_visible_margin_voxels", 4)),
        )
        origin = tuple(float(origin[i]) + float(atlas.origin[i]) for i in range(3))
        scene_label = f"{scene_label}  (visible crop)"

    scale = float(_arg(args, "global_scale", 1.0))
    if scale != 1.0:
        atlas = dataclasses.replace(atlas, voxel_size=float(atlas.voxel_size) * scale)
        origin = tuple(float(component) * scale for component in origin)
        scene_label = f"{scene_label}  (global_scale={scale:g})"

    return HexAtlasSetup(
        atlas=atlas,
        origin=tuple(float(component) for component in origin),
        scene_label=scene_label,
        visible_class_crop=visible_class_crop,
        prepared_texture_rgb=prepared_texture_rgb,
    )


def resolve_hex_solver_modes(args: Any) -> HexSolverModes:
    if _arg(args, "shape_matching_mode", None) is None:
        shape_matching_mode = (
            SHAPE_MATCHING_SOLVE_GATHER
            if bool(_arg(args, "shape_matching_gather", False))
            else SHAPE_MATCHING_SOLVE_SCATTER
        )
    else:
        shape_matching_mode = L0_SHAPE_MATCHING_MODE_BY_NAME[str(_arg(args, "shape_matching_mode", None))]

    return HexSolverModes(
        shape_matching_mode=int(shape_matching_mode),
        hierarchical_shape_matching_mode=int(
            HIERARCHICAL_MODE_BY_NAME[str(_arg(args, "hierarchical_shape_matching", "outer8"))]
        ),
        l2_hierarchical_shape_matching_mode=int(
            L2_HIERARCHICAL_MODE_BY_NAME[str(_arg(args, "l2_hierarchical_shape_matching", "outer8"))]
        ),
        shape_matching_gs_weighting=int(
            GS_WEIGHTING_BY_NAME[str(_arg(args, "shape_matching_gs_weighting", "averaged"))]
        ),
        shape_matching_gs_support_alpha=float(_arg(args, "shape_matching_gs_support_alpha", -1.0)),
    )


def build_hex_core_setup(
    args: Any,
    *,
    prepared_volume: PreparedVolume | None = None,
    startup_phases: list[tuple[str, float]] | None = None,
    atlas_setup: HexAtlasSetup | None = None,
) -> HexCoreSetup:
    phases = [] if startup_phases is None else startup_phases

    if atlas_setup is None:
        with StartupPhase("atlas_load", phases):
            atlas_setup = resolve_hex_atlas_setup(args, prepared_volume=prepared_volume)

    atlas = atlas_setup.atlas
    with StartupPhase("build_hex_particle_grid", phases):
        pg = build_hex_particle_grid(
            atlas,
            origin=atlas_setup.origin,
            particle_radius=float(_arg(args, "particle_radius_scale", 0.18)) * float(atlas.voxel_size),
            kinematic_bones=False,
        )
    model = pg.model
    device = model.device

    with StartupPhase("make_hex_deletion_state", phases, sync_device=device):
        delete_state = make_hex_deletion_state(model, pg.aux)
    with StartupPhase("make_hex_heat_state", phases, sync_device=device):
        heat_state = make_hex_heat_state(model, pg.aux)
    with StartupPhase("clusters_l0", phases, sync_device=device):
        clusters = build_shape_matching_clusters(pg)
    with StartupPhase("clusters_l1_hierarchy", phases, sync_device=device):
        hierarchy = build_hierarchical_shape_matching_clusters(pg)

    solver_modes = resolve_hex_solver_modes(args)
    with StartupPhase("solver_ctor", phases, sync_device=device):
        solver = HexShapeMatchingSolver(
            model,
            clusters,
            iterations=int(_arg(args, "iterations", 8)),
            enable_shape_matching=True,
            enable_self_collisions=bool(_arg(args, "particle_particle_collisions", False)),
            enable_ground_plane=True,
            shape_matching_stiffness=float(_arg(args, "shape_matching_stiffness", 1.0)),
            shape_matching_relaxation=float(_arg(args, "shape_matching_relaxation", 1.0)),
            shape_matching_passes=int(_arg(args, "shape_matching_passes", 1)),
            shape_matching_mode=solver_modes.shape_matching_mode,
            shape_matching_gs_weighting=solver_modes.shape_matching_gs_weighting,
            shape_matching_gs_support_alpha=solver_modes.shape_matching_gs_support_alpha,
            shape_matching_use_computed_prolongation=bool(_arg(args, "shape_matching_computed_prolongation", True)),
            enable_volume_preservation=bool(_arg(args, "volume_preservation", False)),
            volume_preservation_stiffness=float(_arg(args, "volume_preservation_stiffness", 0.0)),
            volume_preservation_passes=int(_arg(args, "volume_preservation_passes", 1)),
            hierarchy=hierarchy,
            hierarchical_shape_matching_mode=solver_modes.hierarchical_shape_matching_mode,
            hierarchical_shape_matching_stiffness=float(_arg(args, "hierarchical_shape_matching_stiffness", 1.0)),
            hierarchical_shape_matching_relaxation=float(_arg(args, "hierarchical_shape_matching_relaxation", 1.0)),
            hierarchical_shape_matching_passes=int(_arg(args, "hierarchical_shape_matching_passes", 1)),
            hierarchical_shape_matching_use_gs=bool(_arg(args, "hierarchical_shape_matching_gs", False)),
            hierarchical_shape_matching_outer8_prolongation=bool(
                _arg(args, "hierarchical_shape_matching_outer8_prolongation", True)
            ),
            hierarchical_shape_matching_outer8_absolute_projection=bool(
                _arg(args, "hierarchical_shape_matching_outer8_absolute_projection", False)
            ),
            l2_hierarchical_shape_matching_mode=solver_modes.l2_hierarchical_shape_matching_mode,
            l2_hierarchical_shape_matching_stiffness=float(
                _arg(args, "l2_hierarchical_shape_matching_stiffness", 1.0)
            ),
            l2_hierarchical_shape_matching_relaxation=float(
                _arg(args, "l2_hierarchical_shape_matching_relaxation", 1.0)
            ),
            l2_hierarchical_shape_matching_passes=int(_arg(args, "l2_hierarchical_shape_matching_passes", 1)),
            l2_hierarchical_shape_matching_use_gs=bool(_arg(args, "l2_hierarchical_shape_matching_gs", False)),
            l2_hierarchical_shape_matching_outer8_prolongation=bool(
                _arg(args, "l2_hierarchical_shape_matching_outer8_prolongation", True)
            ),
            l2_hierarchical_shape_matching_outer8_absolute_projection=bool(
                _arg(args, "hierarchical_shape_matching_outer8_absolute_projection", False)
            ),
            sleep_l0_shape_matching=bool(_arg(args, "sleep_l0_shape_matching", False)),
            sleep_l0_wake_halo_blocks=int(_arg(args, "sleep_l0_wake_halo_blocks", 1)),
            ground_height=float(_arg(args, "ground_height", 0.0)),
        )

    return HexCoreSetup(
        atlas_setup=atlas_setup,
        solver_modes=solver_modes,
        particle_grid=pg,
        model=model,
        device=device,
        n_nodes=int(pg.aux.num_nodes),
        n_cells=int(pg.aux.num_cells),
        delete_state=delete_state,
        heat_state=heat_state,
        clusters=clusters,
        hierarchy=hierarchy,
        solver=solver,
    )
