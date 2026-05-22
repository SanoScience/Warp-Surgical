# SPDX-License-Identifier: Apache-2.0
"""Runtime entry points for OmniSurg Hex."""

from __future__ import annotations

import argparse
import dataclasses
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Protocol

import numpy as np
import warp as wp

from omnisurg.config import ViewerConfig
from omnisurg.rendering.bridge import RenderBridge
from omnisurg.rendering.slang import SLANG_RENDER_BACKENDS

from .data.types import PreparedVolume
from .deletion import make_hex_deletion_state
from .haptic import FallbackInput, HapticUnavailable, InputPose, open_haptic_inputs, open_minimou_inputs
from .heat import make_hex_heat_state
from .hex_grid import build_hex_particle_grid, build_hierarchical_shape_matching_clusters, build_shape_matching_clusters
from .kernels.cell_render import update_cell_render_state
from .kernels.marching_cubes import allocate_mc_buffers, bake_vertex_uv3, upload_mc_tables
from .render import SurfaceRenderer
from .shape_matching_solver import (
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

_HIERARCHICAL_MODE_BY_NAME = {
    "off": HIERARCHICAL_SHAPE_MATCHING_OFF,
    "outer8": HIERARCHICAL_SHAPE_MATCHING_OUTER8,
    "full27": HIERARCHICAL_SHAPE_MATCHING_FULL27,
}
_L2_HIERARCHICAL_MODE_BY_NAME = {
    "off": HIERARCHICAL_SHAPE_MATCHING_OFF,
    "outer8": HIERARCHICAL_SHAPE_MATCHING_OUTER8,
    "full125": HIERARCHICAL_SHAPE_MATCHING_FULL27,
}
_L0_SHAPE_MATCHING_MODE_BY_NAME = {
    "scatter": SHAPE_MATCHING_SOLVE_SCATTER,
    "gather": SHAPE_MATCHING_SOLVE_GATHER,
    "gs": SHAPE_MATCHING_SOLVE_COLORED_GS,
}
_GS_WEIGHTING_BY_NAME = {
    "averaged": SHAPE_MATCHING_GS_WEIGHT_AVERAGED,
    "sqrt": SHAPE_MATCHING_GS_WEIGHT_SQRT,
    "full": SHAPE_MATCHING_GS_WEIGHT_FULL,
}

_INSTRUMENT_COUNT = 2


class _StartupPhase:
    def __init__(
        self,
        name: str,
        sink: list[tuple[str, float]],
        *,
        sync_device=None,
    ) -> None:
        self.name = str(name)
        self.sink = sink
        self.sync_device = sync_device
        self._start = 0.0

    def __enter__(self) -> "_StartupPhase":
        if self.sync_device is not None:
            wp.synchronize_device(self.sync_device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        if self.sync_device is not None:
            wp.synchronize_device(self.sync_device)
        self.sink.append((self.name, time.perf_counter() - self._start))


def _print_startup_report(phases: list[tuple[str, float]], total: float) -> None:
    print("[startup]")
    for name, seconds in phases:
        print(f"  {name:<28} {seconds * 1000.0:8.2f} ms")
    print(f"  {'total':<28} {total * 1000.0:8.2f} ms")


def _apply_gravity(model, enabled: bool, gravity_on: np.ndarray, gravity_off: np.ndarray) -> None:
    model.gravity.assign(gravity_on if enabled else gravity_off)


def _fallback_instrument_inputs() -> list[FallbackInput]:
    return [
        FallbackInput(InputPose(position=(0.0, 0.0, 0.0), valid=True)),
        FallbackInput(InputPose(position=(0.0, 0.0, 0.0), valid=True)),
    ]


def _open_instrument_inputs(backend: str, device_name: str, left_device_name: str) -> list:
    if backend == "off":
        return []
    if backend == "fallback":
        return _fallback_instrument_inputs()
    if backend == "minimou":
        return open_minimou_inputs(count=_INSTRUMENT_COUNT)
    if backend == "openhaptics":
        return open_haptic_inputs([device_name, left_device_name])
    raise ValueError(f"unknown input backend {backend!r}")


def _build_lifecycle_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--atlas", type=str, default="")
    parser.add_argument("--atlas-pad", type=int, default=0)
    parser.add_argument("--cryo-texture", type=str, default="")
    parser.add_argument("--cryo-renderer", choices=("volume", "atlas", "off"), default="volume")
    parser.add_argument("--viewer", choices=("gl", "headless", *sorted(SLANG_RENDER_BACKENDS)), default="gl")
    parser.add_argument("--usd", type=str, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--timer-report-secs", type=float, default=0.0)
    parser.add_argument("--timer-sync", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timer-gpu-activities", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cut-debug-validate", action="store_true")
    parser.add_argument("--exit-after-init", action="store_true")
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input-backend", choices=("fallback", "openhaptics", "minimou", "off"), default="fallback")
    parser.add_argument("--device-name", default="Default Device")
    parser.add_argument("--left-device-name", default="Left Device")
    parser.add_argument("--position-scale", type=float, default=0.001)
    parser.add_argument("--device-x-offsets", type=float, nargs=2, default=(-0.17, 0.17))
    parser.add_argument("--instrument-radius", "--instrument-radius-scale", dest="instrument_radius_scale", type=float, default=5.0)
    parser.add_argument("--instrument-collision-relaxation", type=float, default=0.9)
    parser.add_argument("--instrument-contact-iterations", type=int, default=1)
    parser.add_argument("--instrument-max-correction-scale", type=float, default=1.0)
    parser.add_argument("--show-instruments", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--instrument-follow-camera", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--instrument-tool-modes", nargs=_INSTRUMENT_COUNT, default=None)
    parser.add_argument("--viewer-log-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gl-interop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--size", type=int, default=0)
    parser.add_argument("--voxel", type=float, default=0.005)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--drop-height", type=float, default=0.04)
    parser.add_argument("--ground-height", type=float, default=0.0)
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--gravity-on", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--particle-radius-scale", type=float, default=0.18)
    parser.add_argument("--particle-particle-collisions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shape-matching-stiffness", type=float, default=1.0)
    parser.add_argument("--shape-matching-relaxation", type=float, default=1.0)
    parser.add_argument("--shape-matching-passes", type=int, default=1)
    parser.add_argument("--shape-matching-mode", choices=tuple(_L0_SHAPE_MATCHING_MODE_BY_NAME), default=None)
    parser.add_argument("--shape-matching-gather", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--shape-matching-gs-weighting",
        choices=tuple(_GS_WEIGHTING_BY_NAME),
        default="averaged",
    )
    parser.add_argument("--shape-matching-gs-support-alpha", type=float, default=-1.0)
    parser.add_argument("--shape-matching-computed-prolongation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--volume-preservation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--volume-preservation-stiffness", type=float, default=0.0)
    parser.add_argument("--volume-preservation-passes", type=int, default=1)
    parser.add_argument("--sleep-l0-shape-matching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sleep-l0-wake-halo-blocks", type=int, default=1)
    parser.add_argument(
        "--hierarchical-shape-matching",
        choices=tuple(_HIERARCHICAL_MODE_BY_NAME),
        default="outer8",
    )
    parser.add_argument("--hierarchical-shape-matching-stiffness", type=float, default=1.0)
    parser.add_argument("--hierarchical-shape-matching-relaxation", type=float, default=1.0)
    parser.add_argument("--hierarchical-shape-matching-passes", type=int, default=1)
    parser.add_argument("--hierarchical-shape-matching-gs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hierarchical-shape-matching-outer8-prolongation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hierarchical-shape-matching-outer8-absolute-projection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--l2-hierarchical-shape-matching",
        choices=tuple(_L2_HIERARCHICAL_MODE_BY_NAME),
        default="outer8",
    )
    parser.add_argument("--l2-hierarchical-shape-matching-stiffness", type=float, default=1.0)
    parser.add_argument("--l2-hierarchical-shape-matching-relaxation", type=float, default=1.0)
    parser.add_argument("--l2-hierarchical-shape-matching-passes", type=int, default=1)
    parser.add_argument("--l2-hierarchical-shape-matching-gs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--l2-hierarchical-shape-matching-outer8-prolongation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--node-render-radius-scale", type=float, default=0.22)
    parser.add_argument("--cell-render-radius-scale", type=float, default=0.5)
    parser.add_argument("--drag-radius-scale", type=float, default=8.0)
    parser.add_argument("--drag-pull-stiffness", type=float, default=1.0)
    parser.add_argument("--show-grab-constraints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plane-cut-depth-scale", type=float, default=8.0)
    parser.add_argument("--ray-cut-depth-scale", type=float, default=8.0)
    parser.add_argument("--render-particles", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overlay-line-width", type=float, default=0.00035)
    parser.add_argument("--show-heat-overlay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diathermy-power", type=float, default=400.0)
    parser.add_argument("--heat-diffusion", type=float, default=0.25)
    parser.add_argument("--heat-cooling", type=float, default=0.10)
    parser.add_argument("--heat-substeps", type=int, default=1)
    parser.add_argument("--blade-length", "--blade-length-scale", dest="blade_length_scale", type=float, default=8.0)
    parser.add_argument("--blade-radius", "--blade-radius-scale", dest="blade_radius_scale", type=float, default=0.75)
    parser.add_argument("--slang-procedural-material", type=str, default="")
    parser.add_argument("--slang-procedural-scale", type=float, default=1.0)
    parser.add_argument("--slang-procedural-hot-reload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slang-environment-map", type=str, default="environments/photo_studio_01_1k.hdr")
    parser.add_argument("--slang-environment-intensity", type=float, default=1.0)
    parser.add_argument("--slang-environment-background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slang-environment-rotation-deg", type=float, default=0.0)
    parser.add_argument("--slang-environment-pitch-deg", type=float, default=-90.0)
    parser.add_argument("--color-by-segmentation-map", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--color-by-stress", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stress-color-scale", type=float, default=30.0)
    parser.add_argument("--active-cut-fast-surface", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--active-cut-smooth-normals", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--active-cut-taubin-iterations", type=int, default=0)
    parser.add_argument("--atlas-tile-size", type=int, default=8)
    parser.add_argument("--segmentation-panel-settings", type=str, default="segmentation_panel_settings.json")
    return parser


class _RuntimeSession(Protocol):
    exit_after_init: bool
    return_code: int | None

    def init(self) -> None:
        ...

    def poll_input(self) -> None:
        ...

    def step(self) -> None:
        ...

    def render(self) -> None:
        ...

    def pace(self) -> None:
        ...

    def is_running(self) -> bool:
        ...

    def close(self) -> None:
        ...


class HexRuntimeSession:
    """Stateful in-process lifecycle for the packaged hex runtime."""

    def __init__(self, volume: PreparedVolume, args: argparse.Namespace) -> None:
        self.volume = volume
        self.args = args
        self.exit_after_init = bool(args.exit_after_init)
        self.return_code: int | None = None
        self._closed = False
        self._running = False
        self._frame = 0
        self._max_frames: int | None = None
        self._frame_dt = 1.0 / max(1, int(args.fps))
        self._state = None
        self._next_state = None
        self._solver: HexShapeMatchingSolver | None = None
        self._model = None
        self._viewer = None
        self._render_bridge: RenderBridge | None = None
        self._delete_state = None
        self._heat_state = None
        self._particle_grid = None
        self._render_aux = None
        self._mc_tables = None
        self._mc_buffers = None
        self._mc_factor = 0.0
        self._surface: SurfaceRenderer | None = None
        self._input_devices: list = []
        self._start_wall = 0.0
        self._completed_frames = 0
        self._tri_count = 0

    def init(self) -> None:
        if self._model is not None:
            return

        phases: list[tuple[str, float]] = []
        startup_t0 = time.perf_counter()
        self._start_wall = time.time()

        if self.args.usd is not None and self.args.frames is None:
            raise SystemExit("--frames is required when using --usd")
        if self.args.usd is not None and str(self.args.viewer) in SLANG_RENDER_BACKENDS:
            raise SystemExit("--viewer slang cannot be combined with --usd")
        if self.args.instrument_follow_camera is None:
            self.args.instrument_follow_camera = str(self.args.input_backend) == "fallback"

        if self.args.frames is None:
            self._max_frames = 1 if str(self.args.viewer) == "headless" else None
        else:
            self._max_frames = max(0, int(self.args.frames))

        with _StartupPhase("atlas_load", phases):
            atlas = self.volume.to_hex_atlas()
            base_origin = getattr(self.volume, "origin", (0.0, 0.0, 0.0))
            origin = (
                float(base_origin[0]),
                float(base_origin[1]),
                float(base_origin[2]) + float(self.args.drop_height),
            )
            if float(self.args.global_scale) != 1.0:
                scale = float(self.args.global_scale)
                atlas = dataclasses.replace(atlas, voxel_size=float(atlas.voxel_size) * scale)
                origin = tuple(float(component) * scale for component in origin)

        with _StartupPhase("build_hex_particle_grid", phases):
            pg = build_hex_particle_grid(
                atlas,
                origin=origin,
                particle_radius=float(self.args.particle_radius_scale) * float(atlas.voxel_size),
                kinematic_bones=False,
            )

        model = pg.model
        device = model.device
        with _StartupPhase("make_hex_deletion_state", phases, sync_device=device):
            delete_state = make_hex_deletion_state(model, pg.aux)
        with _StartupPhase("make_hex_heat_state", phases, sync_device=device):
            heat_state = make_hex_heat_state(model, pg.aux)
        with _StartupPhase("clusters_l0", phases, sync_device=device):
            clusters = build_shape_matching_clusters(pg)
        with _StartupPhase("clusters_hierarchy", phases, sync_device=device):
            hierarchy = build_hierarchical_shape_matching_clusters(pg)
        with _StartupPhase("mc_tables_and_buffers", phases, sync_device=device):
            tables = upload_mc_tables(device=device)
            mc_buffers = allocate_mc_buffers(pg.aux.grid_shape, pg.aux.num_cells, device=device)
            mc_factor = float(atlas.voxel_size) * 0.5
            bake_vertex_uv3(mc_buffers, pg.aux.cell_grid_xyz, pg.aux.grid_shape, device=device)
            surface = SurfaceRenderer(
                "hex_shape_matching_surface",
                pg.aux.num_cells,
                device=device,
                max_triangles=mc_buffers.max_triangles,
            )
            render_aux = SimpleNamespace(
                particle_material=pg.aux.cell_material,
                particle_grid_xyz=pg.aux.cell_grid_xyz,
                grid_to_particle=pg.aux.grid_to_cell,
                grid_shape=pg.aux.grid_shape,
            )

        if self.args.shape_matching_mode is None:
            shape_matching_mode = (
                SHAPE_MATCHING_SOLVE_GATHER
                if bool(self.args.shape_matching_gather)
                else SHAPE_MATCHING_SOLVE_SCATTER
            )
        else:
            shape_matching_mode = _L0_SHAPE_MATCHING_MODE_BY_NAME[str(self.args.shape_matching_mode)]

        with _StartupPhase("solver_ctor", phases, sync_device=device):
            solver = HexShapeMatchingSolver(
                model,
                clusters,
                iterations=int(self.args.iterations),
                enable_shape_matching=True,
                enable_self_collisions=bool(self.args.particle_particle_collisions),
                enable_ground_plane=True,
                shape_matching_stiffness=float(self.args.shape_matching_stiffness),
                shape_matching_relaxation=float(self.args.shape_matching_relaxation),
                shape_matching_passes=int(self.args.shape_matching_passes),
                shape_matching_mode=shape_matching_mode,
                shape_matching_gs_weighting=_GS_WEIGHTING_BY_NAME[str(self.args.shape_matching_gs_weighting)],
                shape_matching_gs_support_alpha=float(self.args.shape_matching_gs_support_alpha),
                shape_matching_use_computed_prolongation=bool(self.args.shape_matching_computed_prolongation),
                enable_volume_preservation=bool(self.args.volume_preservation),
                volume_preservation_stiffness=float(self.args.volume_preservation_stiffness),
                volume_preservation_passes=int(self.args.volume_preservation_passes),
                hierarchy=hierarchy,
                hierarchical_shape_matching_mode=_HIERARCHICAL_MODE_BY_NAME[
                    str(self.args.hierarchical_shape_matching)
                ],
                hierarchical_shape_matching_stiffness=float(self.args.hierarchical_shape_matching_stiffness),
                hierarchical_shape_matching_relaxation=float(self.args.hierarchical_shape_matching_relaxation),
                hierarchical_shape_matching_passes=int(self.args.hierarchical_shape_matching_passes),
                hierarchical_shape_matching_use_gs=bool(self.args.hierarchical_shape_matching_gs),
                hierarchical_shape_matching_outer8_prolongation=bool(
                    self.args.hierarchical_shape_matching_outer8_prolongation
                ),
                hierarchical_shape_matching_outer8_absolute_projection=bool(
                    self.args.hierarchical_shape_matching_outer8_absolute_projection
                ),
                l2_hierarchical_shape_matching_mode=_L2_HIERARCHICAL_MODE_BY_NAME[
                    str(self.args.l2_hierarchical_shape_matching)
                ],
                l2_hierarchical_shape_matching_stiffness=float(self.args.l2_hierarchical_shape_matching_stiffness),
                l2_hierarchical_shape_matching_relaxation=float(self.args.l2_hierarchical_shape_matching_relaxation),
                l2_hierarchical_shape_matching_passes=int(self.args.l2_hierarchical_shape_matching_passes),
                l2_hierarchical_shape_matching_use_gs=bool(self.args.l2_hierarchical_shape_matching_gs),
                l2_hierarchical_shape_matching_outer8_prolongation=bool(
                    self.args.l2_hierarchical_shape_matching_outer8_prolongation
                ),
                l2_hierarchical_shape_matching_outer8_absolute_projection=bool(
                    self.args.hierarchical_shape_matching_outer8_absolute_projection
                ),
                sleep_l0_shape_matching=bool(self.args.sleep_l0_shape_matching),
                sleep_l0_wake_halo_blocks=int(self.args.sleep_l0_wake_halo_blocks),
                ground_height=float(self.args.ground_height),
            )

        with _StartupPhase("input_devices", phases):
            try:
                self._input_devices = _open_instrument_inputs(
                    str(self.args.input_backend),
                    str(self.args.device_name),
                    str(self.args.left_device_name),
                )
            except HapticUnavailable as exc:
                print(f"Instrument input unavailable: {exc}")
                self.return_code = 2
                self._running = False
                return

        with _StartupPhase("state_and_viewer", phases, sync_device=device):
            state = pg.state
            next_state = model.state()
            gravity_on = model.gravity.numpy().copy()
            gravity_off = np.zeros_like(gravity_on)
            _apply_gravity(model, bool(self.args.gravity_on), gravity_on, gravity_off)

            if self.args.usd is not None:
                import newton

                viewer = newton.viewer.ViewerUSD(str(self.args.usd), num_frames=self._max_frames)
                viewer.set_model(model)
                render_bridge = RenderBridge.wrap_existing(viewer, backend="usd", device=device)
            else:
                camera_pos = (0.12, -0.18, 0.12) if int(self.args.size) > 0 else (0.32, 0.04, 0.08)
                viewer_config = ViewerConfig(
                    backend=str(self.args.viewer),
                    camera_pos=camera_pos,
                    vsync=True,
                    textures_enabled=bool(str(self.args.cryo_renderer) != "off"),
                    slang_world_up=(0.0, 0.0, 1.0),
                    slang_procedural_material_path=str(self.args.slang_procedural_material or "") or None,
                    slang_procedural_material_scale=max(float(self.args.slang_procedural_scale), 0.000001),
                    slang_procedural_material_hot_reload=bool(self.args.slang_procedural_hot_reload),
                    slang_environment_path=str(self.args.slang_environment_map or "") or None,
                    slang_environment_intensity=max(float(self.args.slang_environment_intensity), 0.0),
                    slang_environment_background=bool(self.args.slang_environment_background),
                    slang_environment_rotation_degrees=float(self.args.slang_environment_rotation_deg),
                    slang_environment_pitch_degrees=float(self.args.slang_environment_pitch_deg),
                )
                render_bridge = RenderBridge(viewer_config, model, device)
                viewer = render_bridge
                if hasattr(viewer, "set_camera"):
                    try:
                        viewer.set_camera(pos=wp.vec3(*camera_pos), pitch=-18.0, yaw=30.0)
                    except TypeError:
                        viewer.set_camera(wp.vec3(*camera_pos), -18.0, 30.0)
                render_bridge.show_ui = True
                self._register_runtime_ui(render_bridge)

        self._model = model
        self._state = state
        self._next_state = next_state
        self._solver = solver
        self._delete_state = delete_state
        self._heat_state = heat_state
        self._particle_grid = pg
        self._render_aux = render_aux
        self._mc_tables = tables
        self._mc_buffers = mc_buffers
        self._mc_factor = mc_factor
        self._surface = surface
        self._viewer = viewer
        self._render_bridge = render_bridge
        self._running = bool(not self.exit_after_init and (self._max_frames is None or self._max_frames > 0))
        if self.exit_after_init:
            self.return_code = 0
        _print_startup_report(phases, time.perf_counter() - startup_t0)

    def _register_runtime_ui(self, render_bridge: RenderBridge) -> None:
        def _runtime_panel(ui) -> None:
            ui.text("OmniSurg Hex")
            ui.separator()
            ui.text(f"frame: {self._frame}")
            ui.text(f"triangles: {self._tri_count}")
            if self._delete_state is not None:
                active_cells = 0
                total_cells = 0
                pg = self._particle_grid
                if pg is not None:
                    total_cells = int(pg.aux.num_cells)
                    active_cells = total_cells - int(getattr(self._delete_state, "deleted_total", 0))
                ui.text(f"active cells: {active_cells}/{total_cells}")
            ui.text(f"viewer: {self.args.viewer}")

        try:
            render_bridge.register_ui_callback(_runtime_panel, position="side")
        except Exception:
            pass

    def poll_input(self) -> None:
        for input_device in self._input_devices:
            poll = getattr(input_device, "poll", None)
            if callable(poll):
                poll()

    def step(self) -> None:
        if not self._running or self._solver is None or self._state is None or self._next_state is None:
            return
        substeps = max(1, int(self.args.substeps))
        sub_dt = self._frame_dt / float(substeps)
        for _ in range(substeps):
            self._solver.step(self._state, self._next_state, None, None, sub_dt)
            self._state, self._next_state = self._next_state, self._state
        wp.synchronize_device(self._model.device)

    def render(self) -> None:
        if not self._running or self._render_bridge is None or self._viewer is None or self._state is None:
            return
        self._render_bridge.begin_frame(self._frame * self._frame_dt)
        if bool(self.args.viewer_log_state):
            self._render_bridge.log_state(self._state)
        self._draw_surface()
        self._render_bridge.end_frame()
        self._frame += 1
        self._completed_frames = self._frame
        if self._max_frames is not None and self._frame >= self._max_frames:
            self._running = False
            self.return_code = 0
        elif not self._render_bridge.is_running():
            self._running = False
            self.return_code = 0

    def _draw_surface(self) -> None:
        if (
            self._surface is None
            or self._particle_grid is None
            or self._render_aux is None
            or self._mc_tables is None
            or self._mc_buffers is None
            or self._render_bridge is None
            or self._state is None
            or self._model is None
        ):
            return

        pg = self._particle_grid
        device = self._model.device
        update_cell_render_state(pg.aux, self._state.particle_q, device=device, compute_stretch=False)
        topology_revision = 0
        if self._delete_state is not None:
            topology_revision = int(getattr(self._delete_state, "topology_revision", 0))
        self._tri_count = int(
            self._surface.update(
                viewer=self._render_bridge,
                aux=self._render_aux,
                particle_q=pg.aux.cell_center_q,
                particle_flags=pg.aux.cell_render_flags,
                orientation=pg.aux.cell_orientation,
                tables=self._mc_tables,
                buffers=self._mc_buffers,
                mc_factor=float(self._mc_factor),
                hidden=False,
                smooth_normals=True,
                taubin_iterations=0,
                topology_revision=topology_revision,
            )
        )

    def pace(self) -> None:
        return None

    def is_running(self) -> bool:
        return bool(self._running and not self._closed)

    def close(self) -> None:
        if self._closed:
            return
        self._running = False
        self._closed = True
        for input_device in reversed(self._input_devices):
            close = getattr(input_device, "close", None)
            if callable(close):
                close()
        self._input_devices = []
        if self._render_bridge is not None:
            self._render_bridge.close()
        if self.return_code is None:
            self.return_code = 0
        elapsed = time.time() - self._start_wall if self._start_wall else 0.0
        if elapsed > 0.0:
            fps = float(self._completed_frames) / elapsed
            print(f"wall: {elapsed:.1f} s  ({fps:.1f} fps)")


@dataclass
class HexAppLauncher:
    """Launch the packaged hex app with a prepared volume."""

    volume: PreparedVolume
    argv: Sequence[str] = field(default_factory=tuple)
    return_code: int | None = None
    _closed: bool = False

    def run(self, argv: Sequence[str] | None = None) -> int:
        args = list(self.argv if argv is None else argv)
        parser = _build_lifecycle_parser()
        session = HexRuntimeSession(self.volume, parser.parse_args(args))
        try:
            session.init()
            while session.is_running():
                session.poll_input()
                session.step()
                session.render()
                session.pace()
            self.return_code = 0 if session.return_code is None else int(session.return_code)
        finally:
            session.close()
        self._closed = True
        return self.return_code

    def run_exit_after_init(self) -> int:
        args = list(self.argv)
        if "--exit-after-init" not in args:
            args.append("--exit-after-init")
        return self.run(args)

    def close(self) -> None:
        self._closed = True


@dataclass
class HexRuntime:
    """Stateful runtime facade used by tests and app orchestration."""

    launcher: HexAppLauncher
    _session: _RuntimeSession | None = None
    _initialized: bool = False
    _closed: bool = False
    return_code: int | None = None
    exit_after_init: bool = False

    @classmethod
    def from_volume(cls, volume: PreparedVolume, argv: Sequence[str] | None = None) -> "HexRuntime":
        return cls(HexAppLauncher(volume, tuple(argv or ())))

    def init(self) -> None:
        if self._initialized:
            return
        parser = _build_lifecycle_parser()
        session = HexRuntimeSession(self.launcher.volume, parser.parse_args(list(self.launcher.argv)))
        self._session = session
        session.init()
        self.exit_after_init = bool(session.exit_after_init)
        if session.return_code is not None:
            self.return_code = int(session.return_code)
        elif self.exit_after_init:
            self.return_code = 0 if session.return_code is None else int(session.return_code)
        self._initialized = True

    def _require_session(self) -> _RuntimeSession:
        if not self._initialized:
            self.init()
        assert self._session is not None
        return self._session

    def poll_input(self) -> None:
        self._require_session().poll_input()

    def step(self) -> None:
        session = self._require_session()
        session.step()
        if session.return_code is not None:
            self.return_code = int(session.return_code)

    def render(self) -> None:
        session = self._require_session()
        session.render()
        if session.return_code is not None:
            self.return_code = int(session.return_code)

    def is_running(self) -> bool:
        if not self._initialized or self._session is None:
            return False
        return bool(self._session.is_running())

    def pace(self) -> None:
        self._require_session().pace()

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            if self._session.return_code is not None:
                self.return_code = int(self._session.return_code)
        self._closed = True
        self.launcher.close()


def __getattr__(name: str):
    if name == "OmniSurgHexApp":
        from .app import OmniSurgHexApp

        return OmniSurgHexApp
    raise AttributeError(name)


__all__ = ["HexAppLauncher", "HexRuntime", "OmniSurgHexApp"]
