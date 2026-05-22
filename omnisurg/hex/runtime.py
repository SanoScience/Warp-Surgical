# SPDX-License-Identifier: Apache-2.0
"""Runtime entry points for OmniSurg Hex."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Protocol

import numpy as np
import warp as wp

from omnisurg.config import ViewerConfig
from omnisurg.rendering.bridge import RenderBridge

from .data.types import PreparedVolume
from .haptic import FallbackInput, HapticUnavailable, InputPose, open_haptic_inputs, open_minimou_inputs
from .kernels.cell_render import update_cell_render_state
from .kernels.marching_cubes import allocate_mc_buffers, bake_vertex_uv3, upload_mc_tables
from .render import SurfaceRenderer
from .runtime_lifecycle import HexFrameLoopState, build_hex_frame_loop_config
from .runtime_config import INSTRUMENT_COUNT as _INSTRUMENT_COUNT
from .runtime_config import add_hex_runtime_arguments, normalize_hex_runtime_args
from .setup import (
    StartupPhase as _StartupPhase,
    build_hex_core_setup,
    print_startup_report as _print_startup_report,
)
from .shape_matching_solver import HexShapeMatchingSolver


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
    add_hex_runtime_arguments(parser)
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
        self._frame_loop: HexFrameLoopState | None = None
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
        self._tri_count = 0

    def init(self) -> None:
        if self._model is not None:
            return
        normalize_hex_runtime_args(self.args)
        frame_loop_config = build_hex_frame_loop_config(self.args)

        phases: list[tuple[str, float]] = []
        startup_t0 = time.perf_counter()

        core_setup = build_hex_core_setup(self.args, prepared_volume=self.volume, startup_phases=phases)
        atlas = core_setup.atlas_setup.atlas
        pg = core_setup.particle_grid
        model = core_setup.model
        device = core_setup.device
        delete_state = core_setup.delete_state
        heat_state = core_setup.heat_state
        solver = core_setup.solver

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

                viewer = newton.viewer.ViewerUSD(str(self.args.usd), num_frames=frame_loop_config.max_frames)
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
        self._frame_loop = HexFrameLoopState(frame_loop_config)
        self.exit_after_init = bool(frame_loop_config.exit_after_init)
        self._running = self._frame_loop.should_run_frame()
        if self.exit_after_init:
            self.return_code = 0
        _print_startup_report(phases, time.perf_counter() - startup_t0)

    def _register_runtime_ui(self, render_bridge: RenderBridge) -> None:
        def _runtime_panel(ui) -> None:
            frame_loop = self._frame_loop
            ui.text("OmniSurg Hex")
            ui.separator()
            ui.text(f"frame: {0 if frame_loop is None else frame_loop.frame}")
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
        if self._frame_loop is None:
            return
        substeps = max(1, int(self.args.substeps))
        sub_dt = self._frame_loop.frame_dt / float(substeps)
        for _ in range(substeps):
            self._solver.step(self._state, self._next_state, None, None, sub_dt)
            self._state, self._next_state = self._next_state, self._state
        wp.synchronize_device(self._model.device)

    def render(self) -> None:
        if not self._running or self._render_bridge is None or self._viewer is None or self._state is None:
            return
        if self._frame_loop is None:
            return
        self._render_bridge.begin_frame(self._frame_loop.frame_time)
        if bool(self.args.viewer_log_state):
            self._render_bridge.log_state(self._state)
        self._draw_surface()
        self._render_bridge.end_frame()
        self._running = self._frame_loop.complete_frame(render_running=self._render_bridge.is_running())
        if not self._running:
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
        if self._frame_loop is not None and not self._frame_loop.config.exit_after_init:
            elapsed = self._frame_loop.elapsed
            if elapsed > 0.0:
                print(f"wall: {elapsed:.1f} s  ({self._frame_loop.average_fps:.1f} fps)")


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
        parsed_args = parser.parse_args(args)
        normalize_hex_runtime_args(parsed_args, parser=parser)
        session = HexRuntimeSession(self.volume, parsed_args)
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
        parsed_args = parser.parse_args(list(self.launcher.argv))
        normalize_hex_runtime_args(parsed_args, parser=parser)
        session = HexRuntimeSession(self.launcher.volume, parsed_args)
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
