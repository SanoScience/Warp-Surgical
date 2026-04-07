import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import warp as wp

from omnisurg.config import BoundsConfig, HapticConfig, SceneConfig, SimulationConfig, ViewerConfig
from omnisurg.input.haptic_collision import HapticSphereCollisionSystem
from omnisurg.input.haptic_proxy import HapticProxyState, create_vec3_staging_buffer, scale_position, update_haptic_proxy
from omnisurg.input.sources import InputRig, InputSource
from omnisurg.instruments.grasper import load_kinematic_grasper
from omnisurg.mesh.assets import load_scene_asset
from omnisurg.mesh.scene import build_scene
from omnisurg.physics.solver import Phase1Solver
from omnisurg.physics.systems import (
    BoundsCollisionSystem,
    DistanceConstraintSystem,
    TrianglePointConstraintSystem,
    VolumeConstraintSystem,
)
from omnisurg.rendering.bridge import RenderBridge


REPO_ROOT = Path(__file__).resolve().parents[1]
GRASPER_ASSET_PATH = REPO_ROOT / "meshes" / "pgrasp.usdc"
PRIMARY_CONTROLLER_ID = "right"
TEXTURE_CANDIDATES = {
    "liver": (
        REPO_ROOT / "textures" / "liver" / "diffuse-base.png",
        REPO_ROOT / "textures" / "liver_mesh_diffuse.png",
    ),
    "fat": (
        REPO_ROOT / "textures" / "fat" / "diffuse-base.png",
        REPO_ROOT / "textures" / "fat_mesh_diffuse.png",
    ),
    "gallbladder": (
        REPO_ROOT / "textures" / "gallbladder" / "diffuse-base.png",
        REPO_ROOT / "textures" / "gallbladder_mesh_diffuse.png",
    ),
}

MESH_COLORS = {
    "liver": (0.63, 0.29, 0.24),
    "fat": (0.90, 0.82, 0.42),
    "gallbladder": (0.19, 0.54, 0.22),
    "tissue": (0.87, 0.83, 0.78),
}

CONTROLLER_ROOT_COLORS = {
    "left": (0.18, 0.78, 1.0),
}


@dataclass(frozen=True)
class ControllerBinding:
    controller_id: str
    uses_physics_proxy: bool


@dataclass
class ControllerRuntimeState:
    active: bool = False
    button: bool = False
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    scaled_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    render_position: wp.array | None = None
    staging: wp.array | None = None
    staging_view: np.ndarray | None = None


CONTROLLER_BINDINGS = {
    "right": ControllerBinding(controller_id="right", uses_physics_proxy=True),
    "left": ControllerBinding(controller_id="left", uses_physics_proxy=False),
}


def _resolve_texture_path(mesh_name: str) -> str | None:
    for candidate in TEXTURE_CANDIDATES.get(mesh_name, ()): 
        if candidate.exists():
            return str(candidate)
    return None


def _resolve_mesh_textures(mesh_names) -> dict[str, str]:
    textures = {}
    for mesh_name in mesh_names:
        texture = _resolve_texture_path(mesh_name)
        if texture is not None:
            textures[mesh_name] = texture
    return textures


class Runtime:
    """Phase 1/2 runtime for soft-body tissue scenes."""

    def __init__(
        self,
        sim_config: SimulationConfig,
        scene_config: SceneConfig,
        haptic_config: HapticConfig,
        viewer_config: ViewerConfig,
        bounds_config: BoundsConfig,
    ):
        self.sim_config = sim_config
        self.scene_config = scene_config
        self.haptic_config = haptic_config
        self.device = wp.get_device()
        self.textures_enabled = viewer_config.textures_enabled
        self.controller_bindings = CONTROLLER_BINDINGS
        self.controller_states = self._create_controller_states()

        asset = load_scene_asset(scene_config)
        scene = build_scene(asset, scene_config, haptic_config, self.device)

        self.model = scene.model
        self.mesh_ranges = scene.mesh_ranges
        self.surface_indices = scene.surface_tri_indices
        self.surface_meshes = scene.surface_meshes
        self.uvs = scene.uvs
        self.mesh_textures = _resolve_mesh_textures(self.surface_meshes.keys())
        self.proxy: HapticProxyState = scene.haptic_proxy

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.solver = Phase1Solver(self.model, iterations=sim_config.constraint_iterations)
        self.solver.register_system(DistanceConstraintSystem(priority=50))
        self.solver.register_system(VolumeConstraintSystem(stiffness=scene_config.volume_stiffness, priority=60))
        if len(self.model.tri_points_connectors) > 0:
            self.solver.register_system(TrianglePointConstraintSystem(priority=70))
        self.solver.register_system(HapticSphereCollisionSystem(self.proxy, priority=80))
        self.solver.register_system(
            BoundsCollisionSystem(
                bounds_min=wp.vec3(*bounds_config.bounds_min),
                bounds_max=wp.vec3(*bounds_config.bounds_max),
                priority=100,
            ),
        )

        self.renderer = RenderBridge(viewer_config, self.model, self.device)
        self.renderer.set_input_callbacks(on_key_press=self._on_key_press)
        self.graspers = {
            controller_id: load_kinematic_grasper(GRASPER_ASSET_PATH, self.device)
            for controller_id in self.controller_bindings
        }
        self.sim_time = 0.0

        self._haptic_staging, self._haptic_staging_view = create_vec3_staging_buffer()
        self._haptic_render_pos = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self._proxy_scaled_staging, self._proxy_scaled_view = create_vec3_staging_buffer()

        self.use_cuda_graph = self.device.is_cuda and viewer_config.backend != "headless"
        self.graph = None
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self._simulate_step()
            self.graph = capture.graph

        self._frame_start = time.perf_counter()

    def _create_controller_states(self) -> dict[str, ControllerRuntimeState]:
        states: dict[str, ControllerRuntimeState] = {}
        for controller_id, binding in self.controller_bindings.items():
            render_position = None
            staging = None
            staging_view = None
            if not binding.uses_physics_proxy:
                render_position = wp.zeros(1, dtype=wp.vec3, device=self.device)
                staging, staging_view = create_vec3_staging_buffer()
            states[controller_id] = ControllerRuntimeState(
                render_position=render_position,
                staging=staging,
                staging_view=staging_view,
            )
        return states

    def poll_input(self, source: InputRig | InputSource):
        controller_samples = self._poll_controller_samples(source)
        for controller_id, binding in self.controller_bindings.items():
            self._apply_controller_sample(controller_id, binding, controller_samples.get(controller_id))

    def _poll_controller_samples(self, source: InputRig | InputSource):
        if isinstance(source, InputRig):
            return source.poll()

        from omnisurg.input.sources import ControllerSample

        sample = ControllerSample.from_sample_dict(source.poll())
        if sample.active or sample.button:
            return {PRIMARY_CONTROLLER_ID: sample}
        return {}

    def _apply_controller_sample(self, controller_id: str, binding: ControllerBinding, sample):
        state = self.controller_states[controller_id]

        if sample is None:
            if binding.uses_physics_proxy:
                wp.copy(self.proxy.center_prev, self.proxy.center_target)
            state.button = False
            return

        state.button = bool(sample.button)
        if sample.rotation is not None:
            state.rotation = np.array(sample.rotation, dtype=np.float32)
            state.active = True

        if sample.position is None:
            if binding.uses_physics_proxy:
                wp.copy(self.proxy.center_prev, self.proxy.center_target)
            return

        raw_with_offset, scaled = self._map_haptic_position(sample.position)
        state.scaled_position = scaled
        state.active = True

        if binding.uses_physics_proxy:
            wp.copy(self.proxy.center_prev, self.proxy.center_target)
            self._haptic_staging_view[0] = raw_with_offset.tolist()
            wp.copy(self.proxy.center_target, self._haptic_staging)
            return

        state.staging_view[0] = scaled.tolist()
        wp.copy(state.render_position, state.staging)

    def _map_haptic_position(self, raw_position) -> tuple[np.ndarray, np.ndarray]:
        offset = np.array(self.haptic_config.position_offset, dtype=np.float32)
        mapped = np.array(raw_position, dtype=np.float32) + offset
        scaled = mapped * float(self.haptic_config.position_scale)
        return mapped, scaled

    def _simulate_step(self):
        scale = self.haptic_config.position_scale
        for i in range(self.sim_config.substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            factor = float(i) / float(self.sim_config.substeps)
            wp.launch(
                update_haptic_proxy,
                dim=1,
                inputs=[
                    self.proxy.center_prev,
                    self.proxy.center_target,
                    self.proxy.center_current,
                    self.proxy.center_scaled,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.proxy.body_id,
                    factor,
                    scale,
                    self.sim_config.substep_dt,
                ],
                device=self.device,
            )

            self.solver.step(self.state_0, self.state_1, None, None, self.sim_config.substep_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _on_key_press(self, symbol: int, modifiers: int):
        try:
            import pyglet
        except Exception:
            return

        if symbol == pyglet.window.key.T:
            self.toggle_textures()

    def toggle_textures(self):
        self.textures_enabled = not self.textures_enabled
        print(f"Textures {'on' if self.textures_enabled else 'off'}")

    def step(self):
        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate_step()

        for controller_id, grasper in self.graspers.items():
            state = self.controller_states[controller_id]
            if grasper is not None and state.active:
                grasper.advance(self.sim_config.frame_dt, state.button)

        self.sim_time += self.sim_config.frame_dt

    def render(self):
        wp.launch(
            scale_position,
            dim=1,
            inputs=[self.proxy.center_current, self._haptic_render_pos, self.haptic_config.position_scale],
            device=self.device,
        )
        wp.copy(self._proxy_scaled_staging, self.proxy.center_scaled)
        self.controller_states[PRIMARY_CONTROLLER_ID].scaled_position = np.array(self._proxy_scaled_view[0], dtype=np.float32)

        for controller_id, binding in self.controller_bindings.items():
            state = self.controller_states[controller_id]
            grasper = self.graspers.get(controller_id)
            if grasper is None or not state.active:
                continue

            root_position = state.scaled_position
            grasper.update_geometry(root_position, state.rotation)

        self.renderer.begin_frame(self.sim_time)
        if self.surface_meshes:
            for mesh_name, indices in self.surface_meshes.items():
                texture = self.mesh_textures.get(mesh_name) if self.textures_enabled else None
                self.renderer.draw_mesh(
                    mesh_name,
                    self.state_0.particle_q,
                    indices,
                    color=None if texture else MESH_COLORS.get(mesh_name, MESH_COLORS["tissue"]),
                    uvs=self.uvs if texture else None,
                    texture=texture,
                )
        else:
            texture = self.mesh_textures.get("tissue") if self.textures_enabled else None
            self.renderer.draw_mesh(
                "tissue",
                self.state_0.particle_q,
                self.surface_indices,
                color=MESH_COLORS["tissue"] if texture is None else None,
                uvs=self.uvs if texture else None,
                texture=texture,
            )

        for controller_id, grasper in self.graspers.items():
            state = self.controller_states[controller_id]
            if grasper is not None and state.active:
                grasper.render(self.renderer, prefix=f"{controller_id}_grasper")

        if self.controller_states[PRIMARY_CONTROLLER_ID].active:
            self.renderer.draw_haptic_sphere(self._haptic_render_pos)
        for controller_id, color in CONTROLLER_ROOT_COLORS.items():
            state = self.controller_states[controller_id]
            if state.active and state.render_position is not None:
                self.renderer.draw_points(
                    f"{controller_id}_controller_root",
                    state.render_position,
                    0.025,
                    color,
                )
        self.renderer.end_frame()

    def pace(self):
        deadline = self._frame_start + self.sim_config.frame_dt
        now = time.perf_counter()
        remaining = deadline - now
        if remaining > 1e-4:
            time.sleep(remaining)
        self._frame_start = time.perf_counter()

    def is_running(self) -> bool:
        return self.renderer.is_running()

    def close(self):
        self.renderer.close()
