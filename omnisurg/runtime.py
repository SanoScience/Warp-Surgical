import time

import warp as wp

from simulation_systems import BoundsCollisionSystem, DistanceConstraintSystem, VolumeConstraintSystem

from omnisurg.assets import load_tet_asset
from omnisurg.config import (
    BoundsConfig,
    HapticConfig,
    SceneConfig,
    SimulationConfig,
    ViewerConfig,
)
from omnisurg.haptic_collision import HapticSphereCollisionSystem
from omnisurg.haptics import InputSource
from omnisurg.render_bridge import RenderBridge
from omnisurg.scene_builder import HapticProxyState, build_scene
from omnisurg.solver import Phase1Solver


@wp.kernel
def _update_haptic_proxy(
    center_prev: wp.array(dtype=wp.vec3f),
    center_target: wp.array(dtype=wp.vec3f),
    center_current: wp.array(dtype=wp.vec3f),
    center_scaled: wp.array(dtype=wp.vec3f),
    body_q: wp.array(dtype=wp.transformf),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    body_id: int,
    factor: float,
    position_scale: float,
    dt: float,
):
    tid = wp.tid()
    if tid >= 1:
        return

    current = wp.lerp(center_prev[0], center_target[0], factor)
    center_current[0] = current

    scaled = current * position_scale
    center_scaled[0] = scaled

    t = body_q[body_id]
    prev_pos = wp.transform_get_translation(t)
    body_q[body_id] = wp.transform(scaled, wp.quat(t[3], t[4], t[5], t[6]))

    lin_vel = (scaled - prev_pos) / dt
    body_qd[body_id] = wp.spatial_vector(lin_vel, wp.vec3f(0.0, 0.0, 0.0))


@wp.kernel
def _scale_position(
    src: wp.array(dtype=wp.vec3f),
    dst: wp.array(dtype=wp.vec3f),
    scale: float,
):
    tid = wp.tid()
    if tid >= 1:
        return
    dst[0] = src[0] * scale


class Runtime:
    """Phase 1 simulation runtime.

    Strict separation:
      - poll_input(): one host->device copy per frame
      - step(): GPU kernel launches only, zero .numpy()
      - render(): pre-allocated buffers only, zero wp.array()
    """

    def __init__(
        self,
        sim_config: SimulationConfig,
        scene_config: SceneConfig,
        haptic_config: HapticConfig,
        viewer_config: ViewerConfig,
        bounds_config: BoundsConfig,
    ):
        self.sim_config = sim_config
        self.haptic_config = haptic_config
        self.device = wp.get_device()

        asset = load_tet_asset(scene_config.asset_name, scene_config.mesh_dir)

        scene = build_scene(asset, scene_config, haptic_config, self.device)
        self.model = scene.model
        self.surface_indices = scene.surface_tri_indices
        self.proxy: HapticProxyState = scene.haptic_proxy

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.solver = Phase1Solver(
            self.model, iterations=sim_config.constraint_iterations,
        )
        self.solver.register_system(DistanceConstraintSystem(priority=50))
        self.solver.register_system(
            VolumeConstraintSystem(
                stiffness=scene_config.volume_stiffness, priority=60,
            ),
        )
        self.solver.register_system(
            HapticSphereCollisionSystem(self.proxy, priority=80),
        )
        self.solver.register_system(
            BoundsCollisionSystem(
                bounds_min=wp.vec3(*bounds_config.bounds_min),
                bounds_max=wp.vec3(*bounds_config.bounds_max),
                priority=100,
            ),
        )

        self.renderer = RenderBridge(viewer_config, self.model, self.device)
        self.sim_time = 0.0

        self._haptic_staging = wp.zeros(1, dtype=wp.vec3, device="cpu")
        self._haptic_render_pos = wp.zeros(1, dtype=wp.vec3, device=self.device)

        self.use_cuda_graph = self.device.is_cuda
        self.graph = None
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self._simulate_step()
            self.graph = capture.graph

        self._frame_start = time.perf_counter()

    def poll_input(self, source: InputSource):
        """Read one sample from the input source and copy to device.

        Stores position in device units (raw + offset).  The position_scale
        from HapticConfig converts to simulation space on the GPU side.
        """
        sample = source.poll()
        if "position" not in sample:
            wp.copy(self.proxy.center_prev, self.proxy.center_target)
            return

        raw = sample["position"]
        off = self.haptic_config.position_offset
        pos = wp.vec3(
            float(raw[0]) + off[0],
            float(raw[1]) + off[1],
            float(raw[2]) + off[2],
        )

        wp.copy(self.proxy.center_prev, self.proxy.center_target)

        self._haptic_staging.numpy()[0] = [pos[0], pos[1], pos[2]]
        wp.copy(self.proxy.center_target, self._haptic_staging)

    def _simulate_step(self):
        """GPU kernel launches for one frame of substeps."""
        scale = self.haptic_config.position_scale
        for i in range(self.sim_config.substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            factor = float(i) / float(self.sim_config.substeps)
            wp.launch(
                _update_haptic_proxy,
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

            self.solver.step(
                self.state_0, self.state_1, None, None, self.sim_config.substep_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance simulation by one frame, using CUDA graph replay when available."""
        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate_step()
        self.sim_time += self.sim_config.frame_dt

    def render(self):
        """Render phase: static surface + dynamic positions, pre-allocated buffers."""
        wp.launch(
            _scale_position,
            dim=1,
            inputs=[
                self.proxy.center_current,
                self._haptic_render_pos,
                self.haptic_config.position_scale,
            ],
            device=self.device,
        )

        self.renderer.begin_frame(self.sim_time)
        self.renderer.draw_mesh("tissue", self.state_0.particle_q, self.surface_indices)
        self.renderer.draw_haptic_sphere(self._haptic_render_pos)
        self.renderer.end_frame()

    def pace(self):
        """Sleep for the remaining frame budget to hold the target FPS."""
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
