import warp as wp

from simulation_systems import BoundsCollisionSystem, DistanceConstraintSystem, VolumeConstraintSystem
from simulation_kernels import set_body_position

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
def _interpolate_haptic_position(
    src: wp.array(dtype=wp.vec3f),
    dst: wp.array(dtype=wp.vec3f),
    out: wp.array(dtype=wp.vec3f),
    factor: float,
):
    tid = wp.tid()
    if tid >= 1:
        return
    out[0] = wp.lerp(src[0], dst[0], factor)


@wp.kernel
def _scale_position(
    src: wp.array(dtype=wp.vec3f),
    dst: wp.array(dtype=wp.vec3f),
    scale: float,
):
    """Copy src * scale into dst.  Used to produce the sim-space render position
    from the device-unit proxy buffer without a host-side allocation."""
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

    def poll_input(self, source: InputSource):
        """Read one sample from the input source and copy to device.

        Stores position in device units (raw + offset).  The built-in 0.01
        scale in set_body_position and collide_triangles_vs_sphere converts
        to simulation space on the GPU side.
        """
        sample = source.poll()
        if "position" not in sample:
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

    def step(self):
        """GPU phase: all kernel launches, zero .numpy(), zero wp.array()."""
        for i in range(self.sim_config.substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            factor = float(i) / float(self.sim_config.substeps)
            wp.launch(
                _interpolate_haptic_position,
                dim=1,
                inputs=[
                    self.proxy.center_prev,
                    self.proxy.center_target,
                    self.proxy.center_current,
                    factor,
                ],
                device=self.device,
            )

            wp.launch(
                set_body_position,
                dim=1,
                inputs=[
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.proxy.body_id,
                    self.proxy.center_current,
                    self.sim_config.substep_dt,
                ],
                device=self.device,
            )

            self.solver.step(
                self.state_0, self.state_1, None, None, self.sim_config.substep_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.sim_config.frame_dt

    def render(self):
        """Render phase: static surface + dynamic positions, pre-allocated buffers."""
        wp.launch(
            _scale_position,
            dim=1,
            inputs=[self.proxy.center_current, self._haptic_render_pos, 0.01],
            device=self.device,
        )

        self.renderer.begin_frame(self.sim_time)
        self.renderer.draw_mesh("tissue", self.state_0.particle_q, self.surface_indices)
        self.renderer.draw_haptic_sphere(self._haptic_render_pos)
        self.renderer.end_frame()

    def is_running(self) -> bool:
        return self.renderer.is_running()

    def close(self):
        self.renderer.close()
