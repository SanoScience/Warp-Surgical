import math

import warp as wp

import newton
import newton.examples

from omnisurg.config import HapticConfig, SimulationConfig
from omnisurg.haptic_kinematic import (
    create_haptic_proxy_state,
    create_vec3_staging_buffer,
    update_haptic_proxy,
)
from omnisurg.haptics import LiveHapticSource, ReplayInputSource
from omnisurg.newton.soft_grid_scene import SoftGridSceneConfig, build_soft_grid_scene


class SoftBodySim:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.device = wp.get_device()

        self.sim_config = SimulationConfig(
            substeps=10,
            fps=120,
            constraint_iterations=10,
        )
        self.haptic_config = HapticConfig(
            collision_radius=0.15,
            position_offset=(100.0, 100.0, 0.0),
            position_scale=0.02,
        )
        self.scene_config = SoftGridSceneConfig()

        self.fps = self.sim_config.fps
        self.frame_dt = self.sim_config.frame_dt
        self.sim_substeps = self.sim_config.substeps
        self.iterations = self.sim_config.constraint_iterations
        self.sim_dt = self.sim_config.substep_dt
        self._pending_substeps = self.sim_substeps
        self._pending_constraint_iterations = self.iterations

        scene = build_soft_grid_scene(self.scene_config, self.haptic_config)
        self.model = scene.model
        self.proxy = self._create_haptic_proxy(scene.haptic_body_id, scene.haptic_start)

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_collision_detection_interval = 1,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
        )

        # self.solver = newton.solvers.SolverXPBD(
        #     model=self.model, 
        #     iterations=self.iterations, 
        #     soft_body_relaxation=1e-6,
        #     soft_contact_relaxation=0.9,
        #     )

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=self.haptic_config.collision_radius,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

        self._haptic_staging, self._haptic_staging_view = create_vec3_staging_buffer()
        self.input_source = self._init_input_source(args)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        self.capture()

    def _create_haptic_proxy(self, body_id: int, haptic_start: wp.vec3):
        scale = self.haptic_config.position_scale
        raw_start = wp.vec3(
            haptic_start[0] / scale,
            haptic_start[1] / scale,
            haptic_start[2] / scale,
        )
        return create_haptic_proxy_state(
            body_id=body_id,
            radius=self.haptic_config.collision_radius,
            device=self.device,
            initial_position=raw_start,
            initial_scaled_position=haptic_start,
        )

    def _map_haptic_position(self, position) -> wp.vec3:
        """Map device coordinates into the simulation's world axes."""

        x = float(position[0])
        y = float(position[1])
        z = float(position[2])
        if self.model.up_axis == newton.Axis.Z:
            return wp.vec3(z, x, y)
        return wp.vec3(x, y, z)

    def _init_input_source(self, args):
        if args.replay:
            print(f"Using replay input: {args.replay}")
            return ReplayInputSource(args.replay)

        try:
            source = LiveHapticSource()
            print("Using live haptic device")
            return source
        except Exception as e:
            print(f"Haptic device not available ({e}), sphere follows a demo trajectory.")
            return None

    def capture(self):
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def _apply_pending_sim_settings(self):
        substeps = max(1, int(self._pending_substeps))
        iterations = max(1, int(self._pending_constraint_iterations))
        if substeps == self.sim_substeps and iterations == self.iterations:
            return

        self.sim_substeps = substeps
        self.iterations = iterations
        self.sim_dt = self.frame_dt / float(self.sim_substeps)

        self.sim_config.substeps = self.sim_substeps
        self.sim_config.constraint_iterations = self.iterations
        self.sim_config.substep_dt = self.sim_dt
        self.solver.iterations = self.iterations

        self.graph = None
        if self.device.is_cuda:
            # Rebuild the captured graph so updated loop counts take effect.
            self.capture()

    def simulate(self):
        for i in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            factor = float(i) / float(self.sim_substeps)
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
                    self.haptic_config.position_scale,
                    self.sim_dt,
                ],
                device=self.device,
            )

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _read_haptic_position(self):
        """Return haptic position in raw units (+ offset). Scale is applied on GPU."""
        off = self.haptic_config.position_offset
        if self.input_source:
            sample = self.input_source.poll()
            if "position" in sample:
                raw = sample["position"]
                return self._map_haptic_position((
                    float(raw[0]) + off[0],
                    float(raw[1]) + off[1],
                    float(raw[2]) + off[2],
                ))
        # Demo: sinusoidal sweep in raw haptic units around the grid center
        s = 1.0 / self.haptic_config.position_scale
        t = self.sim_time
        return self._map_haptic_position((
            0.6 * s + 0.5 * s * math.sin(t * 1.5),
            1.2 * s + 0.2 * s * math.sin(t * 0.7),
            1.2 * s + 0.2 * s * math.cos(t * 1.0),
        ))

    def step(self):
        self._apply_pending_sim_settings()
        new_pos = self._read_haptic_position()
        wp.copy(self.proxy.center_prev, self.proxy.center_target)
        self._haptic_staging_view[0] = [new_pos[0], new_pos[1], new_pos[2]]
        wp.copy(self.proxy.center_target, self._haptic_staging)

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.sim_config.frame_dt

    def test_final(self):
        # Keep the sanity check broad: the soft grid can sag and deform significantly.
        p_lower = wp.vec3(-1.0, -0.5, 0.0)
        p_upper = wp.vec3(3.0, 4.0, 3.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self._apply_pending_sim_settings()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Soft Body Settings")
        changed, substeps = ui.slider_int("Substeps", self._pending_substeps, 1, 64)
        if changed:
            self._pending_substeps = substeps

        changed, iterations = ui.slider_int(
            "Constraint Iterations",
            self._pending_constraint_iterations,
            1,
            32,
        )
        if changed:
            self._pending_constraint_iterations = iterations

        preview_dt_ms = (self.frame_dt / max(1, self._pending_substeps)) * 1000.0
        ui.text(f"Substep dt: {preview_dt_ms:.3f} ms")
        if (
            self._pending_substeps != self.sim_substeps
            or self._pending_constraint_iterations != self.iterations
        ):
            message = "Changes apply on the next frame."
            if self.device.is_cuda:
                message += " CUDA graph will be rebuilt."
            ui.text(message)

    def close(self):
        if self.input_source:
            self.input_source.close()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--replay",
            type=str,
            default=None,
            help="Path to .npy haptic replay trace (N,7) for deterministic testing",
        )
        return parser


if __name__ == "__main__":
    parser = SoftBodySim.create_parser()
    viewer, args = newton.examples.init(parser)
    example = SoftBodySim(viewer, args)
    newton.examples.run(example, args)
