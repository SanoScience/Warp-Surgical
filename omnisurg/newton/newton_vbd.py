import math
import os
import sys

import warp as wp

import newton
import newton.examples

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@wp.kernel
def _update_haptic_sphere(
    pos_prev: wp.array(dtype=wp.vec3f),
    pos_target: wp.array(dtype=wp.vec3f),
    pos_current: wp.array(dtype=wp.vec3f),
    body_q: wp.array(dtype=wp.transformf),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    body_id: int,
    factor: float,
    position_scale: float,
    dt: float,
):
    if wp.tid() == 0:
        current = wp.lerp(pos_prev[0], pos_target[0], factor)
        pos_current[0] = current

        scaled = current * position_scale

        xform = body_q[body_id]
        prev = wp.transform_get_translation(xform)
        body_q[body_id] = wp.transform(scaled, wp.transform_get_rotation(xform))
        vel = (scaled - prev) / dt
        body_qd[body_id] = wp.spatial_vector(vel, wp.vec3f(0.0, 0.0, 0.0))


class SoftBodySim:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.solver_type = args.solver
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        if self.solver_type != "vbd":
            raise ValueError("The hanging softbody example only supports the VBD solver.")

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_ground_plane()

        dim_x = 12
        dim_y = 4
        dim_z = 4
        cell_size = 0.1

        builder.add_soft_grid(
            pos=wp.vec3(0.0, 1.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=1.0e3,
            k_mu=1.0e5,
            k_lambda=1.0e5,
            k_damp=1.0e-1,
            fix_left=True,
        )

        # Haptic-driven kinematic sphere for collision
        self.haptic_radius = 0.15
        haptic_start = wp.vec3(0.5, 1.2, 1.2)

        self.haptic_body_id = builder.add_body(
            xform=wp.transform(haptic_start, wp.quat_identity()),
            mass=0.0,
            armature=0.0,
        )
        builder.add_shape_sphere(
            body=self.haptic_body_id,
            xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
            radius=self.haptic_radius,
            cfg=newton.ModelBuilder.ShapeConfig(density=10),
        )

        builder.color()

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1.0e-4
        self.model.soft_contact_mu = 1.0

        self.model.shape_material_ke.fill_(1.0e5)
        self.model.shape_material_kd.fill_(1.0e-4)
        self.model.shape_material_mu.fill_(1.0)

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
        )

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=self.haptic_radius,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

        # Haptic mapping: raw haptic coords + offset, then * scale on GPU
        # Matches omnisurg/runtime.py convention
        self.haptic_position_scale = 0.02
        self.haptic_position_offset = (0.0, 100.0, 100.0)

        # Initial buffers in raw haptic units (haptic_start / scale)
        raw_start = wp.vec3(
            haptic_start[0] / self.haptic_position_scale,
            haptic_start[1] / self.haptic_position_scale,
            haptic_start[2] / self.haptic_position_scale,
        )
        dev = wp.get_device()
        self.haptic_pos_prev = wp.array([raw_start], dtype=wp.vec3f, device=dev)
        self.haptic_pos_target = wp.array([raw_start], dtype=wp.vec3f, device=dev)
        self.haptic_pos_current = wp.array([raw_start], dtype=wp.vec3f, device=dev)

        self.input_source = None
        try:
            from omnisurg.haptics import LiveHapticSource
            self.input_source = LiveHapticSource()
            print("Haptic device connected")
        except Exception as e:
            print(f"No haptic device ({e}). Sphere follows a demo trajectory.")

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for i in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            factor = float(i) / float(self.sim_substeps)
            wp.launch(_update_haptic_sphere, dim=1, inputs=[
                self.haptic_pos_prev, self.haptic_pos_target,
                self.haptic_pos_current,
                self.state_0.body_q, self.state_0.body_qd,
                self.haptic_body_id, factor,
                self.haptic_position_scale, self.sim_dt,
            ])

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _read_haptic_position(self):
        """Return haptic position in raw units (+ offset). Scale is applied on GPU."""
        off = self.haptic_position_offset
        if self.input_source:
            sample = self.input_source.poll()
            if "position" in sample:
                raw = sample["position"]
                return wp.vec3(
                    float(raw[0]) + off[0],
                    float(raw[1]) + off[1],
                    float(raw[2]) + off[2],
                )
        # Demo: sinusoidal sweep in raw haptic units around the grid center
        s = 1.0 / self.haptic_position_scale
        t = self.sim_time
        return wp.vec3(
            0.6 * s + 0.5 * s * math.sin(t * 1.5),
            1.2 * s + 0.2 * s * math.sin(t * 0.7),
            1.2 * s + 0.2 * s * math.cos(t * 1.0),
        )

    def step(self):
        new_pos = self._read_haptic_position()
        wp.copy(self.haptic_pos_prev, self.haptic_pos_target)
        wp.copy(
            self.haptic_pos_target,
            wp.array([new_pos], dtype=wp.vec3f, device=wp.get_device()),
        )

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        # Test that particles are in a reasonable range (soft body may settle or deform)
        # We check that they haven't exploded or collapsed completely
        # 4 grids, each roughly 1.2 x 0.4 x 0.4 in size, positioned along Y-axis
        # Initial positions: Y from 1.0 to ~3.2, X from 0 to 1.2, Z around 1.0 to 1.4
        # With fix_left=True, grids hang and sag significantly towards the ground
        p_lower = wp.vec3(-1.0, -0.5, 0.0)
        p_upper = wp.vec3(3.0, 4.0, 3.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--solver",
            help="Type of solver (only 'vbd' supports volumetric soft bodies)",
            type=str,
            choices=["vbd"],
            default="vbd",
        )
        return parser


if __name__ == "__main__":
    parser = SoftBodySim.create_parser()
    viewer, args = newton.examples.init(parser)
    example = SoftBodySim(viewer, args)
    newton.examples.run(example, args)
