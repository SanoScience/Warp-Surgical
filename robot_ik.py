
import numpy as np
import warp as wp
import math

wp.config.enable_backward = False

import newton
import newton.examples
import newton.sim
import newton.utils
from newton.utils.recorder import BasicRecorder
from newton.utils.recorder_gui import RecorderImGuiManager

from haptic_device import HapticController
from simulation_kernels import (
    set_body_position,
)

class Example:
    def __init__(self, stage_path="example_quadruped.usd", num_envs=8):

        self.dev_pos_buffer = wp.array([0.0, 0.0, 0.0], dtype=wp.vec3, device=wp.get_device())
        self.dev_pos_prev_buffer = wp.array([0.0, 0.0, 0.0], dtype=wp.vec3, device=wp.get_device())

        # articulation_builder = newton.ModelBuilder()
        # articulation_builder.default_body_armature = 0.01
        # articulation_builder.default_joint_cfg.armature = 0.01
        # articulation_builder.default_joint_cfg.mode = newton.JOINT_MODE_TARGET_POSITION
        # articulation_builder.default_joint_cfg.target_ke = 2000.0
        # articulation_builder.default_joint_cfg.target_kd = 1.0
        # articulation_builder.default_shape_cfg.ke = 1.0e4
        # articulation_builder.default_shape_cfg.kd = 1.0e2
        # articulation_builder.default_shape_cfg.kf = 1.0e2
        # articulation_builder.default_shape_cfg.mu = 1.0
        # newton.utils.parse_urdf(
        #     newton.examples.get_asset("quadruped.urdf"),
        #     articulation_builder,
        #     xform=wp.transform([0.0, 0.0, 0.7], wp.quat_identity()),
        #     floating=True,
        #     enable_self_collisions=False,
        # )
        # articulation_builder.joint_q[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]
        # articulation_builder.joint_target[-12:] = articulation_builder.joint_q[-12:]


        articulation_builder2 = newton.ModelBuilder()
        asset_path = newton.utils.download_asset("franka_description")

        newton.utils.parse_urdf(
            str(asset_path / "urdfs" / "fr3_franka_hand.urdf"),
            articulation_builder2,
            up_axis="Y",
            xform=wp.transform(
                (0, 0, 0),
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5),
            ),
            floating=False,
            scale=10,  # unit: cm
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        #articulation_builder2.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]
        # articulation_builder.joint_target[-6:] = articulation_builder.joint_q[-6:]
        builder = newton.ModelBuilder()
        
        # Add haptic device collision body
        self.haptic_body_id = builder.add_body(
            xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
            mass=0.0,  # Zero mass makes it kinematic
            armature=0.0
        )

        # builder.add_shape_sphere(
        #     body=self.haptic_body_id,
        #     xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
        #     radius=0.5,
        #     cfg=newton.ModelBuilder.ShapeConfig(
        #         density=10
        #     )
        # )

        self.sim_time = 0.0
        fps = 120
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_envs = num_envs

        # offsets = newton.examples.compute_env_offsets(self.num_envs)
        # for i in range(self.num_envs):
        #     builder.add_builder(articulation_builder, xform=wp.transform(offsets[i], wp.quat_identity()))

        builder.add_builder(articulation_builder2, xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()))

        builder.add_ground_plane()

        np.set_printoptions(suppress=True)
        # finalize model
        self.model = builder.finalize()

        self.solver = newton.solvers.XPBDSolver(self.model, iterations=16)
        # self.solver = newton.solvers.FeatherstoneSolver(self.model)
        # self.solver = newton.solvers.SemiImplicitSolver(self.model)
        # self.solver = newton.solvers.MuJoCoSolver(self.model)

        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(self.model, path=stage_path)
            self.recorder = BasicRecorder()
            self.gui = RecorderImGuiManager(self.renderer, self.recorder, self)
            self.renderer.render_2d_callbacks.append(self.gui.render_frame)
        else:
            self.renderer = None
            self.recorder = None
            self.gui = None

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        newton.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.haptic_pos_right = [0.0, 0.0, 0.0]
        # simulate() allocates memory via a clone, so we can't use graph capture if the device does not support mempools
        self.use_cuda_graph = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    @property
    def paused(self):
        if self.renderer:
            return self.renderer.paused
        return False

    @paused.setter
    def paused(self, value):
        if self.renderer:
            if self.renderer.paused == value:
                return
            self.renderer.paused = value
            if self.gui:
                self.gui._clear_contact_points()

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            # if self.renderer and hasattr(self.renderer, "apply_picking_force"):
            #     self.renderer.apply_picking_force(self.state_0)

                        # Update haptic device position
            wp.launch(
                set_body_position,
                dim=1,
                inputs=[self.state_0.body_q, self.state_0.body_qd, self.haptic_body_id, self.dev_pos_buffer, self.sim_dt],
                device=self.state_0.body_q.device,
            )


            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.paused:
            return

        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

        if self.recorder:
            if self.renderer:
                self.renderer.compute_contact_rendering_points(self.state_0.body_q, self.contacts)
                contact_points = [self.renderer.contact_points0, self.renderer.contact_points1]
                self.recorder.record(self.state_0.body_q, contact_points)
            else:
                self.recorder.record(self.state_0.body_q)

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            if not self.paused:
                
                self.renderer.render_sphere(
                    "haptic_proxy_sphere",
                    [self.haptic_pos_right[0], self.haptic_pos_right[1], self.haptic_pos_right[2]],
                    [0.0, 0.0, 0.0, 1.0],
                    0.5,
                )
                
                self.renderer.render(self.state_0)
                #self.renderer.render_computed_contacts(contact_point_radius=1e-2)


            else:
                # in paused mode, the GUI will handle rendering from the recorder
                pass
            self.renderer.end_frame()

    def update_haptic_position(self, position):
        """Update the haptic device position in the simulation."""
        
        haptic_pos = wp.vec3(position[2], position[0], position[1])  # Offset to avoid collision with ground;
        self.haptic_pos_right = [haptic_pos[0], haptic_pos[1], haptic_pos[2]]

        wp.copy(self.dev_pos_buffer, wp.array([haptic_pos], dtype=wp.vec3, device=wp.get_device()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_quadruped.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=30000, help="Total number of frames.")
    parser.add_argument("--num-envs", type=int, default=1, help="Total number of simulated environments.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)
        haptic_controller = HapticController(scale=0.05)

        if example.renderer:
            while example.renderer.is_running():
                haptic_pos = haptic_controller.get_scaled_position()
                example.update_haptic_position(haptic_pos)
                example.step()
                example.render()
        else:
            for _ in range(args.num_frames):
                example.step()
                example.render()

        # if example.renderer:
        #     example.renderer.save()
