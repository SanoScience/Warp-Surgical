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

@wp.kernel
def compute_ik_jacobian(
    body_q: wp.array(dtype=wp.transform),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    end_effector_body_id: int,
    target_pos: wp.vec3,
    joint_targets: wp.array(dtype=float),
    ik_damping: float,
    dt: float
):
    # Simple analytical IK for 6-DOF robot arm
    # This is a simplified version - you may need more sophisticated IK
    tid = wp.tid()
    
    if tid == 0:
        # Get current end-effector position
        current_transform = body_q[end_effector_body_id]
        current_pos = wp.transform_get_translation(current_transform)
        
        # Position error
        pos_error = target_pos - current_pos
        error_magnitude = wp.length(pos_error)
        
        # Simple proportional control with damping
        if error_magnitude > 1e-6:
            # Normalize error
            pos_error_norm = pos_error / error_magnitude
            
            # Simple joint space mapping (this is very basic - real IK would use Jacobian)
            # Map position error to joint space changes
            joint_delta = wp.vec3(0.0, 0.0, 0.0)
            
            # Basic mapping for first 3 joints (simplified)
            joint_delta[0] = pos_error_norm[0] * error_magnitude * 0.1
            joint_delta[1] = pos_error_norm[1] * error_magnitude * 0.1  
            joint_delta[2] = pos_error_norm[2] * error_magnitude * 0.1
            
            # Update joint targets with damping
            for i in range(min(3, len(joint_targets))):
                current_target = joint_targets[i]
                new_target = current_target + joint_delta[i] * dt
                # Apply limits (simple clamp)
                new_target = wp.clamp(new_target, -3.14, 3.14)
                joint_targets[i] = current_target + (new_target - current_target) * ik_damping

class Example:
    def __init__(self, stage_path="example_quadruped.usd", num_envs=8):

        self.dev_pos_buffer = wp.array([0.0, 0.0, 0.0], dtype=wp.vec3, device=wp.get_device())
        self.dev_pos_prev_buffer = wp.array([0.0, 0.0, 0.0], dtype=wp.vec3, device=wp.get_device())
        self.ik_target_buffer = wp.array([0.0, 0.0, 0.0], dtype=wp.vec3, device=wp.get_device())

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
        
        # Set initial joint configuration
        articulation_builder2.joint_q[:7] = [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]
        articulation_builder2.joint_target[:7] = articulation_builder2.joint_q[:7]
        
        # Configure joint control
        for i in range(7):  # First 7 joints are the arm
            articulation_builder2.joint_cfg[i].mode = newton.JointMode.TARGET_POSITION
            articulation_builder2.joint_cfg[i].target_ke = 1000.0
            articulation_builder2.joint_cfg[i].target_kd = 50.0
        
        builder = newton.ModelBuilder()
        
        # Add haptic device collision body
        self.haptic_body_id = builder.add_body(
            xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
            mass=0.0,  # Zero mass makes it kinematic
            armature=0.0
        )

        self.sim_time = 0.0
        fps = 120
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_envs = num_envs

        builder.add_builder(articulation_builder2, xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()))
        builder.add_ground_plane()

        np.set_printoptions(suppress=True)
        # finalize model
        self.model = builder.finalize()
        
        # Find end-effector body ID (typically the last link)
        self.end_effector_body_id = len(self.model.body_mass) - 2  # -1 for ground, -1 for last body
        
        # IK parameters
        self.ik_damping = 0.1
        self.ik_enabled = True
        
        # Joint target buffer for IK
        self.joint_targets = wp.array(self.model.joint_target, dtype=float, device=wp.get_device())

        self.solver = newton.solvers.XPBDSolver(self.model, iterations=16)

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
            
            # Update haptic device position
            wp.launch(
                set_body_position,
                dim=1,
                inputs=[self.state_0.body_q, self.state_0.body_qd, self.haptic_body_id, self.dev_pos_buffer, self.sim_dt],
                device=self.state_0.body_q.device,
            )
            
            # Perform IK if enabled
            if self.ik_enabled:
                wp.launch(
                    compute_ik_jacobian,
                    dim=1,
                    inputs=[
                        self.state_0.body_q,
                        self.state_0.joint_q,
                        self.state_0.joint_qd,
                        self.end_effector_body_id,
                        self.ik_target_buffer.data[0],
                        self.joint_targets,
                        self.ik_damping,
                        self.sim_dt
                    ],
                    device=self.state_0.body_q.device,
                )
                
                # Copy computed targets to control
                wp.copy(self.control.joint_target, self.joint_targets)

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
                    [0.0, 1.0, 0.0, 1.0],  # Green for haptic position
                    0.5,
                )
                
                # Render IK target position
                ik_target = self.ik_target_buffer.numpy()[0]
                self.renderer.render_sphere(
                    "ik_target_sphere",
                    [ik_target[0], ik_target[1], ik_target[2]],
                    [1.0, 0.0, 0.0, 1.0],  # Red for IK target
                    0.3,
                )
                
                self.renderer.render(self.state_0)

            else:
                # in paused mode, the GUI will handle rendering from the recorder
                pass
            self.renderer.end_frame()

    def update_haptic_position(self, position):
        """Update the haptic device position in the simulation."""
        
        haptic_pos = wp.vec3(position[2], position[0], position[1])  # Offset to avoid collision with ground
        self.haptic_pos_right = [haptic_pos[0], haptic_pos[1], haptic_pos[2]]

        wp.copy(self.dev_pos_buffer, wp.array([haptic_pos], dtype=wp.vec3, device=wp.get_device()))
        
        # Update IK target based on haptic position with offset
        ik_target = wp.vec3(haptic_pos[0], haptic_pos[1] + 5.0, haptic_pos[2] + 10.0)  # Offset for reachable workspace
        wp.copy(self.ik_target_buffer, wp.array([ik_target], dtype=wp.vec3, device=wp.get_device()))

    def toggle_ik(self):
        """Toggle IK control on/off."""
        self.ik_enabled = not self.ik_enabled
        print(f"IK control: {'enabled' if self.ik_enabled else 'disabled'}")

    def set_ik_damping(self, damping):
        """Set IK damping factor (0.0 to 1.0)."""
        self.ik_damping = max(0.0, min(1.0, damping))
        print(f"IK damping set to: {self.ik_damping}")


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
