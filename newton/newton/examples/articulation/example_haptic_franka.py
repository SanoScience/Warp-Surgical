# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Haptic Franka
#
# Real-time haptic device control of Franka FR3 robot using IK.
# Control the robot end-effector with an Omni Geomagic haptic device.
#
# Command: python -m newton.examples haptic_franka [--solver-type xpbd|featherstone|mujoco]
#
###########################################################################

import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils

# Add main project folder to path for haptic_device import
main_folder = str(Path(__file__).resolve().parents[4])
print(main_folder)
sys.path.insert(0, main_folder)

try:
    from haptic_device import HapticController
    HAPTIC_AVAILABLE = True
except ImportError:
    HAPTIC_AVAILABLE = False
    print("[WARNING]: HapticController not available. Install OpenHaptics to use this example.")


class HapticConfig:
    """Haptic device control configuration."""

    # Haptic control sensitivity parameters
    position_sensitivity = 0.004  # Scale for position mapping
    rotation_sensitivity = 0.5  # Scale for rotation mapping
    max_position_delta = 0.03  # Maximum position change per step (meters)
    max_rotation_delta = 0.3  # Maximum rotation change per step (radians)

    # Omni Geomagic axis mapping
    # Robot [X, Y, Z] = Haptic [Z, X, Y]
    haptic_axis_map = [2, 0, 1]
    haptic_axis_signs = [-1.0, -1.0, 1.0]

    # Rotation axis remapping (same as position)
    haptic_rot_axis_map = [2, 0, 1]
    haptic_rot_axis_signs = [-1.0, -1.0, 1.0]

    # Calibration settings
    enable_calibration = True
    calibration_wait_time = 3.0  # Seconds to wait for device to be docked

    # Target robot pose when haptic is docked
    target_ee_pos_docked = [0.6, 0.0, 0.3]  # Forward, centered, mid-height
    gripper_pitch_angle_deg = 90.0  # Degrees downward for gripper

    # Gripper control settings
    gripper_pos_closed = 0.0
    gripper_pos_open = 0.04

    @staticmethod
    def get_target_ee_quat_docked():
        """Get target quaternion for gripper pointing downward."""
        angle = np.radians(HapticConfig.gripper_pitch_angle_deg)
        # Quaternion for rotation around Y axis: [x, y, z, w]
        return np.array([
            0.0,
            np.sin(angle / 2),
            0.0,
            np.cos(angle / 2)
        ])


class Example:
    def __init__(self, viewer, args=None):
        if not HAPTIC_AVAILABLE:
            raise RuntimeError("HapticController not available. Cannot run haptic example.")

        # Setup simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        # Get solver type
        solver_type = getattr(args, "solver_type", "xpbd") if args else "xpbd"

        # Adjust substeps based on solver type
        if solver_type == "xpbd":
            self.sim_substeps = 32  # Balanced for real-time haptic control
        elif solver_type == "featherstone":
            self.sim_substeps = 10
        elif solver_type == "mujoco":
            self.sim_substeps = 10
        else:
            self.sim_substeps = 32

        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.args = args

        # Load haptic config
        self.haptic_cfg = HapticConfig()

        # Initialize haptic controller
        self.haptic_controller = HapticController(scale=1.0)
        print("[INFO]: Haptic controller initialized")

        # Build Franka robot
        builder = newton.ModelBuilder()

        if solver_type == "mujoco":
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
        )

        # Set control gains
        if solver_type == "mujoco":
            for i in range(len(builder.joint_target_ke)):
                builder.joint_target_ke[i] = 500.0
                builder.joint_target_kd[i] = 50.0
        elif solver_type == "featherstone":
            for i in range(len(builder.joint_target_ke)):
                builder.joint_target_ke[i] = 400.0
                builder.joint_target_kd[i] = 15.0
        elif solver_type == "xpbd":
            for i in range(len(builder.joint_target_ke)):
                builder.joint_target_ke[i] = 10000.0
                builder.joint_target_kd[i] = 2000.0

        builder.add_ground_plane()

        # Finalize model
        self.model = builder.finalize()
        self.device = wp.get_device()

        print(f"[INFO]: Loaded Franka with {self.model.joint_count} joints")

        # Set initial joint positions
        home_pose = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
        self.model.joint_q = wp.array(home_pose, dtype=wp.float32, device=self.device)
        self.model.joint_qd.zero_()

        # Create solver
        print(f"[INFO]: Creating {solver_type.upper()} solver...")

        if solver_type == "xpbd":
            self.model.ground = True
            body_inv_mass = self.model.body_inv_mass.numpy()
            body_inv_inertia = self.model.body_inv_inertia.numpy()

            base_body_idx = 0
            body_inv_mass[base_body_idx] = 0.0
            body_inv_inertia[base_body_idx] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

            self.model.body_inv_mass = wp.array(body_inv_mass, dtype=wp.float32, device=self.device)
            self.model.body_inv_inertia = wp.array(body_inv_inertia, dtype=wp.mat33, device=self.device)

            self.solver = newton.solvers.SolverXPBD(
                self.model,
                iterations=20,
                angular_damping=0.15,
                joint_linear_compliance=0.0,
                joint_angular_compliance=0.0,
                joint_linear_relaxation=0.8,
                joint_angular_relaxation=0.6,
            )
        elif solver_type == "featherstone":
            self.solver = newton.solvers.SolverFeatherstone(
                self.model,
                update_mass_matrix_interval=self.sim_substeps,
            )
        elif solver_type == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                disable_contacts=True,
            )
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Initialize FK
        if not isinstance(self.solver, newton.solvers.SolverMuJoCo):
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # End-effector index
        self.ee_index = 10

        # Get initial EE pose
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        body_q_np = self.state_0.body_q.numpy()
        ee_transform = body_q_np[self.ee_index]
        ee_pos_w = ee_transform[:3]
        ee_quat_w = ee_transform[3:]  # [w, x, y, z]

        # Create IK objectives
        self.pos_obj = ik.IKPositionObjective(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.vec3(ee_pos_w[0], ee_pos_w[1], ee_pos_w[2])], dtype=wp.vec3),
            weight=3.0,
        )

        self.rot_obj = ik.IKRotationObjective(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(ee_quat_w[1], ee_quat_w[2], ee_quat_w[3], ee_quat_w[0])], dtype=wp.vec4),
            weight=2.0,
        )

        self.joint_limits_obj = ik.IKJointLimitObjective(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=2.0,
        )

        # Create IK solver
        self.joint_q_2d = wp.array(self.model.joint_q.numpy().reshape(1, -1), dtype=wp.float32)

        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limits_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianMode.ANALYTIC,
        )

        # Calibrate haptic device
        self._calibrate_haptic()

        # Initialize target tracking
        target_ee_pos_docked = np.array(self.haptic_cfg.target_ee_pos_docked)
        target_ee_quat_docked = self.haptic_cfg.get_target_ee_quat_docked()

        if self.haptic_cfg.enable_calibration:
            self.target_ee_pos = target_ee_pos_docked.copy()
            # Convert [x, y, z, w] to Warp [w, x, y, z]
            self.target_ee_quat = np.array([target_ee_quat_docked[3], target_ee_quat_docked[0],
                                           target_ee_quat_docked[1], target_ee_quat_docked[2]])
        else:
            self.target_ee_pos = ee_pos_w.copy()
            self.target_ee_quat = ee_quat_w.copy()

        # Initialize previous haptic pose
        self._update_previous_haptic_pose()

        # Create collision pipeline
        if isinstance(self.solver, newton.solvers.SolverMuJoCo):
            self.collision_pipeline = None
            self.contacts = None
        else:
            self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, self.args)
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        # Set initial control targets
        self.control.joint_target_pos = wp.array(home_pose, dtype=wp.float32, device=self.device)
        self.control.joint_target_vel = wp.zeros(self.model.joint_coord_count, dtype=wp.float32, device=self.device)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, 2.0, 0.5), pitch=0.0, yaw=-90.0)

        self.capture()

        print("[INFO]: Haptic IK control ready")

    def _calibrate_haptic(self):
        """Calibrate haptic device by recording docked position."""
        if not self.haptic_cfg.enable_calibration:
            self.haptic_pos_offset = np.array([0.0, 0.0, 0.0])
            self.haptic_rot_offset = np.array([0.0, 0.0, 0.0, 1.0])
            print("[INFO]: Calibration disabled")
            return

        print(f"[INFO]: Calibrating haptic device...")
        print(f"[INFO]: Please dock your Omni Geomagic device and keep it still for {self.haptic_cfg.calibration_wait_time} seconds...")

        import time
        time.sleep(self.haptic_cfg.calibration_wait_time)

        # Get haptic device position when docked
        docked_haptic_pos_device = np.array(self.haptic_controller.get_scaled_position())
        docked_haptic_rot = np.array(self.haptic_controller.get_rotation())

        # Remap to robot coordinate system
        docked_haptic_pos = np.array([
            self.haptic_cfg.haptic_axis_signs[0] * docked_haptic_pos_device[self.haptic_cfg.haptic_axis_map[0]],
            self.haptic_cfg.haptic_axis_signs[1] * docked_haptic_pos_device[self.haptic_cfg.haptic_axis_map[1]],
            self.haptic_cfg.haptic_axis_signs[2] * docked_haptic_pos_device[self.haptic_cfg.haptic_axis_map[2]]
        ])

        # Remap rotation
        docked_haptic_rot_remapped = np.array([
            self.haptic_cfg.haptic_rot_axis_signs[0] * docked_haptic_rot[self.haptic_cfg.haptic_rot_axis_map[0]],
            self.haptic_cfg.haptic_rot_axis_signs[1] * docked_haptic_rot[self.haptic_cfg.haptic_rot_axis_map[1]],
            self.haptic_cfg.haptic_rot_axis_signs[2] * docked_haptic_rot[self.haptic_cfg.haptic_rot_axis_map[2]],
            docked_haptic_rot[3]
        ])
        docked_haptic_rot_remapped = docked_haptic_rot_remapped / np.linalg.norm(docked_haptic_rot_remapped)

        # Compute offset
        target_ee_pos_docked = np.array(self.haptic_cfg.target_ee_pos_docked)
        target_ee_quat_docked = self.haptic_cfg.get_target_ee_quat_docked()

        self.haptic_pos_offset = target_ee_pos_docked - docked_haptic_pos

        # Compute rotation offset using quaternion math
        docked_quat = docked_haptic_rot_remapped  # [x, y, z, w]
        target_quat = target_ee_quat_docked  # [x, y, z, w]

        # q_offset = q_target * q_docked^-1
        # Conjugate: [x, y, z, w]^-1 = [-x, -y, -z, w]
        docked_quat_conj = np.array([-docked_quat[0], -docked_quat[1], -docked_quat[2], docked_quat[3]])

        # Quaternion multiplication: q1 * q2
        def quat_mul(q1, q2):
            w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
            w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
            return np.array([
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
                w1*w2 - x1*x2 - y1*y2 - z1*z2
            ])

        self.haptic_rot_offset = quat_mul(target_quat, docked_quat_conj)

        print(f"[INFO]: Calibration complete!")
        print(f"[INFO]: Position offset: {self.haptic_pos_offset}")

    def _update_previous_haptic_pose(self):
        """Update previous haptic pose for delta computation."""
        raw_prev_pos_device = np.array(self.haptic_controller.get_scaled_position())
        raw_prev_rot_device = np.array(self.haptic_controller.get_rotation())

        # Remap position
        raw_prev_pos = np.array([
            self.haptic_cfg.haptic_axis_signs[0] * raw_prev_pos_device[self.haptic_cfg.haptic_axis_map[0]],
            self.haptic_cfg.haptic_axis_signs[1] * raw_prev_pos_device[self.haptic_cfg.haptic_axis_map[1]],
            self.haptic_cfg.haptic_axis_signs[2] * raw_prev_pos_device[self.haptic_cfg.haptic_axis_map[2]]
        ])
        self.prev_haptic_pos = raw_prev_pos + self.haptic_pos_offset

        # Remap rotation
        raw_prev_rot = np.array([
            self.haptic_cfg.haptic_rot_axis_signs[0] * raw_prev_rot_device[self.haptic_cfg.haptic_rot_axis_map[0]],
            self.haptic_cfg.haptic_rot_axis_signs[1] * raw_prev_rot_device[self.haptic_cfg.haptic_rot_axis_map[1]],
            self.haptic_cfg.haptic_rot_axis_signs[2] * raw_prev_rot_device[self.haptic_cfg.haptic_rot_axis_map[2]],
            raw_prev_rot_device[3]
        ])
        raw_prev_rot = raw_prev_rot / np.linalg.norm(raw_prev_rot)

        # Apply rotation offset
        def quat_mul(q1, q2):
            w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
            w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
            return np.array([
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
                w1*w2 - x1*x2 - y1*y2 - z1*z2
            ])

        self.prev_haptic_rot = quat_mul(raw_prev_rot, self.haptic_rot_offset)

    def _process_haptic_input(self):
        """Process haptic device input and update target EE pose."""
        # Get current haptic pose
        raw_haptic_pos_device = np.array(self.haptic_controller.get_scaled_position())
        raw_haptic_rot_device = np.array(self.haptic_controller.get_rotation())

        # Remap position
        raw_haptic_pos = np.array([
            self.haptic_cfg.haptic_axis_signs[0] * raw_haptic_pos_device[self.haptic_cfg.haptic_axis_map[0]],
            self.haptic_cfg.haptic_axis_signs[1] * raw_haptic_pos_device[self.haptic_cfg.haptic_axis_map[1]],
            self.haptic_cfg.haptic_axis_signs[2] * raw_haptic_pos_device[self.haptic_cfg.haptic_axis_map[2]]
        ])
        haptic_pos = raw_haptic_pos + self.haptic_pos_offset

        # Remap rotation
        raw_haptic_rot = np.array([
            self.haptic_cfg.haptic_rot_axis_signs[0] * raw_haptic_rot_device[self.haptic_cfg.haptic_rot_axis_map[0]],
            self.haptic_cfg.haptic_rot_axis_signs[1] * raw_haptic_rot_device[self.haptic_cfg.haptic_rot_axis_map[1]],
            self.haptic_cfg.haptic_rot_axis_signs[2] * raw_haptic_rot_device[self.haptic_cfg.haptic_rot_axis_map[2]],
            raw_haptic_rot_device[3]
        ])
        raw_haptic_rot = raw_haptic_rot / np.linalg.norm(raw_haptic_rot)

        # Apply rotation offset
        def quat_mul(q1, q2):
            w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
            w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
            return np.array([
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
                w1*w2 - x1*x2 - y1*y2 - z1*z2
            ])

        haptic_rot = quat_mul(raw_haptic_rot, self.haptic_rot_offset)

        # Compute position delta
        delta_pos = haptic_pos - self.prev_haptic_pos
        delta_pos = delta_pos * self.haptic_cfg.position_sensitivity

        # Clamp for safety
        delta_pos_magnitude = np.linalg.norm(delta_pos)
        if delta_pos_magnitude > self.haptic_cfg.max_position_delta:
            delta_pos = delta_pos * (self.haptic_cfg.max_position_delta / delta_pos_magnitude)

        # Compute rotation delta (relative rotation)
        # delta_q = curr_q * prev_q^-1
        prev_quat_conj = np.array([-self.prev_haptic_rot[0], -self.prev_haptic_rot[1],
                                   -self.prev_haptic_rot[2], self.prev_haptic_rot[3]])
        delta_quat = quat_mul(haptic_rot, prev_quat_conj)

        # Convert to axis-angle for scaling
        w, x, y, z = delta_quat[3], delta_quat[0], delta_quat[1], delta_quat[2]
        angle = 2.0 * np.arctan2(np.sqrt(x*x + y*y + z*z), w)

        # Apply sensitivity and clamp
        angle = angle * self.haptic_cfg.rotation_sensitivity
        if angle > self.haptic_cfg.max_rotation_delta:
            angle = self.haptic_cfg.max_rotation_delta

        # Convert back to quaternion
        if angle > 1e-6:
            axis = np.array([x, y, z]) / np.sqrt(x*x + y*y + z*z)
            delta_quat_scaled = np.array([
                axis[0] * np.sin(angle/2),
                axis[1] * np.sin(angle/2),
                axis[2] * np.sin(angle/2),
                np.cos(angle/2)
            ])
        else:
            delta_quat_scaled = np.array([0.0, 0.0, 0.0, 1.0])

        # Update target EE pose
        self.target_ee_pos += delta_pos

        # Convert to [w, x, y, z] for quaternion multiplication
        target_quat_wxyz = np.array([self.target_ee_quat[0], self.target_ee_quat[1],
                                     self.target_ee_quat[2], self.target_ee_quat[3]])
        delta_quat_wxyz = np.array([delta_quat_scaled[3], delta_quat_scaled[0],
                                   delta_quat_scaled[1], delta_quat_scaled[2]])

        def quat_mul_wxyz(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])

        new_target_quat = quat_mul_wxyz(delta_quat_wxyz, target_quat_wxyz)
        self.target_ee_quat = new_target_quat / np.linalg.norm(new_target_quat)

        # Update previous state
        self.prev_haptic_pos = haptic_pos.copy()
        self.prev_haptic_rot = haptic_rot.copy()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            if not isinstance(self.solver, newton.solvers.SolverMuJoCo):
                self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # Process haptic input
        self._process_haptic_input()

        # Update IK objectives
        self.pos_obj.set_target_position(0, wp.vec3(self.target_ee_pos[0],
                                                    self.target_ee_pos[1],
                                                    self.target_ee_pos[2]))
        self.rot_obj.set_target_rotation(0, wp.vec4(self.target_ee_quat[1],
                                                    self.target_ee_quat[2],
                                                    self.target_ee_quat[3],
                                                    self.target_ee_quat[0]))

        # Update joint positions for IK
        current_joint_q = self.model.joint_q.numpy()
        self.joint_q_2d.assign(current_joint_q.reshape(1, -1))

        # Solve IK
        self.ik_solver.step(self.joint_q_2d, self.joint_q_2d, iterations=50)

        # Get solved joint positions
        solved_joint_q = self.joint_q_2d.numpy()[0]

        # Update control targets
        self.control.joint_target_pos = wp.array(solved_joint_q.tolist(), dtype=wp.float32, device=self.device)

        # Update model joint positions (for FK)
        self.model.joint_q = wp.array(solved_joint_q.tolist(), dtype=wp.float32, device=self.device)

        # Run simulation
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        # Visualize target EE position as gizmo
        target_tf = wp.transform(
            wp.vec3(self.target_ee_pos[0], self.target_ee_pos[1], self.target_ee_pos[2]),
            wp.quat(self.target_ee_quat[1], self.target_ee_quat[2], self.target_ee_quat[3], self.target_ee_quat[0])
        )
        self.viewer.log_gizmo("target_ee", target_tf)

        # Compute FK for visualization
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.log_state(self.state_0)
        if self.contacts is not None:
            self.viewer.log_contacts(self.contacts, self.state_0)

        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver-type",
        type=str,
        default="xpbd",
        choices=["xpbd", "featherstone", "mujoco"],
        help="Solver type to use (xpbd, featherstone, or mujoco).",
    )

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
