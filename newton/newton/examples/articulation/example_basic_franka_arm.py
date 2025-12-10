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
# Example Basic Franka Arm
#
# Demonstrates loading a Franka FR3 robot from URDF. The robot randomly
# generates target poses and reaches them within a time limit before
# switching to a new target.
#
# Command: python -m newton.examples basic_franka_arm
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils


class Example:
    def __init__(self, viewer, args=None):
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        # Get solver type early to set appropriate substeps
        solver_type = getattr(args, "solver_type", "xpbd") if args else "xpbd"
        # Adjust substeps based on solver type
        if solver_type == "xpbd":
            self.sim_substeps = 64  # XPBD needs more substeps
        elif solver_type == "featherstone":
            self.sim_substeps = 10  # Featherstone works well with fewer substeps
        elif solver_type == "mujoco":
            self.sim_substeps = 10  # MuJoCo works well with fewer substeps
        else:
            self.sim_substeps = 64  # Default
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        # Get solver type from command line argument, default to "xpbd"
        # (Note: solver_type was already determined above for substeps calculation)
        # Build the Franka FR3 robot from URDF
        builder = newton.ModelBuilder()

        # Register MuJoCo custom attributes BEFORE building (critical for MuJoCo!)
        if solver_type == "mujoco":
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        # Add Franka robot with fixed base, positioned above ground
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(
                wp.vec3(0.0, 0.0, 0.0),  # Position at origin
                wp.quat_identity(),
            ),
            floating=False,  # Fixed base
            enable_self_collisions=False,
        )

        # Set control gains on builder BEFORE finalizing (all solvers)
        # This is the recommended approach for consistency
        if solver_type == "mujoco":
            # Set gains matching UR10 example: ke=500, kd=50
            for i in range(len(builder.joint_target_ke)):
                builder.joint_target_ke[i] = 500.0
                builder.joint_target_kd[i] = 50.0
        elif solver_type == "featherstone":
            # Featherstone needs explicit control gains
            for i in range(len(builder.joint_target_ke)):
                builder.joint_target_ke[i] = 400.0
                builder.joint_target_kd[i] = 15.0
        elif solver_type == "xpbd":
            # Standard gains for XPBD
            for i in range(len(builder.joint_target_ke)):
                builder.joint_target_ke[i] = 10000.0
                builder.joint_target_kd[i] = 2000.0

        # Add ground plane AFTER robot (important!)
        builder.add_ground_plane()

        # Finalize model
        self.model = builder.finalize()

        # Store device for later use
        self.device = wp.get_device()

        print(f"Loaded Franka robot with {self.model.joint_count} joints")
        print(f"Joint coordinates (DOF): {self.model.joint_coord_count}")

        # Set initial joint positions BEFORE creating solver (critical for Featherstone!)
        # Use a safe home position similar to cloth example
        home_pose = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
        self.model.joint_q = wp.array(home_pose, dtype=wp.float32, device=self.device)
        self.model.joint_qd.zero_()
        print(f"Set initial joint positions: {home_pose[:7]}")

        # Create solver based on command line argument
        print(f"Creating {solver_type.upper()} solver...")

        if solver_type == "xpbd":
            self.model.ground = True
            # CRITICAL: Lock the base body by setting inverse mass to zero
            # This makes the base kinematic (immovable) for XPBD solver
            body_inv_mass = self.model.body_inv_mass.numpy()
            body_inv_inertia = self.model.body_inv_inertia.numpy()

            # Find the base body (first body of the robot, not the ground plane)
            # Ground plane is body -1, robot base is body 0
            base_body_idx = 0
            body_inv_mass[base_body_idx] = 0.0  # Zero inverse mass = infinite mass
            body_inv_inertia[base_body_idx] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

            self.model.body_inv_mass = wp.array(body_inv_mass, dtype=wp.float32, device=self.device)
            self.model.body_inv_inertia = wp.array(body_inv_inertia, dtype=wp.mat33, device=self.device)
            print(f"Set base body (idx {base_body_idx}) to kinematic (inv_mass=0)")
            
            # XPBD solver with parameters from working IsaacLab example
            self.solver = newton.solvers.SolverXPBD(
                self.model,
                iterations=16,  # 10 iterations per substep
                angular_damping=0.1,
                joint_linear_compliance=0.0,  # Rigid joint limits
                joint_angular_compliance=0.0,  # Rigid joint limits
                joint_linear_relaxation=0.7,
                joint_angular_relaxation=0.4,
            )
        elif solver_type == "featherstone":
            # Featherstone solver
            self.solver = newton.solvers.SolverFeatherstone(
                self.model,
                update_mass_matrix_interval=self.sim_substeps,
                # angular_damping=0.1,
                # friction_smoothing=4.0,
            )
            print(f"Featherstone: update_mass_matrix_interval=4, substeps={self.sim_substeps}")
        elif solver_type == "mujoco":
            # MuJoCo solver (high performance, accurate physics)
            # Similar to UR10 example: disable contacts for cleaner simulation
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                disable_contacts=True,  # Disable contacts like UR10 example
            )
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

        # Control gains are already set on builder before finalize
        # Print confirmation based on solver type
        if isinstance(self.solver, newton.solvers.SolverMuJoCo):
            print("Using MuJoCo solver with gains (ke=500, kd=50) - set on builder")
        elif isinstance(self.solver, newton.solvers.SolverFeatherstone):
            print("Using Featherstone solver with gains (ke=3000, kd=100) - set on builder")
        elif isinstance(self.solver, newton.solvers.SolverXPBD):
            print(f"Using {type(self.solver).__name__} solver with standard gains (ke=10000, kd=2000) - set on builder")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Initialize forward kinematics (not needed for MuJoCo)
        if not isinstance(self.solver, newton.solvers.SolverMuJoCo):
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # End-effector body index (Franka hand)
        self.ee_index = 10

        # Random target generation parameters
        self.target_duration = 2.0  # Time to reach each target (seconds)
        self.next_target_time = self.target_duration

        # Joint limits for the 7 arm joints (reasonable workspace)
        # These are conservative limits to keep the robot in a safe workspace
        self.joint_limits = [
            (-2.8973, 2.8973),   # Joint 0
            (-1.7628, 1.7628),   # Joint 1
            (-2.8973, 2.8973),   # Joint 2
            (-3.0718, -0.0698),  # Joint 3 (limited to avoid singularities)
            (-2.8973, 2.8973),   # Joint 4
            (-0.0175, 3.7525),   # Joint 5
            (-2.8973, 2.8973),   # Joint 6
        ]

        # Gripper positions (fixed)
        self.gripper_pos = [0.04, 0.04]

        # Initialize forward kinematics with home pose (already set earlier)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Generate initial random target
        self.target_pose = self._generate_random_target()
        print(f"Initial target: {self.target_pose[:7]}")

        # Set control targets
        self.control.joint_target_pos = wp.array(
            self.target_pose, dtype=wp.float32, device=self.device
        )
        self.control.joint_target_vel = wp.zeros(
            self.model.joint_coord_count, dtype=wp.float32, device=self.device
        )

        # Compute target end-effector position
        self.target_ee_transform = self._compute_target_ee_transform()

        # Create collision pipeline (not needed for MuJoCo with disable_contacts=True)
        if isinstance(self.solver, newton.solvers.SolverMuJoCo):
            self.collision_pipeline = None
            self.contacts = None
        else:
            self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, self.args)
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

        # Set camera to view the Franka robot
        self.viewer.set_camera(pos=wp.vec3(0.0, 2.0, 0.5), pitch=0.0, yaw=-90.0)

        self.capture()

    def _generate_random_target(self):
        """Generate a random target pose within joint limits."""
        target = []
        for min_angle, max_angle in self.joint_limits:
            # Generate random angle within limits
            angle = np.random.uniform(min_angle, max_angle)
            target.append(angle)

        # Add gripper positions
        target.extend(self.gripper_pos)

        return target

    def _compute_target_ee_transform(self):
        """Compute the end-effector transform for the target joint configuration."""
        # Create a temporary state to compute forward kinematics for target pose
        target_joint_q = wp.array(self.target_pose, dtype=wp.float32, device=self.device)
        target_joint_qd = wp.zeros(self.model.joint_coord_count, dtype=wp.float32, device=self.device)

        temp_state = self.model.state()
        newton.eval_fk(self.model, target_joint_q, target_joint_qd, temp_state)

        # Get the end-effector transform
        body_q_np = temp_state.body_q.numpy()
        ee_transform = wp.transform(*body_q_np[self.ee_index])

        return ee_transform

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

            # Apply forces from viewer (e.g., user interaction)
            self.viewer.apply_forces(self.state_0)

            # Update contacts (not needed for MuJoCo with disable_contacts=True)
            if not isinstance(self.solver, newton.solvers.SolverMuJoCo):
                self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        # Check if it's time to switch to a new target
        if self.sim_time >= self.next_target_time:
            # Reset robot to home position before generating new target
            home_pose = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
            home_pose_array = wp.array(home_pose, dtype=wp.float32, device=self.device)

            # Update model joint positions and velocities
            self.model.joint_q = home_pose_array
            self.model.joint_qd.zero_()

            # Reset BOTH states (they get swapped during simulation)
            # State 0
            self.state_0.joint_q.assign(home_pose_array)
            self.state_0.joint_qd.zero_()
            self.state_0.body_qd.zero_()
            self.state_0.clear_forces()

            # State 1
            self.state_1.joint_q.assign(home_pose_array)
            self.state_1.joint_qd.zero_()
            self.state_1.body_qd.zero_()
            self.state_1.clear_forces()

            # Re-initialize forward kinematics with home pose for both states
            # For MuJoCo, we still need to compute FK to ensure body_q is correct
            # (MuJoCo will sync its internal state on the next step, but body_q should be correct)
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

            print(f"\n[t={self.sim_time:.1f}s] Reset to home position")

            # Generate new random target
            self.target_pose = self._generate_random_target()
            print(f"New target generated:")
            for i in range(7):
                print(f"  Joint {i}: {self.target_pose[i]:.3f} rad")

            # Update control target
            self.control.joint_target_pos = wp.array(
                self.target_pose, dtype=wp.float32, device=self.device
            )
            # Reset control target velocities to zero (important for MuJoCo)
            self.control.joint_target_vel.zero_()

            # Compute target end-effector position
            self.target_ee_transform = self._compute_target_ee_transform()

            # Schedule next target change
            self.next_target_time = self.sim_time + self.target_duration

    def render(self):
        # Print joint angles every 1 second for debugging
        if int(self.sim_time / 1.0) != int((self.sim_time - self.frame_dt) / 1.0):
            newton.eval_ik(self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd)
            q = self.state_0.joint_q.numpy()

            # Calculate error to target
            errors = [abs(q[i] - self.target_pose[i]) for i in range(7)]
            max_error = max(errors)

            print(f"[t={self.sim_time:.1f}s] Max error: {max_error:.3f} rad, Time to next target: {self.next_target_time - self.sim_time:.1f}s")

        self.viewer.begin_frame(self.sim_time)

        # Visualize the target end-effector position as a gizmo
        self.viewer.log_gizmo("target_ee", self.target_ee_transform)

        self.viewer.log_state(self.state_0)
        if self.contacts is not None:
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver-type",
        type=str,
        default="xpbd",
        choices=["xpbd", "featherstone", "mujoco"],
        help="Solver type to use (xpbd, featherstone, or mujoco).",
    )

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create viewer and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
