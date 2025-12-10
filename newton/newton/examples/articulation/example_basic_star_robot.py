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
# Example Basic STAR Robot
#
# Demonstrates loading a STAR surgical robot from USD and controlling
# it with position control. The robot randomly generates target poses
# and reaches them within a time limit before switching to a new target.
#
# Command: python -m newton.examples basic_star_robot [--solver-type xpbd|featherstone|mujoco]
#
###########################################################################

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples


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

        # Path to STAR robot USD file
        star_usd_path = Path.home() / "sano/Warp-Surgical-isaac_integration/isaacLab/source/isaaclab_assets/isaaclab_assets/robots/star.usd"

        if not star_usd_path.exists():
            raise FileNotFoundError(f"STAR robot USD file not found at: {star_usd_path}")

        # Build the STAR robot from USD
        builder = newton.ModelBuilder()

        # Register MuJoCo custom attributes BEFORE building (critical for MuJoCo!)
        if solver_type == "mujoco":
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        builder.add_usd(
            str(star_usd_path),
            hide_collision_shapes=True,
        )
        builder.add_ground_plane()

        # Set control gains on builder BEFORE finalizing (all solvers)
        if solver_type == "mujoco":
            for i in range(len(builder.joint_target_ke)):
                builder.joint_target_ke[i] = 500.0
                builder.joint_target_kd[i] = 50.0
        elif solver_type == "featherstone":
            for i in range(len(builder.joint_target_ke)):
                builder.joint_target_ke[i] = 400.0
                builder.joint_target_kd[i] = 10.0
        elif solver_type == "xpbd":
            for i in range(len(builder.joint_target_ke)):
                builder.joint_target_ke[i] = 10000.0
                builder.joint_target_kd[i] = 2000.0

        # Finalize model
        self.model = builder.finalize()

        # Store device for later use
        self.device = wp.get_device()

        print(f"Loaded STAR robot with {self.model.joint_count} joints")
        print(f"Joint coordinates (DOF): {self.model.joint_coord_count}")
        print(f"Body count: {self.model.body_count}")
        print(f"Shape count: {self.model.shape_count}")

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
                iterations=16,  # 16 iterations per substep
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
            )
            print(f"Featherstone: update_mass_matrix_interval={self.sim_substeps}, substeps={self.sim_substeps}")
        elif solver_type == "mujoco":
            # MuJoCo solver (high performance, accurate physics)
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                disable_contacts=True,  # Disable contacts for cleaner simulation
            )
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

        # Control gains are already set on builder before finalize
        # Print confirmation based on solver type
        if isinstance(self.solver, newton.solvers.SolverMuJoCo):
            print("Using MuJoCo solver with gains (ke=500, kd=50) - set on builder")
        elif isinstance(self.solver, newton.solvers.SolverFeatherstone):
            print("Using Featherstone solver with gains (ke=400, kd=15) - set on builder")
        elif isinstance(self.solver, newton.solvers.SolverXPBD):
            print(f"Using {type(self.solver).__name__} solver with standard gains (ke=10000, kd=2000) - set on builder")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Initialize forward kinematics (not needed for MuJoCo)
        if not isinstance(self.solver, newton.solvers.SolverMuJoCo):
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Print initial body positions to help locate the robot
        if self.model.body_count > 0:
            print("\nInitial body positions:")
            body_q = self.state_0.body_q.numpy()
            for i in range(min(5, self.model.body_count)):
                print(f"  Body {i}: pos=({body_q[i][0]:.3f}, {body_q[i][1]:.3f}, {body_q[i][2]:.3f})")

        # Get initial joint positions from the model
        initial_joint_q = self.model.joint_q.numpy().copy()
        print(f"Initial joint positions: {initial_joint_q[:min(7, len(initial_joint_q))]}")

        # End-effector body index (assume last body is end-effector)
        self.ee_index = self.model.body_count - 1

        # Random target generation parameters
        self.target_duration = 2.0  # Time to reach each target (seconds)
        self.next_target_time = self.target_duration

        # Get joint limits from model (use defaults if unlimited)
        joint_limit_lower = self.model.joint_limit_lower.numpy()
        joint_limit_upper = self.model.joint_limit_upper.numpy()
        JOINT_LIMIT_UNLIMITED = 1e6  # Sentinel value for unlimited joints

        # Build joint limits list, using defaults if unlimited
        self.joint_limits_full = []
        for i in range(self.model.joint_coord_count):
            lower = joint_limit_lower[i]
            upper = joint_limit_upper[i]
            # Check if limits are unlimited (sentinel values)
            if abs(lower) >= JOINT_LIMIT_UNLIMITED or abs(upper) >= JOINT_LIMIT_UNLIMITED:
                # Use reasonable defaults for unlimited joints
                lower = -1.57  # -90 degrees
                upper = 1.57    # +90 degrees
            self.joint_limits_full.append((lower, upper))

        print(
            f"Full joint limits: "
            f"{self.joint_limits_full[:min(7, len(self.joint_limits_full))]}"
        )

        # Store home position for conservative target generation
        self.home_pose = initial_joint_q.copy()

        # Build conservative joint limits (50% of range centered around home)
        # This ensures targets are more likely to be within reach
        self.joint_limits = []
        for i in range(self.model.joint_coord_count):
            lower_full, upper_full = self.joint_limits_full[i]
            home_angle = self.home_pose[i]
            range_full = upper_full - lower_full

            # Use 50% of the full range, centered around home position
            range_conservative = range_full * 0.5
            lower_conservative = max(
                lower_full, home_angle - range_conservative / 2
            )
            upper_conservative = min(
                upper_full, home_angle + range_conservative / 2
            )

            self.joint_limits.append((lower_conservative, upper_conservative))

        print(
            f"Conservative joint limits (50% range): "
            f"{self.joint_limits[:min(7, len(self.joint_limits))]}"
        )

        # Generate initial random target
        self.target_pose = self._generate_random_target()
        print(f"Initial target: {self.target_pose[:min(7, len(self.target_pose))]}")

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

        # Set camera to view the STAR robot  - robot is at origin, extends upward
        # Position camera close and low to see the robot clearly
        self.viewer.set_camera(pos=wp.vec3(3.0, 1.5, 1.5), pitch=-15.0, yaw=-140.0)

        self.capture()

    def _generate_random_target(self):
        """Generate a random target pose within joint limits."""
        target = []
        for min_angle, max_angle in self.joint_limits:
            # Generate random angle within limits
            angle = np.random.uniform(min_angle, max_angle)
            target.append(angle)

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
            # Get initial joint positions from the model (home position)
            home_pose = self.model.joint_q.numpy().copy()
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
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

            print(f"\n[t={self.sim_time:.1f}s] Reset to home position")

            # Generate new random target
            self.target_pose = self._generate_random_target()
            print("New target generated:")
            num_to_print = min(7, len(self.target_pose))
            for i in range(num_to_print):
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
            num_joints = min(len(q), len(self.target_pose))
            errors = [abs(q[i] - self.target_pose[i]) for i in range(num_joints)]
            max_error = max(errors) if errors else 0.0

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
