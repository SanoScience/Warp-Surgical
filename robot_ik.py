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
def compute_ik_jacobian_fk(
    joint_q: wp.array(dtype=float),
    joint_target: wp.array(dtype=float),
    target_pos: wp.array(dtype=wp.vec3f),
    joint_type: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3f),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    end_effector_joint_id: int,
    ik_gain: float,
    dt: float,
    # Workspace arrays for intermediate calculations
    joint_world_transforms: wp.array(dtype=wp.transform),
    joint_world_positions: wp.array(dtype=wp.vec3f),
    joint_world_axes: wp.array(dtype=wp.vec3f)
):
    """
    Compute IK using forward kinematic and Jacobian transpose method.
    """
    tid = wp.tid()
    
    if tid == 0:
        num_joints = 7 # Assuming a 7-DOF arm

        # --- 1. Forward Kinematics Pass ---
        # Compute world transforms for each joint from base to end-effector
        
        # The first joint's parent is the base, so its transform is relative to the world
        current_transform = wp.transform_identity()

        for i in range(num_joints):
            q_start = joint_q_start[i]
            qd_start = joint_qd_start[i]
            
            # Get local joint transform based on current angle/position
            joint_local_transform = wp.transform_identity()
            joint_t = joint_type[i]

            if joint_t == newton.JOINT_REVOLUTE:
                angle = joint_q[q_start]
                axis = joint_axis[qd_start]
                q_rot = wp.quat_from_axis_angle(axis, angle)
                joint_local_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), q_rot)
            elif joint_t == newton.JOINT_PRISMATIC:
                displacement = joint_q[q_start]
                axis = joint_axis[qd_start]
                translation = axis * displacement
                joint_local_transform = wp.transform(translation, wp.quat_identity())

            # World transform of the joint is: parent_world_transform * local_joint_frame * joint_actuation
            # Note: URDF parser combines parent and joint frames into joint_X_p
            parent_world_transform = current_transform
            joint_frame_transform = joint_X_p[i]
            
            # Transform of the joint frame in world space
            world_joint_frame_transform = wp.transform_multiply(parent_world_transform, joint_frame_transform)
            
            # Transform of the actuated joint in world space
            world_actuated_transform = wp.transform_multiply(world_joint_frame_transform, joint_local_transform)
            
            # The 'current_transform' for the next iteration is the transform of this joint's child body
            current_transform = world_actuated_transform

            # Store results for Jacobian calculation
            joint_world_transforms[i] = world_joint_frame_transform
            joint_world_positions[i] = wp.transform_get_translation(world_joint_frame_transform)
            joint_world_axes[i] = wp.quat_rotate(wp.transform_get_rotation(world_joint_frame_transform), joint_axis[qd_start])

        # Compute final end-effector position
        ee_local_transform = joint_X_c[end_effector_joint_id]
        ee_world_transform = wp.transform_multiply(current_transform, ee_local_transform)
        current_ee_pos = wp.transform_get_translation(ee_world_transform)

        # --- 2. Jacobian Transpose IK ---
        pos_error = target_pos[0] - current_ee_pos
        
        # Iterate backwards from end-effector to base to apply updates
        for i in range(num_joints - 1, -1, -1):
            qd_start = joint_qd_start[i]
            if qd_start >= len(joint_target):
                continue

            joint_t = joint_type[i]
            
            # Vector from current joint to end-effector
            r = current_ee_pos - joint_world_positions[i]
            
            jacobian_col = wp.vec3()
            if joint_t == newton.JOINT_REVOLUTE:
                # Jacobian for revolute joint: Jv = world_axis x (p_ee - p_joint)
                jacobian_col = wp.cross(joint_world_axes[i], r)
            elif joint_t == newton.JOINT_PRISMATIC:
                # Jacobian for prismatic joint: Jv = world_axis
                jacobian_col = joint_world_axes[i]

            # Jacobian transpose method: delta_q = alpha * J^T * error
            joint_contribution = wp.dot(jacobian_col, pos_error)
            delta_q = ik_gain * joint_contribution

            # Update joint target
            current_target = joint_target[qd_start]
            new_target = current_target + delta_q * dt
            
            # Apply joint limits
            # new_target = wp.clamp(new_target, -3.14, 3.14)
            joint_target[qd_start] = new_target


@wp.kernel
def compute_ik_jacobian_fk_dls(
    joint_q: wp.array(dtype=float),
    joint_target: wp.array(dtype=float),
    target_pos: wp.array(dtype=wp.vec3f),
    joint_type: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3f),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    end_effector_joint_id: int,
    ik_gain: float,
    dt: float,
    # Workspace arrays for intermediate calculations
    joint_world_transforms: wp.array(dtype=wp.transform),
    joint_world_positions: wp.array(dtype=wp.vec3f),
    joint_world_axes: wp.array(dtype=wp.vec3f),
    # Debug outputs
    debug_ee_position: wp.array(dtype=wp.vec3f)
):
    """
    Compute IK using forward kinematic and Damped Least Squares (DLS) method.
    """
    tid = wp.tid()
    
    if tid == 0:
        num_joints = 7 # Assuming a 7-DOF arm
        dls_damping = 0.1 # Damping factor for singularity avoidance

        # --- 1. Forward Kinematics Pass ---
        # ... (This part remains the same) ...
        current_transform = wp.transform_identity()

        for i in range(num_joints):
            q_start = joint_q_start[i]
            qd_start = joint_qd_start[i]
            
            joint_local_transform = wp.transform_identity()
            joint_t = joint_type[i]

            if joint_t == newton.JOINT_REVOLUTE:
                angle = joint_q[q_start]
                axis = joint_axis[qd_start]
                q_rot = wp.quat_from_axis_angle(axis, angle)
                joint_local_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), q_rot)
            elif joint_t == newton.JOINT_PRISMATIC:
                displacement = joint_q[q_start]
                axis = joint_axis[qd_start]
                translation = axis * displacement
                joint_local_transform = wp.transform(translation, wp.quat_identity())

            parent_world_transform = current_transform
            joint_frame_transform = joint_X_p[i]
            world_joint_frame_transform = wp.transform_multiply(parent_world_transform, joint_frame_transform)
            world_actuated_transform = wp.transform_multiply(world_joint_frame_transform, joint_local_transform)
            current_transform = world_actuated_transform

            joint_world_transforms[i] = world_joint_frame_transform
            joint_world_positions[i] = wp.transform_get_translation(world_joint_frame_transform)
            joint_world_axes[i] = wp.quat_rotate(wp.transform_get_rotation(world_joint_frame_transform), joint_axis[qd_start])

        # Compute final end-effector position
        ee_local_transform = joint_X_c[end_effector_joint_id]
        ee_world_transform = wp.transform_multiply(current_transform, ee_local_transform)
        current_ee_pos = wp.transform_get_translation(ee_world_transform)
        debug_ee_position[0] = current_ee_pos # Output for debug rendering

        # --- 2. Damped Least Squares (DLS) IK ---
        pos_error = target_pos[0] - current_ee_pos
        
        # Iterate backwards from end-effector to base to apply updates
        for i in range(num_joints - 1, -1, -1):
            qd_start = joint_qd_start[i]
            if qd_start >= len(joint_target):
                continue

            joint_t = joint_type[i]
            
            r = current_ee_pos - joint_world_positions[i]
            
            jacobian_col = wp.vec3()
            if joint_t == newton.JOINT_REVOLUTE:
                jacobian_col = wp.cross(joint_world_axes[i], r)
            elif joint_t == newton.JOINT_PRISMATIC:
                jacobian_col = joint_world_axes[i]

            # DLS formula: delta_q = J^T * (J*J^T + lambda^2*I)^-1 * error
            # For a single column, this simplifies significantly
            j_jt = wp.dot(jacobian_col, jacobian_col)
            lambda_sq = dls_damping * dls_damping
            
            # Simplified inverse for the single-column case
            joint_contribution = wp.dot(jacobian_col, pos_error) / (j_jt + lambda_sq)
            delta_q = ik_gain * joint_contribution

            # Update joint target
            current_target = joint_target[qd_start]
            new_target = current_target + delta_q * dt
            
            # Apply joint limits
            new_target = wp.clamp(new_target, -2.8973, 2.8973)
            joint_target[qd_start] = new_target

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
            parse_visuals_as_colliders=False,
            ignore_inertial_definitions=False
        )
        
        # Set initial joint configuration
        #articulation_builder2.joint_q[:7] = [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]
        #articulation_builder2.joint_target[:7] = articulation_builder2.joint_q[:7]
        
        stiffness = 500
        damping = 2.0 * math.sqrt(stiffness)

        # Configure joint control
        for i in range(7):  # First 7 joints are the arm
            articulation_builder2.joint_dof_mode[i] = newton.JOINT_MODE_TARGET_POSITION
            #articulation_builder2.joint_dof_mode[i] = newton.JOINT_MODE_TARGET_VELOCITY
            #articulation_builder2.joint_dof_mode[i] = newton.JOINT_MODE_NONE
            articulation_builder2.joint_target[i] = 0.0

            articulation_builder2.joint_target_ke[i] = stiffness
            articulation_builder2.joint_target_kd[i] = damping

        for i in range(len(articulation_builder2.body_mass)):
            articulation_builder2.body_mass[i] = 0

        
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

        self.sim_substeps = 20
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_envs = num_envs

        builder.add_builder(articulation_builder2, xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()))
        #plane_body_id = builder.add_body(
        #    xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
        #    mass=0.0,
        #    armature=0.0
        #)
        #builder.add_shape_plane(body=plane_body_id)

        np.set_printoptions(suppress=True)
        # finalize model
        self.model = builder.finalize()
        
        # Find end-effector body ID 
        self.end_effector_joint_id = 6
        
        # IK parameters
        self.ik_damping = 0.95 * 5.0
        self.ik_enabled = True
        
        # Joint target buffer for IK
        self.joint_targets = wp.array(self.model.joint_target, dtype=float, device=wp.get_device())

        # self.solver = newton.solvers.XPBDSolver(self.model, iterations=16)
        self.solver = newton.solvers.FeatherstoneSolver(self.model)

        max_joints = 10
        self.debug_joint_positions = wp.array([wp.vec3(0.0, 0.0, 0.0)] * max_joints, dtype=wp.vec3, device=wp.get_device())
        self.debug_joint_rotations = wp.array([wp.quat_identity()] * max_joints, dtype=wp.quat, device=wp.get_device())
        self.debug_joint_axes = wp.array([wp.vec3(0.0, 0.0, 0.0)] * max_joints, dtype=wp.vec3, device=wp.get_device())
        self.debug_ee_position = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=wp.get_device())
        self.debug_transforms = wp.array([wp.transform_identity()] * max_joints, dtype=wp.transform, device=wp.get_device())
        
        # Workspace arrays for IK kernel
        self.ik_joint_world_transforms = wp.empty(shape=(max_joints,), dtype=wp.transform, device=wp.get_device())
        self.ik_joint_world_positions = wp.empty(shape=(max_joints,), dtype=wp.vec3, device=wp.get_device())
        self.ik_joint_world_axes = wp.empty(shape=(max_joints,), dtype=wp.vec3, device=wp.get_device())
        
        # Add debug flags
        self.show_joint_debug = False
        self.show_fk_chain = False
        self.show_ik_debug = True

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
                    compute_ik_jacobian_fk_dls,
                    dim=1,
                    inputs=[
                        self.state_0.joint_q,
                        self.joint_targets,
                        self.ik_target_buffer,
                        self.model.joint_type,
                        self.model.joint_X_p,
                        self.model.joint_X_c,
                        self.model.joint_axis,
                        self.model.joint_q_start,
                        self.model.joint_qd_start,
                        self.end_effector_joint_id,
                        self.ik_damping * 1.0,
                        self.sim_dt,
                        # Workspace arrays
                        self.ik_joint_world_transforms,
                        self.ik_joint_world_positions,
                        self.ik_joint_world_axes,
                        self.debug_ee_position
                    ],
                    device=self.state_0.joint_q.device,
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

        self.print_debug_info()

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
                    pos=[self.haptic_pos_right[0], self.haptic_pos_right[1], self.haptic_pos_right[2]],
                    rot=wp.quat_identity(),
                    color=[0.0, 1.0, 0.0],  # Green for haptic position
                    radius=0.5,
                )
                
                # Render IK target position
                ik_target = self.ik_target_buffer.numpy()[0]
                self.renderer.render_sphere(
                    "ik_target_sphere",
                    pos=[ik_target[0], ik_target[1], ik_target[2]],
                    rot=wp.quat_identity(),
                    color=[1.0, 0.0, 0.0],  # Red for IK target
                    radius=1.0,
                )

                # Debug visualization for IK and kinematics
                if self.show_joint_debug:
                    self.render_joint_debug()
                
                if self.show_fk_chain:
                    self.render_fk_chain()
                
                if self.show_ik_debug:
                    self.render_ik_debug()

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
        ik_target = wp.vec3(haptic_pos[0], haptic_pos[1] + 3.0, haptic_pos[2] + 3.0)  # Offset for reachable workspace
        wp.copy(self.ik_target_buffer, wp.array([ik_target], dtype=wp.vec3, device=wp.get_device()))

    def toggle_ik(self):
        """Toggle IK control on/off."""
        self.ik_enabled = not self.ik_enabled
        print(f"IK control: {'enabled' if self.ik_enabled else 'disabled'}")

    def set_ik_damping(self, damping):
        """Set IK damping factor (0.0 to 1.0)."""
        self.ik_damping = max(0.0, min(1.0, damping))
        print(f"IK damping set to: {self.ik_damping}")

    def render_joint_debug(self):
        """Render detailed joint information"""
        joint_q = self.state_0.joint_q.numpy()
        joint_targets = self.joint_targets.numpy()
        debug_positions = self.debug_joint_positions.numpy()
        debug_axes = self.debug_joint_axes.numpy()
        
        # Render each joint
        for i in range(min(7, len(debug_positions))):  # First 7 joints
            pos = debug_positions[i]
            
            # Joint position sphere
            color = [0.0, 0.0, 1.0] if i < len(joint_q) else [0.5, 0.5, 0.5]
            self.renderer.render_sphere(
                f"debug_joint_{i}",
                pos=pos,
                rot=wp.quat_identity(),
                color=color,
                radius=1.0,
            )
            
            # Joint axis
            if i < len(debug_axes):
                axis = debug_axes[i]
                axis_end = pos + axis * 3.0
                self.renderer.render_line_strip(
                    f"debug_joint_axis_{i}",
                    vertices=[pos, axis_end],
                    color=[1.0, 1.0, 0.0],  # Yellow
                    radius=0.3,
                )
            
            # Current vs target joint angle (for revolute joints)
            if i < len(joint_q) and i < len(joint_targets):
                current_angle = joint_q[i] if i < len(joint_q) else 0.0
                target_angle = joint_targets[i] if i < len(joint_targets) else 0.0
                
                # Visual indicator of joint angle difference
                angle_diff = abs(target_angle - current_angle)
                error_color = [min(1.0, angle_diff), max(0.0, 1.0 - angle_diff), 0.0]
                
                self.renderer.render_sphere(
                    f"debug_joint_error_{i}",
                    pos=pos + [0, 0, 2],  # Offset above joint
                    rot=wp.quat_identity(),
                    color=error_color,
                    radius=0.5,
                )

    def render_fk_chain(self):
        """Render forward kinematics chain"""
        debug_positions = self.debug_joint_positions.numpy()
        
        # Draw lines connecting joints in the kinematic chain
        for i in range(min(6, len(debug_positions) - 1)):
            start_pos = debug_positions[i]
            end_pos = debug_positions[i + 1]
            
            self.renderer.render_line_strip(
                f"fk_chain_{i}",
                vertices=[start_pos, end_pos],
                color=[0.0, 1.0, 1.0],  # Cyan
                radius=0.5,
            )
        
        # Render computed end-effector position
        ee_pos = self.debug_ee_position.numpy()[0]
        self.renderer.render_sphere(
            "debug_ee_position",
            pos=ee_pos,
            rot=wp.quat_identity(),
            color=[1.0, 0.0, 1.0],  # Magenta
            radius=1.0,
        )
        
        # Draw line from last joint to end-effector
        if len(debug_positions) > 0:
            last_joint_pos = debug_positions[min(6, len(debug_positions) - 1)]
            self.renderer.render_line_strip(
                "ee_link",
                vertices=[last_joint_pos, ee_pos],
                color=[1.0, 0.0, 1.0],  # Magenta
                radius=0.7,
            )

    def render_ik_debug(self):
        """Render IK-specific debug information"""
        ik_target = self.ik_target_buffer.numpy()[0]
        ee_pos = self.debug_ee_position.numpy()[0]
        
        # Draw error vector from end-effector to target
        self.renderer.render_line_strip(
            "ik_error_vector",
            vertices=[ee_pos, ik_target],
            color=[1.0, 0.5, 0.0],  # Orange
            radius=0.4,
        )
        
        # Display numerical debug info as spheres with different colors
        error_magnitude = np.linalg.norm(ik_target - ee_pos)
        
        # Error magnitude visualization
        error_normalized = min(1.0, error_magnitude / 10.0)  # Normalize to 0-1 range
        error_color = [error_normalized, 1.0 - error_normalized, 0.0]
        
        self.renderer.render_sphere(
            "ik_error_magnitude",
            pos=ee_pos + [0, 0, 5],
            rot=wp.quat_identity(),
            color=error_color,
            radius=1.0,
        )

    def print_debug_info(self):
        """Print detailed debug information to console"""
        if self.sim_time % 1.0 < self.frame_dt:  # Print once per second
            joint_q = self.state_0.joint_q.numpy()
            joint_targets = self.joint_targets.numpy()
            ee_pos = self.debug_ee_position.numpy()[0]
            ik_target = self.ik_target_buffer.numpy()[0]
            
            print(f"\n=== Debug Info at t={self.sim_time:.2f} ===")
            print(f"Joint angles: {joint_q[:7]}")
            print(f"Joint targets: {joint_targets[:7]}")
            print(f"EE position: {ee_pos}")
            print(f"IK target: {ik_target}")
            print(f"Error magnitude: {np.linalg.norm(ik_target - ee_pos):.3f}")
            print(f"IK enabled: {self.ik_enabled}")



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
