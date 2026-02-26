import os
import argparse
import atexit
import newton
import newton.examples
from newton.selection import ArticulationView

from newton.ik import IKSolver, IKPositionObjective, IKRotationObjective, IKJointLimitObjective

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as R
import zlib

from data_io import DataLoader, InstrumentData
from ik_utils import build_multi_robot_mimic_objective, lock_constant_joints
import shutil

from parquet_reader import load_parquet_instrument_data, overwrite_parquet_joint_columns
from update_lerobot_meta import update_lerobot_metadata
from recording import RecordingManager

DEFAULT_PSM_URDF = os.path.join(os.path.dirname(__file__), "assets", "psm", "psm_RL.urdf")
URDF_SCALE = 10.0
PARQUET_POS_SCALE = 12.5

# Rotate all loaded data from Y-up into the simulation's default Z-up frame.
SCENE_ROT = R.from_euler("x", 90.0, degrees=True)
SCENE_ROT_QUAT = SCENE_ROT.as_quat().astype(np.float32)
TOOL_TIP_OFFSET = wp.vec3(0.0, 0.0102 * URDF_SCALE, 0.0)  # Offset from yaw link to tool tip (scaled with URDF).
RCM_OFFSET = np.array([0.0, 4.864, 0.0], dtype=np.float32)  # Remote Center of Motion offset from base_link (unscaled)
DEFAULT_CAMERA_POS = np.array([0.5, 1.0, 5.0], dtype=np.float32)
DEFAULT_CAMERA_POS_PARQUET = np.array([0.7, -0.2, -2.0], dtype=np.float32)
GAIN = 0.1

# IK solver parameters
IK_ITERATIONS = 50
IK_STEP_SIZE = 1.0
IK_ROTATION_WEIGHT = 0.05
UI_SCROLLBAR_SIZE = 18.0
UI_GRAB_MIN_SIZE = 18.0
EE_MATCH_TOLERANCE = 0.02

# End effector link name
EE_LINK_NAME = "psm_tool_yaw_link"
JAW_JOINT_NAME = 'psm_tool_gripper2_joint'
LOCKED_JOINT_NAMES = (
    "psm_tool_pitch_joint",
    "psm_tool_yaw_joint",
    "psm_tool_gripper1_joint",
    "psm_tool_gripper2_joint",
)

# Movement step sizes
POSITION_STEP = 0.005   # 5mm for sliders
KEYBOARD_STEP = 0.0005  # 0.5mm per frame for keyboard (runs at 60fps)
JAW_STEP = 0.01         # Jaw angle step per frame

NUM_ROBOTS = 4  # Fixed number of robots/instruments

ROBOT_BASE_ROTATIONS = [
    (20.0, 0.0, 60.0),
    (-20.0, 0.0, 135.0),
    (20.0, 10.0, 30.0),
    (25.0, -25.0, -90.0),
]

PARQUET_ENTRY_POINTS = [
    (0.68, 5.069, -2.438),
    (0.333, 5.154, -2.574),
    (-0.26, 5.151, -2.631),
    (-0.582, 5.148, -2.75),
]

# Mimic joint mapping: follower_joint -> (leader_joint, multiplier, offset)
MIMIC_JOINT_MAPPING = {
    'psm_pitch_end_joint': ('psm_pitch_back_joint', 1.0, 0.0),
    'psm_pitch_bottom_joint': ('psm_pitch_back_joint', -1.0, 0.0),
    'psm_pitch_top_joint': ('psm_pitch_back_joint', -1.0, 0.0),
    'psm_pitch_front_joint': ('psm_pitch_back_joint', 1.0, 0.0),
    'psm_tool_gripper1_joint': ('psm_tool_gripper2_joint', -1.0, 0.0),
}

class DataPlayback:

    def __init__(self, data_dir, viewer, fps=30, parquet_path=None, parquet_recompute=False):
        self.data_dir = data_dir
        self.fps = fps
        self.frame_dt = 1.0 / self.fps
        self.viewer = viewer
        self.sim_time = 0.0
        self.tissue_types = ['Liver', 'Fat', 'Gallbladder']
        self.use_parquet = parquet_path is not None
        self.parquet_recompute = bool(parquet_recompute)
        self.parquet_data = None
        self.parquet_path = parquet_path
        self.parquet_entry_points = None
        self._current_frame_idx = 0
        self._frame_override_active = False
        self._frame_override_idx = 0
        self._parquet_action_joints = None
        self._parquet_observation_joints_pos = None
        self._parquet_observation_joints_vel = None
        self.model_scale = URDF_SCALE
        self.rcm_offset = RCM_OFFSET
        self.tool_tip_offset = wp.vec3(0.0, 0.0102 * self.model_scale, 0.0)

        # Find all frames first (before building model)
        if self.use_parquet:
            self.parquet_data = load_parquet_instrument_data(parquet_path)
            self.num_robots = len(self.parquet_data.instrument_ids)
            self.frames = list(range(self.parquet_data.action.shape[0]))
            instruments = self._load_parquet_instruments(0, init_entry_points=True)
            self.render_tissues = False
        else:
            self.num_robots = NUM_ROBOTS
            self.data_loader = DataLoader(self.data_dir, self.tissue_types, SCENE_ROT)
            self.frames = self.data_loader.find_frames()
            instruments = self.data_loader.load_instruments(self.frames[0])
            self.render_tissues = True
        self.num_frame = 0
        if not self.frames:
            if self.use_parquet:
                raise ValueError(f"No frame data found in parquet file: {parquet_path}")
            raise ValueError(f"No frame data found in {data_dir}")
        print(f"Found {len(self.frames)} frames in {data_dir}")

        # Load first frame's instrument data to get entry points
        print(instruments)
        if len(instruments) < self.num_robots:
            raise ValueError(f"Expected {self.num_robots} instruments, found {len(instruments)}")

        print(f"Creating single model with {self.num_robots} robots")
        if self.use_parquet and instruments:
            print("Initial parquet instrument data (frame 0):")
            for i, instr in enumerate(instruments[: self.num_robots]):
                print(f"  Instr {i}: pos={instr.position}, rot={instr.rotation}, entry={instr.entry_point}")

        # Build single model containing all 4 robots
        entry_points = self._get_initial_entry_points(instruments)
        self.initial_entry_points = [np.array(p, dtype=np.float32) for p in entry_points]
        self.base_rotations_ref = [
            tuple(ROBOT_BASE_ROTATIONS[i]) if i < len(ROBOT_BASE_ROTATIONS) else (0.0, 0.0, 0.0)
            for i in range(self.num_robots)
        ]
        # Local delta rotations applied on top of the reference orientation.
        self.base_rotations_local = [(0.0, 0.0, 0.0) for _ in range(self.num_robots)]
        self.base_rotations_pending = [tuple(r) for r in self.base_rotations_local]
        self.base_positions = self._compute_base_positions(self.initial_entry_points, self.base_rotations_ref)
        self.ee_position_errors = [np.nan for _ in range(self.num_robots)]

        # Gizmo visibility toggles
        if self.use_parquet and self.parquet_recompute:
            self.show_target_gizmos = False
            self.show_ee_gizmos = False
            self.show_entry_gizmos = False
        else:
            self.show_target_gizmos = True
            self.show_ee_gizmos = True
            self.show_entry_gizmos = False

        # Recording state
        self.recorder = RecordingManager(
            viewer=self.viewer,
            output_path=os.path.join(self.data_dir, "playback.mp4"),
            frames_dir=os.path.join(self.data_dir, "frames"),
            fps=self.fps,
        )
        atexit.register(self.recorder.stop)

        # Build model state, kinematics views, and IK solver.
        self._initialize_model(self.initial_entry_points)
        self._register_base_rotation_ui()

        # Initialize Warp
        wp.init()
        wp.set_module_options({"enable_backward": False})

        self.current_frame = 0
        self.paused = False
        self.current_tissues = []
        self._tissue_mesh_cache = {}
        self._tissue_instance_cache = {}
        self._tissue_version_cache = {}
        self._active_tissue_objects = set()

    def _build_model_with_all_robots(self, entry_points: list) -> newton.Model:
        """Build a single model containing all robots."""
        builder = newton.ModelBuilder()

        for i, entry_point in enumerate(entry_points):
            if hasattr(self, "_compose_base_rotation"):
                base_rot_scipy = self._compose_base_rotation(i)
            else:
                base_rotations_ref = ROBOT_BASE_ROTATIONS
                euler = base_rotations_ref[i] if i < len(base_rotations_ref) else (0.0, 0.0, 0.0)
                base_rot_scipy = R.from_euler("xyz", euler, degrees=True)

            quat_np = base_rot_scipy.as_quat().astype(np.float32)
            base_rotation = wp.quat(float(quat_np[0]), float(quat_np[1]), float(quat_np[2]), float(quat_np[3]))

            rcm_offset_world = base_rot_scipy.apply(self.rcm_offset).astype(np.float32)
            base_positions = getattr(self, "base_positions", None)
            if base_positions is not None and len(base_positions) > i:
                base_position = np.array(base_positions[i], dtype=np.float32)
            else:
                # Align the RCM with the entry point after applying the base rotation.
                base_position = np.array(entry_point, dtype=np.float32) - rcm_offset_world

            rcm_point = base_position + rcm_offset_world
            print(f"  Robot {i}: Placing base at {base_position} (RCM at {rcm_point})")

            scale = self.model_scale
            builder.add_urdf(
                DEFAULT_PSM_URDF,
                scale=scale,
                xform=wp.transform(tuple(base_position), base_rotation),
                floating=False,
                collapse_fixed_joints=False,
            )

        model = builder.finalize()
        print(f"Model created with {len(model.body_key)} bodies, {len(model.joint_key)} joints")
        print(f"Articulation keys: {model.articulation_key}")
        return model

    def _get_initial_entry_points(self, instruments):
        return [np.array(instruments[i].entry_point) for i in range(self.num_robots)]

    def _compose_base_rotation(self, i):
        base_rotations_ref = getattr(self, "base_rotations_ref", ROBOT_BASE_ROTATIONS)
        base_rotations_local = getattr(
            self,
            "base_rotations_local",
            [(0.0, 0.0, 0.0) for _ in range(len(base_rotations_ref))],
        )
        ref_euler = base_rotations_ref[i] if i < len(base_rotations_ref) else (0.0, 0.0, 0.0)
        local_euler = base_rotations_local[i] if i < len(base_rotations_local) else (0.0, 0.0, 0.0)
        rot_ref = R.from_euler("xyz", ref_euler, degrees=True)
        rot_local = R.from_euler("xyz", local_euler, degrees=True)
        # Post-multiply so the slider rotation is local to the robot.
        return rot_ref * rot_local

    def _compute_base_positions(self, entry_points, rotations):
        base_positions = []
        for i in range(self.num_robots):
            entry_point = np.array(entry_points[i], dtype=np.float32)
            euler = rotations[i] if i < len(rotations) else (0.0, 0.0, 0.0)
            base_rot_scipy = R.from_euler("xyz", euler, degrees=True)

            rcm_offset_world = base_rot_scipy.apply(self.rcm_offset).astype(np.float32)
            base_positions.append(entry_point - rcm_offset_world)
        return base_positions

    def _set_camera_up_axis(self):
        if hasattr(self.viewer, "camera") and self.viewer.camera is not None:
            from pyglet.math import Vec3  # noqa: PLC0415

            self.viewer.camera.up_axis = 2
            if self.parquet_data != None:
                self.viewer.camera.pos = Vec3(float(DEFAULT_CAMERA_POS_PARQUET[0]), float(DEFAULT_CAMERA_POS_PARQUET[1]), float(DEFAULT_CAMERA_POS_PARQUET[2]))
            else:
                self.viewer.camera.pos = Vec3(float(DEFAULT_CAMERA_POS[0]), float(DEFAULT_CAMERA_POS[1]), float(DEFAULT_CAMERA_POS[2]))
            self.viewer.camera.yaw = -260.0
            self.viewer.camera.pitch = -10.0

    def _initialize_model(self, entry_points):
        self.model = self._build_model_with_all_robots(entry_points)

        # Create state and evaluate FK
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        # Set model for viewer
        self.viewer.set_model(self.model)
        self._set_camera_up_axis()

        # Lock selected tool joints at the middle of their limits.
        self._locked_coord_indices, self._locked_coord_values = lock_constant_joints(
            self.model, LOCKED_JOINT_NAMES
        )
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        # Create ArticulationView for all PSM robots
        # All robots have articulation name "psm" from the URDF
        self.psm_view = ArticulationView(self.model, "psm", verbose=True)
        print(f"ArticulationView: {self.psm_view.count} articulations, {self.psm_view.joint_dof_count} DOFs each")

        # Get end effector link index within each articulation
        self.ee_link_idx = self.psm_view.link_names.index(EE_LINK_NAME)
        print(f"End effector link '{EE_LINK_NAME}' at local index {self.ee_link_idx}")

        # Get body layout info to compute global body indices
        from newton._src.sim.model import ModelAttributeFrequency
        body_layout = self.psm_view.frequency_layouts[ModelAttributeFrequency.BODY]

        # Calculate global body indices for each robot's end effector
        self.ee_body_indices = []
        for i in range(self.num_robots):
            # Global body index = offset + (articulation_idx * stride) + local_link_idx
            ee_body_idx = body_layout.offset + (i * body_layout.stride_within_worlds) + self.ee_link_idx
            self.ee_body_indices.append(ee_body_idx)
            print(f"Robot {i}: EE body index = {ee_body_idx}")

        # Get joint layout for mimic constraints (needed before IK setup)
        joint_layout = self.psm_view.frequency_layouts[ModelAttributeFrequency.JOINT]
        self.joint_offset = joint_layout.offset
        self.joint_stride = joint_layout.stride_within_worlds

        # Get DOF layout for joint positions/velocities
        dof_layout = self.psm_view.frequency_layouts[ModelAttributeFrequency.JOINT_DOF]
        self.dof_offset = dof_layout.offset
        self.dof_stride = dof_layout.stride_within_worlds
        print(f"Joint layout: offset={self.joint_offset}, stride={self.joint_stride}")
        print(f"DOF layout: offset={self.dof_offset}, stride={self.dof_stride}")

        # Initialize per-robot target data
        self.target_positions = []
        self.target_orientations = []
        self.jaw_angles = []

        # Gizmo transforms for visualization (one per instrument)
        self.gizmo_transforms = []
        self.entry_gizmo_transforms = []
        for i in range(self.num_robots):
            self.gizmo_transforms.append(wp.transform_identity())
            self.entry_gizmo_transforms.append(wp.transform_identity())

        # Get initial end effector poses for each robot
        for i in range(self.num_robots):
            initial_tf = self.state.body_q.numpy()[self.ee_body_indices[i]]
            target_position = initial_tf[:3].copy()
            target_orientation = initial_tf[3:7].copy()
            self.target_positions.append(target_position)
            self.target_orientations.append(target_orientation)
            self.jaw_angles.append(0.0)

        # Set up single IK solver with objectives for all robots
        self._setup_unified_ik_solver()

        # Joint q buffer for IK (shape: n_problems x joint_coord_count)
        initial_q = self.model.joint_q.numpy()
        self.joint_q_buffer = wp.array(initial_q.reshape(1, -1), dtype=wp.float32, device=self.model.device)
        self.root_joint_indices = self._compute_root_joint_indices()

    def _rebuild_model(self):
        # Apply new base rotations without resetting the viewer model.
        self._apply_base_rotations_in_place()

    def _compute_root_joint_indices(self):
        joint_parent = self.model.joint_parent.numpy()
        articulation_start = list(self.model.articulation_start.numpy())
        if len(articulation_start) == self.model.articulation_count:
            articulation_start.append(self.model.joint_count)

        root_indices = []
        for i in range(min(self.num_robots, self.model.articulation_count)):
            start = articulation_start[i]
            end = articulation_start[i + 1]
            roots = [j for j in range(start, end) if joint_parent[j] == -1]
            if not roots:
                raise RuntimeError(f"No root joint found for articulation {i}")
            root_indices.append(int(roots[0]))
        return root_indices

    def _apply_base_rotations_in_place(self):
        joint_X_p_np = self.model.joint_X_p.numpy()
        updated_base_positions = []
        for i, root_idx in enumerate(self.root_joint_indices):
            base_rot_curr = self._compose_base_rotation(i)
            rcm_offset_world = base_rot_curr.apply(self.rcm_offset).astype(np.float32)
            entry_point = np.array(self.initial_entry_points[i], dtype=np.float32)
            # Keep the RCM aligned with the entry point as the base rotates.
            base_position = entry_point - rcm_offset_world
            base_quat = base_rot_curr.as_quat().astype(np.float32)
            joint_X_p_np[root_idx, 0:3] = base_position
            joint_X_p_np[root_idx, 3:7] = base_quat
            rcm_point = base_position + rcm_offset_world
            print(f"  Robot {i}: Updated base to {base_position} (RCM at {rcm_point})")
            updated_base_positions.append(base_position)

        self.base_positions = updated_base_positions

        self.model.joint_X_p.assign(wp.array(joint_X_p_np, dtype=wp.transform, device=self.model.device))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

    def _register_base_rotation_ui(self):
        if not hasattr(self.viewer, "register_ui_callback"):
            return
        if getattr(self, "_base_ui_registered", False):
            return
        self.viewer.register_ui_callback(self._base_rotation_ui_callback, position="side")
        self._base_ui_registered = True

    def _tune_ui_style(self, imgui):
        if getattr(self, "_ui_style_tuned", False):
            return
        style = imgui.get_style()
        style.scrollbar_size = max(style.scrollbar_size, UI_SCROLLBAR_SIZE)
        style.grab_min_size = max(style.grab_min_size, UI_GRAB_MIN_SIZE)
        self._ui_style_tuned = True

    def _base_rotation_ui_callback(self, imgui):
        self._tune_ui_style(imgui)
        imgui.separator()
        imgui.text("Base Local Rotation (deg)")

        pending = list(self.base_rotations_pending)
        changed_any = False
        for i in range(self.num_robots):
            rot = list(pending[i])
            imgui.text(f"Robot {i}")
            changed_x, rot_x = imgui.slider_float(f"X {i}", rot[0], -180.0, 180.0, "%.1f")
            changed_y, rot_y = imgui.slider_float(f"Y {i}", rot[1], -180.0, 180.0, "%.1f")
            changed_z, rot_z = imgui.slider_float(f"Z {i}", rot[2], -180.0, 180.0, "%.1f")
            if changed_x or changed_y or changed_z:
                pending[i] = (float(rot_x), float(rot_y), float(rot_z))
                changed_any = True
            eff_euler = self._compose_base_rotation(i).as_euler("xyz", degrees=True)
            imgui.text(f"  Effective: ({eff_euler[0]:.1f}, {eff_euler[1]:.1f}, {eff_euler[2]:.1f})")

        self.base_rotations_pending = pending
        if changed_any:
            self.base_rotations_local = [tuple(r) for r in self.base_rotations_pending]
            self._rebuild_model()

        if imgui.button("Reset Base Rotations"):
            self.base_rotations_local = [(0.0, 0.0, 0.0) for _ in range(self.num_robots)]
            self.base_rotations_pending = [tuple(r) for r in self.base_rotations_local]
            self.base_positions = self._compute_base_positions(self.initial_entry_points, self.base_rotations_ref)
            self._rebuild_model()

        imgui.separator()
        imgui.text("Gizmos")
        _, self.show_target_gizmos = imgui.checkbox("Target gizmos", self.show_target_gizmos)
        _, self.show_ee_gizmos = imgui.checkbox("EE gizmos", self.show_ee_gizmos)
        _, self.show_entry_gizmos = imgui.checkbox("Entry gizmos", self.show_entry_gizmos)

        imgui.separator()
        imgui.text("Recording")
        _, self.recorder.record_ui = imgui.checkbox("Include UI", self.recorder.record_ui)
        if not self.recorder.is_recording:
            if imgui.button("Start Recording"):
                self.recorder.start()
        else:
            if imgui.button("Stop Recording"):
                self.recorder.stop()
        imgui.text(f"Output: {self.recorder.output_path}")

        imgui.separator()
        imgui.text("Frame")
        max_frame = max(0, len(self.frames) - 1)
        current_idx = self._frame_override_idx if self._frame_override_active else self._current_frame_idx
        changed_frame, new_idx = imgui.slider_int("Frame Index", int(current_idx), 0, max_frame)
        changed_lock, self._frame_override_active = imgui.checkbox("Lock frame", self._frame_override_active)
        if changed_frame:
            self._frame_override_idx = int(new_idx)
            self._frame_override_active = True
        if imgui.button("Resume Playback"):
            self._frame_override_active = False

        imgui.separator()
        imgui.text("EE Target Match")
        tol = EE_MATCH_TOLERANCE
        for i, err in enumerate(self.ee_position_errors):
            if not np.isfinite(err):
                imgui.text(f"Robot {i}: n/a")
                continue
            is_match = err <= tol
            color = (0.2, 0.85, 0.2, 1.0) if is_match else (0.9, 0.3, 0.3, 1.0)
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*color))
            status = "OK" if is_match else "off"
            imgui.text(f"Robot {i}: {status} ({err:.4f})")
            imgui.pop_style_color()

    def _setup_unified_ik_solver(self):
        """Set up a single IK solver with objectives for all robots."""
        objectives = []
        self.pos_objectives = []
        self.rot_objectives = []

        # Create position and rotation objectives for each robot
        for i in range(self.num_robots):
            # Position objective for this robot's end effector
            # Each objective has n_problems=1, targeting one position
            pos_target = wp.array(
                self.target_positions[i].reshape(1, 3), dtype=wp.vec3, device=self.model.device
            )
            toolTipOffset = self.tool_tip_offset
            pos_obj = IKPositionObjective(
                self.ee_body_indices[i], toolTipOffset, pos_target, weight=1.0
            )
            self.pos_objectives.append(pos_obj)
            objectives.append(pos_obj)

            # Rotation objective for this robot's end effector
            rot_target = wp.array(
                self.target_orientations[i].reshape(1, 4), dtype=wp.vec4, device=self.model.device
            )
            rot_obj = IKRotationObjective(
                self.ee_body_indices[i], wp.quat_identity(), rot_target, weight=IK_ROTATION_WEIGHT
            )
            self.rot_objectives.append(rot_obj)
            objectives.append(rot_obj)

        # Add joint limit objective (applies to all joints)
        objectives.append(
            IKJointLimitObjective(
                self.model.joint_limit_lower, self.model.joint_limit_upper, weight=10.0
            )
        )

        # Build mimic joint constraints for all robots
        # Since joint names are duplicated across robots, we build the mimic info directly
        mimic_objective = build_multi_robot_mimic_objective(
            self.model, MIMIC_JOINT_MAPPING, self.num_robots, weight=100.0
        )
        if mimic_objective is not None:
            objectives.append(mimic_objective)

        # Create single IK solver with all objectives
        self.ik_solver = IKSolver(
            model=self.model,
            n_problems=1,
            objectives=objectives,
            optimizer="lm",
            jacobian_mode="analytic",
            n_seeds=1,
        )
        print(f"IK solver created with {len(objectives)} objectives")

    def _normalize_quat(self, quat):
        quat = quat.astype(np.float32)
        quat_norm = float(np.linalg.norm(quat)) if np.isfinite(quat).all() else 0.0
        if quat_norm < 1.0e-8:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return quat / quat_norm

    def _get_parquet_action_pose(self, frame_idx, inst_idx):
        action_row = self.parquet_data.action[frame_idx, inst_idx]
        pos = action_row[:3].astype(np.float32)
        rot = self._normalize_quat(action_row[3:7])
        return pos, rot

    def _get_parquet_observation_pose(self, frame_idx, inst_idx):
        pos = self.parquet_data.observation_tip_pos[frame_idx, inst_idx].astype(np.float32)
        rot = self._normalize_quat(self.parquet_data.observation_tip_rot[frame_idx, inst_idx])
        if not np.isfinite(pos).all():
            pos = np.zeros(3, dtype=np.float32)
        return pos, rot

    def _get_joint_positions_per_robot(self):
        joint_q = self.model.joint_q.numpy()
        dof_count = int(self.psm_view.joint_dof_count)
        joint_positions = np.zeros((self.num_robots, dof_count), dtype=np.float32)
        for i in range(self.num_robots):
            start = int(self.dof_offset + i * self.dof_stride)
            joint_positions[i] = joint_q[start:start + dof_count]
        return joint_positions

    def _get_joint_velocities_per_robot(self):
        joint_qd = self.model.joint_qd.numpy()
        dof_count = int(self.psm_view.joint_dof_count)
        joint_velocities = np.zeros((self.num_robots, dof_count), dtype=np.float32)
        for i in range(self.num_robots):
            start = int(self.dof_offset + i * self.dof_stride)
            joint_velocities[i] = joint_qd[start:start + dof_count]
        return joint_velocities

    def _update_parquet_joint_positions(self, frame_idx, joint_positions, joint_velocities):
        dof_count = joint_positions.shape[1]
        if (
            self._parquet_action_joints is None
            or self._parquet_observation_joints_pos is None
            or self._parquet_observation_joints_vel is None
        ):
            num_frames = len(self.frames)
            list_size = self.num_robots * dof_count
            self._parquet_action_joints = np.zeros((num_frames, list_size), dtype=np.float32)
            self._parquet_observation_joints_pos = np.zeros((num_frames, list_size), dtype=np.float32)
            self._parquet_observation_joints_vel = np.zeros((num_frames, list_size), dtype=np.float32)
        for i in range(self.num_robots):
            start = i * dof_count
            self._parquet_action_joints[frame_idx, start:start + dof_count] = joint_positions[i]
            self._parquet_observation_joints_pos[frame_idx, start:start + dof_count] = joint_positions[i]
            self._parquet_observation_joints_vel[frame_idx, start:start + dof_count] = joint_velocities[i]

    def _load_parquet_instruments(self, frame_idx, init_entry_points=False):
        instruments = []
        if init_entry_points or self.parquet_entry_points is None:
            self.parquet_entry_points = []
            for i in range(self.num_robots):
                if i < len(PARQUET_ENTRY_POINTS):
                    entry = np.array(PARQUET_ENTRY_POINTS[i], dtype=np.float32)
                else:
                    entry, _ = self._get_parquet_observation_pose(frame_idx, i)
                self.parquet_entry_points.append(entry) 

        for i in range(self.num_robots):
            pos, rot = self._get_parquet_observation_pose(frame_idx, i)
            pos *= PARQUET_POS_SCALE
            entry = np.array(self.parquet_entry_points[i], dtype=np.float32)
            instruments.append(InstrumentData(pos, rot, entry))
        return instruments

    def load_frame(self, frame_num):
        """Load all tissue data and instruments for a specific frame."""
        if self.use_parquet:
            instruments = self._load_parquet_instruments(frame_num)
            return [], instruments
        return self.data_loader.load_frame(frame_num)

    def _hash_indices(self, indices_np):
        if indices_np.size == 0:
            return 0
        return zlib.adler32(indices_np.view(np.uint8))

    def _get_tissue_mesh_names(self, tissue_name, vertex_count, index_count, indices_hash):
        cache = self._tissue_mesh_cache.get(tissue_name)
        prev_mesh_name = None
        prev_instance_name = None
        if (
            cache is None
            or cache["vertex_count"] != vertex_count
            or cache["index_count"] != index_count
            or cache["indices_hash"] != indices_hash
        ):
            if cache is not None:
                prev_mesh_name = cache["mesh_name"]
                prev_instance_name = cache["instance_name"]
            version = cache["version"] + 1 if cache else 0
            mesh_name = f"/tissues/{tissue_name}_mesh_{version}"
            instance_name = f"/tissues/{tissue_name}_inst_{version}"
            cache = {
                "vertex_count": vertex_count,
                "index_count": index_count,
                "indices_hash": indices_hash,
                "mesh_name": mesh_name,
                "instance_name": instance_name,
                "version": version,
            }
            self._tissue_mesh_cache[tissue_name] = cache
        versions = self._tissue_version_cache.setdefault(tissue_name, {})
        versions[cache["instance_name"]] = cache["mesh_name"]
        return cache["mesh_name"], cache["instance_name"], prev_mesh_name, prev_instance_name

    def _hide_tissue_instance(self, tissue_name):
        cache = self._tissue_mesh_cache.get(tissue_name)
        instance = self._tissue_instance_cache.get(tissue_name)
        if cache is None or instance is None:
            return
        self.viewer.log_instances(
            cache["instance_name"],
            cache["mesh_name"],
            instance["xforms"],
            instance["scales"],
            instance["colors"],
            None,
            hidden=True,
        )

    def _destroy_viewer_object(self, name):
        if not hasattr(self.viewer, "objects"):
            return
        obj = self.viewer.objects.pop(name, None)
        if obj is None:
            return
        try:
            if hasattr(obj, "_tissue_color_buffer"):
                try:
                    from newton._src.viewer.gl.opengl import RendererGL  # noqa: PLC0415

                    gl = RendererGL.gl
                    gl.glDeleteBuffers(1, obj._tissue_color_buffer)
                except Exception:
                    pass
            if hasattr(obj, "destroy"):
                obj.destroy()
        finally:
            del obj

    def _clear_tissues(self):
        # Remove all previously registered tissue meshes/instances.
        for name in list(self._active_tissue_objects):
            self._destroy_viewer_object(name)
        self._active_tissue_objects.clear()
        self._tissue_mesh_cache.clear()
        self._tissue_instance_cache.clear()
        self._tissue_version_cache.clear()

    def _set_mesh_vertex_colors(self, mesh_name, color, vertex_count):
        if not hasattr(self.viewer, "objects"):
            return
        mesh = self.viewer.objects.get(mesh_name)
        if mesh is None or not hasattr(mesh, "vao"):
            return
        try:
            import ctypes  # noqa: PLC0415
            from newton._src.viewer.gl.opengl import RendererGL  # noqa: PLC0415

            gl = RendererGL.gl
            color_np = np.tile(np.array(color, dtype=np.float32), (vertex_count, 1))
            color_buffer = gl.GLuint()
            gl.glGenBuffers(1, color_buffer)
            gl.glBindVertexArray(mesh.vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, color_buffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, color_np.nbytes, color_np.ctypes.data, gl.GL_STATIC_DRAW)
            gl.glVertexAttribPointer(7, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * 4, ctypes.c_void_p(0))
            gl.glEnableVertexAttribArray(7)
            gl.glVertexAttribDivisor(7, 0)
            gl.glBindVertexArray(0)
            mesh._tissue_color_buffer = color_buffer
        except Exception:
            pass

    def _hide_stale_tissue_versions(self, tissue_name, current_instance, instance):
        versions = self._tissue_version_cache.get(tissue_name)
        if not versions:
            return
        for inst_name, mesh_name in versions.items():
            if inst_name == current_instance:
                continue
            try:
                self.viewer.log_instances(
                    inst_name,
                    mesh_name,
                    instance["xforms"],
                    instance["scales"],
                    instance["colors"],
                    None,
                    hidden=True,
                )
            except Exception:
                # Older versions may not exist in the current viewer backend.
                pass

    def _ensure_tissue_instance(self, tissue_name, color):
        instance = self._tissue_instance_cache.get(tissue_name)
        color_np = np.array(color, dtype=np.float32)
        needs_color_update = (
            instance is None
            or "color_np" not in instance
            or not np.allclose(instance["color_np"], color_np)
        )
        if instance is None:
            xforms = wp.array([wp.transform_identity()], dtype=wp.transform, device=self.model.device)
            scales = wp.array([wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3, device=self.model.device)
            colors = wp.array([wp.vec3(float(color_np[0]), float(color_np[1]), float(color_np[2]))], dtype=wp.vec3, device=self.model.device)
            instance = {"xforms": xforms, "scales": scales, "colors": colors, "color_np": color_np}
            self._tissue_instance_cache[tissue_name] = instance
        elif needs_color_update:
            instance["colors"] = wp.array(
                [wp.vec3(float(color_np[0]), float(color_np[1]), float(color_np[2]))],
                dtype=wp.vec3,
                device=self.model.device,
            )
            instance["color_np"] = color_np
        return instance

    def _render_tissues(self, tissues):
        # Clear and rebuild tissue visuals every frame to avoid stale instances.
        self._clear_tissues()
        if not tissues:
            return
        for tissue in tissues:
            if tissue.vertices is None or tissue.triangles is None:
                continue
            if len(tissue.vertices) == 0 or len(tissue.triangles) == 0:
                continue
            indices_np = tissue.triangles.reshape(-1)
            instance = self._ensure_tissue_instance(tissue.name, tissue.color)
            mesh_name = f"/tissues/{tissue.name}_mesh"
            instance_name = f"/tissues/{tissue.name}_inst"
            points = wp.array(tissue.vertices, dtype=wp.vec3, device=self.model.device)
            indices = wp.array(indices_np, dtype=wp.int32, device=self.model.device)
            self.viewer.log_mesh(mesh_name, points, indices)

            self._active_tissue_objects.add(mesh_name)
            if hasattr(self.viewer, "objects"):
                self._set_mesh_vertex_colors(mesh_name, tissue.color, len(tissue.vertices))
            else:
                self.viewer.log_instances(
                    instance_name,
                    mesh_name,
                    instance["xforms"],
                    instance["scales"],
                    instance["colors"],
                    None,
                    hidden=False,
                )
                self._active_tissue_objects.add(instance_name)

    def step(self) -> None:
        if self._frame_override_active:
            frame_idx = int(self._frame_override_idx)
        else:
            frame_idx = self.num_frame
            self.num_frame = (self.num_frame + 1) % len(self.frames)  # Loop back to start
        frame_num = self.frames[frame_idx]
        self._current_frame_idx = frame_idx
        tissues, instruments = self.load_frame(frame_num)
        self.current_tissues = tissues if self.render_tissues else []
        if self.use_parquet and instruments:
            print(f"Parquet observations (frame {frame_num}):")
            for i, instr in enumerate(instruments[: self.num_robots]):
                print(f"  Instr {i}: pos={instr.position}, rot={instr.rotation}, entry={instr.entry_point}")

        # Update target positions/orientations from instrument data
        for i in range(self.num_robots):
            if i < len(instruments):
                self.target_positions[i] = instruments[i].position
                self.target_orientations[i] = instruments[i].rotation
                pos = instruments[i].position
                rot = instruments[i].rotation  # quaternion (x, y, z, w)
                self.gizmo_transforms[i] = wp.transform(
                    wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                    wp.quat(float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))
                )
                entry = instruments[i].entry_point
                self.entry_gizmo_transforms[i] = wp.transform(
                    wp.vec3(float(entry[0]), float(entry[1]), float(entry[2])),
                    wp.quat_identity(),
                )

        # Solve IK for each robot and apply joint positions
        self._solve_ik_for_all_robots()

        self.sim_time += self.frame_dt

    def _solve_ik_for_all_robots(self):
        """Solve IK for all robots and update joint positions."""
        # Update IK targets for each robot
        for i in range(self.num_robots):
            # Update position target
            new_pos = wp.vec3(
                float(self.target_positions[i][0]),
                float(self.target_positions[i][1]),
                float(self.target_positions[i][2])
            )
            self.pos_objectives[i].set_target_position(0, new_pos)

            # Update rotation target
            new_rot = wp.vec4(
                float(self.target_orientations[i][0]),
                float(self.target_orientations[i][1]),
                float(self.target_orientations[i][2]),
                float(self.target_orientations[i][3])
            )
            self.rot_objectives[i].set_target_rotation(0, new_rot)

        # Copy current joint positions to buffer as initial guess
        current_q = self.model.joint_q.numpy().reshape(1, -1)
        self.joint_q_buffer = wp.array(current_q, dtype=wp.float32, device=self.model.device)

        # Solve IK for all robots simultaneously
        self.ik_solver.step(
            self.joint_q_buffer,
            self.joint_q_buffer,
            iterations=IK_ITERATIONS,
            step_size=IK_STEP_SIZE,
        )

        # Copy solved positions back to model
        solved_q = self.joint_q_buffer.numpy()[0]
        if self._locked_coord_indices:
            solved_q[self._locked_coord_indices] = self._locked_coord_values
        self.model.joint_q.assign(wp.array(solved_q, dtype=wp.float32, device=self.model.device))

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        # Evaluate FK with updated joint positions
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.viewer.log_state(self.state)
        if self.use_parquet and self.parquet_recompute:
            joint_positions = self._get_joint_positions_per_robot()
            joint_velocities = self._get_joint_velocities_per_robot()
            self._update_parquet_joint_positions(self._current_frame_idx, joint_positions, joint_velocities)

        # Render gizmos at instrument targets and current end-effector poses
        for i in range(self.num_robots):
            ee_tf = self.state.body_q.numpy()[self.ee_body_indices[i]]
            ee_pos = ee_tf[:3]
            ee_rot = ee_tf[3:7]
            tip_offset = np.array(
                [float(TOOL_TIP_OFFSET[0]), float(TOOL_TIP_OFFSET[1]), float(TOOL_TIP_OFFSET[2])],
                dtype=np.float32,
            )
            tip_pos = ee_pos + R.from_quat(ee_rot).apply(tip_offset).astype(np.float32)
            target_pos = np.array(self.target_positions[i], dtype=np.float32)
            self.ee_position_errors[i] = float(np.linalg.norm(tip_pos - target_pos))
            if self.use_parquet and self.parquet_recompute:
                self.parquet_data.action[self._current_frame_idx, i, 0:3] = tip_pos
                self.parquet_data.action[self._current_frame_idx, i, 3:7] = ee_rot
                self.parquet_data.observation_tip_pos[self._current_frame_idx, i, 0:3] = tip_pos
                self.parquet_data.observation_tip_rot[self._current_frame_idx, i, 0:4] = ee_rot
            if self.show_target_gizmos:
                self.viewer.log_gizmo(f"instrument_{i}", self.gizmo_transforms[i])
            ee_gizmo = wp.transform(
                wp.vec3(float(tip_pos[0]), float(tip_pos[1]), float(tip_pos[2])),
                wp.quat(float(ee_rot[0]), float(ee_rot[1]), float(ee_rot[2]), float(ee_rot[3])),
            )
            if self.show_ee_gizmos:
                self.viewer.log_gizmo(f"instrument_ee_{i}", ee_gizmo)
            if self.show_entry_gizmos:
                self.viewer.log_gizmo(f"instrument_entry_{i}", self.entry_gizmo_transforms[i])

        if self.render_tissues:
            self._render_tissues(self.current_tissues)

        self.viewer.end_frame()
        wp.synchronize()
        self.recorder.capture()

    def save_parquet_overwrite(self, output_dir=None):
        if not self.use_parquet or not self.parquet_recompute:
            return
        if self.parquet_data is None:
            return
        if (
            self._parquet_action_joints is None
            or self._parquet_observation_joints_pos is None
            or self._parquet_observation_joints_vel is None
        ):
            raise RuntimeError("Joint buffers are empty; nothing to save.")
        def _resolve_dataset_root(path):
            path = os.path.abspath(path)
            if path.endswith(".parquet"):
                path = os.path.dirname(path)
            parts = path.split(os.sep)
            if "data" in parts:
                idx = len(parts) - 1 - list(reversed(parts)).index("data")
                root = os.sep.join(parts[:idx])
                data_root = os.path.join(root, "data")
                return root, data_root
            return path, os.path.join(path, "data")

        source_root, source_data_root = _resolve_dataset_root(self.parquet_path)
        if output_dir:
            output_root, output_data_root = _resolve_dataset_root(output_dir)
            os.makedirs(output_data_root, exist_ok=True)
            meta_src = os.path.join(source_root, "meta")
            meta_dst = os.path.join(output_root, "meta")
            if os.path.isdir(meta_src) and not os.path.isdir(meta_dst):
                shutil.copytree(meta_src, meta_dst)
        else:
            output_root, output_data_root = source_root, source_data_root

        overwrite_parquet_joint_columns(
            source_data_root,
            output_data_root,
            self._parquet_action_joints,
            self._parquet_observation_joints_pos,
            self._parquet_observation_joints_vel,
        )

        update_lerobot_metadata(output_root, urdf_path=DEFAULT_PSM_URDF)

def main():
    parser = argparse.ArgumentParser(description='Playback exported tissue data')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='exported_data',
        help='Directory containing the exported .dat files '
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Playback frame rate (default: 30)'
    )
    parser.add_argument(
        '--parquet_path',
        type=str,
        default=None,
        help='Optional parquet file containing instrument action/observation data'
    )
    parser.add_argument(
        '--parquet_recompute',
        action='store_true',
        help='Overwrite parquet action/observation data with IK-computed cartesian poses'
    )
    parser.add_argument(
        '--parquet_output_dir',
        type=str,
        default=None,
        help='Optional output directory for parquet overwrites (defaults to in-place)'
    )

    args = parser.parse_args()

    # Validate data sources
    if args.parquet_path is not None:
        if not os.path.exists(args.parquet_path):
            print(f"Error: Parquet file '{args.parquet_path}' does not exist.")
            return
        os.makedirs(args.data_dir, exist_ok=True)
    else:
        if not os.path.exists(args.data_dir):
            print(f"Error: Data directory '{args.data_dir}' does not exist.")
            print(f"Please create it and place your .dat files there, or specify a different directory with --data_dir")
            return

    try:
        if args.parquet_recompute and args.parquet_path is not None:
            viewer = newton.viewer.ViewerNull()
            newton_args = None
        else:
            viewer, newton_args = newton.examples.init()
        playback = DataPlayback(
            data_dir=args.data_dir,
            fps=args.fps,
            viewer=viewer,
            parquet_path=args.parquet_path,
            parquet_recompute=args.parquet_recompute,
        )
        if isinstance(viewer, newton.viewer.ViewerNull):
            viewer.num_frames = len(playback.frames)
        newton.examples.run(playback, newton_args)
        if args.parquet_recompute:
            playback.save_parquet_overwrite(output_dir=args.parquet_output_dir)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
