# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for haptic device control."""

import numpy as np
from isaaclab.utils import configclass


@configclass
class HapticControlCfg:
    """Configuration for haptic device teleoperation control."""

    # Haptic control sensitivity parameters
    position_sensitivity: float = 0.004  # Scale for position mapping
    rotation_sensitivity: float = 1.0  # Scale for rotation mapping
    max_position_delta: float = 1.0  # Maximum position change per step (meters) - safety limit
    max_rotation_delta: float = 1.0  # Maximum rotation change per step (radians) - safety limit

    # Omni Geomagic axis mapping
    # The Omni device coordinate system:
    #   - Omni X: Left/Right (+ = right)
    #   - Omni Y: Up/Down (+ = up)
    #   - Omni Z: Forward/Backward (+ = forward/away from user, docked = max forward)
    # Robot coordinate system (Isaac Sim):
    #   - Robot X: Forward/Backward (+ = forward)
    #   - Robot Y: Left/Right (+ = left)
    #   - Robot Z: Up/Down (+ = up)
    # Mapping: [haptic_axis_for_robot_x, haptic_axis_for_robot_y, haptic_axis_for_robot_z]
    haptic_axis_map: list = [2, 0, 1]  # Robot [X, Y, Z] = Haptic [Z, X, Y]
    haptic_axis_signs: list = [-1.0, -1.0, 1.0]  # Signs for [Robot X, Robot Y, Robot Z]

    # Rotation axis remapping (same logic as position)
    # Quaternion [x, y, z, w] where x/y/z are rotation around those axes
    # Robot rotation X (roll) = Haptic rotation Z (yaw)
    # Robot rotation Y (pitch) = Haptic rotation X (roll)
    # Robot rotation Z (yaw) = Haptic rotation Y (pitch)
    haptic_rot_axis_map: list = [2, 0, 1]  # Same as position mapping
    haptic_rot_axis_signs: list = [-1.0, -1.0, 1.0]  # Same signs

    # Omni Geomagic device settings
    # The Omni device typically has its docked position at a specific location
    # We'll calibrate this to match the robot's initial pose
    enable_calibration: bool = True  # Set to False to skip calibration
    calibration_wait_time: float = 3.0  # Seconds to wait for device to be docked

    # Target robot pose when haptic is docked:
    # - Forward (high x): near max of workspace range
    # - Centered (y=0): 0.0
    # - Low (low z): near min of workspace range
    # - Rotation: gripper pointing slightly downward
    target_ee_pos_docked: list = [1.0, 0.0, 0.18]  # Forward, centered, low
    gripper_pitch_angle_deg: float = 90.0  # Degrees downward for gripper when starting

    # Gripper control settings
    gripper_pos_closed: float = 0.0  # Gripper position when button pressed (closed)
    gripper_pos_open: float = 0.04  # Gripper position when button released (open)

    def get_target_ee_quat_docked(self) -> np.ndarray:
        """Get target quaternion for gripper pointing downward."""
        gripper_pitch_angle = np.radians(self.gripper_pitch_angle_deg)
        # Quaternion for rotation around Y axis: [x, y, z, w] = [0, sin(θ/2), 0, cos(θ/2)]
        return np.array([
            0.0,                                    # x
            np.sin(gripper_pitch_angle / 2),        # y (pitch axis)
            0.0,                                    # z
            np.cos(gripper_pitch_angle / 2)         # w
        ])


