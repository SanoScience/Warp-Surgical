# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--newton_visualizer", action="store_true", default=False, help="Enable Newton rendering.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np
import warp as wp

from isaaclab.utils.timer import Timer
from isaaclab.utils.math import quat_mul, quat_from_angle_axis, axis_angle_from_quat, quat_apply

Timer.enable = False
Timer.enable_display_output = False

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Add main project folder to path for haptic_device import
import sys
from pathlib import Path
main_folder = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, main_folder)

from haptic_device import HapticController
import newton.ik as ik
from isaaclab.sim._impl.newton_manager import NewtonManager

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Haptic teleoperation agent with IK-based control."""

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        newton_visualizer=args_cli.newton_visualizer,
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Get haptic control configuration from environment config
    # If haptic config doesn't exist, use defaults
    if hasattr(env_cfg, 'haptic'):
        haptic_cfg = env_cfg.haptic
    else:
        # Fallback: create default config if not available
        from isaaclab_tasks.manager_based.manipulation.reach.config.franka.haptic_cfg import HapticControlCfg
        haptic_cfg = HapticControlCfg()
        print("[WARNING]: Haptic config not found in environment, using defaults")
    
    # Extract haptic control parameters from config
    position_sensitivity = haptic_cfg.position_sensitivity
    rotation_sensitivity = haptic_cfg.rotation_sensitivity
    max_position_delta = haptic_cfg.max_position_delta
    max_rotation_delta = haptic_cfg.max_rotation_delta
    haptic_axis_map = haptic_cfg.haptic_axis_map
    haptic_axis_signs = haptic_cfg.haptic_axis_signs
    haptic_rot_axis_map = haptic_cfg.haptic_rot_axis_map
    haptic_rot_axis_signs = haptic_cfg.haptic_rot_axis_signs
    enable_calibration = haptic_cfg.enable_calibration
    calibration_wait_time = haptic_cfg.calibration_wait_time
    gripper_pos_closed = haptic_cfg.gripper_pos_closed
    gripper_pos_open = haptic_cfg.gripper_pos_open

    # Initialize haptic controller
    haptic_controller = HapticController(scale=1.0)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    
    # reset environment
    env.reset()
    
    # Get Newton model and robot info
    model = NewtonManager.get_model()
    robot = env.unwrapped.scene["robot"]
    
    # Find end-effector body index in Newton model
    ee_body_name = env.unwrapped.cfg.commands.ee_pose.body_name
    ee_body_path = f"/World/envs/env_0/Robot/{ee_body_name}"
    ee_body_idx = model.body_key.index(ee_body_path)
    
    # Get the number of controlled joints from action manager
    num_action_joints = env.unwrapped.action_manager.total_action_dim
    
    # Get action configuration for proper action space conversion
    arm_action_term = env.unwrapped.action_manager._terms.get("arm_action")
    if arm_action_term is None:
        raise ValueError("Could not find 'arm_action' term in action manager")
    
    # Get default joint positions (offset) and scale from action config
    action_scale = arm_action_term._scale
    if isinstance(action_scale, torch.Tensor):
        action_scale = action_scale[0, :num_action_joints].cpu().numpy()
    else:
        action_scale = float(action_scale)
    
    default_joint_pos = arm_action_term._offset
    if isinstance(default_joint_pos, torch.Tensor):
        default_joint_pos = default_joint_pos[0, :num_action_joints].cpu()
    else:
        # If offset is a scalar, get default positions from robot
        default_joint_pos = robot.data.default_joint_pos[0, :num_action_joints].cpu()
    
    # Get joint IDs for action joints
    action_joint_ids = arm_action_term._joint_ids
    if isinstance(action_joint_ids, slice):
        # If all joints, create list
        action_joint_ids = list(range(num_action_joints))
    elif isinstance(action_joint_ids, torch.Tensor):
        action_joint_ids = action_joint_ids.cpu().tolist()
    
    print(f"[INFO]: Action scale: {action_scale}")
    print(f"[INFO]: Default joint positions: {default_joint_pos}")
    print(f"[INFO]: Action joint IDs: {action_joint_ids}")
    
    # Check if gripper action is available
    gripper_action_term = env.unwrapped.action_manager._terms.get("gripper_action")
    has_gripper = gripper_action_term is not None
    
    # If no separate gripper action, check if we can find gripper joints in the robot
    gripper_joint_names = None
    gripper_joint_ids = None
    if not has_gripper:
        # Try to find gripper joints (common patterns)
        all_joint_names = robot.joint_names
        gripper_patterns = ["finger", "gripper", "jaw", "grip"]
        gripper_joint_names = [name for name in all_joint_names 
                              if any(pattern in name.lower() for pattern in gripper_patterns)]
        if gripper_joint_names:
            gripper_joint_ids = [all_joint_names.index(name) for name in gripper_joint_names]
            print(f"[INFO]: Found gripper joints: {gripper_joint_names} (IDs: {gripper_joint_ids})")
        else:
            print(f"[INFO]: No gripper joints found. Button will not control gripper.")
    else:
        print(f"[INFO]: Gripper action term found")
    
    # Setup IK solver (single environment for now)
    total_residuals = 3 + 3 + num_action_joints  # position + rotation + joint limits
    
    # Get initial end-effector pose
    robot.update(dt=0.0)  # Ensure robot data is current
    ee_pos_w = robot.data.body_pos_w[0, robot.find_bodies(ee_body_name)[0][0]]
    ee_quat_w = robot.data.body_quat_w[0, robot.find_bodies(ee_body_name)[0][0]]
    
    # Create IK objectives
    pos_obj = ik.IKPositionObjective(
        link_index=ee_body_idx,
        link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=wp.array([wp.vec3(ee_pos_w[0].item(), ee_pos_w[1].item(), ee_pos_w[2].item())], dtype=wp.vec3),
        n_problems=1,
        total_residuals=total_residuals,
        residual_offset=0,
        weight=1.0,
    )
    
    rot_obj = ik.IKRotationObjective(
        link_index=ee_body_idx,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([wp.vec4(ee_quat_w[1].item(), ee_quat_w[2].item(), 
                                           ee_quat_w[3].item(), ee_quat_w[0].item())], dtype=wp.vec4),
        n_problems=1,
        total_residuals=total_residuals,
        residual_offset=3,
        weight=1.0,
    )
    
    joint_limits_obj = ik.IKJointLimitObjective(
        joint_limit_lower=model.joint_limit_lower,
        joint_limit_upper=model.joint_limit_upper,
        n_problems=1,
        total_residuals=total_residuals,
        residual_offset=6,
        weight=10.0,
    )
    
    # Get current joint positions for IK initialization
    current_joint_q = robot.data.joint_pos[0].cpu().numpy()
    joint_q_2d = wp.array(current_joint_q.reshape(1, -1), dtype=wp.float32)
    
    # Create IK solver
    ik_solver = ik.IKSolver(
        model=model,
        joint_q=joint_q_2d,
        objectives=[pos_obj, rot_obj, joint_limits_obj],
        lambda_initial=0.1,
        jacobian_mode=ik.IKJacobianMode.ANALYTIC,
    )
    
    # Get target robot pose from config
    target_ee_pos_docked = np.array(haptic_cfg.target_ee_pos_docked)
    target_ee_quat_docked = haptic_cfg.get_target_ee_quat_docked()
    
    # Calibration: Map haptic device docked position to desired robot pose
    if enable_calibration:
        print(f"[INFO]: Calibrating haptic device...")
        print(f"[INFO]: Please dock your Omni Geomagic device and keep it still for {calibration_wait_time} seconds...")
        import time
        time.sleep(calibration_wait_time)
        
        # Get haptic device position when docked (raw device coordinates)
        docked_haptic_pos_device = np.array(haptic_controller.get_scaled_position())
        docked_haptic_rot = np.array(haptic_controller.get_rotation())
        
        # Remap haptic position axes to robot coordinate system
        docked_haptic_pos = np.array([
            haptic_axis_signs[0] * docked_haptic_pos_device[haptic_axis_map[0]],
            haptic_axis_signs[1] * docked_haptic_pos_device[haptic_axis_map[1]],
            haptic_axis_signs[2] * docked_haptic_pos_device[haptic_axis_map[2]]
        ])
        
        # Remap haptic rotation axes to robot coordinate system
        docked_haptic_rot_remapped = np.array([
            haptic_rot_axis_signs[0] * docked_haptic_rot[haptic_rot_axis_map[0]],
            haptic_rot_axis_signs[1] * docked_haptic_rot[haptic_rot_axis_map[1]],
            haptic_rot_axis_signs[2] * docked_haptic_rot[haptic_rot_axis_map[2]],
            docked_haptic_rot[3]
        ])
        docked_haptic_rot_remapped = docked_haptic_rot_remapped / np.linalg.norm(docked_haptic_rot_remapped)
        
        # Compute offset: target_robot_pose - haptic_docked (in robot space)
        # This offset will be added to haptic positions to map them to robot space
        haptic_pos_offset = target_ee_pos_docked - docked_haptic_pos
        
        # For rotation: compute the offset quaternion
        # We want: target_robot_quat = haptic_quat * rotation_offset
        # So: rotation_offset = haptic_quat^-1 * target_robot_quat
        docked_quat_isaac = torch.tensor([docked_haptic_rot_remapped[3], docked_haptic_rot_remapped[0], 
                                          docked_haptic_rot_remapped[1], docked_haptic_rot_remapped[2]], 
                                         dtype=torch.float32)
        target_quat_isaac = torch.tensor([target_ee_quat_docked[3], target_ee_quat_docked[0], 
                                          target_ee_quat_docked[1], target_ee_quat_docked[2]], 
                                         dtype=torch.float32)
        docked_quat_conj = docked_quat_isaac * torch.tensor([1, -1, -1, -1])
        haptic_rot_offset_isaac = quat_mul(target_quat_isaac.unsqueeze(0), 
                                          docked_quat_conj.unsqueeze(0)).squeeze(0)
        # Convert back to [x, y, z, w] format
        haptic_rot_offset = np.array([haptic_rot_offset_isaac[1].item(), 
                                     haptic_rot_offset_isaac[2].item(),
                                     haptic_rot_offset_isaac[3].item(), 
                                     haptic_rot_offset_isaac[0].item()])
        
        print(f"[INFO]: Calibration complete!")
        print(f"[INFO]: Docked haptic position (device): {docked_haptic_pos_device}")
        print(f"[INFO]: Docked haptic position (remapped): {docked_haptic_pos}")
        print(f"[INFO]: Target robot EE position (when docked): {target_ee_pos_docked}")
        print(f"[INFO]: Position offset: {haptic_pos_offset}")
        print(f"[INFO]: Axis mapping: Robot [X,Y,Z] = Haptic [{haptic_axis_map}] with signs {haptic_axis_signs}")
    else:
        # No calibration - use zero offsets
        haptic_pos_offset = np.array([0.0, 0.0, 0.0])
        haptic_rot_offset = np.array([0.0, 0.0, 0.0, 1.0])
    
    # Initialize haptic tracking (after env.reset())
    # Set target end-effector pose to the desired docked pose
    if enable_calibration:
        # Start from the target docked pose
        target_ee_pos = torch.tensor(target_ee_pos_docked, device=env.unwrapped.device, dtype=torch.float32)
        target_ee_quat = torch.tensor([target_ee_quat_docked[3], target_ee_quat_docked[0], 
                                      target_ee_quat_docked[1], target_ee_quat_docked[2]], 
                                     device=env.unwrapped.device, dtype=torch.float32)
    else:
        # No calibration - start from current robot pose
        target_ee_pos = ee_pos_w.clone()
        target_ee_quat = ee_quat_w.clone()
    
    # Initialize previous haptic pose in calibrated space
    raw_prev_pos_device = np.array(haptic_controller.get_scaled_position())
    raw_prev_rot_device = np.array(haptic_controller.get_rotation())
    
    # Remap haptic position axes to robot coordinate system
    raw_prev_pos = np.array([
        haptic_axis_signs[0] * raw_prev_pos_device[haptic_axis_map[0]],
        haptic_axis_signs[1] * raw_prev_pos_device[haptic_axis_map[1]],
        haptic_axis_signs[2] * raw_prev_pos_device[haptic_axis_map[2]]
    ])
    prev_haptic_pos = raw_prev_pos + haptic_pos_offset
    
    # Remap haptic rotation axes to robot coordinate system
    raw_prev_rot = np.array([
        haptic_rot_axis_signs[0] * raw_prev_rot_device[haptic_rot_axis_map[0]],
        haptic_rot_axis_signs[1] * raw_prev_rot_device[haptic_rot_axis_map[1]],
        haptic_rot_axis_signs[2] * raw_prev_rot_device[haptic_rot_axis_map[2]],
        raw_prev_rot_device[3]
    ])
    raw_prev_rot = raw_prev_rot / np.linalg.norm(raw_prev_rot)
    
    # Apply rotation offset to previous rotation
    raw_prev_quat_isaac = torch.tensor([raw_prev_rot[3], raw_prev_rot[0], 
                                       raw_prev_rot[1], raw_prev_rot[2]], 
                                      dtype=torch.float32)
    rot_offset_isaac = torch.tensor([haptic_rot_offset[3], haptic_rot_offset[0], 
                                     haptic_rot_offset[1], haptic_rot_offset[2]], 
                                    dtype=torch.float32)
    prev_quat_isaac = quat_mul(raw_prev_quat_isaac.unsqueeze(0), 
                              rot_offset_isaac.unsqueeze(0)).squeeze(0)
    prev_haptic_rot = np.array([prev_quat_isaac[1].item(), 
                                prev_quat_isaac[2].item(),
                                prev_quat_isaac[3].item(), 
                                prev_quat_isaac[0].item()])
    
    print("[INFO]: Haptic IK control ready")
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():

            # Update robot data to get current joint positions
            robot.update(dt=0.0)
            
            # Update IK solver's initial joint positions with current robot state
            # This ensures IK starts from the current robot configuration each frame
            current_joint_q = robot.data.joint_pos[0].cpu().numpy()
            joint_q_2d_np = current_joint_q.reshape(1, -1)
            # Update the Warp array in-place (assign expects numpy array or Warp array)
            joint_q_2d.assign(joint_q_2d_np)
            
            # Get current haptic pose (raw device coordinates)
            raw_haptic_pos_device = np.array(haptic_controller.get_scaled_position())
            raw_haptic_rot_device = np.array(haptic_controller.get_rotation())  # [x, y, z, w]
            
            # Remap haptic position axes to robot coordinate system
            # Robot [X, Y, Z] = Haptic [haptic_axis_map[0], haptic_axis_map[1], haptic_axis_map[2]]
            raw_haptic_pos = np.array([
                haptic_axis_signs[0] * raw_haptic_pos_device[haptic_axis_map[0]],
                haptic_axis_signs[1] * raw_haptic_pos_device[haptic_axis_map[1]],
                haptic_axis_signs[2] * raw_haptic_pos_device[haptic_axis_map[2]]
            ])
            
            # Remap haptic rotation axes to robot coordinate system
            # Quaternion [x, y, z, w] where x/y/z components represent rotation around those axes
            # We remap the x/y/z components using the same axis mapping as position
            raw_haptic_rot = np.array([
                haptic_rot_axis_signs[0] * raw_haptic_rot_device[haptic_rot_axis_map[0]],  # Robot rot X
                haptic_rot_axis_signs[1] * raw_haptic_rot_device[haptic_rot_axis_map[1]],  # Robot rot Y
                haptic_rot_axis_signs[2] * raw_haptic_rot_device[haptic_rot_axis_map[2]],  # Robot rot Z
                raw_haptic_rot_device[3]  # w component stays the same
            ])
            # Normalize the quaternion after remapping
            raw_haptic_rot = raw_haptic_rot / np.linalg.norm(raw_haptic_rot)
            
            # Apply calibration offset to map haptic device space to robot space
            haptic_pos = raw_haptic_pos + haptic_pos_offset
            
            # Apply rotation offset to haptic rotation
            raw_quat_isaac = torch.tensor([raw_haptic_rot[3], raw_haptic_rot[0], 
                                          raw_haptic_rot[1], raw_haptic_rot[2]], 
                                         device=env.unwrapped.device, dtype=torch.float32)
            rot_offset_isaac = torch.tensor([haptic_rot_offset[3], haptic_rot_offset[0], 
                                             haptic_rot_offset[1], haptic_rot_offset[2]], 
                                            device=env.unwrapped.device, dtype=torch.float32)
            haptic_quat_isaac = quat_mul(raw_quat_isaac.unsqueeze(0), 
                                        rot_offset_isaac.unsqueeze(0)).squeeze(0)
            # Convert back to [x, y, z, w] format
            haptic_rot = np.array([haptic_quat_isaac[1].item(), 
                                  haptic_quat_isaac[2].item(),
                                  haptic_quat_isaac[3].item(), 
                                  haptic_quat_isaac[0].item()])
            
            # Compute position delta from previous haptic pose (in robot space)
            delta_pos_np = haptic_pos - prev_haptic_pos
            delta_pos = torch.tensor(delta_pos_np, device=env.unwrapped.device, dtype=torch.float32)
            
            # Apply sensitivity (scale the delta)
            delta_pos = delta_pos * position_sensitivity
            
            # Clamp for safety (prevent large jumps)
            delta_pos_magnitude = torch.norm(delta_pos)
            if delta_pos_magnitude > max_position_delta:
                delta_pos = delta_pos * (max_position_delta / delta_pos_magnitude)
            
            # Compute rotation delta
            # Convert quaternions from [x, y, z, w] to [w, x, y, z] for Isaac Lab
            prev_quat = torch.tensor([prev_haptic_rot[3], prev_haptic_rot[0], 
                                     prev_haptic_rot[1], prev_haptic_rot[2]], 
                                    device=env.unwrapped.device, dtype=torch.float32)
            curr_quat = torch.tensor([haptic_rot[3], haptic_rot[0], 
                                     haptic_rot[1], haptic_rot[2]], 
                                    device=env.unwrapped.device, dtype=torch.float32)
            
            # Compute relative rotation: delta_q = curr_q * prev_q^-1
            prev_quat_conj = prev_quat * torch.tensor([1, -1, -1, -1], device=env.unwrapped.device)
            delta_quat = quat_mul(curr_quat.unsqueeze(0), prev_quat_conj.unsqueeze(0)).squeeze(0)
            
            # Convert to axis-angle for scaling
            delta_axis_angle = axis_angle_from_quat(delta_quat.unsqueeze(0)).squeeze(0)
            
            # Apply sensitivity and clamp for safety
            delta_axis_angle = delta_axis_angle * rotation_sensitivity
            angle = torch.norm(delta_axis_angle)
            if angle > max_rotation_delta:
                delta_axis_angle = delta_axis_angle * (max_rotation_delta / angle)
            
            # Convert back to quaternion
            if angle > 1e-6:
                axis = delta_axis_angle / angle
                delta_quat_scaled = quat_from_angle_axis(angle.unsqueeze(0), axis.unsqueeze(0)).squeeze(0)
            else:
                delta_quat_scaled = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)
            
            # Update target end-effector pose incrementally
            target_ee_pos += delta_pos
            target_ee_quat = quat_mul(delta_quat_scaled.unsqueeze(0), target_ee_quat.unsqueeze(0)).squeeze(0)
            
            # Normalize quaternion to prevent drift
            target_ee_quat = target_ee_quat / torch.norm(target_ee_quat)
            
            # Update IK objectives with new target
            pos_obj.set_target_position(0, wp.vec3(target_ee_pos[0].item(), 
                                                   target_ee_pos[1].item(), 
                                                   target_ee_pos[2].item()))
            rot_obj.set_target_rotation(0, wp.vec4(target_ee_quat[1].item(), 
                                                   target_ee_quat[2].item(),
                                                   target_ee_quat[3].item(), 
                                                   target_ee_quat[0].item()))
            
            # Solve IK to get joint positions (increased iterations for better precision)
            ik_solver.solve(iterations=50)
            
            # Get solved joint positions from IK (joint_q_2d is updated in-place by IK solver)
            # joint_q_2d contains all joints in Newton model order, which matches robot.data.joint_pos order
            solved_joint_q_all = wp.to_torch(joint_q_2d)[0].cpu()  # Shape: (num_joints_in_model,)
            
            # Extract action-controlled joints using action_joint_ids
            # action_joint_ids are indices into robot.data.joint_pos, which matches Newton model order
            if isinstance(action_joint_ids, slice):
                solved_joint_q = solved_joint_q_all[action_joint_ids]
            elif isinstance(action_joint_ids, list):
                solved_joint_q = solved_joint_q_all[action_joint_ids]
            else:
                # If it's a tensor, convert to list
                solved_joint_q = solved_joint_q_all[action_joint_ids.cpu().tolist() if isinstance(action_joint_ids, torch.Tensor) else action_joint_ids]
            
            # Convert absolute joint positions to action space format:
            # action = (target_joint_pos - default_joint_pos) / scale
            joint_offset = solved_joint_q - default_joint_pos
            
            if isinstance(action_scale, np.ndarray):
                actions = (joint_offset / torch.tensor(action_scale, dtype=torch.float32)).unsqueeze(0)
            else:
                actions = (joint_offset / action_scale).unsqueeze(0)
            
            actions = actions.to(device=env.unwrapped.device)  # Shape: (1, num_action_joints)
            
            # Check button state for gripper control
            button_pressed = haptic_controller.is_button_pressed()
            
            # Control gripper based on button state
            if has_gripper:
                # Use separate gripper action term
                # Gripper action: 1.0 = close, -1.0 = open (or vice versa)
                gripper_value = 1.0 if button_pressed else -1.0
                # This would need to be handled separately if gripper_action is a different term
                # For now, we just print the state
                if button_pressed:
                    print("[GRIP]: Closing gripper")
            elif gripper_joint_ids is not None:
                # Directly control gripper joints using config values
                gripper_target = gripper_pos_closed if button_pressed else gripper_pos_open
                
                # Set gripper joint targets directly on the robot
                gripper_targets = torch.tensor([[gripper_target] * len(gripper_joint_ids)], 
                                              device=env.unwrapped.device)
                robot.set_joint_position_target(gripper_targets, joint_ids=gripper_joint_ids)
            
            # Update previous state (use calibrated haptic pose for delta computation)
            prev_haptic_pos = haptic_pos.copy()
            prev_haptic_rot = haptic_rot.copy()
            
            # Apply actions (arm only, gripper is controlled separately)
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
