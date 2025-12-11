# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.sim import SimulationCfg
from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg, FeatherstoneSolverCfg, XPBOSolverCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip
from isaaclab_tasks.manager_based.manipulation.reach.config.franka.haptic_cfg import HapticControlCfg


##
# Environment configuration
##


@configclass
class FrankaReachEnvCfg(ReachEnvCfg):
    # sim: SimulationCfg = SimulationCfg(
    #     newton_cfg=NewtonCfg(
    #         solver_cfg=XPBOSolverCfg(
    #             iterations=16,
    #             joint_linear_compliance=0.0,  # Zero compliance for rigid joints (matching Newton example)
    #             joint_angular_compliance=0.0,  # Zero compliance for rigid joints (matching Newton example)
    #             joint_linear_relaxation=0.7,
    #             joint_angular_relaxation=0.4,
    #             angular_damping=0.1,
    #         ),
    #         num_substeps=32,  # Increase substeps for better stability
    #         debug_mode=True,
    #         # use_cuda_graph=False,  # Disable CUDA graph to allow debug output
    #     )
    # )
    sim: SimulationCfg = SimulationCfg(
        newton_cfg=NewtonCfg(
            solver_cfg=FeatherstoneSolverCfg(
                update_mass_matrix_interval=10,  # Update mass matrix every 10 substeps
                # angular_damping=0.1,  # Add damping to prevent instability (matching Star)
                # friction_smoothing=4.0,  # Increase friction smoothing (matching Star)
            ),
            num_substeps=10,  # Increase substeps for better stability
            debug_mode=True,
            # use_cuda_graph=False,  # Disable CUDA graph to allow debug output
        )
    )

    # # Working MuJoCo settings
    # sim: SimulationCfg = SimulationCfg(
    #     newton_cfg=NewtonCfg(
    #         solver_cfg=MJWarpSolverCfg(
    #             njmax=20,
    #             nconmax=20,
    #             ls_iterations=10,
    #             cone="pyramidal",
    #             impratio=1,
    #             ls_parallel=True,
    #             integrator="implicit",
    #             save_to_mjcf="FrankaReachEnv.xml",
    #         ),
    #         num_substeps=10,  # Increase substeps for better stability
    #         debug_mode=True,
    #         # use_cuda_graph=False,
    #     )
    # )

    # Haptic device control configuration
    haptic: HapticControlCfg = HapticControlCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["panda_hand"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "panda_hand"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class FrankaReachEnvCfg_PLAY(FrankaReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
