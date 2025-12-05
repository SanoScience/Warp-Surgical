# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# import mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.sim._impl.solvers_cfg import XPBOSolverCfg
from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg
import math

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets import SO_ARM100_CFG  # isort: skip

##
# Scene definition
##

# ----------------------------------------------------------------
# --------------- LycheeAI live asset ----------------------------
# ----------------------------------------------------------------


@configclass
class SoArm100ReachEnvCfg(ReachEnvCfg):
    sim: SimulationCfg = SimulationCfg(
        # dt=1/60.0,
        newton_cfg=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=20,
                nconmax=20,
                ls_iterations=10,
                cone="pyramidal",
                impratio=1,
                ls_parallel=True,
                integrator="implicit",
                save_to_mjcf="FrankaReachEnv.xml",
            ),
            num_substeps=64,
            debug_mode=True,
        )
    )


    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = SO_ARM100_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["Fixed_Gripper"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["Fixed_Gripper"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["Fixed_Gripper"]

        # TODO: reorient command target

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Shoulder_Rotation", "Shoulder_Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            scale=0.5,
            use_default_offset=True,
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = ["Fixed_Gripper"]
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class SoArm100ReachEnvCfg_PLAY(SoArm100ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


# ----------------------------------------------------------------
# --------------- ROSCON ES 2025 asset ---------------------------
# ----------------------------------------------------------------


@configclass
class SoArm100ReachRosConEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = SO_ARM100_ROSCON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["wrist_2_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["wrist_2_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["wrist_2_link"]

        self.rewards.end_effector_orientation_tracking.weight = 0.0

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.5,
            use_default_offset=True,
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = ["wrist_2_link"]
        # self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class SoArm100ReachRosConEnvCfg_PLAY(SoArm100ReachRosConEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
