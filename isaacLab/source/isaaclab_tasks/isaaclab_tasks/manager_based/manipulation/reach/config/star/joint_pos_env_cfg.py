# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg, FeatherstoneSolverCfg, XPBOSolverCfg
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.assets import AssetBaseCfg, DeformableObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_angle_axis, quat_mul
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
import torch
import math

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets import STAR_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class STARReachEnvCfg(ReachEnvCfg):
    sim: SimulationCfg = SimulationCfg(
        dt=1/60.0,
        newton_cfg=NewtonCfg(
            # use_cuda_graph = False,
            solver_cfg=XPBOSolverCfg(
                iterations=16,
                soft_body_relaxation=0.0001,
            ),
            num_substeps=32,
            debug_mode=True,
        )
    )
    # # Working Mujoco Solver
    # sim: SimulationCfg = SimulationCfg(
    #     dt=1/60.0,
    #     newton_cfg=NewtonCfg(
    #         solver_cfg=MJWarpSolverCfg(
    #             njmax=20,
    #             nconmax=20,
    #             ls_iterations=10,
    #             cone="pyramidal",
    #             impratio=1,
    #             ls_parallel=True,
    #             integrator="implicit",
    #             save_to_mjcf="StarReachEnv.xml",
    #         ),
    #         num_substeps=10,
    #         debug_mode=True,
    #     )
    # )
    # # Working Featherstone Solver
    # sim: SimulationCfg = SimulationCfg(
    #     newton_cfg=NewtonCfg(
    #         solver_cfg=FeatherstoneSolverCfg(
    #             update_mass_matrix_interval=10,  # Update mass matrix every 4 substeps
    #             angular_damping=0.1,  # Add damping to prevent instability
    #             friction_smoothing=4.0, # Increase friction smoothing
    #         ),
    #         num_substeps=10,
    #         debug_mode=True,
    #     )
    # )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Register callback to set model parameters after model is created
        def set_soft_contact_params():
            model = NewtonManager.get_model()
            if model is not None:
                model.soft_contact_ke = 1.0e6  # Increased from 1.0e4 for stiffer contact
                model.soft_contact_kd = 1.0e3  # Increased damping for stability
                model.soft_contact_mu = 0.5
                model.soft_contact_restitution = 0.0  # Reduce bounce
        
        NewtonManager.add_on_start_callback(set_soft_contact_params)

        # switch robot to star
        self.scene.robot = STAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["endo360_needle"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["endo360_needle"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["endo360_needle"]
        # # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "endo360_needle"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        # # simulation settings
        self.viewer.eye = (2.0, 2.0, 1.0)

        # Deformable Liver
        zRot = quat_from_angle_axis(torch.tensor(-90), torch.tensor([0.0, 0.0, 1.0])).tolist()
        self.scene.liver = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Liver",
            spawn=sim_utils.UsdFileCfg(
                usd_path='../meshes/liver/liver.usd',
                # scale=(1.0, 1.0, 1.0)
                scale=(0.1, 0.1, 0.1)
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.15), rot=zRot),
            debug_vis=True,
        )

        # override command generator body
        # Use liver vertex position as target instead of random poses
        # self.commands.ee_pose = mdp.LiverVertexCommandTermCfg(
        #     vertex_index=0,  # Track vertex 199 of liver mesh
        #     resampling_time_range=(4.0, 4.0),
        #     debug_vis=True,
        # )

        # self.events.reset_robot_joints = EventTerm(
        #     func=mdp.reset_joints_by_scale,
        #     mode="reset",
        #     params={
        #         "position_range": (0.5, 1.5),
        #         "velocity_range": (0.0, 0.0),
        #     },
        # )

@configclass
class STARReachEnvCfg_PLAY(STARReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
