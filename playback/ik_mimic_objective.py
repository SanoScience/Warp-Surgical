"""
IK objective for enforcing mimic joint constraints.

Mimic joints follow a leader joint: q_follower = offset + multiplier * q_leader
This objective adds residuals to the IK solver to enforce these constraints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from newton.ik import IKObjective

if TYPE_CHECKING:
    import newton


@wp.kernel
def _mimic_residuals(
    joint_q: wp.array2d(dtype=wp.float32),
    mimic_info: wp.array2d(dtype=wp.float32),  # [follower_q, leader_q, multiplier, offset]
    n_mimic: int,
    weight: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    """Compute residual: r = weight * (q_follower - expected_q)"""
    problem_idx, mimic_idx = wp.tid()
    if mimic_idx >= n_mimic:
        return

    follower_q = wp.int32(mimic_info[mimic_idx, 0])
    leader_q = wp.int32(mimic_info[mimic_idx, 1])
    multiplier = mimic_info[mimic_idx, 2]
    offset = mimic_info[mimic_idx, 3]

    expected = offset + multiplier * joint_q[problem_idx, leader_q]
    error = joint_q[problem_idx, follower_q] - expected
    residuals[problem_idx, start_idx + mimic_idx] = weight * error


@wp.kernel
def _mimic_jac_analytic(
    mimic_info: wp.array2d(dtype=wp.float32),
    n_mimic: int,
    weight: float,
    start_idx: int,
    coord_to_dof: wp.array1d(dtype=wp.int32),
    jacobian: wp.array3d(dtype=wp.float32),
):
    """Jacobian: dr/dq_follower = weight, dr/dq_leader = -weight * multiplier"""
    problem_idx, mimic_idx = wp.tid()
    if mimic_idx >= n_mimic:
        return

    follower_q = wp.int32(mimic_info[mimic_idx, 0])
    leader_q = wp.int32(mimic_info[mimic_idx, 1])
    multiplier = mimic_info[mimic_idx, 2]
    row = start_idx + mimic_idx

    follower_dof = coord_to_dof[follower_q]
    leader_dof = coord_to_dof[leader_q]

    if follower_dof >= 0:
        jacobian[problem_idx, row, follower_dof] = weight
    if leader_dof >= 0:
        jacobian[problem_idx, row, leader_dof] = -weight * multiplier


class IKMimicObjective(IKObjective):
    """
    IK objective that penalizes mimic constraint violations.

    Higher weight = harder constraint (100.0 works well in practice).

    Args:
        model: Newton model (vanilla, no mimic attributes needed)
        mimic_mapping: Dict mapping follower joint name to (leader_name, multiplier, offset)
                       Example: {'psm_pitch_end_joint': ('psm_pitch_back_joint', 1.0, 0.0)}
        weight: Constraint weight (default 100.0)
    """

    def __init__(
        self,
        model: newton.Model,
        mimic_mapping: dict[str, tuple[str, float, float]],
        weight: float = 100.0,
    ) -> None:
        super().__init__()
        self.weight = weight

        # Build mimic info from external configuration
        joint_name_to_idx = {key: i for i, key in enumerate(model.joint_key)}
        q_start = model.joint_q_start.numpy()

        # Build mimic info: [follower_q_idx, leader_q_idx, multiplier, offset]
        mimic_joints = []
        for follower_name, (leader_name, mult, offset) in mimic_mapping.items():
            if follower_name in joint_name_to_idx and leader_name in joint_name_to_idx:
                follower_idx = joint_name_to_idx[follower_name]
                leader_idx = joint_name_to_idx[leader_name]
                mimic_joints.append([
                    float(q_start[follower_idx]),   # follower q index
                    float(q_start[leader_idx]),     # leader q index
                    float(mult),
                    float(offset)
                ])

        self.n_mimic = len(mimic_joints)
        self.mimic_info_np = np.array(mimic_joints, dtype=np.float32) if mimic_joints else np.zeros((1, 4), dtype=np.float32)
        self.mimic_info = None
        self.coord_to_dof = None

        print(f"IKMimicObjective: {self.n_mimic} mimic joints")

    def bind_device(self, device: str) -> None:
        super().bind_device(device)
        self.mimic_info = wp.array(self.mimic_info_np, dtype=wp.float32, device=device)

    def init_buffers(self, model: newton.Model, jacobian_mode: str) -> None:
        self._require_batch_layout()

        # Map joint coordinate indices to DOF indices (for Jacobian)
        coord_to_dof = np.full(model.joint_coord_count, -1, dtype=np.int32)
        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        dof_dim = model.joint_dof_dim.numpy()

        for j in range(model.joint_count):
            for k in range(dof_dim[j, 0] + dof_dim[j, 1]):
                coord_to_dof[q_start[j] + k] = qd_start[j] + k

        self.coord_to_dof = wp.array(coord_to_dof, dtype=wp.int32, device=self.device)

    def residual_dim(self) -> int:
        return self.n_mimic

    def supports_analytic(self) -> bool:
        return True

    def compute_residuals(
        self,
        body_q: wp.array,
        joint_q: wp.array,
        model: newton.Model,
        residuals: wp.array,
        start_idx: int,
        problem_idx: int,
    ) -> None:
        if self.n_mimic == 0:
            return
        wp.launch(_mimic_residuals, dim=[joint_q.shape[0], self.n_mimic],
                  inputs=[joint_q, self.mimic_info, self.n_mimic, self.weight, start_idx],
                  outputs=[residuals], device=self.device)

    def compute_jacobian_analytic(
        self,
        body_q: wp.array,
        joint_q: wp.array,
        model: newton.Model,
        jacobian: wp.array,
        joint_S_s: wp.array,
        start_idx: int,
    ) -> None:
        if self.n_mimic == 0:
            return
        wp.launch(_mimic_jac_analytic, dim=[joint_q.shape[0], self.n_mimic],
                  inputs=[self.mimic_info, self.n_mimic, self.weight, start_idx, self.coord_to_dof],
                  outputs=[jacobian], device=self.device)

    def compute_jacobian_autodiff(
        self,
        tape: wp.Tape,
        model: newton.Model,
        jacobian: wp.array,
        start_idx: int,
        dq_dof: wp.array,
    ) -> None:
        pass  
