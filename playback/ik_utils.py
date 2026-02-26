import numpy as np
import warp as wp

from ik_mimic_objective import IKMimicObjective


def get_joint_coord_indices(model, joint_names):
    joint_names = set(joint_names)
    q_start = model.joint_q_start.numpy()
    coord_indices = {
        int(q_start[joint_idx])
        for joint_idx, name in enumerate(model.joint_key)
        if name in joint_names
    }
    return sorted(coord_indices)


def compute_midpoint_values(model, coord_indices):
    if not coord_indices:
        return np.array([], dtype=np.float32)
    lower_np = model.joint_limit_lower.numpy()
    upper_np = model.joint_limit_upper.numpy()
    joint_q_np = model.joint_q.numpy()
    mids = []
    for coord_idx in coord_indices:
        lo = float(lower_np[coord_idx])
        hi = float(upper_np[coord_idx])
        if np.isfinite(lo) and np.isfinite(hi):
            mids.append(0.5 * (lo + hi))
        else:
            mids.append(float(joint_q_np[coord_idx]))
    return np.array(mids, dtype=np.float32)


def lock_joint_coords(model, coord_indices, coord_values):
    if not coord_indices:
        return
    joint_q_np = model.joint_q.numpy()
    lower_np = model.joint_limit_lower.numpy()
    upper_np = model.joint_limit_upper.numpy()
    for coord_idx, value in zip(coord_indices, coord_values):
        joint_q_np[coord_idx] = value
        lower_np[coord_idx] = value
        upper_np[coord_idx] = value
    model.joint_q.assign(wp.array(joint_q_np, dtype=wp.float32, device=model.device))
    model.joint_limit_lower.assign(wp.array(lower_np, dtype=wp.float32, device=model.device))
    model.joint_limit_upper.assign(wp.array(upper_np, dtype=wp.float32, device=model.device))


def lock_constant_joints(model, joint_names):
    coord_indices = get_joint_coord_indices(model, joint_names)
    coord_values = compute_midpoint_values(model, coord_indices)
    if coord_indices:
        print(f"Locking joints {joint_names} at midpoints for coords {coord_indices}")
    lock_joint_coords(model, coord_indices, coord_values)
    return coord_indices, coord_values


def build_multi_robot_mimic_objective(model, mimic_mapping, num_robots, weight=100.0):
    """Build mimic objective that handles multiple robots with duplicate joint names."""
    joint_name_indices = {}
    for idx, name in enumerate(model.joint_key):
        if name not in joint_name_indices:
            joint_name_indices[name] = []
        joint_name_indices[name].append(idx)

    q_start = model.joint_q_start.numpy()
    mimic_joints = []

    for follower_name, (leader_name, mult, offset) in mimic_mapping.items():
        if follower_name in joint_name_indices and leader_name in joint_name_indices:
            follower_indices = joint_name_indices[follower_name]
            leader_indices = joint_name_indices[leader_name]
            for f_joint_idx, l_joint_idx in zip(follower_indices, leader_indices):
                mimic_joints.append(
                    [
                        float(q_start[f_joint_idx]),
                        float(q_start[l_joint_idx]),
                        float(mult),
                        float(offset),
                    ]
                )

    if not mimic_joints:
        return None

    print(f"Built {len(mimic_joints)} mimic joint constraints for {num_robots} robots")

    mimic_obj = IKMimicObjective(model, {}, weight=weight)
    mimic_obj.n_mimic = len(mimic_joints)
    mimic_obj.mimic_info_np = np.array(mimic_joints, dtype=np.float32)
    return mimic_obj
