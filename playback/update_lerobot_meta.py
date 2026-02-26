import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow.parquet as pq


def _load_info(info_path):
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_info(info_path, info):
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, sort_keys=False)


def _iter_parquet_files(data_root):
    for path in sorted(Path(data_root).rglob("*.parquet")):
        yield path


def _get_list_size_from_parquet(parquet_path, column_name):
    table = pq.read_table(parquet_path, columns=[column_name])
    field = table.schema.field(column_name)
    if not str(field.type).startswith("fixed_size_list"):
        raise ValueError(f"{column_name} is not a fixed_size_list in {parquet_path}")
    return field.type.list_size


def _infer_num_instruments(info):
    names = info["features"]["action.cartesian_absolute"]["names"]
    per_inst = 8
    if len(names) % per_inst != 0:
        raise ValueError("Cannot infer number of instruments from action.cartesian_absolute names.")
    return len(names) // per_inst


def _load_urdf_joint_names(urdf_path):
    if urdf_path is None:
        return None
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = []
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type", "")
        if joint_type == "fixed":
            continue
        name = joint.attrib.get("name")
        if name:
            joints.append(name)
    return joints or None


def _make_joint_names(num_instruments, dof_count, urdf_joint_names=None):
    if urdf_joint_names and len(urdf_joint_names) == dof_count:
        names = []
        for i in range(num_instruments):
            for joint_name in urdf_joint_names:
                names.append(f"inst_{i}_{joint_name}")
        return names
    names = []
    for i in range(num_instruments):
        for j in range(dof_count):
            names.append(f"inst_{i}_joint_{j}")
    return names


def _compute_stats(values):
    stats = {
        "min": np.nanmin(values, axis=0).tolist(),
        "max": np.nanmax(values, axis=0).tolist(),
        "mean": np.nanmean(values, axis=0).tolist(),
        "std": np.nanstd(values, axis=0).tolist(),
        "count": [int(values.shape[0])],
    }
    return stats


def _read_joint_arrays(parquet_path, action_name, obs_pos_name, obs_vel_name):
    table = pq.read_table(parquet_path, columns=[action_name, obs_pos_name, obs_vel_name])
    action = table.column(action_name).combine_chunks().values.to_numpy(zero_copy_only=False)
    obs_pos = table.column(obs_pos_name).combine_chunks().values.to_numpy(zero_copy_only=False)
    obs_vel = table.column(obs_vel_name).combine_chunks().values.to_numpy(zero_copy_only=False)
    action = action.reshape((table.num_rows, -1)).astype(np.float32)
    obs_pos = obs_pos.reshape((table.num_rows, -1)).astype(np.float32)
    obs_vel = obs_vel.reshape((table.num_rows, -1)).astype(np.float32)
    return action, obs_pos, obs_vel


def _update_info_features(
    info,
    action_list_size,
    obs_pos_list_size,
    obs_vel_list_size,
    urdf_joint_names=None,
):
    num_instruments = _infer_num_instruments(info)
    if action_list_size % num_instruments != 0:
        raise ValueError("action.joint_positions list size not divisible by number of instruments.")
    if obs_pos_list_size % num_instruments != 0:
        raise ValueError("observation.state.joint_positions list size not divisible by number of instruments.")
    if obs_vel_list_size % num_instruments != 0:
        raise ValueError("observation.state.joint_velocities list size not divisible by number of instruments.")
    dof_count = action_list_size // num_instruments
    if obs_pos_list_size != action_list_size or obs_vel_list_size != action_list_size:
        raise ValueError("Joint list sizes do not match across action/observation columns.")
    joint_names = _make_joint_names(num_instruments, dof_count, urdf_joint_names=urdf_joint_names)

    info["features"]["action.joint_positions"] = {
        "dtype": "float32",
        "shape": [action_list_size],
        "names": joint_names,
    }
    info["features"]["observation.state.joint_positions"] = {
        "dtype": "float32",
        "shape": [obs_pos_list_size],
        "names": joint_names,
    }
    info["features"]["observation.state.joint_velocities"] = {
        "dtype": "float32",
        "shape": [obs_vel_list_size],
        "names": joint_names,
    }


def _update_episode_stats(stats_path, data_root):
    action_name = "action.joint_positions"
    obs_pos_name = "observation.state.joint_positions"
    obs_vel_name = "observation.state.joint_velocities"

    episode_stats = {}
    order = []
    with open(stats_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            episode_stats[record["episode_index"]] = record
            order.append(record["episode_index"])

    for parquet_path in _iter_parquet_files(data_root):
        episode_name = parquet_path.stem
        if not episode_name.startswith("episode_"):
            continue
        episode_index = int(episode_name.split("_")[1])
        action_vals, obs_pos_vals, obs_vel_vals = _read_joint_arrays(
            parquet_path,
            action_name,
            obs_pos_name,
            obs_vel_name,
        )
        stats = episode_stats.get(episode_index)
        if stats is None:
            stats = {"episode_index": episode_index, "stats": {}}
        stats["stats"][action_name] = _compute_stats(action_vals)
        stats["stats"][obs_pos_name] = _compute_stats(obs_pos_vals)
        stats["stats"][obs_vel_name] = _compute_stats(obs_vel_vals)
        episode_stats[episode_index] = stats

    with open(stats_path, "w", encoding="utf-8") as f:
        for ep in order:
            f.write(json.dumps(episode_stats[ep]) + "\n")


def update_lerobot_metadata(dataset_root, urdf_path=None):
    dataset_root = Path(dataset_root)
    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    data_root = dataset_root / "data"

    info = _load_info(info_path)

    sample_parquet = next(_iter_parquet_files(data_root), None)
    if sample_parquet is None:
        raise RuntimeError("No parquet files found under data/.")

    action_list_size = _get_list_size_from_parquet(sample_parquet, "action.joint_positions")
    obs_pos_list_size = _get_list_size_from_parquet(
        sample_parquet,
        "observation.state.joint_positions",
    )
    obs_vel_list_size = _get_list_size_from_parquet(
        sample_parquet,
        "observation.state.joint_velocities",
    )

    urdf_joint_names = _load_urdf_joint_names(urdf_path)
    if urdf_joint_names is None:
        raise RuntimeError("No joint names found in URDF.")
    _update_info_features(
        info,
        action_list_size,
        obs_pos_list_size,
        obs_vel_list_size,
        urdf_joint_names=urdf_joint_names,
    )
    _save_info(info_path, info)

    _update_episode_stats(stats_path, data_root)


def main():
    parser = argparse.ArgumentParser(description="Update LeRobot v2.1 metadata for joint columns.")
    parser.add_argument("dataset_root", help="Path to dataset root (contains meta/ and data/).")
    default_urdf = os.path.join(os.path.dirname(__file__), "assets", "psm", "psm_RL.urdf")
    parser.add_argument(
        "--urdf_path",
        default=default_urdf,
        help="URDF path to source joint names (defaults to playback/assets/psm/psm_RL.urdf).",
    )
    args = parser.parse_args()

    update_lerobot_metadata(args.dataset_root, urdf_path=args.urdf_path)
    print("Updated info.json and episodes_stats.jsonl for joint columns.")


if __name__ == "__main__":
    main()
