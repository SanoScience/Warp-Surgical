from __future__ import annotations

import json
import os
import shutil
import re
from dataclasses import dataclass

import numpy as np


ACTION_FIELDS = ("x", "y", "z", "qx", "qy", "qz", "qw")
OBS_TIP_POS_FIELDS = ("x", "y", "z")
OBS_TIP_ROT_FIELDS = ("x", "y", "z", "w")

ACTION_RE = re.compile(r"^action\.cartesian_absolute/inst_(\d+)_(x|y|z|qx|qy|qz|qw)$")
OBS_TIP_POS_RE = re.compile(r"^observation\.state/inst_(\d+)_tip_pos_(x|y|z)$")
OBS_TIP_ROT_RE = re.compile(r"^observation\.state/inst_(\d+)_tip_rot_(x|y|z|w)$")


@dataclass
class ParquetInstrumentData:
    instrument_ids: list[int]
    action: np.ndarray
    observation_tip_pos: np.ndarray
    observation_tip_rot: np.ndarray
    action_list: np.ndarray | None
    observation_list: np.ndarray | None
    action_block: int | None
    observation_block: int | None
    source_is_list: bool
    columns_used: dict[str, list[str]]
    columns_missing: dict[str, list[str]]


def _get_parquet_columns(path: str) -> list[str]:
    errors = []
    try:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.dataset as ds  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415

        if os.path.isdir(path):
            dataset = ds.dataset(path, format="parquet")
            return dataset.schema.names
        parquet_file = pq.ParquetFile(path)
        return parquet_file.schema.names
    except Exception as exc:
        errors.append(f"pyarrow.ParquetFile: {exc}")
        try:
            import pandas as pd  # noqa: PLC0415

            return list(pd.read_parquet(path, engine="pyarrow", columns=[]).columns)
        except Exception as exc2:
            errors.append(f"pandas pyarrow engine: {exc2}")
            try:
                import pandas as pd  # noqa: PLC0415

                return list(pd.read_parquet(path, engine="fastparquet", columns=[]).columns)
            except Exception as exc:
                errors.append(f"pandas fastparquet engine: {exc}")
                detail = "; ".join(errors)
                raise RuntimeError(
                    "Unable to read parquet schema. Verify the file path and format, "
                    "and ensure a parquet engine is installed. Errors: "
                    f"{detail}"
                ) from exc


def _read_parquet(path: str, columns: list[str]):
    try:
        import pyarrow.dataset as ds  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415

        if os.path.isdir(path):
            dataset = ds.dataset(path, format="parquet")
            table = dataset.to_table(columns=columns)
        else:
            table = pq.read_table(path, columns=columns)
        return table.to_pandas()
    except Exception:
        try:
            import pandas as pd  # noqa: PLC0415

            return pd.read_parquet(path, engine="pyarrow", columns=columns)
        except Exception:
            import pandas as pd  # noqa: PLC0415

            return pd.read_parquet(path, engine="fastparquet", columns=columns)


def _discover_instrument_ids(columns: list[str]) -> list[int]:
    instrument_ids = set()
    for col in columns:
        match = ACTION_RE.match(col) or OBS_TIP_POS_RE.match(col) or OBS_TIP_ROT_RE.match(col)
        if match:
            instrument_ids.add(int(match.group(1)))
    return sorted(instrument_ids)


def _read_list_columns_table(path: str, columns: list[str]):
    import pyarrow.dataset as ds  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    if os.path.isdir(path):
        dataset = ds.dataset(path, format="parquet")
        return dataset.to_table(columns=columns)
    return pq.read_table(path, columns=columns)


def _fixed_size_list_to_numpy(table, column_name: str) -> np.ndarray:
    import pyarrow as pa  # noqa: PLC0415

    col = table.column(column_name)
    if hasattr(col, "combine_chunks"):
        col = col.combine_chunks()
    if not pa.types.is_fixed_size_list(col.type):
        raise ValueError(f"Column '{column_name}' is not a fixed-size list.")
    list_size = col.type.list_size
    values = col.values.to_numpy(zero_copy_only=False)
    return values.reshape((len(col), list_size))


def _fixed_size_list_array_from_numpy(values: np.ndarray, list_size: int):
    import pyarrow as pa  # noqa: PLC0415

    flat = values.reshape(-1).astype(np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), list_size)


def _update_hf_metadata(table, feature_lengths: dict[str, int]):
    metadata = table.schema.metadata or {}
    hf_key = None
    for key in metadata.keys():
        if key.decode("utf-8", errors="ignore") == "huggingface":
            hf_key = key
            break
    if hf_key is None:
        return table

    try:
        hf_meta = json.loads(metadata[hf_key].decode("utf-8"))
    except Exception:
        return table

    info = hf_meta.get("info", {})
    features = info.get("features", {})
    if not isinstance(features, dict):
        return table

    template = features.get("action.cartesian_absolute")
    def make_feature(length):
        if isinstance(template, dict):
            new = json.loads(json.dumps(template))
            if "length" in new:
                new["length"] = length
            elif "size" in new:
                new["size"] = length
            else:
                new["length"] = length
            return new
        return {"_type": "FixedSizeList", "length": length, "feature": {"dtype": "float32", "_type": "Value"}}

    for name, length in feature_lengths.items():
        features[name] = make_feature(length)

    info["features"] = features
    hf_meta["info"] = info
    metadata[hf_key] = json.dumps(hf_meta).encode("utf-8")
    return table.replace_schema_metadata(metadata)


def _collect_block(df, instrument_ids, prefix, fields):
    num_frames = len(df)
    num_instr = len(instrument_ids)
    block = np.full((num_frames, num_instr, len(fields)), np.nan, dtype=np.float32)
    used = []
    missing = []
    for i, inst_id in enumerate(instrument_ids):
        for j, field in enumerate(fields):
            col = f"{prefix}{inst_id}_{field}"
            if col in df.columns:
                block[:, i, j] = df[col].to_numpy(dtype=np.float32)
                used.append(col)
            else:
                missing.append(col)
    return block, used, missing


def load_parquet_instrument_data(path: str) -> ParquetInstrumentData:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    columns = _get_parquet_columns(path)
    instrument_ids = _discover_instrument_ids(columns)

    list_action_col = "action.cartesian_absolute"
    list_obs_col = "observation.state"
    has_list_cols = list_action_col in columns or list_obs_col in columns

    if not instrument_ids and not has_list_cols:
        raise ValueError(
            "No instrument columns found in parquet file. Expected either flattened columns "
            "like 'action.cartesian_absolute/inst_0_x' or list columns such as "
            "'action.cartesian_absolute' and 'observation.state'."
        )

    if not instrument_ids and has_list_cols:
        table = _read_list_columns_table(path, [c for c in [list_action_col, list_obs_col] if c in columns])
        action_raw = None
        obs_raw = None
        action_len = None
        obs_len = None

        if list_action_col in columns:
            action_raw = _fixed_size_list_to_numpy(table, list_action_col)
            action_len = action_raw.shape[1]
        if list_obs_col in columns:
            obs_raw = _fixed_size_list_to_numpy(table, list_obs_col)
            obs_len = obs_raw.shape[1]

        if action_len is None:
            raise ValueError("Parquet list columns missing 'action.cartesian_absolute'.")

        if action_len % 7 == 0:
            block_action = 7
        elif action_len % 8 == 0:
            block_action = 8
        else:
            raise ValueError(
                f"Unable to infer instrument count from action length {action_len}; "
                "expected divisible by 7 or 8."
            )
        num_instruments = action_len // block_action
        instrument_ids = list(range(num_instruments))

        action_blocked = action_raw.reshape((-1, num_instruments, block_action))
        action = action_blocked[:, :, :7].astype(np.float32)

        if obs_raw is not None:
            if obs_len % num_instruments != 0:
                raise ValueError(
                    f"Observation length {obs_len} not divisible by {num_instruments} instruments."
                )
            block_obs = obs_len // num_instruments
            if block_obs < 7:
                raise ValueError(
                    f"Observation block size {block_obs} is too small to contain pos+rot."
                )
            obs_blocked = obs_raw.reshape((-1, num_instruments, block_obs))
            observation_tip_pos = obs_blocked[:, :, 0:3].astype(np.float32)
            observation_tip_rot = obs_blocked[:, :, 3:7].astype(np.float32)
        else:
            observation_tip_pos = np.full((action.shape[0], num_instruments, 3), np.nan, dtype=np.float32)
            observation_tip_rot = np.full((action.shape[0], num_instruments, 4), np.nan, dtype=np.float32)
            block_obs = None

        columns_used = {"action": [list_action_col], "observation_tip_pos": [list_obs_col], "observation_tip_rot": [list_obs_col]}
        columns_missing = {"action": [], "observation_tip_pos": [], "observation_tip_rot": []}

        return ParquetInstrumentData(
            instrument_ids=instrument_ids,
            action=action,
            observation_tip_pos=observation_tip_pos,
            observation_tip_rot=observation_tip_rot,
            action_list=action_raw,
            observation_list=obs_raw,
            action_block=block_action,
            observation_block=block_obs,
            source_is_list=True,
            columns_used=columns_used,
            columns_missing=columns_missing,
        )

    action_cols = [
        f"action.cartesian_absolute/inst_{inst_id}_{field}"
        for inst_id in instrument_ids
        for field in ACTION_FIELDS
    ]
    obs_pos_cols = [
        f"observation.state/inst_{inst_id}_tip_pos_{field}"
        for inst_id in instrument_ids
        for field in OBS_TIP_POS_FIELDS
    ]
    obs_rot_cols = [
        f"observation.state/inst_{inst_id}_tip_rot_{field}"
        for inst_id in instrument_ids
        for field in OBS_TIP_ROT_FIELDS
    ]
    needed_columns = [c for c in action_cols + obs_pos_cols + obs_rot_cols if c in columns]

    df = _read_parquet(path, needed_columns)

    action, action_used, action_missing = _collect_block(
        df, instrument_ids, "action.cartesian_absolute/inst_", ACTION_FIELDS
    )
    obs_pos_fields = tuple(f"tip_pos_{field}" for field in OBS_TIP_POS_FIELDS)
    obs_rot_fields = tuple(f"tip_rot_{field}" for field in OBS_TIP_ROT_FIELDS)
    obs_pos, obs_pos_used, obs_pos_missing = _collect_block(
        df, instrument_ids, "observation.state/inst_", obs_pos_fields
    )
    obs_rot, obs_rot_used, obs_rot_missing = _collect_block(
        df, instrument_ids, "observation.state/inst_", obs_rot_fields
    )

    columns_used = {
        "action": action_used,
        "observation_tip_pos": obs_pos_used,
        "observation_tip_rot": obs_rot_used,
    }
    columns_missing = {
        "action": action_missing,
        "observation_tip_pos": obs_pos_missing,
        "observation_tip_rot": obs_rot_missing,
    }

    return ParquetInstrumentData(
        instrument_ids=instrument_ids,
        action=action,
        observation_tip_pos=obs_pos,
        observation_tip_rot=obs_rot,
        action_list=None,
        observation_list=None,
        action_block=None,
        observation_block=None,
        source_is_list=False,
        columns_used=columns_used,
        columns_missing=columns_missing,
    )


def overwrite_parquet_list_columns(path: str, action_list: np.ndarray, observation_list: np.ndarray):
    import pyarrow.dataset as ds  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    if os.path.isdir(path):
        dataset = ds.dataset(path, format="parquet")
        table = dataset.to_table()
    else:
        table = pq.read_table(path)

    if "action.cartesian_absolute" not in table.schema.names:
        raise ValueError("Parquet schema missing action.cartesian_absolute column.")
    if "observation.state" not in table.schema.names:
        raise ValueError("Parquet schema missing observation.state column.")

    action_list_size = action_list.shape[1]
    obs_list_size = observation_list.shape[1]
    action_array = _fixed_size_list_array_from_numpy(action_list, action_list_size)
    obs_array = _fixed_size_list_array_from_numpy(observation_list, obs_list_size)

    action_idx = table.schema.get_field_index("action.cartesian_absolute")
    obs_idx = table.schema.get_field_index("observation.state")
    table = table.set_column(action_idx, "action.cartesian_absolute", action_array)
    table = table.set_column(obs_idx, "observation.state", obs_array)

    if os.path.isdir(path):
        ds.write_dataset(table, path, format="parquet", existing_data_behavior="delete_matching")
    else:
        pq.write_table(table, path)


def overwrite_parquet_joint_columns(
    source_path: str,
    output_path: str | None,
    action_joints: np.ndarray,
    observation_joints_pos: np.ndarray,
    observation_joints_vel: np.ndarray,
):
    import pyarrow.dataset as ds  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    if os.path.isdir(source_path):
        dataset = ds.dataset(source_path, format="parquet")
        fragments = list(dataset.get_fragments())
        fragments.sort(key=lambda frag: str(frag.path))

        if output_path is None:
            output_path = source_path
        os.makedirs(output_path, exist_ok=True)
        # Copy non-parquet metadata/files to preserve dataset structure.
        for root, _, files in os.walk(source_path):
            rel_root = os.path.relpath(root, source_path)
            out_root = output_path if rel_root == "." else os.path.join(output_path, rel_root)
            os.makedirs(out_root, exist_ok=True)
            for name in files:
                if name.endswith(".parquet"):
                    continue
                src_file = os.path.join(root, name)
                dst_file = os.path.join(out_root, name)
                shutil.copy2(src_file, dst_file)

        offset = 0
        for fragment in fragments:
            frag_table = fragment.to_table()
            num_rows = frag_table.num_rows
            if offset + num_rows > action_joints.shape[0]:
                raise ValueError("Joint array row count does not match dataset rows.")

            action_slice = action_joints[offset:offset + num_rows]
            obs_pos_slice = observation_joints_pos[offset:offset + num_rows]
            obs_vel_slice = observation_joints_vel[offset:offset + num_rows]
            action_list_size = action_slice.shape[1]
            obs_pos_list_size = obs_pos_slice.shape[1]
            obs_vel_list_size = obs_vel_slice.shape[1]
            action_array = _fixed_size_list_array_from_numpy(action_slice, action_list_size)
            obs_pos_array = _fixed_size_list_array_from_numpy(obs_pos_slice, obs_pos_list_size)
            obs_vel_array = _fixed_size_list_array_from_numpy(obs_vel_slice, obs_vel_list_size)

            action_name = "action.joint_positions"
            obs_pos_name = "observation.state.joint_positions"
            obs_vel_name = "observation.state.joint_velocities"

            if action_name in frag_table.schema.names:
                idx = frag_table.schema.get_field_index(action_name)
                frag_table = frag_table.set_column(idx, action_name, action_array)
            else:
                frag_table = frag_table.append_column(action_name, action_array)

            if obs_pos_name in frag_table.schema.names:
                idx = frag_table.schema.get_field_index(obs_pos_name)
                frag_table = frag_table.set_column(idx, obs_pos_name, obs_pos_array)
            else:
                frag_table = frag_table.append_column(obs_pos_name, obs_pos_array)
            if obs_vel_name in frag_table.schema.names:
                idx = frag_table.schema.get_field_index(obs_vel_name)
                frag_table = frag_table.set_column(idx, obs_vel_name, obs_vel_array)
            else:
                frag_table = frag_table.append_column(obs_vel_name, obs_vel_array)
            frag_table = _update_hf_metadata(
                frag_table,
                {
                    action_name: action_list_size,
                    obs_pos_name: obs_pos_list_size,
                    obs_vel_name: obs_vel_list_size,
                },
            )

            rel_path = os.path.relpath(str(fragment.path), source_path)
            out_file = os.path.join(output_path, rel_path)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            pq.write_table(frag_table, out_file)
            offset += num_rows

        if offset != action_joints.shape[0]:
            raise ValueError("Joint array row count does not match dataset rows.")
        return

    table = pq.read_table(source_path)

    action_list_size = action_joints.shape[1]
    obs_pos_list_size = observation_joints_pos.shape[1]
    obs_vel_list_size = observation_joints_vel.shape[1]
    action_array = _fixed_size_list_array_from_numpy(action_joints, action_list_size)
    obs_pos_array = _fixed_size_list_array_from_numpy(observation_joints_pos, obs_pos_list_size)
    obs_vel_array = _fixed_size_list_array_from_numpy(observation_joints_vel, obs_vel_list_size)

    action_name = "action.joint_positions"
    obs_pos_name = "observation.state.joint_positions"
    obs_vel_name = "observation.state.joint_velocities"

    if action_name in table.schema.names:
        idx = table.schema.get_field_index(action_name)
        table = table.set_column(idx, action_name, action_array)
    else:
        table = table.append_column(action_name, action_array)

    if obs_pos_name in table.schema.names:
        idx = table.schema.get_field_index(obs_pos_name)
        table = table.set_column(idx, obs_pos_name, obs_pos_array)
    else:
        table = table.append_column(obs_pos_name, obs_pos_array)
    if obs_vel_name in table.schema.names:
        idx = table.schema.get_field_index(obs_vel_name)
        table = table.set_column(idx, obs_vel_name, obs_vel_array)
    else:
        table = table.append_column(obs_vel_name, obs_vel_array)
    table = _update_hf_metadata(
        table,
        {
            action_name: action_list_size,
            obs_pos_name: obs_pos_list_size,
            obs_vel_name: obs_vel_list_size,
        },
    )

    if output_path is None or output_path == source_path:
        pq.write_table(table, source_path)
    else:
        pq.write_table(table, output_path)
