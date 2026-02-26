import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq

from parquet_reader import _update_hf_metadata


def _feature_lengths_from_schema(schema):
    lengths = {}
    for name in (
        "action.joint_positions",
        "observation.state.joint_positions",
        "observation.state.joint_velocities",
    ):
        if name in schema.names:
            field = schema.field(name)
            if pa.types.is_fixed_size_list(field.type):
                lengths[name] = field.type.list_size
    return lengths


def _update_file(path):
    table = pq.read_table(path)
    lengths = _feature_lengths_from_schema(table.schema)
    if not lengths:
        print(f"Skipping {path} (no joint columns found)")
        return
    table = _update_hf_metadata(table, lengths)
    pq.write_table(table, path)
    print(f"Updated schema metadata: {path}")


def main():
    parser = argparse.ArgumentParser(description="Add/update HuggingFace schema metadata for joint columns.")
    parser.add_argument("path", help="Parquet file or directory containing parquet files")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        for root, _, files in os.walk(args.path):
            for name in files:
                if name.endswith(".parquet"):
                    _update_file(os.path.join(root, name))
    else:
        _update_file(args.path)


if __name__ == "__main__":
    main()
