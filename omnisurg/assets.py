import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TetMeshAsset:
    """Immutable host-side mesh data. Parsed once, never modified."""

    name: str
    rest_positions: np.ndarray
    tet_indices: np.ndarray
    edge_indices: np.ndarray
    surface_tri_indices: np.ndarray
    uvs: np.ndarray | None = None


def load_tet_asset(name: str, mesh_dir: str = "meshes") -> TetMeshAsset:
    """Parse mesh files into a frozen asset. Pure NumPy, no Warp/Newton."""
    base_path = os.path.join(mesh_dir, name, "")

    positions = []
    with open(base_path + "model.vertices", "r") as f:
        for line in f:
            parts = line.split()
            if parts:
                positions.append([float(x) for x in parts])

    tet_indices_flat = []
    with open(base_path + "model.tetras", "r") as f:
        for line in f:
            parts = line.split()
            if parts:
                tet_indices_flat.extend(int(x) for x in parts)

    edge_indices_flat = []
    with open(base_path + "model.edges", "r") as f:
        for line in f:
            parts = line.split()
            if parts:
                edge_indices_flat.extend(int(x) for x in parts)

    tri_indices_flat = []
    with open(base_path + "model.tris", "r") as f:
        for line in f:
            parts = line.split()
            if parts:
                tri_indices_flat.extend(int(x) for x in parts)

    uvs = []
    uvs_path = base_path + "model.uvs"
    if os.path.exists(uvs_path):
        with open(uvs_path, "r") as f:
            for line in f:
                parts = line.split()
                if parts:
                    uvs.append([float(x) for x in parts])

    rest_positions = np.array(positions, dtype=np.float32)
    tet_indices = np.array(tet_indices_flat, dtype=np.int32).reshape(-1, 4)
    edge_indices = np.array(edge_indices_flat, dtype=np.int32).reshape(-1, 2)
    surface_tri_indices = np.array(tri_indices_flat, dtype=np.int32).reshape(-1, 3)
    uvs_array = np.array(uvs, dtype=np.float32) if uvs else None

    return TetMeshAsset(
        name=name,
        rest_positions=rest_positions,
        tet_indices=tet_indices,
        edge_indices=edge_indices,
        surface_tri_indices=surface_tri_indices,
        uvs=uvs_array,
    )
