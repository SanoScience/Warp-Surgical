from dataclasses import dataclass
import os

import numpy as np

from omnisurg.config import SceneConfig
from omnisurg.mesh.types import TriPointsConnector, parse_connector_file


@dataclass(frozen=True)
class MeshRange:
    vertex_start: int
    vertex_count: int
    edge_start: int
    edge_count: int
    tet_start: int
    tet_count: int
    tri_start: int
    tri_count: int


@dataclass(frozen=True)
class TetMeshAsset:
    """Immutable tet mesh data parsed from the existing mesh assets."""

    name: str
    rest_positions: np.ndarray
    tet_indices: np.ndarray
    edge_indices: np.ndarray
    surface_tri_indices: np.ndarray
    uvs: np.ndarray | None = None
    mesh_ranges: dict[str, MeshRange] | None = None
    connectors: tuple[TriPointsConnector, ...] = ()


def load_tet_asset(name: str, mesh_dir: str = "meshes") -> TetMeshAsset:
    """Load a single tet mesh asset from the existing on-disk mesh layout."""

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
    mesh_range = MeshRange(
        vertex_start=0,
        vertex_count=int(rest_positions.shape[0]),
        edge_start=0,
        edge_count=int(edge_indices.shape[0]),
        tet_start=0,
        tet_count=int(tet_indices.shape[0]),
        tri_start=0,
        tri_count=int(surface_tri_indices.shape[0]),
    )
    return TetMeshAsset(
        name=name,
        rest_positions=rest_positions,
        tet_indices=tet_indices,
        edge_indices=edge_indices,
        surface_tri_indices=surface_tri_indices,
        uvs=np.array(uvs, dtype=np.float32) if uvs else None,
        mesh_ranges={name: mesh_range},
    )


def _merge_assets(asset_names: tuple[str, ...], mesh_dir: str) -> TetMeshAsset:
    merged_positions: list[np.ndarray] = []
    merged_tets: list[np.ndarray] = []
    merged_edges: list[np.ndarray] = []
    merged_tris: list[np.ndarray] = []
    merged_uvs: list[np.ndarray] = []
    mesh_ranges: dict[str, MeshRange] = {}

    vertex_offset = 0
    edge_offset = 0
    tet_offset = 0
    tri_offset = 0

    for asset_name in asset_names:
        asset = load_tet_asset(asset_name, mesh_dir)
        merged_positions.append(asset.rest_positions)
        merged_tets.append(asset.tet_indices + vertex_offset)
        merged_edges.append(asset.edge_indices + vertex_offset)
        merged_tris.append(asset.surface_tri_indices + vertex_offset)
        if asset.uvs is not None:
            merged_uvs.append(asset.uvs)

        mesh_ranges[asset_name] = MeshRange(
            vertex_start=vertex_offset,
            vertex_count=int(asset.rest_positions.shape[0]),
            edge_start=edge_offset,
            edge_count=int(asset.edge_indices.shape[0]),
            tet_start=tet_offset,
            tet_count=int(asset.tet_indices.shape[0]),
            tri_start=tri_offset,
            tri_count=int(asset.surface_tri_indices.shape[0]),
        )

        vertex_offset += int(asset.rest_positions.shape[0])
        edge_offset += int(asset.edge_indices.shape[0])
        tet_offset += int(asset.tet_indices.shape[0])
        tri_offset += int(asset.surface_tri_indices.shape[0])

    connectors: list[TriPointsConnector] = []
    if asset_names == ("liver", "fat", "gallbladder"):
        connectors.extend(
            parse_connector_file(
                os.path.join(mesh_dir, "fat-liver.connector"),
                particle_id_offset=mesh_ranges["fat"].vertex_start,
                tri_id_offset=mesh_ranges["liver"].vertex_start,
            ),
        )
        connectors.extend(
            parse_connector_file(
                os.path.join(mesh_dir, "gallbladder-fat.connector"),
                particle_id_offset=mesh_ranges["gallbladder"].vertex_start,
                tri_id_offset=mesh_ranges["fat"].vertex_start,
            ),
        )

    return TetMeshAsset(
        name="+".join(asset_names),
        rest_positions=np.concatenate(merged_positions, axis=0),
        tet_indices=np.concatenate(merged_tets, axis=0),
        edge_indices=np.concatenate(merged_edges, axis=0),
        surface_tri_indices=np.concatenate(merged_tris, axis=0),
        uvs=np.concatenate(merged_uvs, axis=0) if merged_uvs else None,
        mesh_ranges=mesh_ranges,
        connectors=tuple(connectors),
    )


def load_scene_asset(scene: SceneConfig) -> TetMeshAsset:
    if scene.scene_preset == "single":
        return load_tet_asset(scene.asset_name, scene.mesh_dir)
    if scene.scene_preset == "chole":
        return _merge_assets(("liver", "fat", "gallbladder"), scene.mesh_dir)
    raise ValueError(f"Unsupported scene preset: {scene.scene_preset}")
