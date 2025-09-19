"""Utilities to bridge warp-cgal viewer functionality with Warp-based simulations."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from mesh_loader import MeshData, load_mesh_with_labels
from mesh_processing import MeshProcessor

try:
    from warp_cgal_python import WarpCGALMeshGenerator

    CGAL_AVAILABLE = True
except ImportError:
    WarpCGALMeshGenerator = None
    CGAL_AVAILABLE = False


QualityName = str
MeshDict = Dict[int, MeshData]


_QUALITY_PRESETS: Dict[QualityName, Dict[str, float]] = {
    "fast": {
        "facet_angle": 25.0,
        "facet_size": 14.0,
        "facet_distance": 1.5,
        "cell_radius_edge_ratio": 3.0,
        "cell_size": 14.0,
    },
    "low": {
        "facet_angle": 25.0,
        "facet_size": 10.0,
        "facet_distance": 1.0,
        "cell_radius_edge_ratio": 3.0,
        "cell_size": 10.0,
    },
    "balanced": {
        "facet_angle": 30.0,
        "facet_size": 6.0,
        "facet_distance": 0.5,
        "cell_radius_edge_ratio": 3.0,
        "cell_size": 8.0,
    },
    "high": {
        "facet_angle": 35.0,
        "facet_size": 4.0,
        "facet_distance": 0.35,
        "cell_radius_edge_ratio": 3.0,
        "cell_size": 5.0,
    },
    "very_high": {
        "facet_angle": 35.0,
        "facet_size": 3.0,
        "facet_distance": 0.25,
        "cell_radius_edge_ratio": 3.0,
        "cell_size": 3.5,
    },
    "ultra": {
        "facet_angle": 40.0,
        "facet_size": 2.0,
        "facet_distance": 0.15,
        "cell_radius_edge_ratio": 3.0,
        "cell_size": 2.0,
    },
}

_WEIGHTS_DEFAULTS = {"sigma": 1.0, "relative_error_bound": 1e-6}


class CGALUnavailableError(RuntimeError):
    """Raised when CGAL bindings are required but missing."""


def _require_cgal() -> None:
    if not CGAL_AVAILABLE:
        raise CGALUnavailableError("warp_cgal_python module is not available")


def _as_mesh_dict(result: Union[MeshData, MeshDict]) -> MeshDict:
    if isinstance(result, dict):
        return {int(label): mesh for label, mesh in result.items()}
    return {0: result}


def load_warp_mesh_dataset(mesh_path: Union[str, Path], *, reconstruct_surface: bool = True) -> MeshDict:
    """Load Warp-format mesh data, optionally reconstructing surfaces."""
    mesh_path = str(mesh_path)
    data = _as_mesh_dict(load_mesh_with_labels(mesh_path))

    if reconstruct_surface:
        processor = MeshProcessor()
        for mesh in data.values():
            needs_surface = mesh.tetrahedra is not None and (
                mesh.triangles is None or len(mesh.triangles) == 0
            )
            if needs_surface:
                processor.reconstruct_surface(mesh)
    return data


def flatten_mesh_dataset(meshes: MeshDict) -> Dict[str, np.ndarray]:
    """Flatten labeled meshes into combined arrays suitable for simulation."""
    vertices_parts: List[np.ndarray] = []
    tetra_parts: List[np.ndarray] = []
    tri_parts: List[np.ndarray] = []
    edge_parts: List[np.ndarray] = []
    uv_parts: List[np.ndarray] = []
    subdomain_parts: List[np.ndarray] = []

    vertex_offset = 0
    tet_offset = 0
    tri_offset = 0
    edge_offset = 0

    label_ranges: Dict[int, Dict[str, int]] = {}

    for label in sorted(meshes.keys()):
        mesh = meshes[label]
        verts = np.asarray(mesh.vertices, dtype=np.float32) if mesh.vertices is not None else np.zeros((0, 3), dtype=np.float32)
        tets = np.asarray(mesh.tetrahedra, dtype=np.int32) if mesh.tetrahedra is not None else np.zeros((0, 4), dtype=np.int32)
        tris = np.asarray(mesh.triangles, dtype=np.int32) if mesh.triangles is not None else np.zeros((0, 3), dtype=np.int32)
        edges = np.asarray(mesh.edges, dtype=np.int32) if mesh.edges is not None else np.zeros((0, 2), dtype=np.int32)
        uvs = np.asarray(mesh.uvs, dtype=np.float32) if mesh.uvs is not None else np.zeros((0, 2), dtype=np.float32)

        if verts.ndim != 2 or verts.shape[1] != 3:
            verts = verts.reshape((-1, 3))
        if tets.ndim != 2 or tets.shape[1] != 4:
            tets = tets.reshape((-1, 4))
        if tris.size > 0:
            if tris.ndim != 2 or tris.shape[1] != 3:
                tris = tris.reshape((-1, 3))
        if edges.size > 0:
            if edges.ndim != 2 or edges.shape[1] != 2:
                edges = edges.reshape((-1, 2))
        if uvs.size > 0:
            if uvs.ndim != 2 or uvs.shape[1] not in (2, 3):
                uvs = uvs.reshape((-1, uvs.shape[-1]))

        vertices_parts.append(verts)
        if len(tets):
            tetra_parts.append(tets + vertex_offset)
            if mesh.subdomain_labels is not None and len(mesh.subdomain_labels) == len(tets):
                subdomain_parts.append(np.asarray(mesh.subdomain_labels, dtype=np.int32))
            else:
                subdomain_parts.append(np.full(len(tets), int(label), dtype=np.int32))
        else:
            subdomain_parts.append(np.zeros((0,), dtype=np.int32))

        if len(tris):
            tri_parts.append(tris + vertex_offset)
        if len(edges):
            edge_parts.append(edges + vertex_offset)
        if len(uvs):
            uv_parts.append(uvs)

        label_ranges[int(label)] = {
            "vertex_start": vertex_offset,
            "vertex_count": len(verts),
            "tet_start": tet_offset,
            "tet_count": len(tets),
            "triangle_start": tri_offset,
            "triangle_count": len(tris),
            "edge_start": edge_offset,
            "edge_count": len(edges),
        }

        vertex_offset += len(verts)
        tet_offset += len(tets)
        tri_offset += len(tris)
        edge_offset += len(edges)

    combined = {
        "vertices": np.vstack(vertices_parts) if vertices_parts else np.zeros((0, 3), dtype=np.float32),
        "tetrahedra": np.vstack(tetra_parts) if tetra_parts else np.zeros((0, 4), dtype=np.int32),
        "triangles": np.vstack(tri_parts) if tri_parts else np.zeros((0, 3), dtype=np.int32),
        "edges": np.vstack(edge_parts) if edge_parts else np.zeros((0, 2), dtype=np.int32),
        "uvs": np.vstack(uv_parts) if uv_parts else np.zeros((0, 2), dtype=np.float32),
        "subdomain_labels": np.concatenate(subdomain_parts) if subdomain_parts else np.zeros((0,), dtype=np.int32),
        "label_ranges": label_ranges,
    }

    return combined


def detect_image_labels(image_path: Union[str, Path]) -> List[int]:
    """Return available segmentation labels in a medical volume."""
    _require_cgal()

    generator = WarpCGALMeshGenerator()
    if not generator.load_image(str(image_path)):
        raise RuntimeError(f"Failed to load medical image: {image_path}")
    return list(generator.get_image_labels())


def _prepare_generator(image_path: Union[str, Path], quality: QualityName, cell_size: Optional[float]) -> Tuple[WarpCGALMeshGenerator, Dict[str, float]]:
    generator = WarpCGALMeshGenerator()
    if not generator.load_image(str(image_path)):
        raise RuntimeError(f"Failed to load medical image: {image_path}")

    params = dict(_QUALITY_PRESETS.get(quality, _QUALITY_PRESETS["balanced"]))
    if cell_size is not None:
        params["cell_size"] = cell_size

    generator.set_criteria(**params)
    return generator, params


def generate_mesh_from_image(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    quality: QualityName = "balanced",
    cell_size: Optional[float] = None,
    multi_label: bool = False,
    weighted: bool = False,
    sigma: float = _WEIGHTS_DEFAULTS["sigma"],
    relative_error_bound: float = _WEIGHTS_DEFAULTS["relative_error_bound"],
    force: bool = False,
) -> Dict[str, Union[str, Dict[str, float], Dict[int, Dict[str, float]], List[str]]]:
    """Generate Warp-format tetrahedral meshes from a medical image."""
    _require_cgal()

    output_path = Path(output_path)
    if multi_label:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if force:
            for existing in output_path.parent.glob(f"{output_path.name}_label_*"):
                if existing.is_dir():
                    shutil.rmtree(existing)
    else:
        if output_path.exists() and force:
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    generator, params = _prepare_generator(image_path, quality, cell_size)

    use_weights = weighted or sigma != _WEIGHTS_DEFAULTS["sigma"] or relative_error_bound != _WEIGHTS_DEFAULTS["relative_error_bound"]
    generator.set_weighted_mesh_parameters(
        use_weights=use_weights,
        sigma=sigma,
        relative_error_bound=relative_error_bound,
    )

    if multi_label:
        success = (
            generator.generate_weighted_multi_label_meshes()
            if use_weights
            else generator.generate_multi_label_meshes()
        )
    else:
        success = generator.generate_weighted_mesh() if use_weights else generator.generate_mesh()

    if not success:
        raise RuntimeError("CGAL mesh generation failed")

    if multi_label:
        export_base = str(output_path)
        if not generator.export_multi_label_meshes_warp_format(export_base):
            raise RuntimeError("Failed to export multi-label Warp-format meshes")
        stats = generator.get_multi_label_mesh_statistics()
        labels = list(stats.keys())
        produced_paths = sorted(
            str(path)
            for path in output_path.parent.glob(f"{output_path.name}_label_*")
            if path.is_dir()
        )
    else:
        if not generator.export_mesh_warp_format(str(output_path)):
            raise RuntimeError("Failed to export Warp-format mesh")
        stats = {0: generator.get_mesh_statistics()}
        labels = [0]
        produced_paths = [str(output_path)]

    return {
        "output_paths": produced_paths,
        "statistics": stats,
        "quality": quality,
        "criteria": params,
        "weighted": use_weights,
        "labels": labels,
    }


__all__ = [
    "CGALUnavailableError",
    "load_warp_mesh_dataset",
    "flatten_mesh_dataset",
    "detect_image_labels",
    "generate_mesh_from_image",
]
