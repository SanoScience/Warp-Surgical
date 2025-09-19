from centrelines import CentrelinePointInfo, ClampConstraint
import warp as wp
import newton
import json
import importlib.util
import time
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@wp.struct
class Tetrahedron:
    ids: wp.vec4i
    rest_volume: wp.float32

@wp.struct
class TriPointsConnector:
    particle_id: wp.int32
    rest_dist: wp.float32
    tri_ids: wp.vec3i
    tri_bar: wp.vec3f

def parse_connector_file(filepath, particle_id_offset=0, tri_id_offset=0):
    """Parse connector file and return list of TriPointsConnector objects."""
    connectors = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) != 8:
                raise ValueError(f"Line does not have 8 elements: {line}")
            
            connector = TriPointsConnector()
            connector.particle_id = int(parts[0]) + particle_id_offset
            connector.rest_dist = float(parts[1])
            connector.tri_ids = wp.vec3i(int(parts[2]) + tri_id_offset, int(parts[3]) + tri_id_offset, int(parts[4]) + tri_id_offset)
            connector.tri_bar = wp.vec3f(float(parts[5]), float(parts[6]), float(parts[7]))
            
            connectors.append(connector)
    return connectors

def compute_tet_volume(p0, p1, p2, p3):
    # Returns the signed volume of the tetrahedron
    return abs(
        wp.dot(
            wp.cross(p1 - p0, p2 - p0),
            p3 - p0
        ) / 6.0
    )

def load_mesh_component(base_path, offset=0):
    """Load a single mesh component and return positions, indices, and edges."""
    positions = []
    indices = []
    edges = []
    tri_surface_indices = []
    uvs = []
    
    vertices_file = base_path + "model.vertices"
    indices_file = base_path + "model.tetras"
    edges_file = base_path + "model.edges"
    surface_indices_file = base_path + "model.tris"
    uvs_file = base_path + "model.uvs"


    # Load vertices
    with open(vertices_file, 'r') as f:
        for line in f:
            pos = [float(x) for x in line.split()]
            positions.append(pos)
    
    # Load indices
    with open(indices_file, 'r') as f:
        for line in f:
            indices.extend([int(x) + offset for x in line.split()])
    
    # Load edges
    with open(edges_file, 'r') as f:
        for line in f:
            edges.extend([int(x) + offset for x in line.split()])

    # Load surface triangle indices
    with open(surface_indices_file, 'r') as f:
        for line in f:
            tri_surface_indices.extend([int(x) + offset for x in line.split()])
    
    # Load uvs
    with open(uvs_file, 'r') as f:
        for line in f:
            uv = [float(x) for x in line.split()]
            uvs.append(uv)

    return positions, indices, edges, tri_surface_indices, uvs

def load_mesh_and_build_model(builder: newton.ModelBuilder, particle_mass, vertical_offset=0.0, spring_stiffness=1.0, spring_dampen=0.0, tetra_stiffness_mu=1.0e3, tetra_stiffness_lambda=1.0e3, tetra_dampen=0.0):
    """Load all mesh components and build the simulation model with ranges."""
    all_positions = []
    all_indices = []
    all_edges = []
    all_tri_surface_indices = []
    all_connectors = []
    all_uvs = []
    
    mesh_ranges = {
        'liver': {'vertex_start': 0, 'vertex_count': 0, 'index_start': 0, 'index_count': 0},
        'fat': {'vertex_start': 0, 'vertex_count': 0, 'index_start': 0, 'index_count': 0},
        'gallbladder': {'vertex_start': 0, 'vertex_count': 0, 'index_start': 0, 'index_count': 0}
    }
    
    # Track current offsets
    current_vertex_offset = 0
    current_index_offset = 0
    current_edge_offset = 0
    current_tet_offset = 0
    
    # Liver
    liver_positions, liver_indices, liver_edges, liver_tris, liver_uvs = load_mesh_component('meshes/liver/', 0)
    mesh_ranges['liver'] = {
        'vertex_start': current_vertex_offset,
        'vertex_count': len(liver_positions),
        'index_start': current_index_offset,
        'index_count': len(liver_tris),
        'edge_start': current_edge_offset // 2,
        'edge_count': len(liver_edges) // 2,
        'tet_start': current_tet_offset,
        'tet_count': len(liver_indices) // 4
    }
    
    all_positions.extend(liver_positions)
    all_indices.extend(liver_indices)
    all_edges.extend(liver_edges)
    all_tri_surface_indices.extend(liver_tris)
    all_uvs.extend(liver_uvs)
    
    current_vertex_offset = len(all_positions)
    current_edge_offset = len(all_edges)
    current_index_offset = len(all_tri_surface_indices)
    current_tet_offset = len(all_indices) // 4
    
    # Fat
    fat_particle_offset = len(all_positions)
    fat_positions, fat_indices, fat_edges, fat_tris, fat_uvs = load_mesh_component('meshes/fat/', fat_particle_offset)
    mesh_ranges['fat'] = {
        'vertex_start': current_vertex_offset,
        'vertex_count': len(fat_positions),
        'index_start': current_index_offset,
        'index_count': len(fat_tris),
        'edge_start': current_edge_offset // 2,
        'edge_count': len(fat_edges) // 2,
        'tet_start': current_tet_offset,
        'tet_count': len(fat_indices) // 4
    }
    
    all_positions.extend(fat_positions)
    all_indices.extend(fat_indices)
    all_edges.extend(fat_edges)
    all_tri_surface_indices.extend(fat_tris)
    all_uvs.extend(fat_uvs)
    
    current_vertex_offset = len(all_positions)
    current_edge_offset = len(all_edges)
    current_index_offset = len(all_tri_surface_indices)
    current_tet_offset = len(all_indices) // 4
    

    # Gallbladder
    gallbladder_particle_offset = len(all_positions)
    gallbladder_positions, gallbladder_indices, gallbladder_edges, gallbladder_tris, gallbladder_uvs = load_mesh_component('meshes/gallbladder/', gallbladder_particle_offset)
    mesh_ranges['gallbladder'] = {
        'vertex_start': current_vertex_offset,
        'vertex_count': len(gallbladder_positions),
        'index_start': current_index_offset,
        'index_count': len(gallbladder_tris),
        'edge_start': current_edge_offset // 2,
        'edge_count': len(gallbladder_edges) // 2,
        'tet_start': current_tet_offset,
        'tet_count': len(gallbladder_indices) // 4
    }
    
    all_positions.extend(gallbladder_positions)
    all_indices.extend(gallbladder_indices)
    all_edges.extend(gallbladder_edges)
    all_tri_surface_indices.extend(gallbladder_tris)
    all_uvs.extend(gallbladder_uvs)
    
    # Load connectors
    fat_liver_connectors = parse_connector_file('meshes/fat-liver.connector', fat_particle_offset, 0)
    gallbladder_fat_connectors = parse_connector_file('meshes/gallbladder-fat.connector', gallbladder_particle_offset, fat_particle_offset)
    all_connectors.extend(fat_liver_connectors)
    all_connectors.extend(gallbladder_fat_connectors)
    
    # Add particles to model

    for position in all_positions:
        pos = wp.vec3(position)
        pos[1] += vertical_offset
        #very ugly hardcoded position and radius below
        if is_particle_within_radius(pos, [0.5, 1.5, -5.0], 1.0):
            builder.add_particle(pos, wp.vec3(0, 0, 0), mass=0, radius=0.01)
        else:
            builder.add_particle(pos, wp.vec3(0, 0, 0), mass=particle_mass, radius=0.01)
    
    
    # Add springs
    for i in range(0, len(all_edges), 2):
        builder.add_spring(all_edges[i], all_edges[i + 1], spring_stiffness, spring_dampen, 0)
    
    
    # Add triangles
    for i in range(0, len(all_tri_surface_indices), 3):
        ids = [all_tri_surface_indices[i], all_tri_surface_indices[i + 1], all_tri_surface_indices[i + 2]]
        builder.add_triangle(ids[0], ids[1], ids[2])

    all_tetrahedra = []

    # Add tetrahedrons (neo-hookean + custom volume constraint)    
    for i in range(0, len(all_indices), 4):
        ids = [all_indices[i], all_indices[i+1], all_indices[i+2], all_indices[i+3]]
        p0 = wp.vec3(all_positions[ids[0]])
        p1 = wp.vec3(all_positions[ids[1]])
        p2 = wp.vec3(all_positions[ids[2]])
        p3 = wp.vec3(all_positions[ids[3]])
        rest_volume = compute_tet_volume(p0, p1, p2, p3)
        
        tet = Tetrahedron()
        tet.ids = wp.vec4i(ids[0], ids[1], ids[2], ids[3])
        tet.rest_volume = rest_volume

        all_tetrahedra.append(tet)
        builder.add_tetrahedron(all_indices[i], all_indices[i + 1], all_indices[i + 2], all_indices[i + 3], tetra_stiffness_mu, tetra_stiffness_lambda, tetra_dampen)



    return wp.array(all_connectors, dtype=TriPointsConnector, device=wp.get_device()), all_tri_surface_indices, all_uvs, mesh_ranges, wp.array(all_tetrahedra, dtype=Tetrahedron, device=wp.get_device())

def load_background_mesh():
    positions = []
    tri_indices = []
    uvs = []
    
    base_path = "meshes/cavity/"
    vertices_file = base_path + "cavity.vertices"
    surface_indices_file = base_path + "cavity.tris"
    uvs_file = base_path + "cavity.uvs"
    
     # Load vertices
    with open(vertices_file, 'r') as f:
        for line in f:
            pos = [float(x) for x in line.split()]
            pos[0] += 0.5
            pos[1] -= 3.6
            positions.append(pos)
    

    # Load surface triangle indices
    offset = 0
    with open(surface_indices_file, 'r') as f:
        for line in f:
            tri_indices.extend([int(x) + offset for x in line.split()])
    
    # Load uvs
    with open(uvs_file, 'r') as f:
        for line in f:
            uv = [float(x) for x in line.split()]
            uvs.append(uv)

    return wp.array(positions, dtype=wp.vec3f, device=wp.get_device()),  wp.array(tri_indices, dtype=wp.int32, device=wp.get_device()), wp.array(uvs, dtype=wp.vec2f, device=wp.get_device()),  wp.zeros(len(positions), dtype=wp.vec4f, device=wp.get_device())

def parse_centreline_file(filepath, point_offset, edge_offset):
    """
    Parse a centreline file and return:
      - a list of CentrelinePointInfo
      - a flat list of all point ids (int)
      - a flat list of all point dists (float)
      - a flat list of all edge ids (int)
    """
    centreline_infos = []
    all_point_cnstrs = []
    all_edge_ids = []

    with open(filepath, 'r') as f:
        point_start_id = 0
        edge_start_id = 0
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            idx = 0
            # Point ids and dists
            point_count = int(parts[idx])
            idx += 1
            point_cnstr = []
            for _ in range(point_count):
                id = int(parts[idx]) + point_offset
                idx += 1
                dist = float(parts[idx])
                idx += 1

                cnstr = ClampConstraint()
                cnstr.id = id
                cnstr.dist = dist
                point_cnstr.append(cnstr)

            # Edge ids
            edge_count = int(parts[idx])
            idx += 1
            edge_ids = []
            for _ in range(edge_count):
                edge_ids.append(int(parts[idx]) + edge_offset)
                idx += 1

            # Stiffness, rest_length_mul, radius
            stiffness = float(parts[idx])
            idx += 1
            rest_length_mul = float(parts[idx])
            idx += 1
            radius = float(parts[idx])
            idx += 1

            info = CentrelinePointInfo()
            info.point_start_id = point_start_id
            info.point_count = point_count
            info.edge_start_id = edge_start_id
            info.edge_count = edge_count
            info.stiffness = stiffness
            info.rest_length_mul = rest_length_mul
            info.radius = radius


            centreline_infos.append(info)
            all_point_cnstrs.extend(point_cnstr)
            all_edge_ids.extend(edge_ids)

            point_start_id += point_count
            edge_start_id += edge_count

    return centreline_infos, all_point_cnstrs, all_edge_ids

def is_particle_within_radius(particle_pos, centre, radius):
    pos = wp.vec3(particle_pos[0], particle_pos[1], particle_pos[2])
    centre_pos = wp.vec3(centre[0], centre[1], centre[2])
    distance = wp.length(pos - centre_pos)
    return distance < radius


@dataclass
class WarpMeshConfig:
    """Configuration describing how to source warp-format meshes via warp-cgal."""

    mesh_path: Optional[str] = None
    multi_label: bool = False
    remesh_image: Optional[str] = None
    remesh_output: Optional[str] = None
    cgal_root: Optional[str] = None
    criteria: Dict[str, float] = field(default_factory=dict)
    subdomain_map: Dict[str, int] = field(default_factory=dict)

    def is_active(self) -> bool:
        return bool(self.mesh_path or self.remesh_image)


_WARP_CGAL_MODULE_CACHE: Dict[str, Dict[str, Any]] = {}


def _normalize_cache_key(path: Path) -> str:
    resolved = str(path.resolve())
    return resolved.replace('\\', '_').replace('/', '_')


def _load_external_module(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _resolve_cgal_root(config: WarpMeshConfig) -> Path:
    if config.cgal_root:
        candidate = Path(config.cgal_root).expanduser()
    else:
        candidate = Path(__file__).resolve().parent.parent / "warp-cgal"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Unable to locate warp-cgal directory. Checked: {candidate}"
        )
    return candidate


def _get_warp_cgal_modules(root: Path) -> Dict[str, Any]:
    cache_key = _normalize_cache_key(root)
    if cache_key in _WARP_CGAL_MODULE_CACHE:
        return _WARP_CGAL_MODULE_CACHE[cache_key]

    viewer_path = root / "viewer"
    if not viewer_path.exists():
        raise FileNotFoundError(
            f"warp-cgal viewer directory not found at {viewer_path}"
        )

    viewer_mesh_loader = _load_external_module(
        f"warp_cgal_viewer_mesh_loader_{cache_key}", viewer_path / "mesh_loader.py"
    )

    original_mesh_loader_module = sys.modules.get("mesh_loader")
    sys.modules["mesh_loader"] = viewer_mesh_loader
    try:
        viewer_mesh_processing = _load_external_module(
            f"warp_cgal_viewer_mesh_processing_{cache_key}", viewer_path / "mesh_processing.py"
        )
    finally:
        if original_mesh_loader_module is not None:
            sys.modules["mesh_loader"] = original_mesh_loader_module
        else:
            sys.modules.pop("mesh_loader", None)

    modules = {
        "mesh_loader": viewer_mesh_loader,
        "mesh_processing": viewer_mesh_processing,
        "warp_cgal_python": _load_external_module(
            f"warp_cgal_python_{cache_key}", root / "warp_cgal_python.py"
        ),
    }

    _WARP_CGAL_MODULE_CACHE[cache_key] = modules
    return modules


def _build_generation_criteria(overrides: Dict[str, float]) -> Dict[str, float]:
    defaults = {
        "facet_angle": 30.0,
        "facet_size": 6.0,
        "facet_distance": 0.5,
        "cell_radius_edge_ratio": 3.0,
        "cell_size": 8.0,
    }
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in defaults:
            raise KeyError(f"Unknown mesh criteria parameter: {key}")
        defaults[key] = float(value)
    return defaults


def _clear_warp_folder(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for filename in ("model.vertices", "model.tetras", "model.edges", "model.tris", "model.uvs", "model.labels"):
        file_path = folder / filename
        if file_path.exists():
            file_path.unlink()


def _generate_mesh_if_requested(config: WarpMeshConfig, root: Path, modules: Dict[str, Any]) -> Optional[Path]:
    if not config.remesh_image:
        return None

    image_path = Path(config.remesh_image).expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found for remeshing: {image_path}")

    if config.remesh_output:
        output_dir = Path(config.remesh_output).expanduser()
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = root / "temp_generated_mesh_warp" / f"generated_{timestamp}"

    _clear_warp_folder(output_dir)

    generator_cls = getattr(modules["warp_cgal_python"], "WarpCGALMeshGenerator")
    generator = generator_cls()
    if not generator.load_image(str(image_path)):
        raise RuntimeError(f"Failed to load image for remeshing: {image_path}")

    criteria = _build_generation_criteria(config.criteria)
    generator.set_criteria(
        facet_angle=criteria["facet_angle"],
        facet_size=criteria["facet_size"],
        facet_distance=criteria["facet_distance"],
        cell_radius_edge_ratio=criteria["cell_radius_edge_ratio"],
        cell_size=criteria["cell_size"],
    )

    if not generator.generate_mesh():
        raise RuntimeError("warp-cgal failed to generate mesh from image")

    if not generator.export_mesh_warp_format(str(output_dir)):
        raise RuntimeError(f"Failed to export warp-format mesh to {output_dir}")

    return output_dir


def _load_warp_mesh_data(modules: Dict[str, Any], mesh_path: Path, multi_label: bool):
    loader = modules["mesh_loader"].MeshLoader
    if multi_label:
        raise NotImplementedError("Multi-label warp mesh loading is not yet supported in the simulation bridge")
    return loader.load_warp_format(str(mesh_path))


def _ensure_array(data: Optional[np.ndarray], expected_rank: int, description: str) -> np.ndarray:
    if data is None:
        raise ValueError(f"Warp mesh is missing {description} data")
    array = np.asarray(data)
    if array.ndim == expected_rank:
        return array
    if array.ndim == expected_rank - 1 and expected_rank == 2:
        return array.reshape((-1, array.shape[0]))
    raise ValueError(f"Unexpected shape for {description}: {array.shape}")


def _is_contiguous(indices: List[int]) -> bool:
    return all(indices[i] + 1 == indices[i + 1] for i in range(len(indices) - 1)) if indices else False


def _compute_mesh_ranges(
    vertex_count: int,
    triangle_index_count: int,
    edge_count: int,
    tet_count: int,
    subdomain_labels: Optional[np.ndarray],
    config: WarpMeshConfig,
) -> Dict[str, Dict[str, Any]]:
    mesh_ranges: Dict[str, Dict[str, Any]] = {
        "cgal_mesh": {
            "vertex_start": 0,
            "vertex_count": vertex_count,
            "index_start": 0,
            "index_count": triangle_index_count,
            "edge_start": 0,
            "edge_count": edge_count,
            "tet_start": 0,
            "tet_count": tet_count,
        }
    }

    if subdomain_labels is None or not config.subdomain_map:
        return mesh_ranges

    label_array = np.asarray(subdomain_labels, dtype=np.int32)
    for name, label in config.subdomain_map.items():
        indices = np.nonzero(label_array == int(label))[0].tolist()
        info: Dict[str, Any] = {
            "label": int(label),
            "tet_count": len(indices),
            "tet_indices": indices,
            "vertex_start": None,
            "vertex_count": 0,
            "index_start": None,
            "index_count": 0,
            "edge_start": None,
            "edge_count": 0,
        }
        if indices and _is_contiguous(indices):
            info["tet_start"] = indices[0]
        else:
            info["tet_start"] = None
        mesh_ranges[name] = info

    return mesh_ranges


def load_warp_mesh_and_build_model(
    builder: newton.ModelBuilder,
    particle_mass: float,
    config: WarpMeshConfig,
    vertical_offset: float = 0.0,
    spring_stiffness: float = 1.0,
    spring_dampen: float = 0.0,
    tetra_stiffness_mu: float = 1.0e3,
    tetra_stiffness_lambda: float = 1.0e3,
    tetra_dampen: float = 0.0,
):
    """Load a warp-format mesh via warp-cgal and build the simulation model."""

    if config is None or not config.is_active():
        raise ValueError("Warp mesh configuration is inactive; provide mesh_path or remesh_image")

    root = _resolve_cgal_root(config)
    modules = _get_warp_cgal_modules(root)

    mesh_dir = config.mesh_path
    generated = False
    if config.remesh_image:
        generated_path = _generate_mesh_if_requested(config, root, modules)
        if generated_path is None:
            raise RuntimeError("Remeshing was requested but no mesh was generated")
        mesh_dir = str(generated_path)
        generated = True

    if not mesh_dir:
        raise ValueError("No warp mesh path resolved after processing configuration")

    mesh_path = Path(mesh_dir).expanduser()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Warp mesh directory not found: {mesh_path}")

    mesh_data = _load_warp_mesh_data(modules, mesh_path, config.multi_label)

    vertices = _ensure_array(mesh_data.vertices, 2, "vertex")
    tetrahedra = _ensure_array(mesh_data.tetrahedra, 2, "tetrahedron")
    triangles = np.asarray(mesh_data.triangles) if mesh_data.triangles is not None else np.zeros((0, 3), dtype=np.int32)
    edges = np.asarray(mesh_data.edges) if mesh_data.edges is not None else np.zeros((0, 2), dtype=np.int32)
    uvs = np.asarray(mesh_data.uvs) if mesh_data.uvs is not None else np.zeros((0, 2), dtype=np.float32)

    vertex_list = vertices.tolist()
    triangle_list = triangles.astype(np.int32).tolist()
    edge_list = edges.astype(np.int32).tolist()
    uv_list = uvs.astype(np.float32).tolist()

    for position in vertex_list:
        pos = wp.vec3(position)
        pos[1] += vertical_offset
        if is_particle_within_radius(pos, [0.5, 1.5, -5.0], 1.0):
            builder.add_particle(pos, wp.vec3(0, 0, 0), mass=0, radius=0.01)
        else:
            builder.add_particle(pos, wp.vec3(0, 0, 0), mass=particle_mass, radius=0.01)

    for edge in edge_list:
        builder.add_spring(int(edge[0]), int(edge[1]), spring_stiffness, spring_dampen, 0)

    surface_tris: List[int] = []
    for tri in triangle_list:
        ids = [int(tri[0]), int(tri[1]), int(tri[2])]
        builder.add_triangle(ids[0], ids[1], ids[2])
        surface_tris.extend(ids)

    tetrahedra_structs: List[Tetrahedron] = []
    for tet in tetrahedra.astype(np.int32):
        ids = [int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])]
        p0 = wp.vec3(vertex_list[ids[0]])
        p1 = wp.vec3(vertex_list[ids[1]])
        p2 = wp.vec3(vertex_list[ids[2]])
        p3 = wp.vec3(vertex_list[ids[3]])
        tet_struct = Tetrahedron()
        tet_struct.ids = wp.vec4i(ids[0], ids[1], ids[2], ids[3])
        tet_struct.rest_volume = compute_tet_volume(p0, p1, p2, p3)
        tetrahedra_structs.append(tet_struct)
        builder.add_tetrahedron(ids[0], ids[1], ids[2], ids[3], tetra_stiffness_mu, tetra_stiffness_lambda, tetra_dampen)

    connectors_wp = wp.zeros(0, dtype=TriPointsConnector, device=wp.get_device())
    tetrahedra_wp = wp.array(tetrahedra_structs, dtype=Tetrahedron, device=wp.get_device())

    subdomain_labels = None
    if getattr(mesh_data, "subdomain_labels", None) is not None:
        label_array = np.asarray(mesh_data.subdomain_labels, dtype=np.int32)
        if len(label_array) == len(tetrahedra_structs):
            subdomain_labels = label_array
            tet_labels_wp = wp.array(label_array.tolist(), dtype=wp.int32, device=wp.get_device())
        else:
            tet_labels_wp = None
    else:
        tet_labels_wp = None

    mesh_ranges = _compute_mesh_ranges(
        vertex_count=len(vertex_list),
        triangle_index_count=len(surface_tris),
        edge_count=len(edge_list),
        tet_count=len(tetrahedra_structs),
        subdomain_labels=subdomain_labels,
        config=config,
    )

    metadata = {
        "mesh_path": str(mesh_path),
        "generated": generated,
        "vertex_count": len(vertex_list),
        "tet_count": len(tetrahedra_structs),
        "subdomain_ids": sorted(np.unique(subdomain_labels).tolist()) if subdomain_labels is not None else [],
        "subdomain_map": dict(config.subdomain_map),
    }

    return (
        connectors_wp,
        surface_tris,
        uv_list,
        mesh_ranges,
        tetrahedra_wp,
        tet_labels_wp,
        metadata,
    )
