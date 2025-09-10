from centrelines import CentrelinePointInfo, ClampConstraint
import warp as wp
import newton
import os
import pickle
import time

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
    cache_path = filepath + ".cache"
    
    # Try to load from cache first
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) > os.path.getmtime(filepath):
        cached_data = load_from_cache(cache_path)
        if cached_data is not None:
            raw_connectors_data = cached_data
            
            # Apply offsets to cached data
            connectors = []
            for data in raw_connectors_data:
                connector = TriPointsConnector()
                connector.particle_id = data[0] + particle_id_offset
                connector.rest_dist = data[1]
                connector.tri_ids = wp.vec3i(data[2] + tri_id_offset, data[3] + tri_id_offset, data[4] + tri_id_offset)
                connector.tri_bar = wp.vec3f(data[5], data[6], data[7])
                connectors.append(connector)
            
            return connectors
    
    # Load from ASCII file if cache miss or invalid
    connectors = []
    raw_connectors_data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) != 8:
                raise ValueError(f"Line does not have 8 elements: {line}")
            
            # Store raw data for caching (without offsets)
            raw_data = (
                int(parts[0]),      # particle_id
                float(parts[1]),    # rest_dist
                int(parts[2]),      # tri_id_0
                int(parts[3]),      # tri_id_1
                int(parts[4]),      # tri_id_2
                float(parts[5]),    # tri_bar_x
                float(parts[6]),    # tri_bar_y
                float(parts[7])     # tri_bar_z
            )
            raw_connectors_data.append(raw_data)
            
            # Create connector with offsets for return
            connector = TriPointsConnector()
            connector.particle_id = int(parts[0]) + particle_id_offset
            connector.rest_dist = float(parts[1])
            connector.tri_ids = wp.vec3i(int(parts[2]) + tri_id_offset, int(parts[3]) + tri_id_offset, int(parts[4]) + tri_id_offset)
            connector.tri_bar = wp.vec3f(float(parts[5]), float(parts[6]), float(parts[7]))
            
            connectors.append(connector)
    
    # Save to cache (without offsets)
    save_to_cache(cache_path, raw_connectors_data)
    
    return connectors

def get_cache_path(base_path):
    """Get the cache file path for a given mesh directory."""
    folder_name = os.path.basename(os.path.normpath(base_path))
    return os.path.join(base_path, f"{folder_name}.cache")

def is_cache_valid(base_path, cache_path):
    """Check if cache is valid by comparing modification times."""
    if not os.path.exists(cache_path):
        return False
    
    cache_mtime = os.path.getmtime(cache_path)
    
    # Check all relevant mesh files
    mesh_files = [
        "model.vertices", "model.tetras", "model.edges", 
        "model.tris", "model.uvs"
    ]
    
    for filename in mesh_files:
        file_path = os.path.join(base_path, filename)
        if os.path.exists(file_path):
            if os.path.getmtime(file_path) > cache_mtime:
                return False
    
    return True

def load_from_cache(cache_path):
    """Load mesh data from binary cache file."""
    print(f"Loading mesh data from cache {cache_path}")
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading cache {cache_path}: {e}")
        return None

def save_to_cache(cache_path, data):
    """Save mesh data to binary cache file."""
    print(f"Saving mesh data to cache {cache_path}")
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Error saving cache {cache_path}: {e}")

def compute_tet_volume(p0, p1, p2, p3):
    # Returns the signed volume of the tetrahedron
    return abs(
        wp.dot(
            wp.cross(p1 - p0, p2 - p0),
            p3 - p0
        ) / 6.0
    )

def compute_triangle_normal(p0, p1, p2):
    """Compute the normal vector of a triangle using cross product."""
    v1 = wp.vec3(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    v2 = wp.vec3(p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
    normal = wp.cross(v1, v2)
    return wp.normalize(normal)

def get_tet_face_triangles(tet_indices):
    """Get all possible triangle faces of tetrahedra with consistent winding.
    
    For each tetrahedron with vertices [a, b, c, d], the four faces are:
    - Face 0: [a, c, b] (opposite to vertex d)
    - Face 1: [a, b, d] (opposite to vertex c) 
    - Face 2: [a, d, c] (opposite to vertex b)
    - Face 3: [b, c, d] (opposite to vertex a)
    
    The winding is chosen so normals point outward from the tetrahedron.
    """
    tet_faces = {}
    
    for i in range(0, len(tet_indices), 4):
        a, b, c, d = tet_indices[i:i+4]
        
        # Four faces of the tetrahedron with outward-pointing normals
        faces = [
            tuple(sorted([a, c, b])),  # Face opposite to d
            tuple(sorted([a, b, d])),  # Face opposite to c
            tuple(sorted([a, d, c])),  # Face opposite to b
            tuple(sorted([b, c, d]))   # Face opposite to a
        ]
        
        # Store the ordered vertices for proper winding
        face_windings = [
            (a, c, b),  # Face opposite to d
            (a, b, d),  # Face opposite to c
            (a, d, c),  # Face opposite to b
            (b, c, d)   # Face opposite to a
        ]
        
        for face, winding in zip(faces, face_windings):
            if face in tet_faces:
                # Face is shared between two tetrahedra - it's internal
                tet_faces[face] = None
            else:
                # Face is on the surface
                tet_faces[face] = winding
    
    # Return only surface faces with their correct winding
    surface_faces = {face: winding for face, winding in tet_faces.items() if winding is not None}
    return surface_faces

def fix_triangle_winding_with_tets(positions, tri_indices, tet_indices):
    """Fix triangle winding using tetrahedral mesh information.
    
    This is more reliable than the centroid method as it uses the actual
    tetrahedral structure to determine outward-facing normals.
    """
    if not tet_indices:
        print("No tetrahedral data available, using centroid-based method...")
        return fix_triangle_winding_centroid(positions, tri_indices)
    
    print("Fixing triangle winding using tetrahedral mesh information...")
    
    # Get surface faces from tetrahedra with correct winding
    surface_faces = get_tet_face_triangles(tet_indices)
    print(f"Found {len(surface_faces)} surface faces from tetrahedra")
    
    fixed_indices = []
    fixed_count = 0
    
    for i in range(0, len(tri_indices), 3):
        v0_idx = tri_indices[i]
        v1_idx = tri_indices[i + 1] 
        v2_idx = tri_indices[i + 2]
        
        # Create sorted tuple for lookup
        face_key = tuple(sorted([v0_idx, v1_idx, v2_idx]))
        
        if face_key in surface_faces:
            # Use the correct winding from tetrahedral analysis
            correct_winding = surface_faces[face_key]
            fixed_indices.extend(correct_winding)
            fixed_count += 1
        else:
            # Triangle not found in tetrahedral surface - keep original
            # This might be a manually added surface triangle
            fixed_indices.extend([v0_idx, v1_idx, v2_idx])
    
    print(f"Fixed winding for {fixed_count}/{len(tri_indices) // 3} triangles using tetrahedral data")
    return fixed_indices

def fix_triangle_winding_centroid(positions, tri_indices):
    """Fix triangle winding order using centroid-based method (fallback).
    
    This function ensures all triangles have counter-clockwise winding
    when viewed from outside. It uses the assumption that triangles
    should have outward-facing normals.
    """
    print("Fixing triangle winding order using centroid method...")
    fixed_indices = []
    
    for i in range(0, len(tri_indices), 3):
        v0_idx = tri_indices[i]
        v1_idx = tri_indices[i + 1]
        v2_idx = tri_indices[i + 2]
        
        # Get vertex positions
        p0 = positions[v0_idx]
        p1 = positions[v1_idx]
        p2 = positions[v2_idx]
        
        # Calculate triangle centroid
        centroid = [(p0[0] + p1[0] + p2[0]) / 3.0,
                   (p0[1] + p1[1] + p2[1]) / 3.0,
                   (p0[2] + p1[2] + p2[2]) / 3.0]
        
        # Calculate triangle normal
        normal = compute_triangle_normal(p0, p1, p2)
        
        # Calculate vector from origin to centroid (assuming mesh is centered around origin)
        to_centroid = wp.vec3(centroid[0], centroid[1], centroid[2])
        
        # If normal points inward (dot product < 0), flip the triangle
        if wp.dot(normal, to_centroid) < 0:
            # Reverse winding order by swapping last two vertices
            fixed_indices.extend([v0_idx, v2_idx, v1_idx])
        else:
            # Keep original order
            fixed_indices.extend([v0_idx, v1_idx, v2_idx])
    
    print(f"Fixed winding for {len(tri_indices) // 3} triangles using centroid method")
    return fixed_indices

def load_mesh_component(base_path, offset=0):
    """Load a single mesh component and return positions, indices, and edges."""
    cache_path = get_cache_path(base_path)
    
    # Try to load from cache first
    if is_cache_valid(base_path, cache_path):
        cached_data = load_from_cache(cache_path)
        if cached_data is not None:
            positions, raw_indices, raw_edges, raw_tri_surface_indices, uvs = cached_data
            
            # Apply offset to cached data
            indices = [x + offset for x in raw_indices]
            edges = [x + offset for x in raw_edges] 
            tri_surface_indices = [x + offset for x in raw_tri_surface_indices]
            
            return positions, indices, edges, tri_surface_indices, uvs
    
    # Load from ASCII files if cache miss or invalid
    print(f"Loading ASCII mesh component from {base_path}")
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
    
    # Load raw indices (without offset for caching)
    raw_indices = []
    with open(indices_file, 'r') as f:
        for line in f:
            raw_indices.extend([int(x) for x in line.split()])
    
    # Load raw edges (without offset for caching)
    raw_edges = []
    with open(edges_file, 'r') as f:
        for line in f:
            raw_edges.extend([int(x) for x in line.split()])

    # Load raw surface triangle indices (without offset for caching)
    raw_tri_surface_indices = []
    with open(surface_indices_file, 'r') as f:
        for line in f:
            raw_tri_surface_indices.extend([int(x) for x in line.split()])
    
    # Fix triangle winding order using tetrahedral data
    raw_tri_surface_indices = fix_triangle_winding_with_tets(positions, raw_tri_surface_indices, raw_indices)
    
    # Load uvs
    with open(uvs_file, 'r') as f:
        for line in f:
            uv = [float(x) for x in line.split()]
            uvs.append(uv)

    # Save to cache (with fixed triangle indices, without offset)
    cache_data = (positions, raw_indices, raw_edges, raw_tri_surface_indices, uvs)
    save_to_cache(cache_path, cache_data)
    
    # Apply offset for return values
    indices = [x + offset for x in raw_indices]
    edges = [x + offset for x in raw_edges]
    tri_surface_indices = [x + offset for x in raw_tri_surface_indices]

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

def load_cgal_and_build_model(builder: newton.ModelBuilder, particle_mass, vertical_offset=0.0, spring_stiffness=1.0, spring_dampen=0.0, tetra_stiffness_mu=1.0e3, tetra_stiffness_lambda=1.0e3, tetra_dampen=0.0):
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
    
    # Check if multi-label meshes exist, otherwise fall back to single mesh
    import os
    multi_label_dirs = []
    for i in range(1, 10):  # Check for up to 9 labels
        label_dir = f'meshes/liver_cgal_label_{i}/'
        if os.path.exists(label_dir):
            multi_label_dirs.append((i, label_dir))
    
    if multi_label_dirs:
        print(f"Found {len(multi_label_dirs)} multi-label meshes, loading only first label for debugging...")
        # Load only the first label for debugging
        label_num, label_dir = multi_label_dirs[0]  # Only first label
        print(f"  Loading label {label_num} from {label_dir}")
        mesh_positions, mesh_indices, mesh_edges, mesh_tris, mesh_uvs = load_mesh_component(label_dir, 0)
        
        all_positions.extend(mesh_positions)
        all_indices.extend(mesh_indices)
        all_edges.extend(mesh_edges)
        all_tri_surface_indices.extend(mesh_tris)
        all_uvs.extend(mesh_uvs)
        
        current_vertex_offset = len(all_positions)
        current_edge_offset = len(all_edges)
        current_index_offset = len(all_tri_surface_indices)
        current_tet_offset = len(all_indices) // 4
        
        # Set liver mesh range for single label
        mesh_ranges['liver']['vertex_start'] = 0
        mesh_ranges['liver']['vertex_count'] = current_vertex_offset
        mesh_ranges['liver']['index_start'] = 0
        mesh_ranges['liver']['index_count'] = current_tet_offset
    else:
        print("No multi-label meshes found, loading single mesh...")
        # Fallback to single mesh loading
        mesh_positions, mesh_indices, mesh_edges, mesh_tris, mesh_uvs = load_mesh_component('meshes/liver_cgal/', 0)
        
        all_positions.extend(mesh_positions)
        all_indices.extend(mesh_indices)
        all_edges.extend(mesh_edges)
        all_tri_surface_indices.extend(mesh_tris)
        all_uvs.extend(mesh_uvs)
        
        current_vertex_offset = len(all_positions)
        current_edge_offset = len(all_edges)
        current_index_offset = len(all_tri_surface_indices)
        current_tet_offset = len(all_indices) // 4
        
        # Set liver mesh range for single mesh
        mesh_ranges['liver']['vertex_start'] = 0
        mesh_ranges['liver']['vertex_count'] = current_vertex_offset
        mesh_ranges['liver']['index_start'] = 0
        mesh_ranges['liver']['index_count'] = current_tet_offset
    
    
    
    # Add particles to model
    print("Adding particles to model...")
    for position in all_positions:
        pos = wp.vec3(position) * 0.01
        #pos[1] += vertical_offset
        #very ugly hardcoded position and radius below
        # if is_particle_within_radius(pos, [0.5, 1.5, -5.0], 1.0):
        #     builder.add_particle(pos, wp.vec3(0, 0, 0), mass=0, radius=0.01)
        # else:
        builder.add_particle(pos, wp.vec3(0, 0, 0), mass=particle_mass, radius=0.01)
    
    
    # Add springs
    print("Adding springs to model...")
    for i in range(0, len(all_edges), 2):
        builder.add_spring(all_edges[i], all_edges[i + 1], spring_stiffness, spring_dampen, 0)
    
    
    # Add triangles
    print("Adding triangles to model...")
    for i in range(0, len(all_tri_surface_indices), 3):
        ids = [all_tri_surface_indices[i], all_tri_surface_indices[i + 1], all_tri_surface_indices[i + 2]]
        builder.add_triangle(ids[0], ids[1], ids[2])

    all_tetrahedra = []

    # Add tetrahedrons (neo-hookean + custom volume constraint)
    # print("Adding tetrahedra to model...", len(all_indices))
    # for i in range(0, len(all_indices), 4):
    #     ids = [all_indices[i], all_indices[i+1], all_indices[i+2], all_indices[i+3]]
    #     p0 = wp.vec3(all_positions[ids[0]])
    #     p1 = wp.vec3(all_positions[ids[1]])
    #     p2 = wp.vec3(all_positions[ids[2]])
    #     p3 = wp.vec3(all_positions[ids[3]])
    #     rest_volume = compute_tet_volume(p0, p1, p2, p3)
        
    #     tet = Tetrahedron()
    #     tet.ids = wp.vec4i(ids[0], ids[1], ids[2], ids[3])
    #     tet.rest_volume = rest_volume

    #     all_tetrahedra.append(tet)
    #     builder.add_tetrahedron(all_indices[i], all_indices[i + 1], all_indices[i + 2], all_indices[i + 3], tetra_stiffness_mu, tetra_stiffness_lambda, tetra_dampen)



    return wp.array(all_connectors, dtype=TriPointsConnector, device=wp.get_device()), all_tri_surface_indices, all_uvs, mesh_ranges, wp.array(all_tetrahedra, dtype=Tetrahedron, device=wp.get_device())


def load_background_mesh():
    base_path = "meshes/cavity/"
    cache_path = get_cache_path(base_path)
    
    # Try to load from cache first
    if is_cache_valid(base_path, cache_path):
        cached_data = load_from_cache(cache_path)
        if cached_data is not None:
            positions, tri_indices, uvs = cached_data
            
            return (wp.array(positions, dtype=wp.vec3f, device=wp.get_device()),  
                   wp.array(tri_indices, dtype=wp.int32, device=wp.get_device()), 
                   wp.array(uvs, dtype=wp.vec2f, device=wp.get_device()),  
                   wp.zeros(len(positions), dtype=wp.vec4f, device=wp.get_device()))
    
    # Load from ASCII files if cache miss or invalid
    positions = []
    tri_indices = []
    uvs = []
    
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
    
    # Fix triangle winding order (no tetrahedral data available for background)
    tri_indices = fix_triangle_winding_centroid(positions, tri_indices)
    
    # Load uvs
    with open(uvs_file, 'r') as f:
        for line in f:
            uv = [float(x) for x in line.split()]
            uvs.append(uv)
    
    # Save to cache
    cache_data = (positions, tri_indices, uvs)
    save_to_cache(cache_path, cache_data)

    return (wp.array(positions, dtype=wp.vec3f, device=wp.get_device()),  
           wp.array(tri_indices, dtype=wp.int32, device=wp.get_device()), 
           wp.array(uvs, dtype=wp.vec2f, device=wp.get_device()),  
           wp.zeros(len(positions), dtype=wp.vec4f, device=wp.get_device()))

def parse_centreline_file(filepath, point_offset, edge_offset):
    """
    Parse a centreline file and return:
      - a list of CentrelinePointInfo
      - a flat list of all point ids (int)
      - a flat list of all point dists (float)
      - a flat list of all edge ids (int)
    """
    cache_path = filepath + ".cache"
    
    # Try to load from cache first
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) > os.path.getmtime(filepath):
        cached_data = load_from_cache(cache_path)
        if cached_data is not None:
            raw_centreline_data = cached_data
            
            # Apply offsets to cached data
            centreline_infos = []
            all_point_cnstrs = []
            all_edge_ids = []
            
            point_start_id = 0
            edge_start_id = 0
            
            for line_data in raw_centreline_data:
                point_count, points_data, edge_count, edges_data, stiffness, rest_length_mul, radius = line_data
                
                # Create point constraints with offsets
                point_cnstr = []
                for point_id, dist in points_data:
                    cnstr = ClampConstraint()
                    cnstr.id = point_id + point_offset
                    cnstr.dist = dist
                    point_cnstr.append(cnstr)
                
                # Apply edge offsets
                edge_ids = [edge_id + edge_offset for edge_id in edges_data]
                
                # Create info
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
    
    # Load from ASCII file if cache miss or invalid
    centreline_infos = []
    all_point_cnstrs = []
    all_edge_ids = []
    raw_centreline_data = []

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
            points_data = []
            for _ in range(point_count):
                id = int(parts[idx])
                idx += 1
                dist = float(parts[idx])
                idx += 1
                
                # Store raw data for caching
                points_data.append((id, dist))

                # Create constraint with offset
                cnstr = ClampConstraint()
                cnstr.id = id + point_offset
                cnstr.dist = dist
                point_cnstr.append(cnstr)

            # Edge ids
            edge_count = int(parts[idx])
            idx += 1
            edge_ids = []
            edges_data = []
            for _ in range(edge_count):
                edge_id = int(parts[idx])
                edges_data.append(edge_id)  # Store raw data for caching
                edge_ids.append(edge_id + edge_offset)
                idx += 1

            # Stiffness, rest_length_mul, radius
            stiffness = float(parts[idx])
            idx += 1
            rest_length_mul = float(parts[idx])
            idx += 1
            radius = float(parts[idx])
            idx += 1

            # Store raw data for caching
            line_data = (point_count, points_data, edge_count, edges_data, stiffness, rest_length_mul, radius)
            raw_centreline_data.append(line_data)

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
    
    # Save to cache (without offsets)
    save_to_cache(cache_path, raw_centreline_data)

    return centreline_infos, all_point_cnstrs, all_edge_ids

def is_particle_within_radius(particle_pos, centre, radius):
    pos = wp.vec3(particle_pos[0], particle_pos[1], particle_pos[2])
    centre_pos = wp.vec3(centre[0], centre[1], centre[2])
    distance = wp.length(pos - centre_pos)
    return distance < radius
