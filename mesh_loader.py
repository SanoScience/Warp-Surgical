from centrelines import CentrelinePointInfo, ClampConstraint
import warp as wp
import newton

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
