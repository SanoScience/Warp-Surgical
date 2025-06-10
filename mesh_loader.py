import warp as wp
import warp.sim

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

def load_mesh_and_build_model(builder: wp.sim.ModelBuilder, vertical_offset=0.0):
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
    
    # Liver
    liver_positions, liver_indices, liver_edges, liver_tris, liver_uvs = load_mesh_component('meshes/liver/', 0)
    mesh_ranges['liver'] = {
        'vertex_start': current_vertex_offset,
        'vertex_count': len(liver_positions),
        'index_start': current_index_offset,
        'index_count': len(liver_tris)
    }
    
    all_positions.extend(liver_positions)
    all_indices.extend(liver_indices)
    all_edges.extend(liver_edges)
    all_tri_surface_indices.extend(liver_tris)
    all_uvs.extend(liver_uvs)
    
    current_vertex_offset = len(all_positions)
    current_index_offset = len(all_tri_surface_indices)
    
    # Fat
    fat_particle_offset = len(all_positions)
    fat_positions, fat_indices, fat_edges, fat_tris, fat_uvs = load_mesh_component('meshes/fat/', fat_particle_offset)
    mesh_ranges['fat'] = {
        'vertex_start': current_vertex_offset,
        'vertex_count': len(fat_positions),
        'index_start': current_index_offset,
        'index_count': len(fat_tris)
    }
    
    all_positions.extend(fat_positions)
    all_indices.extend(fat_indices)
    all_edges.extend(fat_edges)
    all_tri_surface_indices.extend(fat_tris)
    all_uvs.extend(fat_uvs)
    
    current_vertex_offset = len(all_positions)
    current_index_offset = len(all_tri_surface_indices)
    
    # Gallbladder
    gallbladder_particle_offset = len(all_positions)
    gallbladder_positions, gallbladder_indices, gallbladder_edges, gallbladder_tris, gallbladder_uvs = load_mesh_component('meshes/gallbladder/', gallbladder_particle_offset)
    mesh_ranges['gallbladder'] = {
        'vertex_start': current_vertex_offset,
        'vertex_count': len(gallbladder_positions),
        'index_start': current_index_offset,
        'index_count': len(gallbladder_tris)
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
    mass_total = 10.0
    #mass = mass_total / len(all_positions)
    mass = 0.1

    for position in all_positions:
        pos = wp.vec3(position)
        pos[1] += vertical_offset
        #very ugly hardcoded position and radius below
        if is_particle_within_radius(pos, [0.5, 1.5, -5.0], 1.0):
            builder.add_particle(pos, wp.vec3(0, 0, 0), mass=0, radius=0.01)
        else:
            builder.add_particle(pos, wp.vec3(0, 0, 0), mass=mass, radius=0.01)
    
    
    # Add springs
    for i in range(0, len(all_edges), 2):
        builder.add_spring(all_edges[i], all_edges[i + 1], 1.0e3, 0.0, 0)
    
    # Add tetrahedrons
    for i in range(0, len(all_indices), 4):
        builder.add_tetrahedron(all_indices[i], all_indices[i + 1], all_indices[i + 2], all_indices[i + 3])
    
    return wp.array(all_connectors, dtype=TriPointsConnector, device=wp.get_device()), all_tri_surface_indices, all_uvs, mesh_ranges


def is_particle_within_radius(particle_pos, centre, radius):
    pos = wp.vec3(particle_pos[0], particle_pos[1], particle_pos[2])
    centre_pos = wp.vec3(centre[0], centre[1], centre[2])
    distance = wp.length(pos - centre_pos)
    return distance < radius
