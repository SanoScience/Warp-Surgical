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

def load_mesh_component(vertices_file, indices_file, edges_file, offset=0):
    """Load a single mesh component and return positions, indices, and edges."""
    positions = []
    indices = []
    edges = []
    
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
    
    return positions, indices, edges

def load_mesh_and_build_model(builder: wp.sim.ModelBuilder, vertical_offset=0.0):
    """Load all mesh components and build the simulation model."""
    all_positions = []
    all_indices = []
    all_edges = []
    all_connectors = []
    
    # Liver
    liver_positions, liver_indices, liver_edges = load_mesh_component(
        'meshes/liver.vertices', 'meshes/liver.indices', 'meshes/liver.edges', 0)
    liver_particle_offset = 0
    all_positions.extend(liver_positions)
    all_indices.extend(liver_indices)
    all_edges.extend(liver_edges)
    
    # Fat
    fat_particle_offset = len(all_positions)
    fat_positions, fat_indices, fat_edges = load_mesh_component(
        'meshes/fat.vertices', 'meshes/fat.indices', 'meshes/fat.edges', fat_particle_offset)
    all_positions.extend(fat_positions)
    all_indices.extend(fat_indices)
    all_edges.extend(fat_edges)
    
    # Gallbladder
    gallbladder_particle_offset = len(all_positions)
    gallbladder_positions, gallbladder_indices, gallbladder_edges = load_mesh_component(
        'meshes/gallbladder.vertices', 'meshes/gallbladder.indices', 'meshes/gallbladder.edges', gallbladder_particle_offset)
    all_positions.extend(gallbladder_positions)
    all_indices.extend(gallbladder_indices)
    all_edges.extend(gallbladder_edges)
    
    # Load connectors
    fat_liver_connectors = parse_connector_file('meshes/fat-liver.connector', fat_particle_offset, liver_particle_offset)
    gallbladder_fat_connectors = parse_connector_file('meshes/gallbladder-fat.connector', gallbladder_particle_offset, fat_particle_offset)
    all_connectors.extend(fat_liver_connectors)
    all_connectors.extend(gallbladder_fat_connectors)
    
    # Add particles to model
    mass_total = 10.0
    mass = mass_total / len(all_positions)
    
    for position in all_positions:
        pos = wp.vec3(position)
        pos[1] += vertical_offset
        builder.add_particle(pos, wp.vec3(0, 0, 0), mass=mass, radius=0.02)
    
    # Add springs
    for i in range(0, len(all_edges), 2):
        builder.add_spring(all_edges[i], all_edges[i + 1], 1.0e3, 0.0, 0)
    
    # Add tetrahedrons
    for i in range(0, len(all_indices), 4):
        builder.add_tetrahedron(all_indices[i], all_indices[i + 1], all_indices[i + 2], all_indices[i + 3])
    
    return wp.array(all_connectors, dtype=TriPointsConnector, device=wp.get_device())
