import warp as wp
from mesh_loader import Tetrahedron

@wp.kernel
def deactivate_tets_by_stretch_kernel(
    tet_active: wp.array(dtype=wp.int32),         # [num_tets]
    tets: wp.array(dtype=Tetrahedron),            # [num_tets]
    tet_to_edges: wp.array(dtype=wp.int32, ndim=2), # [num_tets, 6]
    edges: wp.array(dtype=wp.int32),              # [num_springs * 2]
    edges_rest_len: wp.array(dtype=wp.float32),   # [num_springs]
    particle_q: wp.array(dtype=wp.vec3f),         # [num_particles]
    vertex_colors: wp.array(dtype=wp.vec4f),      # [num_particles]
    stretch_threshold: float,
    num_tets: int,
    blood_amount: float
):
    tid = wp.tid()
    if tid >= num_tets:
        return

    max_stretch = 1.0
    for i in range(6):
        edge_id = tet_to_edges[tid, i]
        a = edges[edge_id * 2 + 0]
        b = edges[edge_id * 2 + 1]

        pos_a = particle_q[a]
        pos_b = particle_q[b]
        current_length = wp.length(pos_a - pos_b)
        rest_length = edges_rest_len[edge_id]

        stretch = current_length / rest_length
        if stretch > max_stretch:
            max_stretch = stretch

    if max_stretch > stretch_threshold:
        tet_active[tid] = 0
        tet = tets[tid]

        # Add blood to vertex colors (channel A)
        for i in range(4):
            pid = tet.ids[i]
            color = vertex_colors[pid]
            vertex_colors[pid] = wp.vec4(color[0], color[1], color[2], wp.clamp(color[3] + blood_amount, 0.0, 1.0))

@wp.kernel
def deactivate_edges_by_stretch_kernel(
    spring_stiffness: wp.array(dtype=wp.float32),   # [num_springs]
    spring_indices: wp.array(dtype=wp.int32),       # [num_springs * 2]
    spring_rest_length: wp.array(dtype=wp.float32), # [num_springs]
    particle_q: wp.array(dtype=wp.vec3f),           # [num_particles]
    stretch_threshold: float,
    num_springs: int
):
    eid = wp.tid()
    if eid >= num_springs:
        return

    a = spring_indices[eid * 2 + 0]
    b = spring_indices[eid * 2 + 1]
    pos_a = particle_q[a]
    pos_b = particle_q[b]
    current_length = wp.length(pos_a - pos_b)
    rest_length = spring_rest_length[eid]
    stretch = current_length / rest_length

    if stretch > stretch_threshold:
        spring_stiffness[eid] = 0.0

def stretching_breaking_process(sim):
    # Tetrahedrons
    wp.launch(
        deactivate_tets_by_stretch_kernel,
        dim=sim.model.tetrahedra_wp.shape[0],
        inputs=[
            sim.model.tet_active,
            sim.model.tetrahedra_wp,
            sim.tet_to_edges,
            sim.model.spring_indices,
            sim.model.spring_rest_length,
            sim.state_0.particle_q,
            sim.vertex_colors,
            1.5,  # stretch_threshold
            sim.model.tetrahedra_wp.shape[0],
            0.2   # blood_amount
        ],
        device=wp.get_device()
    )
    # Edges
    num_springs = sim.model.spring_indices.shape[0] // 2
    wp.launch(
        deactivate_edges_by_stretch_kernel,
        dim=num_springs,
        inputs=[
            sim.model.spring_stiffness,
            sim.model.spring_indices,
            sim.model.spring_rest_length,
            sim.state_0.particle_q,
            1.5,  # stretch_threshold
            num_springs
        ],
        device=wp.get_device()
    )
