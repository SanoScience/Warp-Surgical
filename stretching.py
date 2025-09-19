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

@wp.kernel
def update_vertex_stretch_blend_kernel(
    vertex_to_edges: wp.array(dtype=wp.int32, ndim=2),   # [num_vertices, max_edges]
    vertex_edge_counts: wp.array(dtype=wp.int32),        # [num_vertices]
    spring_indices: wp.array(dtype=wp.int32),            # [num_springs * 2]
    spring_rest_length: wp.array(dtype=wp.float32),      # [num_springs]
    particle_q: wp.array(dtype=wp.vec3f),                # [num_particles]
    vertex_colors: wp.array(dtype=wp.vec4f),             # [num_particles]
    num_vertices: int,
    max_edges: int
):
    vid = wp.tid()
    if vid >= num_vertices:
        return

    edge_count = vertex_edge_counts[vid]
    if edge_count == 0:
        vertex_colors[vid] = wp.vec4(0.0, 0.0, 0.0, vertex_colors[vid][3])
        return

    stretch_sum = float(0.0)
    for i in range(edge_count):
        eid = vertex_to_edges[vid, i]
        a = spring_indices[eid * 2 + 0]
        b = spring_indices[eid * 2 + 1]
        pos_a = particle_q[a]
        pos_b = particle_q[b]
        current_length = wp.length(pos_a - pos_b)
        rest_length = spring_rest_length[eid]
        stretch = current_length / rest_length
        stretch_sum += stretch

    avg_stretch = stretch_sum / float(edge_count)

    # Normalize: 1.0 = no stretch, 1.5 = max stretch, clamp to [0,1]
    blend = wp.clamp((avg_stretch - 1.0) * 2.0, 0.0, 1.0)
    color = vertex_colors[vid]
    vertex_colors[vid] = wp.vec4(color[0], color[1], blend, color[3])

def stretching_breaking_process(sim):

    fat_range = sim.mesh_ranges.get("fat")
    if not fat_range:
        return

    tet_start = fat_range.get("tet_start")
    tet_count = fat_range.get("tet_count", 0)
    edge_start = fat_range.get("edge_start")
    edge_count = fat_range.get("edge_count", 0)

    if tet_start is None or edge_start is None or tet_count <= 0 or edge_count <= 0:
        return

    # Tetrahedrons
    wp.launch(
        deactivate_tets_by_stretch_kernel,
        dim=tet_count,
        inputs=[
            sim.model.tet_active[tet_start:tet_start+tet_count],
            sim.model.tetrahedra_wp[tet_start:tet_start+tet_count],
            sim.tet_to_edges[tet_start:tet_start+tet_count],
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
        dim=edge_count,
        inputs=[
            sim.model.spring_stiffness[edge_start:edge_start+edge_count],
            sim.model.spring_indices[edge_start*2:(edge_start+edge_count)*2],
            sim.model.spring_rest_length[edge_start:edge_start+edge_count],
            sim.state_0.particle_q,
            1.5,  # stretch_threshold
            num_springs
        ],
        device=wp.get_device()
    )
    # Update vertex stretch blend
    wp.launch(
        update_vertex_stretch_blend_kernel,
        dim=sim.model.particle_count,
        inputs=[
            sim.vertex_to_edges,
            sim.vertex_edge_counts,
            sim.model.spring_indices,
            sim.model.spring_rest_length,
            sim.state_0.particle_q,
            sim.vertex_colors,
            sim.model.particle_count,
            sim.vertex_edges_max
        ],
        device=wp.get_device()
    )
