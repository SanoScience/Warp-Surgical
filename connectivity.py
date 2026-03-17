from mesh_loader import Tetrahedron
import warp as wp

#region Kernels
@wp.kernel
def build_vertex_neighbor_table(
    tet_active: wp.array(dtype=wp.int32),         # [num_tets]
    tets: wp.array(dtype=Tetrahedron),            # [num_tets]
    vertex_neighbors: wp.array(dtype=wp.int32, ndim = 2),   # [num_vertices, max_neighbors]
    vertex_neighbor_counts: wp.array(dtype=wp.int32),       # [num_vertices]
    num_tets: int,
    max_neighbors: int
):
    tid = wp.tid()
    if tid >= num_tets:
        return

    if tet_active[tid] == 0:
        return

    tet = tets[tid]
    for i in range(4):
        v = tet.ids[i]

        for j in range(4):
            if i == j:
                continue

            n = tet.ids[j]

            # Atomically add neighbor if not already present
            # (no duplicate check)
            idx = wp.atomic_add(vertex_neighbor_counts, v, 1)
            if idx < max_neighbors:
                vertex_neighbors[v, idx] = n


@wp.kernel
def build_tet_edge_table(
    tets: wp.array(dtype=Tetrahedron),                # [num_tets]
    springs: wp.array(dtype=wp.int32),                # [num_springs * 2], flat array
    tet_to_edges: wp.array(dtype=wp.int32, ndim=2),   # [num_tets, 6]
    tet_edge_counts: wp.array(dtype=wp.int32),        # [num_tets]
    num_tets: int,
    num_springs: int,
):
    tid = wp.tid()
    if tid >= num_tets:
        return

    tet = tets[tid]
    tet_ids = tet.ids
    count = int(0)
    for i in range(num_springs):

        spring_a = springs[i * 2 + 0]
        spring_b = springs[i * 2 + 1]
        # Edge 0: (0,1)
        a = tet_ids[0]
        b = tet_ids[1]
        if ((spring_a == a and spring_b == b) or (spring_a == b and spring_b == a)):
            idx = wp.atomic_add(tet_edge_counts, tid, 1)
            if idx < 6:
                tet_to_edges[tid, idx] = i
            count += 1
        # Edge 1: (0,2)
        a = tet_ids[0]
        b = tet_ids[2]
        if ((spring_a == a and spring_b == b) or (spring_a == b and spring_b == a)):
            idx = wp.atomic_add(tet_edge_counts, tid, 1)
            if idx < 6:
                tet_to_edges[tid, idx] = i
            count += 1
        # Edge 2: (0,3)
        a = tet_ids[0]
        b = tet_ids[3]
        if ((spring_a == a and spring_b == b) or (spring_a == b and spring_b == a)):
            idx = wp.atomic_add(tet_edge_counts, tid, 1)
            if idx < 6:
                tet_to_edges[tid, idx] = i
            count += 1
        # Edge 3: (1,2)
        a = tet_ids[1]
        b = tet_ids[2]
        if ((spring_a == a and spring_b == b) or (spring_a == b and spring_b == a)):
            idx = wp.atomic_add(tet_edge_counts, tid, 1)
            if idx < 6:
                tet_to_edges[tid, idx] = i
            count += 1
        # Edge 4: (1,3)
        a = tet_ids[1]
        b = tet_ids[3]
        if ((spring_a == a and spring_b == b) or (spring_a == b and spring_b == a)):
            idx = wp.atomic_add(tet_edge_counts, tid, 1)
            if idx < 6:
                tet_to_edges[tid, idx] = i
            count += 1
        # Edge 5: (2,3)
        a = tet_ids[2]
        b = tet_ids[3]
        if ((spring_a == a and spring_b == b) or (spring_a == b and spring_b == a)):
            idx = wp.atomic_add(tet_edge_counts, tid, 1)
            if idx < 6:
                tet_to_edges[tid, idx] = i
            count += 1


@wp.kernel
def build_vertex_edge_table(
    spring_indices: wp.array(dtype=wp.int32),              # [num_springs * 2]
    vertex_to_edges: wp.array(dtype=wp.int32, ndim=2),     # [num_vertices, max_edges]
    vertex_edge_counts: wp.array(dtype=wp.int32),           # [num_vertices]
    num_springs: int,
    max_edges: int
):
    eid = wp.tid()
    if eid >= num_springs:
        return

    a = spring_indices[eid * 2 + 0]
    b = spring_indices[eid * 2 + 1]

    idx_a = wp.atomic_add(vertex_edge_counts, a, 1)
    if idx_a < max_edges:
        vertex_to_edges[a, idx_a] = eid

    idx_b = wp.atomic_add(vertex_edge_counts, b, 1)
    if idx_b < max_edges:
        vertex_to_edges[b, idx_b] = eid
#endregion

def setup_connectivity(warpsim):
    vertex_count = warpsim.model.particle_count
    vertex_neighbour_count = 32

    warpsim.vertex_to_vneighbours = wp.zeros((vertex_count, vertex_neighbour_count), dtype=wp.int32, device=wp.get_device())
    warpsim.vertex_vneighbor_counts = wp.zeros(vertex_count, dtype=wp.int32, device=wp.get_device())
    warpsim.vneighbours_max = vertex_neighbour_count

    # Vertex to edge mapping setup
    vertex_edge_count = 32
    warpsim.vertex_to_edges = wp.zeros((vertex_count, vertex_edge_count), dtype=wp.int32, device=wp.get_device())
    warpsim.vertex_edge_counts = wp.zeros(vertex_count, dtype=wp.int32, device=wp.get_device())
    warpsim.vertex_edges_max = vertex_edge_count

    # Tetrahedron to edge mapping setup
    num_tets = warpsim.model.tetrahedra_wp.shape[0]
    num_springs = warpsim.model.spring_indices.shape[0] // 2

    warpsim.tet_to_edges = wp.zeros((num_tets, 6), dtype=wp.int32, device=wp.get_device())
    warpsim.tet_edge_counts = wp.zeros(num_tets, dtype=wp.int32, device=wp.get_device())

def generate_connectivity(warpsim):
        num_tets = warpsim.model.tetrahedra_wp.shape[0]
        num_springs = warpsim.model.spring_indices.shape[0] // 2

        wp.launch(
            build_tet_edge_table,
            dim=num_tets,
            inputs=[
                warpsim.model.tetrahedra_wp,
                warpsim.model.spring_indices,  # flat int32 array
                warpsim.tet_to_edges,
                warpsim.tet_edge_counts,
                num_tets,
                num_springs
            ],
            device=wp.get_device()
        )

        # Build vertex to edge table
        wp.launch(
            build_vertex_edge_table,
            dim=num_springs,
            inputs=[
                warpsim.model.spring_indices,
                warpsim.vertex_to_edges,
                warpsim.vertex_edge_counts,
                num_springs,
                warpsim.vertex_edges_max
            ],
            device=wp.get_device()
        )
def recompute_connectivity(warpsim):
    wp.copy(warpsim.vertex_vneighbor_counts, wp.zeros(warpsim.model.particle_count, dtype=wp.int32, device=wp.get_device()))
            
    wp.launch(
        build_vertex_neighbor_table,
        dim=warpsim.model.tetrahedra_wp.shape[0],
        inputs=[
            warpsim.model.tet_active,
            warpsim.model.tetrahedra_wp,
            warpsim.vertex_to_vneighbours,
            warpsim.vertex_vneighbor_counts,
            warpsim.model.tetrahedra_wp.shape[0],
            warpsim.vneighbours_max
        ],
        device=wp.get_device()
    )