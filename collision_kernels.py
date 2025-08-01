import warp as wp
from newton.geometry.kernels import (
    triangle_closest_point,
    vertex_adjacent_to_triangle,
    VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX,
    TRI_COLLISION_BUFFER_OVERFLOW_INDEX,
    TRI_CONTACT_FEATURE_VERTEX_A,
    TRI_CONTACT_FEATURE_VERTEX_B,
    TRI_CONTACT_FEATURE_VERTEX_C,
    TRI_CONTACT_FEATURE_EDGE_AB,
    TRI_CONTACT_FEATURE_EDGE_AC,
    TRI_CONTACT_FEATURE_EDGE_BC,
    TRI_CONTACT_FEATURE_FACE_INTERIOR,

)

@wp.kernel
def collide_particles_vs_sphere(
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    inv_masses: wp.array(dtype=wp.float32),
    sphere_center: wp.array(dtype=wp.vec3f),
    sphere_radius: wp.float32,
    restitution: wp.float32,
    dt: wp.float32,
    deltas: wp.array(dtype=wp.vec3f)
):
    tid = wp.tid()
    if tid >= len(positions):
        return
    pos = positions[tid]
    #vel = velocities[tid]
    inv_mass = inv_masses[tid]

    # Sphere collision detection
    to_sphere = pos - sphere_center[0] * 0.01  # Scale the sphere center if needed
    dist = wp.length(to_sphere)
    if inv_mass > 0 and dist < sphere_radius:
        # Collision response
        penetration = sphere_radius - dist
        # if inv_mass > 0.0:
        #     # Apply restitution
        #     vel += wp.normalize(to_sphere) * penetration * restitution
        # else:
        # Static body, just move it out of the sphere
        #pos += wp.normalize(to_sphere) * penetration
        deltas[tid] += wp.normalize(to_sphere) * penetration
    # Update positions and velocities
    #positions[tid] = pos
    #velocities[tid] = vel



@wp.kernel
def vertex_triangle_collision_det(
    query_radius: float,
    bvh_id: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    vertex_colliding_triangles_offsets: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_buffer_sizes: wp.array(dtype=wp.int32),
    # outputs
    vertex_colliding_triangles: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_count: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_min_dist: wp.array(dtype=float),
    triangle_colliding_vertices_min_dist: wp.array(dtype=float),
    resize_flags: wp.array(dtype=wp.int32),
):
    """
    This function applies discrete collision detection between vertices and triangles. It uses pre-allocated spaces to
    record the collision data. Unlike `vertex_triangle_collision_detection_kernel`, this collision detection kernel
    works only in one way, i.e., it only records vertices' colliding triangles to `vertex_colliding_triangles`.

    This function assumes that all the vertices are on triangles, and can be indexed from the pos argument.

    Note:

        The collision date buffer is pre-allocated and cannot be changed during collision detection, therefore, the space
        may not be enough. If the space is not enough to record all the collision information, the function will set a
        certain element in resized_flag to be true. The user can reallocate the buffer based on vertex_colliding_triangles_count
        and vertex_colliding_triangles_count.

    Attributes:
        bvh_id (int): the bvh id you want to collide with
        query_radius (float): the contact radius. vertex-triangle pairs whose distance are less than this will get detected
        pos (array): positions of all the vertices that make up triangles
        vertex_colliding_triangles (array): flattened buffer of vertices' collision triangles, every two elements records
            the vertex index and a triangle index it collides to
        vertex_colliding_triangles_count (array): number of triangles each vertex collides
        vertex_colliding_triangles_offsets (array): where each vertex' collision buffer starts
        vertex_colliding_triangles_buffer_sizes (array): size of each vertex' collision buffer, will be modified if resizing is needed
        vertex_colliding_triangles_min_dist (array): each vertex' min distance to all (non-neighbor) triangles
        triangle_colliding_vertices_min_dist (array): each triangle's min distance to all (non-self) vertices
        resized_flag (array): size == 3, (vertex_buffer_resize_required, triangle_buffer_resize_required, edge_buffer_resize_required)
    """

    v_index = wp.tid()
    v = pos[v_index]
    vertex_buffer_offset = vertex_colliding_triangles_offsets[v_index]
    vertex_buffer_size = vertex_colliding_triangles_offsets[v_index + 1] - vertex_buffer_offset

    lower = wp.vec3(v[0] - query_radius, v[1] - query_radius, v[2] - query_radius)
    upper = wp.vec3(v[0] + query_radius, v[1] + query_radius, v[2] + query_radius)

    query = wp.bvh_query_aabb(bvh_id, lower, upper)

    tri_index = wp.int32(0)
    vertex_num_collisions = wp.int32(0)
    min_dis_to_tris = query_radius
    while wp.bvh_query_next(query, tri_index):
        t1 = tri_indices[tri_index, 0]
        t2 = tri_indices[tri_index, 1]
        t3 = tri_indices[tri_index, 2]
        if vertex_adjacent_to_triangle(v_index, t1, t2, t3):
            continue

        u1 = pos[t1]
        u2 = pos[t2]
        u3 = pos[t3]

        closest_p, bary, feature_type = triangle_closest_point(u1, u2, u3, v)

        dist = wp.length(closest_p - v)

        if dist < query_radius:
            # record v-f collision to vertex
            min_dis_to_tris = wp.min(min_dis_to_tris, dist)
            if vertex_num_collisions < vertex_buffer_size:
                vertex_colliding_triangles[2 * (vertex_buffer_offset + vertex_num_collisions)] = v_index
                vertex_colliding_triangles[2 * (vertex_buffer_offset + vertex_num_collisions) + 1] = tri_index
            else:
                resize_flags[VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX] = 1

            vertex_num_collisions = vertex_num_collisions + 1

            wp.atomic_min(triangle_colliding_vertices_min_dist, tri_index, dist)

    vertex_colliding_triangles_count[v_index] = vertex_num_collisions
    vertex_colliding_triangles_min_dist[v_index] = min_dis_to_tris