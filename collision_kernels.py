import warp as wp
from newton._src.geometry.kernels import (
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

@wp.func
def triangle_normal(v0: wp.vec3f, v1: wp.vec3f, v2: wp.vec3f) -> wp.vec3f:
    """
    Calculate the normal vector of a triangle from three vertices.
    
    Args:
        v0: First vertex of the triangle
        v1: Second vertex of the triangle  
        v2: Third vertex of the triangle
        
    Returns:
        Normalized normal vector of the triangle
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = wp.cross(edge1, edge2)
    return wp.normalize(normal)

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
def collide_triangles_vs_sphere(
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    inv_masses: wp.array(dtype=wp.float32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    sphere_center: wp.array(dtype=wp.vec3f),
    sphere_radius: wp.float32,
    restitution: wp.float32,
    dt: wp.float32,
    delta_accumulator: wp.array(dtype=wp.vec3f),
    delta_counter: wp.array(dtype=wp.int32)
):
    tid = wp.tid()
    if tid >= tri_indices.shape[0]:
        return
    
    t1 = tri_indices[tid, 0]
    t2 = tri_indices[tid, 1]
    t3 = tri_indices[tid, 2]

    p1 = positions[t1]
    p2 = positions[t2]
    p3 = positions[t3]

    w1 = inv_masses[t1]
    w2 = inv_masses[t2]
    w3 = inv_masses[t3]
    w = w1 + w2 + w3

    # Skip if all vertices are static (infinite mass)
    if w <= 0.0:
        return

    sp = sphere_center[0] * 0.01

    closest_p, bary, feature_type = triangle_closest_point(p1, p2, p3, sp)
    
    to_sphere = closest_p - sp
    dist = wp.length(to_sphere)
    
    if dist < sphere_radius:
        penetration = sphere_radius - dist
        
        # Avoid division by zero
        if dist > 1e-8:
            correction_dir = to_sphere / dist  # Normalized direction from sphere to triangle
        else:
            # Use triangle normal as fallback if closest point is exactly at sphere center
            tri_normal = triangle_normal(p1, p2, p3)
            correction_dir = tri_normal
        
        # Distribute correction based on barycentric coordinates and masses
        # Barycentric coordinates from triangle_closest_point: bary = (u, v, w) where u+v+w=1
        # p = u*p1 + v*p2 + w*p3, but we need to extract individual weights
        
        # For now, distribute evenly weighted by inverse mass
        total_correction = correction_dir * penetration
        d1 = total_correction * (w1 / w)
        d2 = total_correction * (w2 / w)
        d3 = total_correction * (w3 / w)

        wp.atomic_add(delta_accumulator, t1, d1)
        wp.atomic_add(delta_accumulator, t2, d2)
        wp.atomic_add(delta_accumulator, t3, d3)

        wp.atomic_add(delta_counter, t1, 1)
        wp.atomic_add(delta_counter, t2, 1)
        wp.atomic_add(delta_counter, t3, 1)

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