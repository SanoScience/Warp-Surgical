import warp as wp
from newton._src.geometry.kernels import triangle_closest_point


@wp.func
def triangle_normal(v0: wp.vec3f, v1: wp.vec3f, v2: wp.vec3f) -> wp.vec3f:
    edge1 = v1 - v0
    edge2 = v2 - v0
    return wp.normalize(wp.cross(edge1, edge2))


@wp.kernel
def collide_triangles_vs_sphere(
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    inv_masses: wp.array(dtype=wp.float32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    sphere_center: wp.array(dtype=wp.vec3f),
    sphere_radius: wp.float32,
    sphere_center_scale: wp.float32,
    restitution: wp.float32,
    dt: wp.float32,
    cull_radius: wp.float32,
    delta_accumulator: wp.array(dtype=wp.vec3f),
    delta_counter: wp.array(dtype=wp.int32),
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

    sphere_pos = sphere_center[0] * sphere_center_scale

    if cull_radius > 0.0:
        centroid = (p1 + p2 + p3) / 3.0
        if wp.length(centroid - sphere_pos) > cull_radius:
            return

    w1 = inv_masses[t1]
    w2 = inv_masses[t2]
    w3 = inv_masses[t3]
    weight = w1 + w2 + w3
    if weight <= 0.0:
        return

    closest_p, bary, feature_type = triangle_closest_point(p1, p2, p3, sphere_pos)
    to_sphere = closest_p - sphere_pos
    dist = wp.length(to_sphere)
    if dist >= sphere_radius:
        return

    penetration = sphere_radius - dist
    if dist > 1e-8:
        correction_dir = to_sphere / dist
    else:
        correction_dir = triangle_normal(p1, p2, p3)

    total_correction = correction_dir * penetration
    d1 = total_correction * (w1 / weight)
    d2 = total_correction * (w2 / weight)
    d3 = total_correction * (w3 / weight)

    wp.atomic_add(delta_accumulator, t1, d1)
    wp.atomic_add(delta_accumulator, t2, d2)
    wp.atomic_add(delta_accumulator, t3, d3)
    wp.atomic_add(delta_counter, t1, 1)
    wp.atomic_add(delta_counter, t2, 1)
    wp.atomic_add(delta_counter, t3, 1)
