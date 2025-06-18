import warp as wp
from mesh_loader import TriPointsConnector

@wp.kernel
def clear_jacobian_accumulator(delta_accumulator: wp.array(dtype=wp.vec3f),
                               count_accumulator: wp.array(dtype=wp.int32)):
    i = wp.tid()
    delta_accumulator[i] = wp.vec3f(0.0, 0.0, 0.0)
    count_accumulator[i] = 0

@wp.kernel
def apply_jacobian_deltas(positions: wp.array(dtype=wp.vec3f),
                         delta_accumulator: wp.array(dtype=wp.vec3f),
                         count_accumulator: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if count_accumulator[i] > 0:
        normalized_delta = delta_accumulator[i] / wp.float32(count_accumulator[i])
        positions[i] = positions[i] + normalized_delta

@wp.kernel
def apply_tri_points_constraints_jacobian(positions: wp.array(dtype=wp.vec3f),
                                         connectors: wp.array(dtype=TriPointsConnector),
                                         delta_accumulator: wp.array(dtype=wp.vec3f)):
    i = wp.tid()
    conn = connectors[i]

    tri_pos = positions[conn.tri_ids[0]] * conn.tri_bar[0] + \
              positions[conn.tri_ids[1]] * conn.tri_bar[1] + \
              positions[conn.tri_ids[2]] * conn.tri_bar[2]
    
    dir = positions[conn.particle_id] - tri_pos
    length = wp.length(dir)
    if length < 1e-7:
        return
    
    kS = 0.1
    invMassP = 0.75
    invMassT = 1.0 - invMassP

    C = length - conn.rest_dist * 0.1
    s = invMassP + invMassT * conn.tri_bar[0] * conn.tri_bar[0] + invMassT * conn.tri_bar[1] * conn.tri_bar[1] + invMassT * conn.tri_bar[2] * conn.tri_bar[2]
    dP = (C / s) * (dir / length) * kS

    # Accumulate deltas
    wp.atomic_add(delta_accumulator, conn.particle_id, -dP * invMassP)
    wp.atomic_add(delta_accumulator, conn.tri_ids[0], dP * conn.tri_bar[0] * invMassT)
    wp.atomic_add(delta_accumulator, conn.tri_ids[1], dP * conn.tri_bar[1] * invMassT)
    wp.atomic_add(delta_accumulator, conn.tri_ids[2], dP * conn.tri_bar[2] * invMassT)

    #wp.atomic_add(count_accumulator, conn.particle_id, 1)
    #wp.atomic_add(count_accumulator, conn.tri_ids[0], 1)
    #wp.atomic_add(count_accumulator, conn.tri_ids[1], 1)
    #wp.atomic_add(count_accumulator, conn.tri_ids[2], 1)

@wp.kernel
def set_body_position(body_q: wp.array(dtype=wp.transformf), 
                      body_qd: wp.array(dtype=wp.spatial_vectorf),
                      body_id: int, 
                      posParameter: wp.array(dtype=wp.vec3f),
                      dt: wp.float32):
    t = body_q[body_id]
    prev_pos = wp.transform_get_translation(t)
    new_pos = posParameter[0] * 0.01

    # Set translation part while preserving rotation
    body_q[body_id] = wp.transform(new_pos, wp.quat(t[3], t[4], t[5], t[6]))

    # Update velocity
    lin_vel = (new_pos - prev_pos) / dt
    body_qd[body_id] = wp.spatial_vector(lin_vel, wp.vec3f(0.0, 0.0, 0.0))

@wp.kernel
def paint_vertices_near_haptic(
    vertex_positions: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transformf),
    vertex_colors: wp.array(dtype=wp.vec3),
    paint_radius: wp.float32,
    paint_color: wp.array(dtype=wp.vec3),  # Now reads from array
    paint_strength: wp.array(dtype=wp.float32),  # Now reads from array
    falloff_power: wp.float32
):
    """Paint vertex colors based on distance to haptic proxy position."""
    tid = wp.tid()
    if tid >= len(vertex_positions):
        return


    # Get haptic position (assuming single element array)
    haptic_transform = body_q[0]
    haptic_pos = wp.transform_get_translation(haptic_transform)
    vertex_pos = vertex_positions[tid]
    
    # Calculate distance from vertex to haptic position
    distance = wp.length(vertex_pos - haptic_pos)
    
    # Only paint if within radius
    if distance <= paint_radius:
        # Calculate falloff factor (1.0 at center, 0.0 at radius edge)
        falloff = 1.0 - wp.pow(distance / paint_radius, falloff_power)
        
        # Get current vertex color
        current_color = vertex_colors[tid]
        
        # Read paint color and strength from arrays
        color = paint_color[0]
        strength = paint_strength[0]
        
        # Blend with paint color based on falloff and strength
        new_color = current_color + color * falloff * strength
        
        # Clamp to [0, 1] range
        new_color = wp.vec3(
            wp.clamp(new_color[0], 0.0, 1.0),
            wp.clamp(new_color[1], 0.0, 1.0),
            wp.clamp(new_color[2], 0.0, 1.0)
        )
        
        vertex_colors[tid] = new_color


from mesh_loader import Tetrahedron

@wp.kernel
def apply_volume_constraints_jacobian(
    positions: wp.array(dtype=wp.vec3f),
    tetrahedra: wp.array(dtype=Tetrahedron),
    delta_accumulator: wp.array(dtype=wp.vec3f),
    stiffness: wp.float32
):
    tid = wp.tid()
    tet = tetrahedra[tid]
    ids = tet.ids

    p0 = positions[ids[0]]
    p1 = positions[ids[1]]
    p2 = positions[ids[2]]
    p3 = positions[ids[3]]

    # Compute current volume
    v = wp.dot(wp.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0

    # Constraint: C = v - rest_volume = 0
    C = v - tet.rest_volume

    # Compute gradients
    grad0 = wp.cross(p1 - p2, p3 - p2) / 6.0
    grad1 = wp.cross(p2 - p0, p3 - p0) / 6.0
    grad2 = wp.cross(p0 - p1, p3 - p1) / 6.0
    grad3 = wp.cross(p1 - p0, p2 - p0) / 6.0

    sum_grad = wp.length_sq(grad0) + wp.length_sq(grad1) + wp.length_sq(grad2) + wp.length_sq(grad3)
    if sum_grad < 1e-8:
        return

    # Lagrange multiplier (projective dynamics style)
    s = stiffness * C / sum_grad

    # Compute deltas
    d0 = -grad0 * s
    d1 = -grad1 * s
    d2 = -grad2 * s
    d3 = -grad3 * s

    # Atomically accumulate deltas and counts
    wp.atomic_add(delta_accumulator, ids[0], d0)
    wp.atomic_add(delta_accumulator, ids[1], d1)
    wp.atomic_add(delta_accumulator, ids[2], d2)
    wp.atomic_add(delta_accumulator, ids[3], d3)

    #wp.atomic_add(count_accumulator, ids[0], 1)
    #wp.atomic_add(count_accumulator, ids[1], 1)
    #wp.atomic_add(count_accumulator, ids[2], 1)
    #wp.atomic_add(count_accumulator, ids[3], 1)