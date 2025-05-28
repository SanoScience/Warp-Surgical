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
                                         delta_accumulator: wp.array(dtype=wp.vec3f),
                                         count_accumulator: wp.array(dtype=wp.int32)):
    i = wp.tid()
    conn = connectors[i]

    tri_pos = positions[conn.tri_ids[0]] * conn.tri_bar[0] + \
              positions[conn.tri_ids[1]] * conn.tri_bar[1] + \
              positions[conn.tri_ids[2]] * conn.tri_bar[2]
    
    dir = positions[conn.particle_id] - tri_pos
    length = wp.length(dir)
    if length < 0.00001:
        return
    
    kS = 0.8
    invMassP = 0.75
    invMassT = 1.0 - invMassP

    C = length - conn.rest_dist * 0.1
    s = invMassP + invMassT * conn.tri_bar[0] * conn.tri_bar[0] + invMassT * conn.tri_bar[1] * conn.tri_bar[1] + invMassT * conn.tri_bar[2] * conn.tri_bar[2]
    dP = (C / s) * (dir / length) * kS

    # Accumulate deltas and counts separately
    wp.atomic_add(delta_accumulator, conn.particle_id, -dP * invMassP)
    wp.atomic_add(count_accumulator, conn.particle_id, 1)
    
    wp.atomic_add(delta_accumulator, conn.tri_ids[0], dP * conn.tri_bar[0] * invMassT)
    wp.atomic_add(count_accumulator, conn.tri_ids[0], 1)
    
    wp.atomic_add(delta_accumulator, conn.tri_ids[1], dP * conn.tri_bar[1] * invMassT)
    wp.atomic_add(count_accumulator, conn.tri_ids[1], 1)
    
    wp.atomic_add(delta_accumulator, conn.tri_ids[2], dP * conn.tri_bar[2] * invMassT)
    wp.atomic_add(count_accumulator, conn.tri_ids[2], 1)

@wp.kernel
def set_body_position(body_q: wp.array(dtype=wp.transformf), 
                      body_qd: wp.array(dtype=wp.spatial_vectorf),
                      body_id: int, posParameter: wp.array(dtype=wp.vec3f)):
    t = body_q[body_id]
    pos = posParameter[0]

    # Set translation part while preserving rotation
    body_q[body_id] = wp.transform(pos * 0.01, wp.quat(t[3], t[4], t[5], t[6]))
