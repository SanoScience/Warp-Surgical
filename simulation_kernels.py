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

from mesh_loader import Tetrahedron

@wp.kernel
def solve_distance_constraints(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    invmass: wp.array(dtype=float),
    spring_indices: wp.array(dtype=int),
    spring_rest_lengths: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    dt: float,
    lambdas: wp.array(dtype=float),
    delta: wp.array(dtype=wp.vec3),
    delta_counter: wp.array(dtype=wp.int32),
):
    tid = wp.tid()

    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]

    ke = spring_stiffness[tid]
    #ke = 1.0
    kd = spring_damping[tid]
    rest = spring_rest_lengths[tid]

    xi = x[i]
    xj = x[j]

    vi = v[i]
    vj = v[j]

    wi = invmass[i]
    wj = invmass[j]
    
    w = wi + wj
    if w <= 0.0 or ke <= 0.0:
        return
    
    xij = xi - xj
    vij = vi - vj

    l = wp.length(xij)
    if l == 0.0:
        return

    n = xij / l

    C = l - rest
    grad = n

    dlambda = -1.0 * (C / w) * ke

    dxi = wi * dlambda *  grad
    dxj = wj * dlambda * -grad

    wp.atomic_add(delta, i, dxi)
    wp.atomic_add(delta, j, dxj)

    wp.atomic_add(delta_counter, i, 1)
    wp.atomic_add(delta_counter, j, 1)

@wp.kernel
def apply_deltas(
    delta: wp.array(dtype=wp.vec3),
    delta_counter: wp.array(dtype=wp.int32),
    target: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if delta_counter[tid] > 0:
        target[tid] += delta[tid] / wp.float32(delta_counter[tid])

@wp.kernel
def apply_deltas_and_zero_accumulators(
    delta: wp.array(dtype=wp.vec3),
    delta_counter: wp.array(dtype=wp.int32),
    target: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if delta_counter[tid] > 0:
        target[tid] += delta[tid] / wp.float32(delta_counter[tid])

    # Zero out the accumulators
    delta[tid] = wp.vec3(0.0, 0.0, 0.0)
    delta_counter[tid] = 0

@wp.kernel
def solve_volume_constraints(
    positions: wp.array(dtype=wp.vec3f),
    invmass: wp.array(dtype=float),
    tetrahedra: wp.array(dtype=Tetrahedron),
    tetrahedra_active: wp.array(dtype=wp.int32),
    stiffness: wp.float32,
    delta_accumulator: wp.array(dtype=wp.vec3f),
    delta_counter: wp.array(dtype=wp.int32)
):
    tid = wp.tid()
    if tetrahedra_active[tid] == 0:
        return

    tet = tetrahedra[tid]
    ids = tet.ids

    p0 = positions[ids[0]]
    p1 = positions[ids[1]]
    p2 = positions[ids[2]]
    p3 = positions[ids[3]]

    w0 = invmass[ids[0]]
    w1 = invmass[ids[1]]
    w2 = invmass[ids[2]]
    w3 = invmass[ids[3]]
    w = w0 + w1 + w2 + w3

    # Compute current volume
    v = wp.dot(wp.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0

    # Constraint: C = v - rest_volume = 0
    C = v - tet.rest_volume

    # Compute gradients
    grad0 = wp.cross(p1 - p2, p3 - p2) / 6.0
    grad1 = wp.cross(p2 - p0, p3 - p0) / 6.0
    grad2 = wp.cross(p0 - p1, p3 - p1) / 6.0
    grad3 = wp.cross(p1 - p0, p2 - p0) / 6.0


    # Mass-weighted denominator
    sum_grad = w0 * wp.length_sq(grad0) + w1 * wp.length_sq(grad1) + w2 * wp.length_sq(grad2) + w3 * wp.length_sq(grad3)
   
    #sum_grad = wp.length_sq(grad0) + wp.length_sq(grad1) + wp.length_sq(grad2) + wp.length_sq(grad3)
    if sum_grad < 1e-8:
        return

    # Lagrange multiplier (projective dynamics style)
    s = stiffness * C / sum_grad

    # # Compute deltas
    # d0 = -grad0 * s
    # d1 = -grad1 * s
    # d2 = -grad2 * s
    # d3 = -grad3 * s

    # Compute mass-weighted deltas
    d0 = -grad0 * s * w0
    d1 = -grad1 * s * w1
    d2 = -grad2 * s * w2
    d3 = -grad3 * s * w3

    # Atomically accumulate deltas and counts
    wp.atomic_add(delta_accumulator, ids[0], d0)
    wp.atomic_add(delta_accumulator, ids[1], d1)
    wp.atomic_add(delta_accumulator, ids[2], d2)
    wp.atomic_add(delta_accumulator, ids[3], d3)

    wp.atomic_add(delta_counter, ids[0], 1)
    wp.atomic_add(delta_counter, ids[1], 1)
    wp.atomic_add(delta_counter, ids[2], 1)
    wp.atomic_add(delta_counter, ids[3], 1)

@wp.kernel
def floor_collision(
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    inv_masses: wp.array(dtype=wp.float32),
    floor_height: wp.float32,
    restitution: wp.float32,
    friction: wp.float32,         # <-- Add friction parameter
    dt: wp.float32
    ):
    tid = wp.tid()
    if tid >= len(positions):
        return
    pos = positions[tid]
    vel = velocities[tid]
    inv_mass = inv_masses[tid]
    if pos[1] < floor_height:
        # Collision detected
        penetration_depth = floor_height - pos[1]
        if inv_mass > 0.0: 
            # Apply restitution
            vel[1] = -vel[1] * restitution
            pos[1] = floor_height + penetration_depth

            # --- Friction ---
            # Compute tangential velocity (x and z)
            if friction != 0.0:
                tangential = wp.vec2f(vel[0], vel[2])
                tangential_len = wp.length(tangential)
                if tangential_len > 1e-6:
                    friction_impulse = min(friction * abs(vel[1]), tangential_len)
                    scale = max(0.0, tangential_len - friction_impulse) / tangential_len
                    vel[0] *= scale
                    vel[2] *= scale
        else:
            # Static body, just move it up
            pos[1] = floor_height + penetration_depth

        # Update positions and velocities
        positions[tid] = pos
        velocities[tid] = vel
 
@wp.kernel
def bounds_collision(
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    inv_masses: wp.array(dtype=wp.float32),
    bounds_min: wp.vec3f,
    bounds_max: wp.vec3f,
    restitution: wp.float32,
    friction: wp.float32,
    dt: wp.float32
):
    tid = wp.tid()
    if tid >= len(positions):
        return
    pos = positions[tid]
    vel = velocities[tid]
    inv_mass = inv_masses[tid]

    for axis in range(3):
        if pos[axis] < bounds_min[axis]:
            penetration = bounds_min[axis] - pos[axis]
            if inv_mass > 0.0:
                vel[axis] = -vel[axis] * restitution
                pos[axis] = bounds_min[axis] + penetration

                # Friction on tangential axes
                # if friction != 0.0:
                #     tangential_axes = [i for i in range(3) if i != axis]
                #     tangential = wp.vec2f(vel[tangential_axes[0]], vel[tangential_axes[1]])
                #     tangential_len = wp.length(tangential)
                #     if tangential_len > 1e-6:
                #         friction_impulse = min(friction * abs(vel[axis]), tangential_len)
                #         scale = max(0.0, tangential_len - friction_impulse) / tangential_len
                #         vel[tangential_axes[0]] *= scale
                #         vel[tangential_axes[1]] *= scale
            else:
                pos[axis] = bounds_min[axis] + penetration

        elif pos[axis] > bounds_max[axis]:
            penetration = pos[axis] - bounds_max[axis]
            if inv_mass > 0.0:
                vel[axis] = -vel[axis] * restitution
                pos[axis] = bounds_max[axis] - penetration

                # Friction on tangential axes
                # if friction != 0.0:
                #     tangential_axes = [i for i in range(3) if i != axis]
                #     tangential = wp.vec2f(vel[tangential_axes[0]], vel[tangential_axes[1]])
                #     tangential_len = wp.length(tangential)
                #     if tangential_len > 1e-6:
                #         friction_impulse = min(friction * abs(vel[axis]), tangential_len)
                #         scale = max(0.0, tangential_len - friction_impulse) / tangential_len
                #         vel[tangential_axes[0]] *= scale
                #         vel[tangential_axes[1]] *= scale
            else:
                pos[axis] = bounds_max[axis] - penetration

    positions[tid] = pos
    velocities[tid] = vel

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