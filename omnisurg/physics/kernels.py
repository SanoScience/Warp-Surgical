import warp as wp

from omnisurg.mesh.types import Tetrahedron


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
    c = l - rest
    grad = n
    dlambda = -1.0 * (c / w) * ke

    dxi = wi * dlambda * grad
    dxj = wj * dlambda * -grad

    wp.atomic_add(delta, i, dxi)
    wp.atomic_add(delta, j, dxj)
    wp.atomic_add(delta_counter, i, 1)
    wp.atomic_add(delta_counter, j, 1)


@wp.kernel
def apply_deltas_and_zero_accumulators(
    delta: wp.array(dtype=wp.vec3),
    delta_counter: wp.array(dtype=wp.int32),
    target: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if delta_counter[tid] > 0:
        target[tid] += delta[tid] / wp.float32(delta_counter[tid])

    delta[tid] = wp.vec3(0.0, 0.0, 0.0)
    delta_counter[tid] = 0


@wp.kernel
def apply_fused_3_accumulators(
    da: wp.array(dtype=wp.vec3),
    ca: wp.array(dtype=wp.int32),
    db: wp.array(dtype=wp.vec3),
    cb: wp.array(dtype=wp.int32),
    dc: wp.array(dtype=wp.vec3),
    cc: wp.array(dtype=wp.int32),
    target: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    result = wp.vec3(0.0, 0.0, 0.0)
    if ca[tid] > 0:
        result = result + da[tid] / wp.float32(ca[tid])
    if cb[tid] > 0:
        result = result + db[tid] / wp.float32(cb[tid])
    if cc[tid] > 0:
        result = result + dc[tid] / wp.float32(cc[tid])

    target[tid] = target[tid] + result

    da[tid] = wp.vec3(0.0, 0.0, 0.0)
    ca[tid] = 0
    db[tid] = wp.vec3(0.0, 0.0, 0.0)
    cb[tid] = 0
    dc[tid] = wp.vec3(0.0, 0.0, 0.0)
    cc[tid] = 0


@wp.kernel
def solve_volume_constraints(
    positions: wp.array(dtype=wp.vec3f),
    invmass: wp.array(dtype=float),
    tetrahedra: wp.array(dtype=Tetrahedron),
    tetrahedra_active: wp.array(dtype=wp.int32),
    stiffness: wp.float32,
    delta_accumulator: wp.array(dtype=wp.vec3f),
    delta_counter: wp.array(dtype=wp.int32),
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

    v = wp.dot(wp.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0
    c = v - tet.rest_volume

    grad0 = wp.cross(p1 - p2, p3 - p2) / 6.0
    grad1 = wp.cross(p2 - p0, p3 - p0) / 6.0
    grad2 = wp.cross(p0 - p1, p3 - p1) / 6.0
    grad3 = wp.cross(p1 - p0, p2 - p0) / 6.0

    sum_grad = (
        w0 * wp.length_sq(grad0)
        + w1 * wp.length_sq(grad1)
        + w2 * wp.length_sq(grad2)
        + w3 * wp.length_sq(grad3)
    )
    if sum_grad < 1e-8:
        return

    scale = stiffness * c / sum_grad

    d0 = -grad0 * scale * w0
    d1 = -grad1 * scale * w1
    d2 = -grad2 * scale * w2
    d3 = -grad3 * scale * w3

    wp.atomic_add(delta_accumulator, ids[0], d0)
    wp.atomic_add(delta_accumulator, ids[1], d1)
    wp.atomic_add(delta_accumulator, ids[2], d2)
    wp.atomic_add(delta_accumulator, ids[3], d3)

    wp.atomic_add(delta_counter, ids[0], 1)
    wp.atomic_add(delta_counter, ids[1], 1)
    wp.atomic_add(delta_counter, ids[2], 1)
    wp.atomic_add(delta_counter, ids[3], 1)


@wp.kernel
def bounds_collision(
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    inv_masses: wp.array(dtype=wp.float32),
    bounds_min: wp.vec3f,
    bounds_max: wp.vec3f,
    restitution: wp.float32,
    friction: wp.float32,
    dt: wp.float32,
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
            else:
                pos[axis] = bounds_min[axis] + penetration
        elif pos[axis] > bounds_max[axis]:
            penetration = pos[axis] - bounds_max[axis]
            if inv_mass > 0.0:
                vel[axis] = -vel[axis] * restitution
                pos[axis] = bounds_max[axis] - penetration
            else:
                pos[axis] = bounds_max[axis] - penetration

    positions[tid] = pos
    velocities[tid] = vel
