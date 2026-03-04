import warp as wp

@wp.kernel
def find_vertices_near_haptic(
    particle_q: wp.array(dtype=wp.vec3f),      # [num_particles]
    haptic_pos: wp.array(dtype=wp.vec3f),      # [1]
    radius: float,
    out_vertex_ids: wp.array(dtype=wp.int32),  # [max_capacity]
    out_count: wp.array(dtype=wp.int32),       # [1]
    num_particles: int,
    max_capacity: int
):
    vid = wp.tid()
    if vid >= num_particles:
        return

    hpos = haptic_pos[0] * 0.01
    dist = wp.length(particle_q[vid] - hpos)
    if dist < radius:
        idx = wp.atomic_add(out_count, 0, 1)
        if idx < max_capacity:
            out_vertex_ids[idx] = vid

@wp.kernel
def store_grasp_offsets(
    particle_q: wp.array(dtype=wp.vec3f),           # [num_particles]
    grasped_ids: wp.array(dtype=wp.int32),          # [num_grasped]
    grasped_count: int,
    dev_pos: wp.array(dtype=wp.vec3f),              # [1]
    grasp_offsets: wp.array(dtype=wp.vec3f),        # [num_grasped]
):
    gid = wp.tid()
    if gid >= grasped_count:
        return

    pid = grasped_ids[gid]
    hpos = dev_pos[0] * 0.01
    grasp_offsets[gid] = particle_q[pid] - hpos

@wp.kernel
def solve_grasp_distance_constraints(
    particle_q: wp.array(dtype=wp.vec3f),           # [num_particles]
    particle_inv_mass: wp.array(dtype=wp.float32),  # [num_particles]
    grasped_ids: wp.array(dtype=wp.int32),          # [num_grasped]
    grasped_count_array: wp.array(dtype=wp.int32),  # [1] - holds actual count
    dev_pos: wp.array(dtype=wp.vec3f),              # [1]
    grasp_offsets: wp.array(dtype=wp.vec3f),        # [num_grasped]
    stiffness: float,
    delta_accumulator: wp.array(dtype=wp.vec3f),
    delta_counter: wp.array(dtype=wp.int32),
):
    gid = wp.tid()
    grasped_count = grasped_count_array[0]
    if gid >= grasped_count:
        return

    pid = grasped_ids[gid]

    wi = particle_inv_mass[pid]
    if wi <= 0.0:
        return

    hpos = dev_pos[0] * 0.01
    target_pos = hpos + grasp_offsets[gid]

    diff = particle_q[pid] - target_pos
    dist = wp.length(diff)

    if dist < 1e-7:
        return

    n = diff / dist
    C = dist

    dlambda = -(C / wi) * stiffness
    delta = wi * dlambda * n

    wp.atomic_add(delta_accumulator, pid, delta)
    wp.atomic_add(delta_counter, pid, 1)

def grasp_start(sim):
    assert(not(sim.grasping_active))

    # Find vertices to grasp
    wp.launch(
        find_vertices_near_haptic,
        dim=sim.model.particle_count,
        inputs=[
            sim.state_0.particle_q,
            sim.integrator.dev_pos_buffer,
            sim.radius_grasping,
            sim.grasped_particles_buffer,
            sim.grasped_particles_counter,
            sim.model.particle_count,
            sim.grasp_capacity
        ],
        device=wp.get_device()
    )

    grasped_count = min(int(sim.grasped_particles_counter.numpy()[0]), sim.grasp_capacity)

    if grasped_count > 0:
        # Store initial offsets from haptic device to grasped particles
        wp.launch(
            store_grasp_offsets,
            dim=grasped_count,
            inputs=[
                sim.state_0.particle_q,
                sim.grasped_particles_buffer,
                grasped_count,
                sim.integrator.dev_pos_buffer,
                sim.grasp_offsets_buffer
            ],
            device=wp.get_device()
        )

    sim.grasping_active = True
    sim.integrator.grasping_active = True

def grasp_end(sim):
    wp.copy(sim.grasped_particles_counter, wp.zeros(1, dtype=wp.int32, device=wp.get_device()))
    sim.grasping_active = False
    sim.integrator.grasping_active = False

def grasp_process(sim):
    pass