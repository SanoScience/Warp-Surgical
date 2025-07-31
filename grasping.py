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
def move_and_lock_grasped_particles(
    particle_q: wp.array(dtype=wp.vec3f),           # [num_particles]
    particle_qd: wp.array(dtype=wp.vec3f),          # [num_particles]
    particle_inv_mass: wp.array(dtype=wp.float32),  # [num_particles]
    grasped_ids: wp.array(dtype=wp.int32),          # [num_grasped]
    grasped_count: int,
    dev_pos: wp.array(dtype=wp.vec3f),              # [1]
    dev_pos_prev: wp.array(dtype=wp.vec3f),         # [1]
):
    gid = wp.tid()
    if gid >= grasped_count:
        return

    pid = grasped_ids[gid]
    delta = (dev_pos[0] - dev_pos_prev[0]) * 0.01
    particle_q[pid] += delta
    # TODO: Write velocity
    particle_inv_mass[pid] = 0.0

@wp.kernel
def unlock_grasped_particles(
    particle_inv_mass: wp.array(dtype=wp.float32),  # [num_particles]
    grasped_ids: wp.array(dtype=wp.int32),          # [num_grasped]
    grasped_count: int,
):
    gid = wp.tid()
    if gid >= grasped_count:
        return

    pid = grasped_ids[gid]
    particle_inv_mass[pid] = 1.0

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
    sim.grasping_active = True

def grasp_end(sim):
    # Unlock grasped particles
    grasped_count = min(int(sim.grasped_particles_counter.numpy()[0]), sim.grasp_capacity)

    wp.launch(
        unlock_grasped_particles,
        dim=grasped_count,
        inputs=[
            sim.model.particle_inv_mass,
            sim.grasped_particles_buffer,
            grasped_count
        ],
        device=wp.get_device()
    )

    wp.copy(sim.grasped_particles_counter, wp.zeros(1, dtype=wp.int32, device=wp.get_device()))
    sim.grasping_active = False

def grasp_process(sim):
    grasped_count = min(int(sim.grasped_particles_counter.numpy()[0]), sim.grasp_capacity)

    wp.launch(
        move_and_lock_grasped_particles,
        dim=grasped_count,
        inputs=[
            sim.state_0.particle_q,
            sim.state_0.particle_qd,
            sim.model.particle_inv_mass,
            sim.grasped_particles_buffer,
            grasped_count,
            sim.integrator.dev_pos_buffer,
            sim.integrator.dev_pos_prev_buffer
        ],
        device=wp.get_device()
    )