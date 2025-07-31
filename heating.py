from mesh_loader import Tetrahedron
import warp as wp

@wp.kernel
def heat_conduction_kernel(
    vertex_neighbors: wp.array(dtype=wp.int32, ndim = 2),   # [num_vertices, max_neighbors]
    vertex_neighbor_counts: wp.array(dtype=wp.int32), # [num_vertices]
    vertex_colors: wp.array(dtype=wp.vec3f),          # [num_vertices] - heat stored in red channel
    heat_diffusion_rate: float,                       # diffusion coefficient (0.0 - 1.0)
    max_neighbors: int,
    vertex_count: int,
    dt: float                                         # time step
):
    vid = wp.tid()
    if vid >= vertex_count:
        return
    
    current_color = vertex_colors[vid]
    current_heat = current_color[0]  # Heat stored in red channel

    if current_heat <= 0.0:
        return
    
    neighbor_count = vertex_neighbor_counts[vid]
    if neighbor_count == 0:
        return
    
    heat_per_neighbor = current_heat * heat_diffusion_rate * dt / float(neighbor_count)
    
    # Conduct heat to neighbors
    for i in range(neighbor_count):
        if i >= max_neighbors:
            break
            
        neighbor_id = vertex_neighbors[vid, i]
        if neighbor_id < 0 or neighbor_id >= vertex_count:
            continue
            
        # Add heat to neighbor (non-atomic approximation)
        neighbor_color = vertex_colors[neighbor_id]
        new_heat = wp.min(neighbor_color[0] + heat_per_neighbor, 1.0)  # Clamp to max 1.0
        vertex_colors[neighbor_id] = wp.vec3(new_heat, neighbor_color[1], neighbor_color[2])

@wp.kernel
def heat_burn_and_cool_kernel(
    vertex_colors: wp.array(dtype=wp.vec3f),      # [num_vertices]
    burn_threshold: float,                        # threshold for burning
    burn_rate: float,                             # rate to increase burn (green channel) per step
    passive_cool_rate: float,                     # rate to decrease heat (red channel) per step
    vertex_count: int
):
    vid = wp.tid()
    if vid >= vertex_count:
        return

    color = vertex_colors[vid]
    heat = color[0]
    burn = color[1]

    # If heat is above threshold, increase burn
    if heat > burn_threshold:
        burn = wp.min(burn + burn_rate, 1.0)

    # Passive heat loss
    heat = wp.max(heat - passive_cool_rate, 0.0)
    vertex_colors[vid] = wp.vec3(heat, burn, color[2])

@wp.kernel
def deactivate_tets_by_burn_kernel(
    tet_active: wp.array(dtype=wp.int32),         # [num_tets]
    tets: wp.array(dtype=Tetrahedron),            # [num_tets]
    vertex_colors: wp.array(dtype=wp.vec3f),      # [num_vertices]
    burn_threshold: float,                        # threshold for deactivation (e.g., 0.95)
    num_tets: int
):
    tid = wp.tid()
    if tid >= num_tets:
        return

    tet = tets[tid]

    # Calculate burn average over vertices
    burn_sum = 0.0
    for i in range(4):
        v_id = tet.ids[i]
        burn_sum += vertex_colors[v_id][1]

    avg_burn = burn_sum * 0.25
    if avg_burn > burn_threshold:
        tet_active[tid] = 0

@wp.kernel
def paint_vertices_near_haptic(
    vertex_positions: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transformf),
    vertex_colors: wp.array(dtype=wp.vec3),
    paint_radius: wp.float32,
    paint_color: wp.array(dtype=wp.vec3),
    paint_strength: wp.array(dtype=wp.float32),
    falloff_power: wp.float32
):
    """Paint vertex colors based on distance to haptic proxy position."""
    tid = wp.tid()
    if tid >= len(vertex_positions):
        return

    haptic_transform = body_q[0]
    haptic_pos = wp.transform_get_translation(haptic_transform)
    vertex_pos = vertex_positions[tid]
    
    distance = wp.length(vertex_pos - haptic_pos)
    
    if distance <= paint_radius:
        # Calculate falloff factor (1.0 at center, 0.0 at radius edge)
        falloff = 1.0 - wp.pow(distance / paint_radius, falloff_power)
        
        current_color = vertex_colors[tid]

        color = paint_color[0]
        strength = paint_strength[0]
        
        new_color = current_color + color * falloff * strength
        
        # Clamp to [0, 1] range
        new_color = wp.vec3(
            wp.clamp(new_color[0], 0.0, 1.0),
            wp.clamp(new_color[1], 0.0, 1.0),
            wp.clamp(new_color[2], 0.0, 1.0)
        )
        
        vertex_colors[tid] = new_color


def heating_start(sim):
    sim.heating_active = True
    set_paint_strength(sim, 1.0 * sim.frame_dt)

def heating_end(sim):
    sim.heating_active = False
    set_paint_strength(sim, 0.0)

def heating_conduction_process(sim):
    dt = sim.frame_dt

    # Diffuse heat to surrounding vertices
    # NOTE: Instead of vertex-vertex connection, consider averaging over each tet? Would this avoid race conditions?
    wp.launch(
        heat_conduction_kernel,
        dim=sim.model.particle_count,
        inputs=[
            sim.vertex_to_vneighbours,
            sim.vertex_vneighbor_counts,
            sim.vertex_colors,
            4.0 * 60 * dt,  # heat_diffusion_rate
            sim.vneighbours_max,
            sim.model.particle_count,
            sim.frame_dt
        ],
        device=wp.get_device()
    )
    
    # High heat values start accumulating in the green channel as burn
    wp.launch(
        heat_burn_and_cool_kernel,
        dim=sim.model.particle_count,
        inputs=[
            sim.vertex_colors,
            0.95,   # burn_threshold
            0.01 * dt * 60,   # burn_rate
            0.02 * dt * 60,  # passive_cool_rate
            sim.model.particle_count
        ],
        device=wp.get_device()
    )

    # Deactivate tets with high burn
    wp.launch(
        deactivate_tets_by_burn_kernel,
        dim=sim.model.tetrahedra_wp.shape[0],
        inputs=[
            sim.model.tet_active,
            sim.model.tetrahedra_wp,
            sim.vertex_colors,
            0.95,  # burn_threshold
            sim.model.tetrahedra_wp.shape[0]
        ],
        device=wp.get_device()
    )


def heating_active_process(sim):
    paint_vertices_near_haptic_proxy(sim, paint_radius=sim.radius_heating, falloff_power=2.0)
    set_paint_strength(sim, 0.05)

def set_paint_strength(sim, strength):
    sim.paint_strength_buffer.assign([strength])

def paint_vertices_near_haptic_proxy(sim, paint_radius=0.25, falloff_power=2.0):
    """Paint vertex colors near the haptic proxy position."""
    # We always paint the red channel only (heat)
    wp.launch(
        paint_vertices_near_haptic,
        dim=sim.model.particle_count,
        inputs=[
            sim.state_0.particle_q,  # vertex positions
            sim.state_0.body_q,      # haptic position
            sim.vertex_colors,       # vertex colors to modify
            paint_radius,             # paint radius
            sim.paint_color_buffer,  # paint color (from array)
            sim.paint_strength_buffer,  # paint strength (from array)
            falloff_power             # falloff power for smooth edges
        ],
        device=wp.get_device()
    )