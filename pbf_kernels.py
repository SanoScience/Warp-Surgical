"""Kernels shared across PBF examples."""

import warp as wp

# Core PBF kernels

@wp.kernel
def pbf_predict_positions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    predicted_positions: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: float,
    max_velocity: float
):
    """Predict positions with velocity clamping."""
    tid = wp.tid()
    
    # Apply external forces (gravity)
    vel = velocities[tid] + gravity * dt
    
    # Clamp velocity to prevent explosion
    vel_magnitude = wp.length(vel)
    if vel_magnitude > max_velocity:
        vel = vel * (max_velocity / vel_magnitude)
    
    velocities[tid] = vel
    predicted_positions[tid] = positions[tid] + vel * dt

@wp.kernel
def pbf_compute_density(
    predicted_positions: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=wp.float32),
    hash_grid: wp.uint64,
    smoothing_radius: float
):
    """Compute particle density using Poly6 kernel."""
    tid = wp.tid()
    pos_i = predicted_positions[tid]
    density = float(0.0)
    
    # Query neighbors using spatial hash
    for neighbor_id in wp.hash_grid_query(hash_grid, pos_i, smoothing_radius):
        pos_j = predicted_positions[neighbor_id]
        r = wp.length(pos_i - pos_j)
        
        if r < smoothing_radius:
            # Poly6 kernel (properly normalized)
            q = r / smoothing_radius
            if q <= 1.0:
                w = 315.0 / (64.0 * wp.pi * wp.pow(smoothing_radius, 9.0))
                w *= wp.pow(smoothing_radius * smoothing_radius - r * r, 3.0)
                density = density + w
    
    # Clamp density to reasonable bounds
    densities[tid] = wp.clamp(density, 100.0, 5000.0)

@wp.kernel
def pbf_compute_lambda(
    predicted_positions: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=wp.float32),
    lambdas: wp.array(dtype=wp.float32),
    hash_grid: wp.uint64,
    smoothing_radius: float,
    rest_density: float,
    constraint_epsilon: float
):
    """Compute constraint Lagrange multipliers."""
    tid = wp.tid()
    pos_i = predicted_positions[tid]
    density_i = densities[tid]
    
    # Constraint value
    C_i = (density_i / rest_density) - 1.0
    
    # Skip if constraint is already satisfied
    if wp.abs(C_i) < 0.01:
        lambdas[tid] = 0.0
        return
    
    # Compute gradient sum
    grad_sum = float(0.0)
    grad_i = wp.vec3(0.0, 0.0, 0.0)
    
    for neighbor_id in wp.hash_grid_query(hash_grid, pos_i, smoothing_radius):
        pos_j = predicted_positions[neighbor_id]
        r_vec = pos_i - pos_j
        r = wp.length(r_vec)
        
        if r > 0.001 and r < smoothing_radius:  # Avoid division by zero
            # Spiky kernel gradient
            grad_magnitude = -45.0 / (wp.pi * wp.pow(smoothing_radius, 6.0))
            grad_magnitude *= wp.pow(smoothing_radius - r, 2.0) / r
            
            grad_j = grad_magnitude * r_vec / rest_density
            
            if neighbor_id == tid:
                grad_i = grad_i + grad_j
            else:
                grad_sum = grad_sum + wp.length_sq(grad_j)
    
    grad_sum = grad_sum + wp.length_sq(grad_i)
    
    # Compute lambda with regularization
    if grad_sum > constraint_epsilon:
        lambda_val = -C_i / (grad_sum + constraint_epsilon)
        # Clamp lambda to reasonable range
        lambdas[tid] = wp.clamp(lambda_val, -0.5, 0.5)
    else:
        lambdas[tid] = 0.0

@wp.kernel
def pbf_compute_delta_positions(
    predicted_positions: wp.array(dtype=wp.vec3),
    lambdas: wp.array(dtype=wp.float32),
    delta_positions: wp.array(dtype=wp.vec3),
    hash_grid: wp.uint64,
    smoothing_radius: float,
    rest_density: float
):
    """Compute position corrections."""
    tid = wp.tid()
    pos_i = predicted_positions[tid]
    lambda_i = lambdas[tid]
    delta_pos = wp.vec3(0.0, 0.0, 0.0)
    
    for neighbor_id in wp.hash_grid_query(hash_grid, pos_i, smoothing_radius):
        pos_j = predicted_positions[neighbor_id]
        lambda_j = lambdas[neighbor_id]
        r_vec = pos_i - pos_j
        r = wp.length(r_vec)
        
        if r > 0.001 and r < smoothing_radius:  # Avoid division by zero
            # Spiky kernel gradient
            grad_magnitude = -45.0 / (wp.pi * wp.pow(smoothing_radius, 6.0))
            grad_magnitude *= wp.pow(smoothing_radius - r, 2.0) / r
            
            delta_pos = delta_pos + (lambda_i + lambda_j) * grad_magnitude * r_vec / rest_density
    
    # Limit delta magnitude to prevent large jumps
    max_delta = smoothing_radius * 0.3  # Conservative limit
    delta_magnitude = wp.length(delta_pos)
    if delta_magnitude > max_delta:
        delta_pos = delta_pos * (max_delta / delta_magnitude)
    
    delta_positions[tid] = delta_pos

@wp.kernel
def pbf_apply_delta_positions(
    predicted_positions: wp.array(dtype=wp.vec3),
    delta_positions: wp.array(dtype=wp.vec3)
):
    """Apply position corrections."""
    tid = wp.tid()
    predicted_positions[tid] = predicted_positions[tid] + delta_positions[tid]

@wp.kernel
def pbf_apply_boundaries(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    domain_min: wp.vec3,
    domain_max: wp.vec3,
    restitution: float
):
    """Apply boundary conditions with velocity damping."""
    tid = wp.tid()
    pos = positions[tid]
    vel = velocities[tid]
    
    # Boundary collision and response with extra damping
    margin = 0.01  # Small margin to prevent sticking
    
    if pos[0] < domain_min[0] + margin:
        pos[0] = domain_min[0] + margin
        vel[0] = -vel[0] * restitution
    elif pos[0] > domain_max[0] - margin:
        pos[0] = domain_max[0] - margin
        vel[0] = -vel[0] * restitution
        
    if pos[1] < domain_min[1] + margin:
        pos[1] = domain_min[1] + margin
        vel[1] = -vel[1] * restitution
        vel[1] = wp.max(vel[1], 0.0)  # Prevent bouncing into ground
    elif pos[1] > domain_max[1] - margin:
        pos[1] = domain_max[1] - margin
        vel[1] = -vel[1] * restitution
        
    if pos[2] < domain_min[2] + margin:
        pos[2] = domain_min[2] + margin
        vel[2] = -vel[2] * restitution
    elif pos[2] > domain_max[2] - margin:
        pos[2] = domain_max[2] - margin
        vel[2] = -vel[2] * restitution
    
    positions[tid] = pos
    velocities[tid] = vel

@wp.kernel
def pbf_update_velocities_positions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    predicted_positions: wp.array(dtype=wp.vec3),
    damping: float,
    dt: float,
    max_velocity: float
):
    """Update velocities and positions with clamping."""
    tid = wp.tid()
    
    # Update velocity based on position change
    new_vel = (predicted_positions[tid] - positions[tid]) / dt
    new_vel = new_vel * damping
    
    # Clamp velocity
    vel_magnitude = wp.length(new_vel)
    if vel_magnitude > max_velocity:
        new_vel = new_vel * (max_velocity / vel_magnitude)
    
    velocities[tid] = new_vel
    positions[tid] = predicted_positions[tid]

# Debug kernels

@wp.kernel
def pbf_predict_positions_debug(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    predicted_positions: wp.array(dtype=wp.vec3),
    unstable_flags: wp.array(dtype=wp.int32),
    gravity: wp.vec3,
    dt: float,
    max_velocity: float
):
    """Debug version of position prediction with stability checks."""
    tid = wp.tid()
    
    # Skip if marked as unstable
    if unstable_flags[tid] == 1:
        predicted_positions[tid] = positions[tid]
        return
    
    # Apply external forces (gravity)
    vel = velocities[tid] + gravity * dt
    
    # Clamp velocity to prevent explosion
    vel_magnitude = wp.length(vel)
    if vel_magnitude > max_velocity:
        vel = vel * (max_velocity / vel_magnitude)
        unstable_flags[tid] = 1  # Mark as potentially unstable
    
    velocities[tid] = vel
    
    # Predict position
    predicted_pos = positions[tid] + vel * dt
    
    # Additional safety check - if predicted position is too far, clamp it
    displacement = wp.length(predicted_pos - positions[tid])
    if displacement > max_velocity * dt * 2.0:  # Allow some tolerance
        direction = (predicted_pos - positions[tid]) / displacement
        predicted_pos = positions[tid] + direction * (max_velocity * dt * 2.0)
        unstable_flags[tid] = 1
    
    predicted_positions[tid] = predicted_pos

@wp.kernel
def pbf_compute_density_debug(
    predicted_positions: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=wp.float32),
    unstable_flags: wp.array(dtype=wp.int32),
    hash_grid: wp.uint64,
    smoothing_radius: float,
    max_density: float,
    min_density: float
):
    """Debug version of density computation with bounds checking."""
    tid = wp.tid()
    
    # Skip if marked as unstable
    if unstable_flags[tid] == 1:
        densities[tid] = 1000.0  # Set to rest density
        return
    
    pos_i = predicted_positions[tid]
    density = float(0.0)
    neighbor_count = int(0)
    
    # Query neighbors using spatial hash
    for neighbor_id in wp.hash_grid_query(hash_grid, pos_i, smoothing_radius):
        pos_j = predicted_positions[neighbor_id]
        r = wp.length(pos_i - pos_j)
        
        if r < smoothing_radius:
            neighbor_count += 1
            # Poly6 kernel
            q = r / smoothing_radius
            if q <= 1.0:
                h2 = smoothing_radius * smoothing_radius
                w = 315.0 / (64.0 * wp.pi * wp.pow(smoothing_radius, 9.0))
                w *= wp.pow(h2 - r * r, 3.0)
                density = density + w
    
    # Only mark as unstable if density is extremely high (indicating particle overlap)
    if density > max_density * 2.0:  # More lenient threshold
        density = max_density
        unstable_flags[tid] = 1
    elif density < min_density * 0.5:  # More lenient threshold
        density = min_density
    
    # Only mark as unstable if completely isolated (0 neighbors including self)
    if neighbor_count == 0:
        unstable_flags[tid] = 1
    
    densities[tid] = density

@wp.kernel
def pbf_compute_lambda_debug(
    predicted_positions: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=wp.float32),
    lambdas: wp.array(dtype=wp.float32),
    unstable_flags: wp.array(dtype=wp.int32),
    hash_grid: wp.uint64,
    smoothing_radius: float,
    rest_density: float,
    constraint_epsilon: float
):
    """Debug version of lambda computation with stability checks."""
    tid = wp.tid()
    
    # Skip if marked as unstable
    if unstable_flags[tid] == 1:
        lambdas[tid] = 0.0
        return
    
    pos_i = predicted_positions[tid]
    density_i = densities[tid]
    
    # Constraint value
    C_i = (density_i / rest_density) - 1.0
    
    # Skip if constraint is already satisfied
    if wp.abs(C_i) < 0.01:
        lambdas[tid] = 0.0
        return
    
    # Compute gradient sum
    grad_sum = float(0.0)
    grad_i = wp.vec3(0.0, 0.0, 0.0)
    
    for neighbor_id in wp.hash_grid_query(hash_grid, pos_i, smoothing_radius):
        if unstable_flags[neighbor_id] == 1:
            continue  # Skip unstable neighbors
            
        pos_j = predicted_positions[neighbor_id]
        r_vec = pos_i - pos_j
        r = wp.length(r_vec)
        
        if r > 0.001 and r < smoothing_radius:  # Avoid division by zero
            # Spiky kernel gradient
            grad_magnitude = -45.0 / (wp.pi * wp.pow(smoothing_radius, 6.0))
            grad_magnitude *= wp.pow(smoothing_radius - r, 2.0) / r
            
            grad_j = grad_magnitude * r_vec / rest_density
            
            if neighbor_id == tid:
                grad_i = grad_i + grad_j
            else:
                grad_sum = grad_sum + wp.length_sq(grad_j)
    
    grad_sum = grad_sum + wp.length_sq(grad_i)
    
    # Compute lambda with regularization
    denominator = grad_sum + constraint_epsilon
    if denominator > 0.0001:  # Avoid division by very small numbers
        lambda_val = -C_i / denominator
        # Clamp lambda to reasonable range
        max_lambda = 1.0
        if wp.abs(lambda_val) > max_lambda:
            lambda_val = wp.sign(lambda_val) * max_lambda
            unstable_flags[tid] = 1
        lambdas[tid] = lambda_val
    else:
        lambdas[tid] = 0.0
        unstable_flags[tid] = 1

@wp.kernel
def pbf_compute_delta_positions_debug(
    predicted_positions: wp.array(dtype=wp.vec3),
    lambdas: wp.array(dtype=wp.float32),
    delta_positions: wp.array(dtype=wp.vec3),
    unstable_flags: wp.array(dtype=wp.int32),
    hash_grid: wp.uint64,
    smoothing_radius: float,
    rest_density: float,
    max_delta: float
):
    """Debug version of delta position computation with bounds."""
    tid = wp.tid()
    
    # Skip if marked as unstable
    if unstable_flags[tid] == 1:
        delta_positions[tid] = wp.vec3(0.0, 0.0, 0.0)
        return
    
    pos_i = predicted_positions[tid]
    lambda_i = lambdas[tid]
    delta_pos = wp.vec3(0.0, 0.0, 0.0)
    
    for neighbor_id in wp.hash_grid_query(hash_grid, pos_i, smoothing_radius):
        if unstable_flags[neighbor_id] == 1:
            continue  # Skip unstable neighbors
            
        pos_j = predicted_positions[neighbor_id]
        lambda_j = lambdas[neighbor_id]
        r_vec = pos_i - pos_j
        r = wp.length(r_vec)
        
        if r > 0.001 and r < smoothing_radius:  # Avoid division by zero
            # Spiky kernel gradient
            grad_magnitude = -45.0 / (wp.pi * wp.pow(smoothing_radius, 6.0))
            grad_magnitude *= wp.pow(smoothing_radius - r, 2.0) / r
            
            correction = (lambda_i + lambda_j) * grad_magnitude * r_vec / rest_density
            delta_pos = delta_pos + correction
    
    # Clamp delta position to prevent large corrections
    delta_magnitude = wp.length(delta_pos)
    if delta_magnitude > max_delta:
        delta_pos = delta_pos * (max_delta / delta_magnitude)
        unstable_flags[tid] = 1
    
    delta_positions[tid] = delta_pos

@wp.kernel
def pbf_apply_delta_positions_debug(
    predicted_positions: wp.array(dtype=wp.vec3),
    delta_positions: wp.array(dtype=wp.vec3),
    unstable_flags: wp.array(dtype=wp.int32)
):
    """Debug version of position update."""
    tid = wp.tid()
    
    # Skip if marked as unstable
    if unstable_flags[tid] == 1:
        return
    
    predicted_positions[tid] = predicted_positions[tid] + delta_positions[tid]

@wp.kernel
def pbf_update_velocities_positions_debug(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    predicted_positions: wp.array(dtype=wp.vec3),
    unstable_flags: wp.array(dtype=wp.int32),
    damping: float,
    dt: float,
    max_velocity: float
):
    """Debug version of velocity and position update with clamping."""
    tid = wp.tid()
    
    # For unstable particles, gradually move them back to stable positions
    if unstable_flags[tid] == 1:
        # Reduce velocity significantly
        velocities[tid] = velocities[tid] * 0.1
        # Small movement towards predicted position
        positions[tid] = positions[tid] * 0.95 + predicted_positions[tid] * 0.05
        return
    
    # Update velocity based on position change
    new_vel = (predicted_positions[tid] - positions[tid]) / dt
    new_vel = new_vel * damping
    
    # Clamp velocity
    vel_magnitude = wp.length(new_vel)
    if vel_magnitude > max_velocity:
        new_vel = new_vel * (max_velocity / vel_magnitude)
    
    velocities[tid] = new_vel
    positions[tid] = predicted_positions[tid]

@wp.kernel
def reset_stability_flags(
    unstable_flags: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    domain_min: wp.vec3,
    domain_max: wp.vec3
):
    """Reset stability flags for particles that have stabilized."""
    tid = wp.tid()
    
    pos = positions[tid]
    vel = velocities[tid]
    
    # Check if particle is in reasonable bounds
    in_bounds = (pos[0] >= domain_min[0] and pos[0] <= domain_max[0] and
                 pos[1] >= domain_min[1] and pos[1] <= domain_max[1] and
                 pos[2] >= domain_min[2] and pos[2] <= domain_max[2])
    
    # Check if velocity is reasonable
    vel_magnitude = wp.length(vel)
    reasonable_velocity = vel_magnitude < 2.0
    
    # Reset flag if particle seems stable
    if in_bounds and reasonable_velocity:
        unstable_flags[tid] = 0

# Alternative kernels used in the PBF2 example

@wp.kernel
def pbf2_predict_positions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    predicted_positions: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: wp.float32,
    max_velocity: wp.float32
):
    tid = wp.tid()

    # Apply gravity
    vel = velocities[tid]
    vel = vel + gravity * dt

    # Clamp velocity
    vel_mag = wp.length(vel)
    if vel_mag > max_velocity:
        vel = vel * (max_velocity / vel_mag)
        velocities[tid] = vel

    # Predict position
    predicted_positions[tid] = positions[tid] + vel * dt

@wp.kernel
def pbf2_compute_density(
    predicted_positions: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=wp.float32),
    hash_grid: wp.uint64,
    smoothing_radius: wp.float32
):
    tid = wp.tid()

    pos_i = predicted_positions[tid]
    density = float(0.0)

    # Query neighbors using spatial hash
    for j in wp.hash_grid_query(hash_grid, pos_i, smoothing_radius):
        pos_j = predicted_positions[j]
        r = wp.length(pos_i - pos_j)

        if r < smoothing_radius:
            # Poly6 kernel (properly normalized)
            h = smoothing_radius
            q = r / h
            poly6_factor = 315.0 / (64.0 * wp.pi * wp.pow(h, 9.0))
            if q <= 1.0:
                kernel_value = poly6_factor * wp.pow(1.0 - q * q, 3.0)
                density = density + kernel_value

    densities[tid] = density

@wp.kernel
def pbf2_compute_lambda(
    predicted_positions: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=wp.float32),
    lambdas: wp.array(dtype=wp.float32),
    hash_grid: wp.uint64,
    smoothing_radius: wp.float32,
    rest_density: wp.float32,
    constraint_epsilon: wp.float32
):
    tid = wp.tid()

    pos_i = predicted_positions[tid]
    density_i = densities[tid]

    # Density constraint
    constraint = density_i / rest_density - 1.0

    if constraint > 0.0:
        # Compute gradient magnitude squared
        grad_i = wp.vec3(0.0, 0.0, 0.0)
        sum_grad_squared = float(0.0)

        for j in wp.hash_grid_query(hash_grid, pos_i, smoothing_radius):
            pos_j = predicted_positions[j]
            r_vec = pos_i - pos_j
            r = wp.length(r_vec)

            if r > 0.001 and r < smoothing_radius:  # Avoid division by zero
                # Spiky gradient kernel
                h = smoothing_radius
                q = r / h
                spiky_factor = -45.0 / (wp.pi * wp.pow(h, 6.0))
                grad_magnitude = spiky_factor * wp.pow(1.0 - q, 2.0) / r
                grad_j = r_vec * grad_magnitude / rest_density

                if j == tid:
                    grad_i = grad_i + grad_j
                else:
                    grad_i = grad_i - grad_j
                    sum_grad_squared = sum_grad_squared + wp.dot(grad_j, grad_j)

        sum_grad_squared = sum_grad_squared + wp.dot(grad_i, grad_i)

        # Compute lambda
        if sum_grad_squared > 0.0:
            lambda_val = -constraint / (sum_grad_squared + constraint_epsilon)
        else:
            lambda_val = 0.0
    else:
        lambda_val = 0.0

    lambdas[tid] = lambda_val

@wp.kernel
def pbf2_compute_position_deltas(
    predicted_positions: wp.array(dtype=wp.vec3),
    lambdas: wp.array(dtype=wp.float32),
    position_deltas: wp.array(dtype=wp.vec3),
    hash_grid: wp.uint64,
    smoothing_radius: wp.float32,
    rest_density: wp.float32
):
    tid = wp.tid()

    pos_i = predicted_positions[tid]
    lambda_i = lambdas[tid]
    delta_pos = wp.vec3(0.0, 0.0, 0.0)

    for j in wp.hash_grid_query(hash_grid, pos_i, smoothing_radius):
        if j == tid:
            continue

        pos_j = predicted_positions[j]
        lambda_j = lambdas[j]
        r_vec = pos_i - pos_j
        r = wp.length(r_vec)

        if r > 0.001 and r < smoothing_radius:  # Avoid division by zero
            # Spiky gradient kernel
            h = smoothing_radius
            q = r / h
            spiky_factor = -45.0 / (wp.pi * wp.pow(h, 6.0))
            grad_magnitude = spiky_factor * wp.pow(1.0 - q, 2.0) / r
            grad_w = r_vec * grad_magnitude

            # Position correction
            delta_pos = delta_pos + (lambda_i + lambda_j) * grad_w / rest_density

    position_deltas[tid] = delta_pos

@wp.kernel
def pbf2_apply_position_deltas(
    predicted_positions: wp.array(dtype=wp.vec3),
    position_deltas: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()

    # Apply position corrections
    predicted_positions[tid] = predicted_positions[tid] + position_deltas[tid]

@wp.kernel
def pbf2_apply_boundaries(
    predicted_positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    domain_min: wp.vec3,
    domain_max: wp.vec3,
    restitution: wp.float32,
    particle_radius: wp.float32
):
    tid = wp.tid()

    pos = predicted_positions[tid]
    vel = velocities[tid]

    # X boundaries
    if pos[0] - particle_radius < domain_min[0]:
        pos = wp.vec3(domain_min[0] + particle_radius, pos[1], pos[2])
        vel = wp.vec3(-vel[0] * restitution, vel[1], vel[2])
    elif pos[0] + particle_radius > domain_max[0]:
        pos = wp.vec3(domain_max[0] - particle_radius, pos[1], pos[2])
        vel = wp.vec3(-vel[0] * restitution, vel[1], vel[2])

    # Y boundaries (floor and ceiling)
    if pos[1] - particle_radius < domain_min[1]:
        pos = wp.vec3(pos[0], domain_min[1] + particle_radius, pos[2])
        vel = wp.vec3(vel[0], -vel[1] * restitution, vel[2])
    elif pos[1] + particle_radius > domain_max[1]:
        pos = wp.vec3(pos[0], domain_max[1] - particle_radius, pos[2])
        vel = wp.vec3(vel[0], -vel[1] * restitution, vel[2])

    # Z boundaries
    if pos[2] - particle_radius < domain_min[2]:
        pos = wp.vec3(pos[0], pos[1], domain_min[2] + particle_radius)
        vel = wp.vec3(vel[0], vel[1], -vel[2] * restitution)
    elif pos[2] + particle_radius > domain_max[2]:
        pos = wp.vec3(pos[0], pos[1], domain_max[2] - particle_radius)
        vel = wp.vec3(vel[0], vel[1], -vel[2] * restitution)

    predicted_positions[tid] = pos
    velocities[tid] = vel

@wp.kernel
def pbf2_update_velocities_positions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    predicted_positions: wp.array(dtype=wp.vec3),
    damping: wp.float32,
    dt: wp.float32,
    max_velocity: wp.float32
):
    tid = wp.tid()

    # Update velocity based on position change (PBD style)
    old_pos = positions[tid]
    new_pos = predicted_positions[tid]
    vel = (new_pos - old_pos) / dt

    # Apply damping
    vel = vel * damping

    # Clamp velocity
    vel_mag = wp.length(vel)
    if vel_mag > max_velocity:
        vel = vel * (max_velocity / vel_mag)

    velocities[tid] = vel
    positions[tid] = new_pos

# Simple sphere collision kernels

@wp.kernel
def sphere_collision_constraints(
    predicted_positions: wp.array(dtype=wp.vec3),
    position_deltas: wp.array(dtype=wp.vec3),
    constraint_counts: wp.array(dtype=wp.int32),
    hash_grid: wp.uint64,
    collision_radius: wp.float32,
    restitution: wp.float32
):
    tid = wp.tid()

    pos_i = predicted_positions[tid]
    delta_sum = wp.vec3(0.0, 0.0, 0.0)
    constraint_count = int(0)  # Dynamic variable

    # Query neighbors using spatial hash
    for j in wp.hash_grid_query(hash_grid, pos_i, collision_radius):
        if j == tid:
            continue

        pos_j = predicted_positions[j]

        # Distance between particles
        delta = pos_i - pos_j
        dist = wp.length(delta)

        # Check collision (particles overlap)
        if dist > 0.001 and dist < collision_radius:  # Avoid division by zero
            # Collision constraint: maintain minimum distance
            constraint_violation = collision_radius - dist
            if constraint_violation > 0.0:
                # Collision normal (from j to i)
                normal = delta / dist

                # Simple position correction (half the correction for each particle)
                correction = normal * constraint_violation * 0.5 * restitution
                delta_sum = delta_sum + correction
                constraint_count = constraint_count + 1

    position_deltas[tid] = delta_sum
    constraint_counts[tid] = constraint_count

@wp.kernel
def sphere_apply_position_deltas(
    predicted_positions: wp.array(dtype=wp.vec3),
    position_deltas: wp.array(dtype=wp.vec3),
    constraint_counts: wp.array(dtype=wp.int32),
    stiffness: wp.float32
):
    tid = wp.tid()

    # Apply accumulated position corrections
    delta = position_deltas[tid]
    count = constraint_counts[tid]

    if count > 0:
        # Average the corrections and apply with stiffness
        avg_correction = delta * (stiffness / wp.float32(count))
        predicted_positions[tid] = predicted_positions[tid] + avg_correction

# Aliases for compatibility across examples
pbf_predict_positions_stable = pbf_predict_positions
pbf_compute_density_stable = pbf_compute_density
pbf_compute_lambda_stable = pbf_compute_lambda
pbf_compute_delta_positions_stable = pbf_compute_delta_positions
pbf_apply_boundaries_stable = pbf_apply_boundaries
pbf_update_velocities_positions_stable = pbf_update_velocities_positions
pbf_compute_position_deltas = pbf_compute_delta_positions
pbf_apply_position_deltas = pbf_apply_delta_positions
