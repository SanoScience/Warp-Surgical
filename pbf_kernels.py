"""
Position Based Fluids (PBF) Kernels using Warp spatial hashing
Based on Macklin & Müller 2013 "Position Based Fluids"
"""

import warp as wp
import math

# Cubic spline kernel for PBF (same as SPH)
@wp.func
def cubic_kernel(r: float, h: float) -> float:
    """Cubic spline smoothing kernel"""
    q = r / h
    if q >= 2.0:
        return 0.0
    elif q >= 1.0:
        return 15.0 / (64.0 * math.pi * h * h * h) * wp.pow(2.0 - q, 3.0)
    else:
        return 15.0 / (64.0 * math.pi * h * h * h) * (2.0 * wp.pow(2.0 - q, 3.0) - 4.0 * wp.pow(1.0 - q, 3.0))

@wp.func  
def cubic_kernel_gradient(r_vec: wp.vec3f, h: float) -> wp.vec3f:
    """Gradient of cubic spline kernel"""
    r = wp.length(r_vec)
    if r < 1e-6 or r >= 2.0 * h:
        return wp.vec3f(0.0, 0.0, 0.0)
    
    q = r / h
    grad_mag = 0.0
    
    if q >= 1.0:
        grad_mag = -45.0 / (64.0 * math.pi * h * h * h * h) * wp.pow(2.0 - q, 2.0)
    else:
        grad_mag = -45.0 / (64.0 * math.pi * h * h * h * h) * (2.0 * wp.pow(2.0 - q, 2.0) - 4.0 * wp.pow(1.0 - q, 2.0))
    
    return grad_mag * r_vec / r

@wp.kernel
def compute_density(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3f),
    densities: wp.array(dtype=wp.float32),
    num_particles: int,
    smoothing_length: float,
    particle_mass: float
):
    """Compute density at each particle using SPH summation"""
    i = wp.tid()
    if i >= num_particles:
        return
    
    xi = positions[i]
    density = float(0.0)  # Explicitly declare as dynamic variable
    
    # Query neighbors using Warp spatial hash
    neighbors = wp.hash_grid_query(grid, xi, smoothing_length)
    for index in neighbors:
        j = int(index)
        if j >= num_particles:
            continue
            
        xj = positions[j]
        r = wp.length(xi - xj)
        
        # Add kernel contribution
        density = density + particle_mass * cubic_kernel(r, smoothing_length)
    
    densities[i] = density

@wp.kernel
def compute_lambda(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3f),
    densities: wp.array(dtype=wp.float32),
    lambdas: wp.array(dtype=wp.float32),
    num_particles: int,
    smoothing_length: float,
    rest_density: float,
    particle_mass: float,
    epsilon: float = 100.0
):
    """Compute lambda multipliers for density constraints"""
    i = wp.tid()
    if i >= num_particles:
        return
    
    xi = positions[i]
    density_i = densities[i]
    
    # Density constraint value
    C_i = (density_i / rest_density) - 1.0
    
    # Compute gradient magnitudes for constraint gradient
    sum_grad_sq = float(0.0)  # Dynamic variable
    grad_i = wp.vec3f(0.0, 0.0, 0.0)
    
    neighbors = wp.hash_grid_query(grid, xi, smoothing_length)
    for index in neighbors:
        j = int(index)
        if j >= num_particles:
            continue
            
        xj = positions[j]
        r_vec = xi - xj
        
        # Gradient of constraint with respect to particle j
        grad_j = (particle_mass / rest_density) * cubic_kernel_gradient(r_vec, smoothing_length)
        
        if i == j:
            grad_i = grad_j
        
        # Accumulate gradient magnitude squared
        sum_grad_sq = sum_grad_sq + wp.dot(grad_j, grad_j) / (particle_mass * particle_mass)
    
    # Compute lambda (Lagrange multiplier)
    if sum_grad_sq > epsilon:
        lambdas[i] = -C_i / (sum_grad_sq + epsilon)
    else:
        lambdas[i] = 0.0

@wp.kernel  
def compute_position_correction(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3f),
    lambdas: wp.array(dtype=wp.float32),
    position_corrections: wp.array(dtype=wp.vec3f),
    num_particles: int,
    smoothing_length: float,
    rest_density: float,
    particle_mass: float,
    artificial_pressure_strength: float = 0.0001,
    artificial_pressure_radius: float = 0.1
):
    """Compute position corrections from lambda values"""
    i = wp.tid()
    if i >= num_particles:
        return
    
    xi = positions[i]
    lambda_i = lambdas[i]
    correction = wp.vec3f(0.0, 0.0, 0.0)  # Initialize properly
    
    neighbors = wp.hash_grid_query(grid, xi, smoothing_length)
    for index in neighbors:
        j = int(index)
        if j >= num_particles or i == j:
            continue
            
        xj = positions[j]
        lambda_j = lambdas[j]
        r_vec = xi - xj
        r = wp.length(r_vec)
        
        # Gradient of kernel
        grad_W = cubic_kernel_gradient(r_vec, smoothing_length)
        
        # Artificial pressure term to prevent particle clumping (tensile instability)
        s_corr = float(0.0)  # Dynamic variable
        if r < artificial_pressure_radius * smoothing_length:
            # Compute artificial pressure
            W_ij = cubic_kernel(r, smoothing_length)
            W_delta_q = cubic_kernel(artificial_pressure_radius * smoothing_length, smoothing_length)
            if W_delta_q > 1e-6:
                s_corr = -artificial_pressure_strength * wp.pow(W_ij / W_delta_q, 4.0)
        
        # Position correction contribution
        correction = correction + (lambda_i + lambda_j + s_corr) * grad_W / rest_density
    
    position_corrections[i] = correction

@wp.kernel
def apply_position_corrections(
    positions: wp.array(dtype=wp.vec3f),
    position_corrections: wp.array(dtype=wp.vec3f),
    num_particles: int
):
    """Apply computed position corrections"""
    i = wp.tid()
    if i >= num_particles:
        return
    
    positions[i] = positions[i] + position_corrections[i]

@wp.kernel
def compute_xsph_viscosity(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    velocity_corrections: wp.array(dtype=wp.vec3f),
    num_particles: int,
    smoothing_length: float,
    particle_mass: float,
    viscosity_strength: float = 0.01
):
    """Apply XSPH viscosity for velocity smoothing"""
    i = wp.tid()
    if i >= num_particles:
        return
    
    xi = positions[i]
    vi = velocities[i]
    velocity_sum = wp.vec3f(0.0, 0.0, 0.0)  # Initialize properly
    
    neighbors = wp.hash_grid_query(grid, xi, smoothing_length)
    for index in neighbors:
        j = int(index)
        if j >= num_particles or i == j:
            continue
            
        xj = positions[j]
        vj = velocities[j]
        r = wp.length(xi - xj)
        
        # XSPH velocity update
        W_ij = cubic_kernel(r, smoothing_length)
        velocity_sum = velocity_sum + particle_mass * (vj - vi) * W_ij
    
    velocity_corrections[i] = viscosity_strength * velocity_sum

@wp.kernel
def apply_velocity_corrections(
    velocities: wp.array(dtype=wp.vec3f),
    velocity_corrections: wp.array(dtype=wp.vec3f),
    num_particles: int
):
    """Apply computed velocity corrections"""
    i = wp.tid()
    if i >= num_particles:
        return
    
    velocities[i] = velocities[i] + velocity_corrections[i]

@wp.kernel
def integrate_particles(
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    predicted_positions: wp.array(dtype=wp.vec3f),
    forces: wp.array(dtype=wp.vec3f),
    num_particles: int,
    dt: float,
    particle_mass: float = 1.0
):
    """Explicit Euler integration step"""
    i = wp.tid()
    if i >= num_particles:
        return
    
    # Apply forces (gravity, etc.)
    acceleration = forces[i] / particle_mass
    
    # Update velocity
    velocities[i] = velocities[i] + acceleration * dt
    
    # Predict new position  
    predicted_positions[i] = positions[i] + velocities[i] * dt

@wp.kernel
def update_velocities_from_positions(
    positions: wp.array(dtype=wp.vec3f),
    predicted_positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    num_particles: int,
    dt: float
):
    """Update velocities based on position changes"""
    i = wp.tid()
    if i >= num_particles:
        return
    
    # Compute velocity from position change
    velocities[i] = (positions[i] - predicted_positions[i]) / dt

@wp.kernel
def apply_boundary_constraints(
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    num_particles: int,
    bounds_min: wp.vec3f,
    bounds_max: wp.vec3f,
    damping: float = 0.5
):
    """Apply simple box boundary constraints"""
    i = wp.tid()
    if i >= num_particles:
        return
    
    pos = positions[i]
    vel = velocities[i]
    
    # Check each axis
    for axis in range(3):
        if pos[axis] < bounds_min[axis]:
            pos = wp.vec3f(pos[0] if axis != 0 else bounds_min[axis],
                          pos[1] if axis != 1 else bounds_min[axis], 
                          pos[2] if axis != 2 else bounds_min[axis])
            vel = wp.vec3f(vel[0] if axis != 0 else -vel[0] * damping,
                          vel[1] if axis != 1 else -vel[1] * damping,
                          vel[2] if axis != 2 else -vel[2] * damping)
        elif pos[axis] > bounds_max[axis]:
            pos = wp.vec3f(pos[0] if axis != 0 else bounds_max[axis],
                          pos[1] if axis != 1 else bounds_max[axis],
                          pos[2] if axis != 2 else bounds_max[axis])
            vel = wp.vec3f(vel[0] if axis != 0 else -vel[0] * damping,
                          vel[1] if axis != 1 else -vel[1] * damping,
                          vel[2] if axis != 2 else -vel[2] * damping)
    
    positions[i] = pos
    velocities[i] = vel