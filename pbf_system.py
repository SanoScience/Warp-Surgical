"""
Position Based Fluids (PBF) System
Manages fluid simulation using Warp spatial hashing and PBF constraint solving
"""

import warp as wp
import numpy as np
from pbf_kernels import (
    compute_density,
    compute_lambda, 
    compute_position_correction,
    apply_position_corrections,
    compute_xsph_viscosity,
    apply_velocity_corrections,
    integrate_particles,
    update_velocities_from_positions,
    apply_boundary_constraints
)

@wp.struct
class FluidParticle:
    """Fluid particle data structure"""
    active: wp.int32
    lifetime: wp.float32
    spawn_time: wp.float32

class PBFSystem:
    """Position Based Fluids simulation system using Warp spatial hashing"""
    
    def __init__(self, max_particles=4096, smoothing_length=0.012, rest_density=1000.0, particle_mass=0.001, device=None):
        if device is None:
            device = wp.get_device()
        
        self.device = device
        self.max_particles = max_particles
        self.smoothing_length = smoothing_length
        self.rest_density = rest_density
        self.particle_mass = particle_mass
        self.num_active_particles = 0
        
        # PBF simulation parameters
        self.constraint_iterations = 5
        self.viscosity_strength = 0.01
        self.artificial_pressure_strength = 0.0001
        self.artificial_pressure_radius = 0.1
        self.boundary_damping = 0.5
        
        # Particle lifetime parameters  
        self.max_lifetime = 10.0  # Maximum particle lifetime in seconds
        self.spawn_velocity_magnitude = 0.5
        
        # Initialize particle arrays
        self._init_particle_arrays()
        
        # Initialize spatial hash grid
        # Grid size based on smoothing length - each cell should be roughly smoothing_length sized
        self.grid_size = 64  # Reasonable grid resolution
        self.grid = wp.HashGrid(self.grid_size, self.grid_size, self.grid_size, device=device)
        
        # Forces (gravity, etc.)
        gravity = wp.vec3f(0.0, -9.81, 0.0)
        self.forces = wp.full(self.max_particles, gravity, dtype=wp.vec3f, device=device)
        
        # Simulation bounds (should be set based on simulation domain)
        self.bounds_min = wp.vec3f(-2.0, -3.0, -2.0)
        self.bounds_max = wp.vec3f(2.0, 2.0, 2.0)
        
        # Active particle counter
        self.active_count = wp.zeros(1, dtype=wp.int32, device=device)
    
    def _init_particle_arrays(self):
        """Initialize all particle data arrays"""
        # Core particle state
        self.positions = wp.zeros(self.max_particles, dtype=wp.vec3f, device=self.device)
        self.velocities = wp.zeros(self.max_particles, dtype=wp.vec3f, device=self.device)
        self.predicted_positions = wp.zeros(self.max_particles, dtype=wp.vec3f, device=self.device)
        
        # PBF computation arrays
        self.densities = wp.zeros(self.max_particles, dtype=wp.float32, device=self.device)
        self.lambdas = wp.zeros(self.max_particles, dtype=wp.float32, device=self.device)
        self.position_corrections = wp.zeros(self.max_particles, dtype=wp.vec3f, device=self.device)
        self.velocity_corrections = wp.zeros(self.max_particles, dtype=wp.vec3f, device=self.device)
        
        # Particle lifecycle management
        self.particle_info = wp.zeros(self.max_particles, dtype=FluidParticle, device=self.device)
        self.next_particle_id = wp.zeros(1, dtype=wp.int32, device=self.device)
        
        # Active status array for kernels (extracted from particle_info)
        self.particle_active = wp.zeros(self.max_particles, dtype=wp.int32, device=self.device)
        
        # Initialize all particles as inactive
        wp.launch(
            kernel=self._init_particles_kernel,
            dim=self.max_particles,
            inputs=[self.particle_info],
            device=self.device
        )
    
    @wp.kernel
    def _init_particles_kernel(particle_info: wp.array(dtype=FluidParticle)):
        """Initialize all particles as inactive"""
        i = wp.tid()
        particle_info[i].active = 0
        particle_info[i].lifetime = 0.0
        particle_info[i].spawn_time = 0.0
    
    def spawn_particle(self, position, velocity=None, current_time=0.0):
        """Spawn a new fluid particle at the given position"""
        if velocity is None:
            # Random velocity with some upward component
            import random
            velocity = wp.vec3f(
                (random.random() - 0.5) * self.spawn_velocity_magnitude,
                random.random() * self.spawn_velocity_magnitude * 0.5 + 0.2,
                (random.random() - 0.5) * self.spawn_velocity_magnitude
            )
        
        # Launch kernel to spawn particle
        wp.launch(
            kernel=self._spawn_particle_kernel,
            dim=1,
            inputs=[
                self.positions,
                self.velocities,
                self.particle_info,
                self.next_particle_id,
                self.max_particles,
                position,
                velocity,
                current_time,
                self.max_lifetime
            ],
            device=self.device
        )
    
    @wp.kernel  
    def _spawn_particle_kernel(
        positions: wp.array(dtype=wp.vec3f),
        velocities: wp.array(dtype=wp.vec3f),
        particle_info: wp.array(dtype=FluidParticle),
        next_particle_id: wp.array(dtype=wp.int32),
        max_particles: int,
        spawn_position: wp.vec3f,
        spawn_velocity: wp.vec3f,
        current_time: float,
        max_lifetime: float
    ):
        """Kernel to spawn a new particle"""
        if wp.tid() == 0:
            # Get next available particle slot
            idx = wp.atomic_add(next_particle_id, 0, 1) % max_particles
            
            # Initialize particle
            positions[idx] = spawn_position
            velocities[idx] = spawn_velocity
            particle_info[idx].active = 1
            particle_info[idx].spawn_time = current_time
            particle_info[idx].lifetime = max_lifetime
    
    def update_particle_lifetimes(self, dt, current_time):
        """Update particle lifetimes and deactivate expired particles"""
        wp.launch(
            kernel=self._update_lifetimes_kernel,
            dim=self.max_particles,
            inputs=[self.particle_info, dt],
            device=self.device
        )
    
    @wp.kernel
    def _update_lifetimes_kernel(particle_info: wp.array(dtype=FluidParticle), dt: float):
        """Update particle lifetimes"""
        i = wp.tid()
        if particle_info[i].active == 1:
            particle_info[i].lifetime -= dt
            if particle_info[i].lifetime <= 0.0:
                particle_info[i].active = 0
    
    def count_active_particles(self):
        """Count the number of active particles"""
        # Reset counter first
        self.active_count.zero_()
        
        # Count active particles
        wp.launch(
            kernel=self._count_active_kernel,
            dim=self.max_particles,
            inputs=[self.particle_info, self.active_count],
            device=self.device
        )
        return int(self.active_count.numpy()[0])
    
    @wp.kernel
    def _count_active_kernel(particle_info: wp.array(dtype=FluidParticle), active_count: wp.array(dtype=wp.int32)):
        """Count active particles"""
        i = wp.tid()
        
        if particle_info[i].active == 1:
            wp.atomic_add(active_count, 0, 1)
    
    def build_spatial_grid(self):
        """Build spatial hash grid for neighbor finding"""
        # Build grid with all particle positions (kernels will check active status)
        self.grid.build(self.positions, self.smoothing_length)
    
    def _get_active_positions(self):
        """Get positions of only active particles (helper for spatial grid)"""
        # For simplicity, we'll use all particle positions but the kernels will check active status
        # In a more optimized version, we'd compact the array to only active particles
        return self.positions
    
    def simulate_step(self, dt, current_time):
        """Perform one PBF simulation step"""
        # Update particle lifetimes
        self.update_particle_lifetimes(dt, current_time)
        
        # Count active particles
        num_active = self.count_active_particles()
        if num_active == 0:
            return
        
        # Step 1: Apply external forces and predict positions
        wp.launch(
            kernel=integrate_particles,
            dim=self.max_particles,
            inputs=[
                self.positions,
                self.velocities,
                self.predicted_positions,
                self.forces,
                self.max_particles,
                dt,
                self.particle_mass
            ],
            device=self.device
        )
        
        # Copy predicted positions to actual positions for constraint solving
        wp.copy(self.positions, self.predicted_positions)
        
        # Step 2: Build spatial hash grid
        self.build_spatial_grid()
        
        # Step 3: Solve density constraints iteratively
        for _ in range(self.constraint_iterations):
            # Compute densities
            wp.launch(
                kernel=compute_density,
                dim=self.max_particles,
                inputs=[
                    self.grid.id,
                    self.positions,
                    self.densities,
                    self.max_particles,
                    self.smoothing_length,
                    self.particle_mass
                ],
                device=self.device
            )
            
            # Compute lambda multipliers
            wp.launch(
                kernel=compute_lambda,
                dim=self.max_particles,
                inputs=[
                    self.grid.id,
                    self.positions,
                    self.densities,
                    self.lambdas,
                    self.max_particles,
                    self.smoothing_length,
                    self.rest_density,
                    self.particle_mass,
                    100.0  # epsilon for stability
                ],
                device=self.device
            )
            
            # Compute position corrections
            wp.launch(
                kernel=compute_position_correction,
                dim=self.max_particles,
                inputs=[
                    self.grid.id,
                    self.positions,
                    self.lambdas,
                    self.position_corrections,
                    self.max_particles,
                    self.smoothing_length,
                    self.rest_density,
                    self.particle_mass,
                    self.artificial_pressure_strength,
                    self.artificial_pressure_radius
                ],
                device=self.device
            )
            
            # Apply position corrections
            wp.launch(
                kernel=apply_position_corrections,
                dim=self.max_particles,
                inputs=[self.positions, self.position_corrections, self.max_particles],
                device=self.device
            )
        
        # Step 4: Update velocities from position changes
        wp.launch(
            kernel=update_velocities_from_positions,
            dim=self.max_particles,
            inputs=[
                self.positions,
                self.predicted_positions,
                self.velocities,
                self.max_particles,
                dt
            ],
            device=self.device
        )
        
        # Step 5: Apply XSPH viscosity
        wp.launch(
            kernel=compute_xsph_viscosity,
            dim=self.max_particles,
            inputs=[
                self.grid.id,
                self.positions,
                self.velocities,
                self.velocity_corrections,
                self.max_particles,
                self.smoothing_length,
                self.particle_mass,
                self.viscosity_strength
            ],
            device=self.device
        )
        
        wp.launch(
            kernel=apply_velocity_corrections,
            dim=self.max_particles,
            inputs=[self.velocities, self.velocity_corrections, self.max_particles],
            device=self.device
        )
        
        # Step 6: Apply boundary constraints
        wp.launch(
            kernel=apply_boundary_constraints,
            dim=self.max_particles,
            inputs=[
                self.positions,
                self.velocities,
                self.max_particles,
                self.bounds_min,
                self.bounds_max,
                self.boundary_damping
            ],
            device=self.device
        )
    
    def get_active_particle_positions(self):
        """Get positions of active particles for rendering"""
        # Create a filtered array of active particle positions
        positions_array = self.positions.numpy()
        particle_info_array = self.particle_info.numpy()
        
        active_positions = []
        for i in range(self.max_particles):
            if particle_info_array[i]['active'] == 1:
                active_positions.append(positions_array[i])
        
        return np.array(active_positions) if active_positions else np.array([]).reshape(0, 3)
    
    def get_active_particle_data(self):
        """Get all data for active particles (positions, velocities, etc.)"""
        positions_array = self.positions.numpy()
        velocities_array = self.velocities.numpy() 
        particle_info_array = self.particle_info.numpy()
        
        active_data = {
            'positions': [],
            'velocities': [],
            'indices': []
        }
        
        for i in range(self.max_particles):
            if particle_info_array[i]['active'] == 1:
                active_data['positions'].append(positions_array[i])
                active_data['velocities'].append(velocities_array[i])
                active_data['indices'].append(i)
        
        # Convert to numpy arrays
        active_data['positions'] = np.array(active_data['positions']) if active_data['positions'] else np.array([]).reshape(0, 3)
        active_data['velocities'] = np.array(active_data['velocities']) if active_data['velocities'] else np.array([]).reshape(0, 3)
        active_data['indices'] = np.array(active_data['indices'])
        
        return active_data
    
    def set_simulation_bounds(self, bounds_min, bounds_max):
        """Set the simulation boundary box"""
        self.bounds_min = wp.vec3f(bounds_min[0], bounds_min[1], bounds_min[2])
        self.bounds_max = wp.vec3f(bounds_max[0], bounds_max[1], bounds_max[2])
    
    def clear_all_particles(self):
        """Deactivate all particles"""
        wp.launch(
            kernel=self._clear_particles_kernel,
            dim=self.max_particles,
            inputs=[self.particle_info],
            device=self.device
        )
    
    @wp.kernel
    def _clear_particles_kernel(particle_info: wp.array(dtype=FluidParticle)):
        """Deactivate all particles"""
        i = wp.tid()
        particle_info[i].active = 0
    
    @wp.kernel
    def _extract_active_status_kernel(
        particle_info: wp.array(dtype=FluidParticle),
        particle_active: wp.array(dtype=wp.int32)
    ):
        """Extract active status from particle_info for use in other kernels"""
        i = wp.tid()
        particle_active[i] = particle_info[i].active
    
    def _update_active_status(self):
        """Update the active status array from particle info"""
        wp.launch(
            kernel=self._extract_active_status_kernel,
            dim=self.max_particles,
            inputs=[self.particle_info, self.particle_active],
            device=self.device
        )