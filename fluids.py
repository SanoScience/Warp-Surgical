import math
from integrator_pbf import PBFIntegrator
import warp as wp
import numpy as np
from newton import ParticleFlags


#region Kernels

@wp.kernel
def scale_vec3_array(
    values: wp.array(dtype=wp.vec3f),
    count: int,
    scale: float,
):
    tid = wp.tid()
    if tid >= count:
        return
    values[tid] = values[tid] * scale

@wp.kernel
def fill_float32_3d(arr: wp.array(dtype=wp.float32, ndim=3), value: float):
    i, j, k = wp.tid()
    if i < arr.shape[0] and j < arr.shape[1] and k < arr.shape[2]:
        arr[i, j, k] = value

@wp.kernel
def compute_aabb_from_particles(
    positions: wp.array(dtype=wp.vec3f),
    active: wp.array(dtype=wp.int32),
    num_particles: int,
    aabb_min: wp.array(dtype=wp.float32),  # [3] - separate x,y,z components required for atomic_min/max
    aabb_max: wp.array(dtype=wp.float32)   # [3] - ^ as above
):
    tid = wp.tid()
    if tid >= num_particles:
        return
    
    if active[tid] == 0:
        return
    
    pos = positions[tid]
    
    # Atomic min/max operations on individual components
    wp.atomic_min(aabb_min, 0, pos[0])
    wp.atomic_min(aabb_min, 1, pos[1])
    wp.atomic_min(aabb_min, 2, pos[2])
    
    wp.atomic_max(aabb_max, 0, pos[0])
    wp.atomic_max(aabb_max, 1, pos[1])
    wp.atomic_max(aabb_max, 2, pos[2])

@wp.kernel
def compute_sdf_field(
    field: wp.array(dtype=wp.float32, ndim=3),
    field_dims: wp.vec3i,
    field_origin: wp.vec3f,
    field_spacing: wp.float32,
    particle_positions: wp.array(dtype=wp.vec3f),
    particle_active: wp.array(dtype=wp.int32),
    num_particles: int,
    particle_radius: wp.float32
):
    i, j, k = wp.tid()
    
    if i >= field_dims[0] or j >= field_dims[1] or k >= field_dims[2]:
        return
    
    # Convert grid coordinates to world position
    world_pos = field_origin + wp.vec3f(
        wp.float32(i) * field_spacing,
        wp.float32(j) * field_spacing,
        wp.float32(k) * field_spacing
    )
    
    # Find minimum distance to any active particle
    min_distance = float(1e6)
    
    for p in range(num_particles):
        if particle_active[p] == 0:
            continue
            
        particle_pos = particle_positions[p]
        dist = wp.length(world_pos - particle_pos)
        
        # SDF of sphere: distance to surface
        sdf_dist = dist - particle_radius
        
        if sdf_dist < min_distance:
            min_distance = sdf_dist
    
    # Store SDF
    field[i, j, k] = -min_distance


#endregion
#region Simulation

def initialize_fluid_particles(self):
        """Initialize fluid particles in a cube formation."""
        import numpy as np
        
        # Create a small cube of particles
        particles_per_axis = self.fluid_particles_per_axis
        spacing = self.fluid_particle_spacing
        half_extent = 0.5 * (particles_per_axis - 1) * spacing
        
        positions = []
        velocities = []
        active_flags = []
        
        count = 0
        for i in range(particles_per_axis):
            if count >= self.fluid_particle_count:
                break
            for j in range(particles_per_axis):
                if count >= self.fluid_particle_count:
                    break
                for k in range(particles_per_axis):
                    if count >= self.fluid_particle_count:
                        break
                    
                    local_pos = np.array([
                        (i * spacing) - half_extent,
                        (j * spacing) - half_extent,
                        (k * spacing) - half_extent
                    ], dtype=np.float32)
                    
                    positions.append(self.fluid_spawn_center + local_pos)
                    velocities.append([0.0, 0.0, 0.0])
                    active_flags.append(1)
                    count += 1
        
        # Pad with inactive particles if needed
        inactive_pos = self.fluid_spawn_center.astype(np.float32)
        while len(positions) < self.fluid_particle_count:
            positions.append(inactive_pos)
            velocities.append([0.0, 0.0, 0.0])
            active_flags.append(0)
        
        wp_pos = wp.array(positions, dtype=wp.vec3f, device=wp.get_device())
        wp_vel = wp.array(velocities, dtype=wp.vec3f, device=wp.get_device())
        wp_active = wp.array(active_flags, dtype=wp.uint32, device=wp.get_device())

        # Copy to GPU
        wp.copy(self.fluid_positions, wp_pos)
        wp.copy(self.fluid_velocities, wp_vel)
        wp.copy(self.fluid_active, wp_active)

def simulate_fluid(self):
        """Simulate PBF fluid particles."""
        from integrator_pbf import _pbf_predict_positions, _pbf_compute_density, _pbf_compute_lambdas, _pbf_compute_position_deltas, _pbf_accumulate_delta, _pbf_update_velocities, _pbf_handle_boundaries
        
        gravity = wp.vec3(0.0, -9.81, 0.0)
        
        # Predict positions
        wp.launch(
            kernel=_pbf_predict_positions,
            dim=self.fluid_particle_count,
            inputs=[
                self.fluid_positions,
                self.fluid_velocities,
                self.fluid_forces,
                self.fluid_inv_masses,
                self.fluid_flags,
                gravity,
                self.substep_dt,
                0.1  # max velocity
            ],
            outputs=[
                self._fluid_temp_positions,
                self._fluid_temp_velocities
            ],
            device=wp.get_device(),
        )

        # Build spatial grid
        self.fluid_particle_grid.build(self._fluid_temp_positions, self.fluid_smoothing_radius)

        # PBF constraint iterations
        for _ in range(self.fluid_solver_iterations):
            # Compute density
            wp.launch(
                kernel=_pbf_compute_density,
                dim=self.fluid_particle_count,
                inputs=[
                    self.fluid_particle_grid.id,
                    self._fluid_temp_positions,
                    self.fluid_masses,
                    self.fluid_flags,
                    self.fluid_smoothing_radius,
                    self.pbf_integrator._poly6_coeff,
                    self.fluid_rest_density,
                ],
                outputs=[self._fluid_temp_density],
                device=wp.get_device(),
            )
            
            # Compute lambdas
            wp.launch(
                kernel=_pbf_compute_lambdas,
                dim=self.fluid_particle_count,
                inputs=[
                    self.fluid_particle_grid.id,
                    self._fluid_temp_positions,
                    self.fluid_masses,
                    self.fluid_flags,
                    self._fluid_temp_density,
                    self.fluid_smoothing_radius,
                    self.pbf_integrator._spiky_coeff,
                    self.fluid_rest_density,
                    1.0e-5,  # relaxation
                ],
                outputs=[self._fluid_temp_lambdas],
                device=wp.get_device(),
            )
            
            # Compute position deltas
            wp.launch(
                kernel=_pbf_compute_position_deltas,
                dim=self.fluid_particle_count,
                inputs=[
                    self.fluid_particle_grid.id,
                    self._fluid_temp_positions,
                    self.fluid_masses,
                    self.fluid_flags,
                    self._fluid_temp_lambdas,
                    self._fluid_temp_density,
                    self.fluid_smoothing_radius,
                    self.pbf_integrator._spiky_coeff,
                    self.pbf_integrator._poly6_coeff,
                    0.0,  # surface_tension
                    0.0,  # tensile_instability
                    self.pbf_integrator._scorr_w0,
                    4.0,  # scorr_power
                    self.fluid_rest_density,
                ],
                outputs=[self._fluid_temp_delta],
                device=wp.get_device(),
            )
            
            # Apply deltas
            wp.launch(
                kernel=_pbf_accumulate_delta,
                dim=self.fluid_particle_count,
                inputs=[self._fluid_temp_positions, self._fluid_temp_delta],
                outputs=[self._fluid_temp_positions],
                device=wp.get_device(),
            )
            
            # Handle boundaries
            wp.launch(
                kernel=_pbf_handle_boundaries,
                dim=self.fluid_particle_count,
                inputs=[
                    self._fluid_temp_positions,
                    self._fluid_temp_velocities,
                    self.fluid_flags,
                    wp.vec3(
                        float(self.fluid_bounds_min[0]),
                        float(self.fluid_bounds_min[1]),
                        float(self.fluid_bounds_min[2]),
                    ),
                    wp.vec3(
                        float(self.fluid_bounds_max[0]),
                        float(self.fluid_bounds_max[1]),
                        float(self.fluid_bounds_max[2]),
                    ),
                    self.fluid_boundary_padding,
                    self.fluid_boundary_restitution,
                    self.fluid_boundary_friction,
                ],
                outputs=[self._fluid_temp_positions, self._fluid_temp_velocities],
                device=wp.get_device(),
            )

            # Rebuild grid with corrected positions for the next iteration
            self.fluid_particle_grid.build(self._fluid_temp_positions, self.fluid_smoothing_radius)
        
        # Update velocities
        wp.launch(
            kernel=_pbf_update_velocities,
            dim=self.fluid_particle_count,
            inputs=[
                self.fluid_positions,
                self._fluid_temp_positions,
                self.fluid_flags,
                self.fluid_inv_masses,
                self.substep_dt,
                0.1,  # max velocity
            ],
            outputs=[self._fluid_temp_velocities],
            device=wp.get_device(),
        )

        # Apply mild damping to suppress high-frequency jitter
        wp.launch(
            kernel=scale_vec3_array,
            dim=self.fluid_particle_count,
            inputs=[
                self._fluid_temp_velocities,
                self.fluid_particle_count,
                0.99,
            ],
            device=wp.get_device(),
        )

        assert self.fluid_positions.shape[0] == self.fluid_particle_count
        assert self._fluid_temp_positions.shape[0] == self.fluid_particle_count

        # Update positions and velocities
        wp.copy(self.fluid_positions, self._fluid_temp_positions)
        wp.copy(self.fluid_velocities, self._fluid_temp_velocities)
        
        # Store final density for rendering
        wp.copy(self.fluid_density, self._fluid_temp_density)


#endregion
# region Meshing
def compute_fluid_field(self):
        """Compute scalar field from fluid particles for marching cubes."""
        # Count active fluid particles
        active_count = int(np.sum(self.fluid_active.numpy()))
        if active_count == 0:
            return None
        
        # Reuse AABB computation infrastructure
        wp.copy(self.bleeding_field_aabb_min, wp.array([1e6, 1e6, 1e6], dtype=wp.float32, device=wp.get_device()))
        wp.copy(self.bleeding_field_aabb_max, wp.array([-1e6, -1e6, -1e6], dtype=wp.float32, device=wp.get_device()))
        
        # Compute AABB from active fluid particles
        wp.launch(
            compute_aabb_from_particles,
            dim=self.fluid_particle_count,
            inputs=[
                self.fluid_positions,
                self.fluid_active,
                self.fluid_particle_count,
                self.bleeding_field_aabb_min,
                self.bleeding_field_aabb_max
            ],
            device=wp.get_device()
        )
        
        # AABB values
        aabb_min_array = self.bleeding_field_aabb_min.numpy()
        aabb_max_array = self.bleeding_field_aabb_max.numpy()
        
        aabb_min = wp.vec3f(aabb_min_array[0], aabb_min_array[1], aabb_min_array[2])
        aabb_max = wp.vec3f(aabb_max_array[0], aabb_max_array[1], aabb_max_array[2])
        
        # Add margin
        # Ensure the field extends far enough to capture the fluid surface (account for particle radius)
        required_margin = max(self.bleeding_field_margin, self.fluid_particle_radius * 1.25)
        margin_vec = wp.vec3f(required_margin, required_margin, required_margin)
        field_min = aabb_min - margin_vec
        field_max = aabb_max + margin_vec
        
        # Compute field dimensions and spacing
        field_size = field_max - field_min
        max_extent = max(field_size[0], field_size[1], field_size[2])
        
        if max_extent <= 0:
            return None
            
        self.bleeding_field_spacing = max_extent / float(self.bleeding_field_resolution)
        
        # Calculate actual grid dimensions
        self.bleeding_field_dims = wp.vec3i(
            int(math.ceil(field_size[0] / self.bleeding_field_spacing)) + 1,
            int(math.ceil(field_size[1] / self.bleeding_field_spacing)) + 1,
            int(math.ceil(field_size[2] / self.bleeding_field_spacing)) + 1
        )
        
        self.bleeding_field_origin = field_min
        
        # Allocate field if needed
        field_shape = (self.bleeding_field_dims[0], self.bleeding_field_dims[1], self.bleeding_field_dims[2])
        if self.bleeding_scalar_field is None or self.bleeding_scalar_field.shape != field_shape:
            self.bleeding_scalar_field = wp.zeros(field_shape, dtype=wp.float32, device=wp.get_device())
        else:
            wp.launch(
                fill_float32_3d,
                dim=self.bleeding_scalar_field.shape,
                inputs=[self.bleeding_scalar_field, 0.0],
                device=wp.get_device()
            )
        
        # Compute SDF field using fluid particles
        wp.launch(
            compute_sdf_field,
            dim=self.bleeding_field_dims,
            inputs=[
                self.bleeding_scalar_field,
                self.bleeding_field_dims,
                self.bleeding_field_origin,
                self.bleeding_field_spacing,
                self.fluid_positions,
                self.fluid_active,
                self.fluid_particle_count,
                self.fluid_particle_radius
            ],
            device=wp.get_device()
        )
        
        return {
            'field': self.bleeding_scalar_field,
            'dims': self.bleeding_field_dims,
            'origin': self.bleeding_field_origin,
            'spacing': self.bleeding_field_spacing
        }

def generate_bleeding_mesh(self):
    """Generate mesh from bleeding particles using marching cubes."""
    field_data = self.compute_bleeding_field()
    if field_data is None:
        self.bleeding_mesh_triangle_count = 0
        print("No field data")
        return None
    
    field = field_data['field']
    dims = field_data['dims']
    
    print(f"Generating mesh with dims: {dims}")
    
    # Initialize/resize
    if (self.bleeding_marching_cubes is None or 
        self.bleeding_marching_cubes.nx != dims[0] - 1 or
        self.bleeding_marching_cubes.ny != dims[1] - 1 or
        self.bleeding_marching_cubes.nz != dims[2] - 1):
        
        # Estimate maximum vertices and triangles
        total_cubes = (dims[0] - 1) * (dims[1] - 1) * (dims[2] - 1)
        max_verts = min(total_cubes * 12, 100000)  # Up to 12 vertices per cube, cap at 100k
        max_tris = min(total_cubes * 5, 200000)   # Up to 5 triangles per cube, cap at 200k
        
        print(f"Creating marching cubes: cubes={total_cubes}, max_verts={max_verts}, max_tris={max_tris}")
        
        if self.bleeding_marching_cubes is None:
            self.bleeding_marching_cubes = wp.MarchingCubes(
                nx=dims[0],
                ny=dims[1], 
                nz=dims[2],
                max_verts=max_verts,
                max_tris=max_tris,
                device=wp.get_device()
            )
            print("Created new marching cubes object")
        else:
            self.bleeding_marching_cubes.resize(
                nx=dims[0],
                ny=dims[1],
                nz=dims[2],
                max_verts=max_verts,
                max_tris=max_tris
            )
            print("Resized existing marching cubes object")
    
    print(f"Running marching cubes with threshold: {self.bleeding_isosurface_threshold}")

    # Extract isosurface
    self.bleeding_marching_cubes.surface(field, self.bleeding_isosurface_threshold)
    
    # Get the generated mesh
    self.bleeding_mesh_vertices = self.bleeding_marching_cubes.verts
    self.bleeding_mesh_indices = self.bleeding_marching_cubes.indices
    
    # Count actual triangles generated
    indices_array = self.bleeding_mesh_indices.numpy()
    self.bleeding_mesh_triangle_count = len(indices_array) // 3
    
    print(f"Marching cubes result: {self.bleeding_mesh_triangle_count} triangles, {len(self.bleeding_mesh_vertices.numpy())} vertices")
    
    if self.bleeding_mesh_triangle_count > 0:
        '''
        wp.launch(
            reverse_triangle_winding,
            dim=self.bleeding_mesh_triangle_count,
            inputs=[
                self.bleeding_mesh_indices,
                self.bleeding_mesh_triangle_count
            ],
            device=wp.get_device()
        )
        '''

        return {
            'vertices': self.bleeding_mesh_vertices,
            'indices': self.bleeding_mesh_indices,
            'triangle_count': self.bleeding_mesh_triangle_count,
            'origin': field_data['origin'],
            'spacing': field_data['spacing']
        }
    else:
        self.bleeding_mesh_triangle_count = 0
        return None
    
def generate_fluid_mesh(self):
    """Generate mesh from fluid particles using marching cubes."""
    # Reuse the bleeding mesh generation infrastructure but for fluid particles
    field_data = compute_fluid_field(self)
    if field_data is None:
        return None
    
    field = field_data['field']
    dims = field_data['dims']
    
    # Initialize/resize marching cubes if needed
    if (self.bleeding_marching_cubes is None or 
        self.bleeding_marching_cubes.nx != dims[0] - 1 or
        self.bleeding_marching_cubes.ny != dims[1] - 1 or
        self.bleeding_marching_cubes.nz != dims[2] - 1):
        
        total_cubes = (dims[0] - 1) * (dims[1] - 1) * (dims[2] - 1)
        max_verts = min(total_cubes * 12, 100000)
        max_tris = min(total_cubes * 5, 200000)
        
        if self.bleeding_marching_cubes is None:
            self.bleeding_marching_cubes = wp.MarchingCubes(
                nx=dims[0],
                ny=dims[1], 
                nz=dims[2],
                max_verts=max_verts,
                max_tris=max_tris,
                device=wp.get_device()
            )
        else:
            self.bleeding_marching_cubes.resize(
                nx=dims[0],
                ny=dims[1],
                nz=dims[2],
                max_verts=max_verts,
                max_tris=max_tris
            )
    
    # Extract isosurface
    self.bleeding_marching_cubes.surface(field, 0.0)
    
    # Get the generated mesh
    vertices = self.bleeding_marching_cubes.verts
    indices = self.bleeding_marching_cubes.indices
    
    triangle_count = len(indices.numpy()) // 3
    
    #print("tri count: " + str(triangle_count))

    if triangle_count > 0:
        return {
            'vertices': vertices,
            'indices': indices,
            'triangle_count': triangle_count,
            'origin': field_data['origin'],
            'spacing': field_data['spacing']
        }
    else:
        return None
#endregion

#region Setup

def setup_fluids_data(self):
    self.fluid_particle_count = 1000
    self.fluid_particle_radius = 0.01
    self.fluid_particle_spacing = self.fluid_particle_radius * 2.2
    self.fluid_smoothing_radius = self.fluid_particle_radius * 3.0
    self.fluid_rest_density = 95.0
    self.fluid_viscosity = 0.05
    self.fluid_spawn_center = np.array([0.5, 1.0, -4.0], dtype=np.float32)
    self.fluid_particles_per_axis = int(np.ceil(self.fluid_particle_count ** (1 / 3)))
    spawn_half_extent = 0.03 * (self.fluid_particles_per_axis - 1)
    half_extent_vec = np.array(
        [
            spawn_half_extent + 0.008,
            spawn_half_extent + 0.025,
            spawn_half_extent + 0.008,
        ],
        dtype=np.float32,
    )
    self.fluid_bounds_min = (self.fluid_spawn_center - half_extent_vec).astype(np.float32)
    self.fluid_bounds_max = (self.fluid_spawn_center + half_extent_vec).astype(np.float32)
    self.fluid_boundary_padding = 0.01
    self.fluid_boundary_restitution = 0.1
    self.fluid_boundary_friction = 0.3
    self.fluid_solver_iterations = 5
    particle_mass = self.fluid_rest_density * (self.fluid_particle_spacing ** 3)
    
    # PBF particle state arrays
    self.fluid_positions = wp.zeros(self.fluid_particle_count, dtype=wp.vec3f, device=wp.get_device())
    self.fluid_velocities = wp.zeros(self.fluid_particle_count, dtype=wp.vec3f, device=wp.get_device())
    self.fluid_forces = wp.zeros(self.fluid_particle_count, dtype=wp.vec3f, device=wp.get_device())
    self.fluid_masses = wp.full(self.fluid_particle_count, particle_mass, dtype=wp.float32, device=wp.get_device())
    self.fluid_inv_masses = wp.full(self.fluid_particle_count, 1.0 / particle_mass, dtype=wp.float32, device=wp.get_device())
    self.fluid_flags = wp.full(self.fluid_particle_count, ParticleFlags.ACTIVE, dtype=wp.int32, device=wp.get_device())  # All active
    self.fluid_active = wp.full(self.fluid_particle_count, 1, dtype=wp.int32, device=wp.get_device())
    
    self.bleed_next_id = wp.zeros(1, dtype=wp.int32, device=wp.get_device())

    self._fluid_temp_positions = wp.zeros(self.fluid_particle_count, dtype=wp.vec3f, device=wp.get_device())
    self._fluid_temp_velocities = wp.zeros(self.fluid_particle_count, dtype=wp.vec3f, device=wp.get_device())
    self._fluid_temp_density = wp.zeros(self.fluid_particle_count, dtype=wp.float32, device=wp.get_device())
    self._fluid_temp_lambdas = wp.zeros(self.fluid_particle_count, dtype=wp.float32, device=wp.get_device())
    self._fluid_temp_delta = wp.zeros(self.fluid_particle_count, dtype=wp.vec3f, device=wp.get_device())

    # Initialize fluid particle positions in a small cube
    initialize_fluid_particles(self)
    
    # Create PBF integrator
    
    self.pbf_integrator = PBFIntegrator(
        self.model,
        smoothing_radius=self.fluid_smoothing_radius,
        rest_density=self.fluid_rest_density,
        relaxation=1.0e-6,
        iterations=self.fluid_solver_iterations,
        viscosity=self.fluid_viscosity,
        vorticity=0.0,
        surface_tension=1.0,
        boundary_min=wp.vec3(
            float(self.fluid_bounds_min[0]),
            float(self.fluid_bounds_min[1]),
            float(self.fluid_bounds_min[2]),
        ),
        boundary_max=wp.vec3(
            float(self.fluid_bounds_max[0]),
            float(self.fluid_bounds_max[1]),
            float(self.fluid_bounds_max[2]),
        ),
        boundary_padding=self.fluid_boundary_padding,
        restitution=self.fluid_boundary_restitution,
        friction=self.fluid_boundary_friction,
    )
    

    # Create particle grid for PBF
    self.fluid_particle_grid = wp.HashGrid(dim_x=64, dim_y=64, dim_z=64, device=wp.get_device())
    # Pre-reserve buffers so CUDA graph capture doesn't allocate during builds
    self.fluid_particle_grid.reserve(self.fluid_particle_count * 2)
    
    # Fluid density for rendering
    self.fluid_density = wp.zeros(self.fluid_particle_count, dtype=wp.float32, device=wp.get_device())

def setup_fluids_rendering(self):
    # Rendering caches
    self.background_mesh_shape_id = None
    self.background_mesh_uploaded = False
    self.fluid_mesh_shape_id = None
    self.fluid_mesh_last_vertex_count = 0
    self.fluid_mesh_last_index_count = 0
    self.fluid_mesh_vertices_world = None
    self.fluid_mesh_indices_current = None
    self._fluid_empty_vertices = None
    self._fluid_empty_indices = None


    # Bleed marching cubes
    self.bleeding_field_resolution = 96  # Max grid resolution per axis
    self.bleeding_field_margin = 0.01    # Margin around AABB
    self.bleeding_particle_sdf_radius = 0.007  # Particle radius for SDF
    
    self.bleeding_field_aabb_min = wp.zeros(3, dtype=wp.float32, device=wp.get_device())
    self.bleeding_field_aabb_max = wp.zeros(3, dtype=wp.float32, device=wp.get_device())
    self.bleeding_scalar_field = None
    self.bleeding_field_dims = wp.vec3i(0, 0, 0)
    self.bleeding_field_origin = wp.vec3f(0.0, 0.0, 0.0)
    self.bleeding_field_spacing = 0.0

    self.bleeding_marching_cubes = None
    self.bleeding_mesh_vertices = None
    self.bleeding_mesh_indices = None
    self.bleeding_mesh_triangle_count = 0
    self.bleeding_isosurface_threshold = 0.0

#endregion