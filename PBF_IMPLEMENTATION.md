# Position Based Fluids (PBF) Implementation

## Overview

This document describes the Position Based Fluids implementation integrated into the surgical simulation bleeding system. The PBF system replaces the simple particle-based bleeding with a more physically accurate incompressible fluid simulation.

## Key Features

- **Incompressible Fluid Constraints**: Maintains fluid density through iterative constraint solving
- **Spatial Hash Grid**: Efficient neighbor finding using Warp's spatial hash data structure  
- **GPU-Accelerated**: All kernels run on GPU for high performance
- **Seamless Integration**: Replaces existing bleeding system with minimal changes to API

## Architecture

### Core Components

1. **Particle System**: 
   - Positions, velocities, predicted positions
   - Density and lambda (Lagrange multiplier) fields
   - Position correction deltas

2. **Spatial Hashing**: 
   - 64x64x64 hash grid for fast neighbor queries
   - Configurable smoothing radius (default: 0.016m)

3. **PBF Solver**:
   - 4 constraint iterations per timestep  
   - Poly6 kernel for density computation
   - Spiky kernel for pressure gradient

### Parameters

```python
pbf_rest_density = 1000.0        # kg/m³ (water density)
pbf_smoothing_radius = 0.016     # meters
pbf_constraint_epsilon = 600.0   # constraint regularization
pbf_solver_iterations = 4        # iterations per timestep  
pbf_damping = 0.99              # velocity damping factor
pbf_gravity = (0, -9.81, 0)     # m/s² acceleration
```

## Implementation Details

### Kernel Pipeline

1. **`pbf_predict_positions`**: Apply forces (gravity) and predict new positions
2. **`pbf_compute_density`**: Calculate particle density using Poly6 kernel
3. **`pbf_compute_lambda`**: Compute Lagrange multipliers for density constraints
4. **`pbf_compute_delta_positions`**: Calculate position corrections using Spiky kernel gradient
5. **`pbf_update_predicted_positions`**: Apply position corrections
6. **`pbf_apply_boundaries`**: Handle collisions with simulation boundaries
7. **`pbf_update_velocities_and_positions`**: Update final velocities and positions

### Integration Points

- **Bleeding Emission**: `emit_bleed_particles()` creates new fluid particles at cut locations
- **Fluid Update**: `update_pbf_bleeding_particles()` runs the full PBF solver pipeline  
- **Rendering**: Existing marching cubes system generates fluid surface mesh

## Usage

The PBF system is automatically used when running the simulation. The key integration points are:

```python
# In WarpSim.__init__():
self.pbf_hash_grid = wp.HashGrid(dim_x=64, dim_y=64, dim_z=64, device=wp.get_device())

# In WarpSim.render():  
self.update_pbf_bleeding_particles()  # Replaces simple particle update

# Methods available:
sim.pbf_solve_fluid_constraints()     # Core PBF solver
sim.update_pbf_bleeding_particles()   # Full update pipeline
```

## Performance

- **Kernel Compilation**: ~2.8s first run (cached afterwards)
- **Runtime Performance**: GPU-accelerated, scales with particle count
- **Memory Usage**: Additional arrays for PBF solver state (~6 arrays per particle)
- **Spatial Hashing**: O(1) neighbor queries vs O(n²) brute force

## Testing

Run the test suite to verify correct implementation:

```bash
python test_pbf_integration.py
```

Tests verify:
- Kernel compilation and execution
- Proper density computation 
- Lambda calculation for constraints
- Position correction pipeline
- Integration with existing codebase

## References

- Macklin, M. & Müller, M. (2013). "Position based fluids." *ACM Transactions on Graphics*
- Müller, M. et al. (2003). "Particle-based fluid simulation for interactive applications"
- NVIDIA Warp documentation: https://nvidia.github.io/warp/