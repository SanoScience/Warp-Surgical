# Position Based Fluids (PBF) OpenGL Example

## Overview

This example demonstrates a complete Position Based Fluids (PBF) simulation with real-time OpenGL particle visualization. It's based on the seminal paper "Position based fluids" by Macklin & Müller (2013) and follows the structure of NVIDIA Warp's SPH example while implementing PBF algorithms.

## Key Features

- **Position Based Fluids**: Incompressible fluid simulation using constraint-based approach
- **Real-time OpenGL Rendering**: Interactive particle visualization with camera controls
- **GPU Acceleration**: All computations run on GPU using NVIDIA Warp
- **Spatial Hash Grid**: Efficient neighbor finding for particle interactions
- **Dam Break Scenario**: Classic fluid simulation setup
- **Interactive Controls**: Pause/resume, reset, camera movement

## Files

- `example_pbf_opengl.py`: Main PBF simulation with OpenGL rendering
- `test_pbf_example.py`: Comprehensive test suite and performance benchmark  
- `README_PBF_EXAMPLE.md`: This documentation

## Requirements

```bash
# Core requirements (already installed)
pip install warp-lang numpy

# For OpenGL visualization  
pip install PyOpenGL PyOpenGL_accelerate
```

## Usage

### Running the Visual Simulation

```bash
python example_pbf_opengl.py
```

### Running Tests

```bash
python test_pbf_example.py
```

## Controls

- **Mouse**: Click and drag to rotate camera
- **Space**: Pause/unpause simulation
- **R**: Reset simulation to initial state
- **Escape**: Exit simulation

## Simulation Parameters

```python
SIM_PARAMS = {
    'particle_radius': 0.025,      # Particle size for rendering
    'smoothing_radius': 0.05,      # SPH kernel support radius
    'rest_density': 1000.0,        # Target fluid density (kg/m³)
    'constraint_iterations': 4,     # PBF solver iterations per step
    'damping': 0.98,               # Velocity damping factor
    'gravity': -9.81,              # Gravitational acceleration (m/s²)
    'dt': 1.0/60.0,               # Timestep (60 FPS)
    'grid_dim': 64,               # Spatial hash grid resolution
    'domain_min': [-2, -1, -2],   # Simulation boundary (min)
    'domain_max': [2, 3, 2]       # Simulation boundary (max)
}
```

## Algorithm Overview

### PBF Pipeline

1. **Predict Positions**: Apply external forces (gravity) and predict new particle positions
2. **Build Spatial Hash**: Update spatial hash grid for efficient neighbor queries
3. **Constraint Projection** (iterated 4 times):
   - Compute particle densities using Poly6 kernel
   - Calculate Lagrange multipliers for density constraints
   - Compute position corrections using Spiky kernel gradient
   - Apply position corrections
4. **Boundary Handling**: Apply collision response with simulation boundaries
5. **Velocity Update**: Compute new velocities from position changes and apply damping

### Key Differences from SPH

- **Constraint-based**: Uses position constraints instead of pressure forces
- **Iterative Solver**: Multiple constraint projection iterations per timestep
- **Better Stability**: More stable than SPH, especially for incompressible flows
- **Improved Visual Quality**: Better particle cohesion and surface tension effects

## Performance

Test results on NVIDIA GeForce RTX 3070:

| Particles | Performance | Time/Step |
|-----------|-------------|-----------|
| 256       | 2474 FPS    | 0.40 ms   |
| 512       | 1709 FPS    | 0.59 ms   |
| 1024      | 1217 FPS    | 0.82 ms   |
| 2048      | 884 FPS     | 1.13 ms   |

The simulation scales well and can handle thousands of particles in real-time.

## Code Structure

### Core Classes

- **`PBFSimulation`**: Main simulation class handling particle system and PBF solver
- **`OpenGLRenderer`**: OpenGL-based particle renderer with camera controls

### Warp Kernels

- **`pbf_predict_positions`**: Applies forces and predicts new positions
- **`pbf_compute_density`**: Computes particle density using Poly6 kernel
- **`pbf_compute_lambda`**: Calculates Lagrange multipliers for constraints
- **`pbf_compute_delta_positions`**: Computes position corrections
- **`pbf_apply_delta_positions`**: Applies position corrections
- **`pbf_apply_boundaries`**: Handles boundary collisions
- **`pbf_update_velocities_positions`**: Updates final particle state

## Visualization Features

- **Particle Rendering**: Particles rendered as colored points with height-based gradient
- **Boundary Visualization**: Wireframe box showing simulation boundaries
- **Real-time UI**: FPS counter, simulation status, and control instructions
- **Interactive Camera**: Mouse-controlled 3D camera with orbit controls

## Extending the Example

### Adding New Forces

```python
@wp.kernel
def apply_custom_force(
    velocities: wp.array(dtype=wp.vec3),
    force: wp.vec3,
    dt: float
):
    tid = wp.tid()
    velocities[tid] = velocities[tid] + force * dt
```

### Modifying Particle Properties

```python
# In PBFSimulation.__init__():
self.particle_colors = wp.zeros(num_particles, dtype=wp.vec3, device=self.device)
self.particle_temperatures = wp.zeros(num_particles, dtype=wp.float32, device=self.device)
```

### Adding Surface Reconstruction

Consider integrating marching cubes for fluid surface rendering:

```python
# Add to simulation
self.marching_cubes = wp.MarchingCubes(nx=64, ny=64, nz=64, device=self.device)
```

## References

1. Macklin, M. & Müller, M. (2013). "Position based fluids." *ACM Transactions on Graphics*, 32(4), 104.
2. Müller, M. et al. (2007). "Position based dynamics." *Journal of Visual Communication and Image Representation*, 18(2), 109-118.
3. NVIDIA Warp Documentation: https://nvidia.github.io/warp/
4. Original SPH Paper: Müller, M. et al. (2003). "Particle-based fluid simulation for interactive applications." *Proceedings of the 2003 ACM SIGGRAPH/Eurographics symposium on Computer animation*.

## Troubleshooting

### Common Issues

1. **OpenGL Import Error**: Install PyOpenGL with `pip install PyOpenGL PyOpenGL_accelerate`
2. **Slow Performance**: Reduce particle count or constraint iterations
3. **Compilation Issues**: Ensure CUDA toolkit is installed for GPU acceleration
4. **Display Issues**: Try running without OpenGL first using the test script

### Platform-Specific Notes

- **Windows**: Ensure Visual Studio C++ build tools are installed
- **Linux**: May need additional OpenGL development packages
- **macOS**: Metal backend may provide better performance than OpenGL

## Contributing

Feel free to extend this example with:
- Surface tension effects
- Viscosity modeling
- Multi-phase fluids  
- Fluid-solid coupling
- Advanced rendering techniques
- Performance optimizations