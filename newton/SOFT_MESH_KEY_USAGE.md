# Soft Mesh Key Tracking - Usage Guide

This document describes the new soft mesh key tracking feature that allows you to identify and track deformable bodies (soft meshes) in Newton, similar to how `body_key` works for rigid bodies.

## Overview

The soft mesh key tracking system provides:
- **String identifiers** for each soft mesh (similar to `body_key` for rigid bodies)
- **Particle index ranges** `(start_idx, end_idx)` inclusive for each soft mesh
- **USD path mapping** when loading from USD files via `parse_usd()`

## API Reference

### ModelBuilder

#### New Attributes
```python
builder.soft_mesh_key: list[str]
# List of string identifiers for each soft mesh

builder.soft_mesh_particle_range: list[tuple[int, int]]
# List of (start_idx, end_idx) particle ranges for each soft mesh

builder.soft_mesh_count: int
# Property that returns the number of soft meshes
```

#### Updated Methods

**`add_soft_mesh()`**
```python
soft_mesh_id = builder.add_soft_mesh(
    pos=wp.vec3(0.0, 0.0, 0.0),
    rot=wp.quat_identity(),
    scale=1.0,
    vel=wp.vec3(0.0, 0.0, 0.0),
    vertices=vertices,
    indices=indices,
    density=1000.0,
    k_mu=1000.0,
    k_lambda=1000.0,
    k_damp=0.1,
    key="my_soft_body",  # NEW: Optional string identifier
)
# Returns: int (soft mesh ID)
```

**`add_soft_grid()`**
```python
soft_mesh_id = builder.add_soft_grid(
    pos=wp.vec3(0.0, 0.0, 0.0),
    rot=wp.quat_identity(),
    vel=wp.vec3(0.0, 0.0, 0.0),
    dim_x=5,
    dim_y=5,
    dim_z=5,
    cell_x=0.1,
    cell_y=0.1,
    cell_z=0.1,
    density=1000.0,
    k_mu=1000.0,
    k_lambda=1000.0,
    k_damp=0.1,
    key="my_grid",  # NEW: Optional string identifier
)
# Returns: int (soft mesh ID)
```

### Model

#### New Attributes
```python
model.soft_mesh_key: list[str]
# Soft mesh identifiers (transferred from ModelBuilder)

model.soft_mesh_particle_range: list[tuple[int, int]]
# Particle index ranges for each soft mesh
```

### parse_usd()

#### New Return Value
```python
result = newton.parse_usd(builder, "model.usd")

result["path_soft_mesh_map"]: dict[str, int]
# Mapping from USD prim path to soft mesh ID
# Example: {"/World/SoftBody": 0, "/World/Cloth": 1}
```

## Usage Examples

### Example 1: Manual Soft Mesh Creation

```python
import warp as wp
import newton

# Create builder
builder = newton.ModelBuilder()

# Define tetrahedral mesh
vertices = [
    wp.vec3(0.0, 0.0, 0.0),
    wp.vec3(1.0, 0.0, 0.0),
    wp.vec3(0.0, 1.0, 0.0),
    wp.vec3(0.0, 0.0, 1.0),
]
indices = [0, 1, 2, 3]

# Add soft mesh with custom key
soft_mesh_id = builder.add_soft_mesh(
    pos=wp.vec3(0.0, 0.0, 2.0),
    rot=wp.quat_identity(),
    scale=1.0,
    vel=wp.vec3(0.0, 0.0, 0.0),
    vertices=vertices,
    indices=indices,
    density=1000.0,
    k_mu=1000.0,
    k_lambda=1000.0,
    k_damp=0.1,
    key="bear_mesh",  # Custom identifier
)

print(f"Soft mesh ID: {soft_mesh_id}")
print(f"Soft mesh key: {builder.soft_mesh_key[soft_mesh_id]}")
print(f"Particle range: {builder.soft_mesh_particle_range[soft_mesh_id]}")

# Finalize model
model = builder.finalize()

# Access soft mesh info from model
start_idx, end_idx = model.soft_mesh_particle_range[soft_mesh_id]
print(f"Soft mesh '{model.soft_mesh_key[soft_mesh_id]}' uses particles {start_idx} to {end_idx}")
```

### Example 2: Loading from USD with Deformable Bodies

```python
import newton

# Create builder
builder = newton.ModelBuilder()

# Parse USD file containing deformable bodies
result = newton.parse_usd(builder, "scene_with_soft_bodies.usd", verbose=True)

# Access the soft mesh mapping
path_soft_mesh_map = result["path_soft_mesh_map"]

# Example: Get soft mesh ID by USD path
if "/World/SoftBear" in path_soft_mesh_map:
    soft_mesh_id = path_soft_mesh_map["/World/SoftBear"]
    print(f"Soft bear has ID: {soft_mesh_id}")
    print(f"Soft bear key: {builder.soft_mesh_key[soft_mesh_id]}")
    
    # Get particle range
    start_idx, end_idx = builder.soft_mesh_particle_range[soft_mesh_id]
    num_particles = end_idx - start_idx + 1
    print(f"Soft bear has {num_particles} particles (indices {start_idx} to {end_idx})")

# Finalize model
model = builder.finalize()

# Access all soft meshes
for i, key in enumerate(model.soft_mesh_key):
    start, end = model.soft_mesh_particle_range[i]
    print(f"Soft mesh {i}: '{key}' - particles [{start}, {end}]")
```

### Example 3: Working with Particle Data

```python
import warp as wp
import newton

builder = newton.ModelBuilder()

# Add soft mesh
soft_mesh_id = builder.add_soft_mesh(
    pos=wp.vec3(0.0, 0.0, 1.0),
    rot=wp.quat_identity(),
    scale=1.0,
    vel=wp.vec3(0.0, 0.0, 0.0),
    vertices=vertices,
    indices=indices,
    density=1000.0,
    k_mu=1000.0,
    k_lambda=1000.0,
    k_damp=0.1,
    key="soft_cube",
)

model = builder.finalize()
state = model.state()

# Get particles belonging to this soft mesh
start_idx, end_idx = model.soft_mesh_particle_range[soft_mesh_id]

# Access particle positions for this soft mesh
particle_positions = state.particle_q.numpy()[start_idx:end_idx+1]
print(f"Soft mesh '{model.soft_mesh_key[soft_mesh_id]}' has {len(particle_positions)} particles")

# Modify particles of a specific soft mesh
for i in range(start_idx, end_idx + 1):
    # Apply some transformation to particles of this soft mesh
    state.particle_q.numpy()[i] += [0.0, 0.1, 0.0]  # Move up
```

### Example 4: Multi-Environment Setup

```python
import newton

# Create a robot with soft components
robot_builder = newton.ModelBuilder()
robot_builder.add_soft_mesh(..., key="robot_soft_gripper")

# Create main environment
main_builder = newton.ModelBuilder()

# Add ground
main_builder.add_ground_plane()

# Add multiple robot instances to different environments
for env_id in range(3):
    main_builder.add_builder(robot_builder, environment=env_id)

# Finalize
model = main_builder.finalize()

# All soft mesh keys are preserved with unique identifiers
print("Soft meshes in the scene:")
for i, key in enumerate(model.soft_mesh_key):
    start, end = model.soft_mesh_particle_range[i]
    print(f"  {i}: {key} - particles [{start}, {end}]")
```

## Key Features

1. **Automatic Key Generation**: If no key is provided, soft meshes are automatically named as `"soft_mesh_0"`, `"soft_mesh_1"`, etc.

2. **Inclusive Ranges**: Particle ranges are stored as `(start_idx, end_idx)` where both indices are inclusive. To get the number of particles: `num_particles = end_idx - start_idx + 1`

3. **USD Integration**: When using `parse_usd()`, the USD prim path is automatically used as the key, and `path_soft_mesh_map` provides the mapping.

4. **Multi-Environment Support**: Soft mesh keys are properly handled when using `add_builder()` for multi-environment scenarios.

5. **Parallel to Rigid Bodies**: The design mirrors the `body_key` system for rigid bodies, providing a consistent API across both rigid and deformable objects.

## Notes

- Soft mesh IDs are zero-indexed and assigned sequentially as soft meshes are added
- The particle ranges are inclusive on both ends: `[start_idx, end_idx]`
- When merging builders, particle indices are automatically adjusted
- Keys are transferred from ModelBuilder to Model during `finalize()`


