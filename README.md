# Warp-Surgical

## Installation

Follow these steps to set up the environment:

1. Navigate to the IsaacLab folder:
```bash
cd isaacLab
```

2. Create a conda environment from the environment file:
```bash
conda env create -f environment.yml
```

3. Activate the conda environment:
```bash
conda activate isaaclab
```

4. Install PyTorch with CUDA 12.8 support:
```bash
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

5. Install Isaac Sim:
```bash
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
```

6. Install IsaacLab dependencies:
```bash
./isaaclab.sh -i
```

7. Navigate to the Newton folder and install it:
```bash
cd ../newton
pip install -e .
```

8. Return to the IsaacLab folder:
```bash
cd ../isaacLab
```

9. Run the test environment:
```bash
./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Reach-STAR-v0 --num_envs 1
```

## Deformable Body Implementation

This project integrates Newton physics engine with Isaac Lab for deformable body simulation. The implementation includes several key optimizations to ensure correct transform handling and collision behavior.

### Transform Baking

**Problem**: USD XformOp ordering (translate, rotate, scale) causes incorrect positioning when transforms are composed, especially when scale is applied. For deformable bodies, this resulted in position shifts that scaled with the scale factor.

**Solution**: All transforms (scale, rotation, translation) are **pre-applied (baked) directly into mesh vertices** during USD spawning, before Newton reads the geometry.

**Implementation** (`isaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py`):
- When spawning a deformable mesh with `UsdFileCfg`:
  - Applies transformation to both `points` (visual vertices) and `extMesh:vertices` (physics vertices)
  - Transformation order: `p_world = translation + rotation × (scale × p_local)`
  - Removes all XformOps from the parent prim
- Newton then uses **identity transform** (pos=0, rot=identity, scale=1.0)
- Result: Position specified in `init_state.pos` is exactly where the mesh appears

**Example**:
```python
DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Liver",
    spawn=sim_utils.UsdFileCfg(
        usd_path='path/to/liver.usd',
        scale=(0.1, 0.1, 0.1)  # Baked into vertices
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(
        pos=(0.5, 0.0, 0.25)  # Exact position in world
    ),
)
```

### Particle Collision Radius

**Problem**: Newton's default particle radius was 0.1 meters (10 cm), which is too large for scaled meshes. A liver scaled to 0.1× would have 10cm collision spheres, causing unrealistic hovering above surfaces.

**Solution**: Particle collision radius is **automatically computed from mesh geometry**.

**Implementation** (`newton/newton/_src/sim/builder.py`):
- In `add_soft_mesh()`, computes average edge length from tetrahedra
- Sets `particle_radius = avg_edge_length × 0.5`
- For a 0.1-scaled liver: edge ≈ 0.01m → radius ≈ 0.005m (5mm)
- Result: Collision behavior scales appropriately with mesh size

### Material Properties

Deformable bodies use elastic material model with Lamé parameters:

**From USD** (if authored):
- `physics:youngsModulus` and `physics:poissonsRatio` → converted to Lamé parameters
- `physics:mu` (shear modulus) and `physics:lambda` (first Lamé parameter)
- `physics:damping` (damping coefficient)

**Default values**:
- `k_mu = 1000.0` (shear modulus)
- `k_lambda = 1000.0` (first Lamé parameter)  
- `k_damp = 0.1` (damping)

**Conversion**:
```
k_mu = 0.5 × E / (1 + ν)
k_lambda = E × ν / ((1 + ν) × (1 - 2ν))
```
Where E = Young's modulus, ν = Poisson's ratio

### Density Scaling

**Important**: When geometry is scaled, density must be adjusted to maintain physical correctness.

**Implementation** (`isaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py`):
- If scale is applied to a USD file, the `physics:density` attribute is divided by volume scale
- Volume scale = `scale_x × scale_y × scale_z`
- For uniform scale 0.1: `new_density = original_density / 0.001`
- This ensures mass scales correctly with volume

**Note**: For deformable bodies with pre-baked transforms, density adjustment happens during USD spawn, not in Newton.

### Soft Contact Parameters

Contact behavior between soft and rigid bodies is controlled by:

**Global parameters** (set via `NewtonManager.get_model()`):
- `soft_contact_ke`: Contact stiffness (default: 1.0e6)
- `soft_contact_kd`: Contact damping (default: 1.0e3)
- `soft_contact_mu`: Friction coefficient (default: 0.5)
- `soft_contact_restitution`: Bounce coefficient (default: 0.0)

**Per-simulation parameters**:
- `soft_contact_margin`: Contact detection distance (default: 0.01m)

**Example configuration**:
```python
def set_soft_contact_params():
    model = NewtonManager.get_model()
    if model is not None:
        model.soft_contact_ke = 1.0e6  # Stiffer contact
        model.soft_contact_kd = 1.0e3  # Higher damping
        model.soft_contact_mu = 0.5    # Friction
        model.soft_contact_restitution = 0.0  # No bounce

NewtonManager.add_on_start_callback(set_soft_contact_params)
```

### Solver Configuration

The XPBOSolver is used for soft body dynamics with these key parameters:

**In `SimulationCfg.newton_cfg`**:
```python
newton_cfg=NewtonCfg(
    solver_cfg=XPBOSolverCfg(
        iterations=32,  # XPBD iterations per substep
        soft_body_relaxation=0.0001,  # Relaxation for soft body constraints
    ),
    num_substeps=32,  # Physics substeps per frame
    debug_mode=True,  # Enable debug visualization
)
```

**Recommendations**:
- `iterations`: 16-32 for stable soft body simulation
- `num_substeps`: 16-32 depending on stiffness (stiffer materials need more)
- `soft_body_relaxation`: 0.0001-0.001 for numerical stability

### Physics Simulation Flow

1. **USD Spawn**: Mesh vertices transformed to world space, XformOps removed
2. **Newton Import**: Reads pre-transformed vertices, uses identity transform
3. **Particle Creation**: Each vertex becomes a particle with computed radius
4. **Collision Detection**: Particles collide with rigid shapes using margin
5. **Physics Step**: XPBD solver updates particle positions
6. **USD Sync**: Particle positions copied directly to USD (no inverse transform needed)

### Troubleshooting

**Mesh appears at wrong position**:
- Check that `init_state.pos` matches desired location
- Verify transforms are being baked (look for `extMesh:vertices` in USD)

**Mesh hovering above surfaces**:
- Particle radius too large → automatically computed from mesh geometry
- Contact margin too large → adjust `soft_contact_margin` in collision call
- Insufficient contact stiffness → increase `soft_contact_ke`

**Mesh is too soft/stiff**:
- Adjust `physics:youngsModulus` in USD file
- Or set `k_mu` and `k_lambda` parameters directly
- Increase `iterations` and `num_substeps` for stiffer materials

**Mesh penetrates geometry**:
- Increase `soft_contact_ke` (contact stiffness)
- Increase `num_substeps` for better temporal resolution
- Check that geometry has proper collision mesh

