# Findings: Surgical Simulator Architecture Review

## Surgical Simulator Codebase

### File Structure
```
warp-surgical/
├── main.py                    # Entry point (~75 lines)
├── warp_simulation.py         # WarpSim class (~2000+ lines) - GOD OBJECT
├── PBDSolver.py              # Custom XPBD solver (~700 lines)
├── haptic_device.py          # Haptic controller wrapper
├── mesh_loader.py            # Mesh parsing and model building
├── collision_kernels.py      # GPU collision kernels
├── simulation_kernels.py     # GPU constraint kernels
├── render_surgsim_opengl.py  # OpenGL renderer
├── render_opengl.py          # Alternative renderer
├── grasping.py               # Grasping behavior
├── heating.py                # Thermal/cauterization
├── stretching.py             # Tissue cutting/breaking
├── centrelines.py            # Vessel/duct tracking
├── surface_reconstruction.py # Dynamic mesh extraction
└── meshes/, textures/        # Assets
```

### Key Components
1. **WarpSim** - Monolithic simulation manager handling physics, rendering, haptics, instruments, bleeding, centrelines
2. **PBDSolver** - Extends Newton's SolverXPBD with custom constraints
3. **HapticController** - OpenHaptics device interface
4. **Rendering** - Dual backends (OpenGL real-time, USD offline)

### Design Patterns Used
- **God Object**: WarpSim (~50+ methods, mixed concerns)
- **GPU Kernel Pattern**: Heavy use of Warp kernels
- **State Swap (Ping-Pong)**: state_0/state_1 for simulation
- **Accumulator Pattern**: Delta accumulation for constraints

---

## Newton Codebase

### Key Patterns & Conventions
1. **Builder Pattern**: ModelBuilder for scene construction → finalize() → Model
2. **State-Solver Separation**: Solver is stateless, data flows through State objects
3. **Model/State/Control/Contacts**: Clear separation of concerns
4. **Example Structure**:
   ```python
   class Example:
       def __init__(self, viewer, args):
           builder = newton.ModelBuilder()
           # ... build scene ...
           self.model = builder.finalize()
           self.solver = newton.solvers.SolverXPBD(self.model)
           self.state_0 = self.model.state()
           self.state_1 = self.model.state()
           self.control = self.model.control()
           self.collision_pipeline = newton.examples.create_collision_pipeline(...)
           self.viewer.set_model(self.model)

       def simulate(self):
           for _ in range(self.sim_substeps):
               self.state_0.clear_forces()
               self.viewer.apply_forces(self.state_0)
               self.contacts = self.model.collide(self.state_0, ...)
               self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
               self.state_0, self.state_1 = self.state_1, self.state_0

       def render(self):
           self.viewer.begin_frame(self.sim_time)
           self.viewer.log_state(self.state_0)
           self.viewer.end_frame()
   ```

### Base Classes & Interfaces
- **SolverBase**: Abstract base with `step()`, `integrate_particles()`, `integrate_bodies()`
- **Model**: Static simulation definition (geometry, constraints, materials)
- **State**: Time-varying data (positions, velocities, forces)
- **Control**: Time-varying inputs (joint forces, activations)
- **Contacts**: Collision information

---

## Architectural Gaps

### 1. God Object (WarpSim)
**Current**: Single 2000+ line class handles everything
**Newton Pattern**: Separate Model, State, Control, Contacts, Solver, Viewer

### 2. Simulation Loop Structure
**Current**: Custom loop in main.py, simulation logic scattered
**Newton Pattern**: Example class with clear `step()`, `simulate()`, `render()` methods

### 3. State Management
**Current**: Scattered arrays (tet_active, centreline_states, clip_attached, etc.)
**Newton Pattern**: Structured Model/State objects with defined attributes

### 4. Solver Architecture
**Current**: PBDSolver overrides step() with 400+ lines of sequential kernel launches
**Newton Pattern**: Modular constraint solvers, clean separation

### 5. Rendering Integration
**Current**: Rendering logic embedded in WarpSim
**Newton Pattern**: Viewer is separate, uses `log_state()` interface

### 6. Custom Attributes
**Current**: Ad-hoc properties on Model (tetrahedra_wp, tri_points_connectors, tet_active)
**Newton Pattern**: `add_custom_attribute()` with frequency/assignment metadata

---

## Code Smells

### Critical
1. **WarpSim God Object** - 50+ methods, 2000+ LOC, unmaintainable
2. **Magic Numbers** - 0.01, 0.1, 0.05, 0.075, 0.2 scattered everywhere
3. **Hardcoded Bounds** - `wp.vec3(-2.0, 0.0, -8.0)` in PBDSolver
4. **HACK Comments** - "Temporary HACK: hide shaft" indicates unresolved issues

### High Priority
1. **Scattered State** - 15+ separate arrays for simulation state
2. **Commented-out Code** - 100+ lines of disabled constraints
3. **Scale Factor Confusion** - Multiple scales (0.01, 0.02, 1.0, 2.5, 20.0)
4. **No Error Handling** - Silent failures on missing assets

### Medium Priority
1. **Quaternion Math Duplication** - axis_angle_to_quat, _matrix_to_quaternion reimplemented
2. **Transform Recomputation** - Instrument transforms rebuilt every frame
3. **No Configuration** - Parameters hardcoded, no config system
4. **Kernel Complexity** - 10+ parameters per kernel signature
