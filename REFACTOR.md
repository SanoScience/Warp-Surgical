# OmniSurg Parallel Refactor Plan

## Migration Strategy

This refactor will use a **parallel strangler approach**:

- Keep all current root-level files intact (`warp_simulation.py`, `mesh_loader.py`, `PBDSolver.py`, etc.)
- Build a new implementation gradually under `omnisurg/`
- Port functionality in vertical slices
- Defer deletion/moves of legacy files until an explicit cutover decision

## Scope and Rules

- No in-place moves of legacy modules during migration
- No second project root: keep using the existing repository `pyproject.toml`
- No asset directory move yet (`meshes/`, `textures/` stay where they are)
- Every new module must run against current assets and dependencies
- Backward compatibility of internal imports is not a goal; coexistence is

---

## Current Codebase Snapshot

- Main monolith: `warp_simulation.py` (~2000 lines)
- Core solver: `PBDSolver.py` (+ `simulation_kernels.py`, `collision_kernels.py`)
- Mesh and structs: `mesh_loader.py`
- Tool logic: `grasping.py`, `heating.py`, `stretching.py`, plus inline cutting/clipping in `WarpSim`
- Entrypoint: `main.py`

Main pain points remain the same: oversized `WarpSim`, mixed responsibilities, hard-coded asset paths, and no automated tests.

---

## Target Structure (Parallel Build)

```text
omnisurg/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── config.py
│   ├── state.py
│   ├── simulation.py
│   └── assets.py
├── physics/
│   ├── __init__.py
│   ├── pbd_solver.py
│   ├── simulation_kernels.py
│   └── collision_kernels.py
├── mesh/
│   ├── __init__.py
│   ├── structs.py
│   ├── loader.py
│   └── surface.py
├── tools/
│   ├── __init__.py
│   ├── base.py
│   ├── grasping.py
│   ├── heating.py
│   ├── stretching.py
│   ├── cutting.py
│   └── clipping.py
├── anatomy/
│   ├── __init__.py
│   ├── centrelines.py
│   └── bleeding.py
├── instruments/
│   ├── __init__.py
│   └── loader.py
├── input/
│   ├── __init__.py
│   ├── haptic.py
│   └── keyboard.py
└── rendering/
    ├── __init__.py
    ├── renderer.py
    └── texture_cache.py

examples/
├── cholecystectomy.py
├── haptic_test.py
└── robot_sim.py

tests/
├── test_mesh_loader.py
├── test_physics_solver.py
├── test_tools.py
└── test_smoke_loop.py
```

Note: this is **new code only**. Legacy files stay at repository root during migration.

---

## Step-by-Step Plan (Vertical Slices)

### Phase 0: Baseline and Guardrails
1. Add a lightweight regression harness for legacy behavior (headless where possible).
2. Record baseline metrics for short runs (particle count, active tet count, frame timing, error logs).
3. Define parity tolerances per subsystem (exact vs epsilon-based).

### Phase 1: Scaffold New Package
4. Create `omnisurg/` package structure with `__init__.py` files.
5. Add `omnisurg/core/config.py` for simulation/tool/bleeding configuration dataclasses.
6. Add `omnisurg/core/assets.py` with centralized asset path resolution pointing to existing `meshes/` and `textures/` directories.

### Phase 2: Slice 1 - Mesh Pipeline
7. Create `omnisurg/mesh/structs.py` and port `Tetrahedron` and `TriPointsConnector`.
8. Create `omnisurg/mesh/loader.py` by porting relevant logic from `mesh_loader.py`.
9. Create `omnisurg/mesh/surface.py` by porting `surface_reconstruction.py` logic.
10. Add tests for mesh parsing and surface extraction parity.

### Phase 3: Slice 2 - Physics Stack
11. Create `omnisurg/physics/simulation_kernels.py` by porting kernels from `simulation_kernels.py`.
12. Create `omnisurg/physics/collision_kernels.py` by porting from `collision_kernels.py`.
13. Create `omnisurg/physics/pbd_solver.py` by porting `PBDSolver.py` and updating imports.
14. Keep `KRNSolver.py` unchanged in legacy root; optionally add a new-package counterpart later.
15. Add solver smoke tests and one-step parity checks.

### Phase 4: Slice 3 - First End-to-End Tool Path
16. Create minimal `omnisurg/core/simulation.py` that can step physics + render with one tool path.
17. Port `grasping` into `omnisurg/tools/grasping.py` and wire it through a `SurgicalTool` interface.
18. Validate behavior against legacy for grasp interactions on a fixed scene.

### Phase 5: Slice 4 - Additional Tools and Anatomy
19. Port `heating.py` and `stretching.py` into `omnisurg/tools/`.
20. Extract and port cutting/clipping logic from `WarpSim.render()` into dedicated modules.
21. Port centreline and bleeding logic to `omnisurg/anatomy/`.
22. Port instrument USD loading to `omnisurg/instruments/loader.py`.

### Phase 6: Input and Rendering Abstractions
23. Port haptics to `omnisurg/input/haptic.py`.
24. Add keyboard mapping module and central input dispatch.
25. Add rendering interfaces (`renderer.py`, `texture_cache.py`) and wire into new simulation.

### Phase 7: New Entrypoint
26. Create `examples/cholecystectomy.py` using only `omnisurg/*` modules.
27. Add smoke tests for startup and short simulation runs.
28. Keep `main.py` and all legacy scripts untouched.

### Phase 8: Cutover (Explicit, Later Decision)
29. Compare legacy vs new metrics over longer scripted scenarios.
30. Decide on cutover readiness.
31. Only after explicit approval: archive/remove legacy root modules and relocate assets if desired.

---

## Definition of Done per Slice

Each migration slice is complete only when all conditions pass:

- New module compiles/imports without relying on moved legacy files
- Legacy files remain unchanged and runnable
- Slice-specific tests pass
- A short parity report is captured against the legacy baseline
- Known deviations are documented with rationale

---

## Design Principles

- Composition over monoliths
- Configuration as data (dataclasses)
- Explicit asset resolution (no scattered hard-coded paths)
- Small, testable modules
- Incremental parity before feature expansion

---

## Out of Scope (for now)

- Moving `meshes/` and `textures/` to `assets/`
- Deleting legacy root modules
- Reorganizing `legacy/` directory contents
- Broad performance optimization passes before parity
