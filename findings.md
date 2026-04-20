# Findings

## Context
- Previous source review showed CUDA graph capture is enabled in `omnisurg/runtime.py`.
- Existing checked-in Nsight artifacts mention both solver kernels and viewer kernels, so rendering is expected to be part of the GPU critical path.
- Nsight Systems CLI is available at `C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.5.2\target-windows-x64\nsys.exe`.
- Existing profile artifacts available in the repo: `omnisurg_profile.nsys-rep` and `omnisurg_profile_opt.nsys-rep`.
- Fresh current-code captures created: `current_omnisurg_gl.nsys-rep`, `current_omnisurg_surgsim.nsys-rep`, and `current_omnisurg_gl_nodes.nsys-rep`.

## Questions To Answer
- How much time is spent in CUDA API submission vs GPU kernel execution?
- Which kernels dominate GPU time?
- How much of the frame cost comes from the viewer backend?
- Is the app more launch-bound, render-bound, or compute-bound?

## Measured Results
- Default Nsight graph tracing (`--cuda-graph-trace=graph`) hides internal solver graph nodes and mostly shows `cudaGraphLaunch` plus render-related kernels.
- A focused node-level recapture (`--cuda-graph-trace=node`) exposed the solver kernels for current `gl`.
- Current `gl` node-level capture shows GPU kernel time dominated by solver work:
  - `solve_volume_constraints`: 7.89 ms total over 60 frames
  - `apply_deltas_and_zero_accumulators`: 5.27 ms total
  - `solve_distance_constraints`: 4.81 ms total
  - `apply_particle_deltas`: 2.29 ms total
  - `collide_triangles_vs_sphere`: 2.11 ms total
  - `integrate_particles`: 2.00 ms total
- In the same node-level capture, render/update CUDA kernels are small by comparison:
  - `accumulate_vertex_normals`: 0.22 ms total over 60 frames
  - `fill_vertex_data`: 0.13 ms total
  - `update_vbo_transforms`: 0.13 ms total
  - `update_vbo_transforms_from_points`: 0.11 ms total
  - `normalize_vertex_normals`: 0.09 ms total
- Current `surgsim` capture shows CPU/API time dominated by CUDA-GL interop:
  - `cuGraphicsMapResources`: 214.0 ms total over 200 frames
  - `cudaMemcpyAsync`: 88.3 ms total
  - `cudaGraphLaunch_v10000`: 80.7 ms total
  - `cuGraphicsUnmapResources`: 18.6 ms total
- Current `gl` capture shows CPU/API time dominated by graph launch and copies:
  - `cudaMemcpyAsync`: 131.1 ms total over 200 frames
  - `cudaGraphLaunch_v10000`: 87.2 ms total
  - `cuLaunchKernel`: 36.1 ms total
- Old checked-in non-graph-heavy baseline had much larger CPU API totals from launch/memset activity, so the current CUDA graph path materially reduced submission overhead.

## 2026-04-10 Anatomy Mesh VTK Converter
- Anatomy asset folders such as `meshes/liver/` contain `model.vertices`, `model.tetras`, `model.tris`, `model.edges`, and `model.uvs`.
- Sample liver counts implied by file lengths:
  - vertices: 1987 rows of XYZ positions
  - uvs: 1987 rows of UV coordinates
  - tris: 3939 rows of triangle connectivity
  - tetras: 6624 rows of tetra connectivity
- Connectivity appears to be 0-based already, which matches VTK expectations from the VTK file-format notes.
- The VTK reference indicates volumetric anatomy should map naturally to `UnstructuredGrid` (`.vtu`) and surface-only exports to `PolyData` (`.vtp`).
- Existing repo loader logic likely lives in `omnisurg/mesh/assets.py` and `mesh_loader.py`, so the converter should mirror their parsing and optional-file behavior instead of inventing a new interpretation.
