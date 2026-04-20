# OmniSurg CPU/GPU Performance Analysis

## Goal
Measure and analyze the CPU/GPU performance of `omnisurg` using Nsight Systems, then identify the main bottlenecks and actionable optimization targets.

## Phases
| Phase | Status | Notes |
|---|---|---|
| 1. Prepare profiling setup | complete | Confirmed `nsys.exe` at `target-windows-x64/nsys.exe` and existing profile artifacts in repo |
| 2. Capture representative runs | complete | Collected fresh current-code captures for `gl`, `surgsim`, and a node-level CUDA graph run |
| 3. Extract CPU/GPU stats | complete | Extracted `cuda_api_sum`, `cuda_gpu_kern_sum`, and memory summaries from saved and fresh profiles |
| 4. Compare simulation vs render costs | complete | Current data shows solver GPU kernels are small; CPU/API + graphics interop dominate |
| 5. Summarize findings | complete | Ready to deliver prioritized bottlenecks and recommendations |

## Errors Encountered
| Error | Attempt | Resolution |
|---|---|---|
| Planning helper script not found at `C:\Users\korze\.claude\skills\planning-with-files\scripts\session-catchup.py` | 1 | Create local planning files directly and continue |
| Fresh `nsys` capture failed due to PowerShell executable quoting | 1 | Re-ran with PowerShell call operator `&` and completed capture successfully |

## 2026-04-10 Anatomy Mesh VTK Converter

### Goal
Develop a Python converter that reads OmniSurg anatomy asset folders such as `meshes/liver/` and writes VTK output using `meshio`.

### Phases
| Phase | Status | Notes |
|---|---|---|
| 1. Inspect mesh bundle format | complete | Confirmed `model.vertices`, `model.tetras`, `model.tris`, `model.edges`, and `model.uvs` with 0-based indexing |
| 2. Inspect existing loader behavior | in_progress | Need to mirror repo parsing details and optional file handling |
| 3. Implement converter | pending | Add a reusable Python CLI/script for folder-to-VTK export |
| 4. Verify on liver asset | pending | Run converter and confirm a `.vtu` file is produced |
| 5. Check tests/lints | pending | Run focused validation and inspect diagnostics |
