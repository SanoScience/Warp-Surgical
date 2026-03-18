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
