# Progress Log

## 2026-03-18
- Started CPU/GPU performance analysis for `omnisurg`.
- Confirmed no active long-running terminal session for the app.
- Attempted planning session catchup script; path was unavailable.
- Created `task_plan.md`, `findings.md`, and `progress.md`.
- Confirmed Nsight Systems install and located CLI at `target-windows-x64/nsys.exe`.
- Extracted stats from existing `omnisurg_profile.nsys-rep` and `omnisurg_profile_opt.nsys-rep`.
- Collected fresh current-code captures for `--viewer surgsim` and `--viewer gl` with `--num_frames 200`.
- Collected an additional current `gl` capture with `--cuda-graph-trace=node` and `--num_frames 60` to expose internal solver kernels hidden by default graph tracing.
