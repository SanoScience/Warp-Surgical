# Repository Guidelines

## Project Structure & Module Organization
Core simulation code lives in `main.py`, `warp_simulation.py`, and the solver modules (`KRNSolver.py`, `PBDSolver.py`). Rendering backends sit in `render_opengl.py` and `render_surgsim_opengl.py`. Geometry and material resources are organised under `meshes/`, `tetmesh_abdomen/`, `tetmesh_liver/`, and `textures/`, while `assets/` holds HUD/UI artefacts. Place device drivers and physical interfaces in `haptics/`; experimental or retired flows belong in `legacy/`. Keep large binary artefacts out of version control—stage them via Git LFS or reference the shared asset store.

## Build, Test, and Development Commands
Use `uv sync` once per environment to install pinned dependencies, including local editable checkouts of `../warp` and `../newton`. Launch simulations with `uv run python main.py --scene meshes/liver.usd` (swap USD paths as needed). Regenerate solver kernels during debugging with `uv run python collision_kernels.py`. Run the regression suite via `uv run pytest`, and sanity-check haptic connectivity using `uv run python haptic_device.py --probe`.

## Coding Style & Naming Conventions
Target Python 3.12 with 4-space indentation. Follow the prevailing snake_case for functions, PascalCase for classes, and UPPER_CASE for module-level constants. Mirror module naming patterns (`*_kernels.py`, `*_loader.py`) when extending subsystems. Prefer explicit type hints for new public APIs and document side effects in a short Google-style docstring. Keep rendering shaders and asset manifests alongside the modules that load them.

## Testing Guidelines
Extend the `tests/` package (create it if missing) with pytest-based modules named `test_<feature>.py`. Aim for coverage on solver kernels, scene loaders, and device adapters; include at least one numeric tolerance assertion per physics routine. Use `uv run pytest --maxfail=1 -q` before sending changes, and capture sample USD or mesh fixtures under `meshes/tests/` to keep reproducibility tight.

## Commit & Pull Request Guidelines
Match the existing concise, imperative commit style (`"USD loading changes"`, `"Bleeding marching cubes"`). Group related edits into single commits and reference issue IDs in the body when applicable. Pull requests should outline the problem, describe solution highlights, list validation commands, and attach screenshots or short clips when visuals change. Confirm branches are rebased on `main` and that large assets remain untracked before requesting review.
