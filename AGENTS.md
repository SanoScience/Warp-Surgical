# Repository Guidelines

## Project Structure & Module Organization
- Python sources live at repo root (e.g., `main.py`, `warp_simulation.py`, `render_opengl.py`).
- Demos: `main.py` (full) and `main_simple.py` (minimal). Haptics in `haptics/`.
- Assets: meshes in `meshes/`, textures in `textures/`, robots in `assets/Robots/`, tetrahedral sets in `tetmesh_*`.
- Legacy experiments in `legacy/`. Keep new work outside unless clearly experimental.

## Build, Test, and Development Commands
- Create env + install deps (UV): `uv sync`
- Run demos:
  - Full: `uv run python main.py`
  - Minimal: `uv run python main_simple.py`
  - Haptics: `uv run python haptics/haptic_game.py`
- Ad‑hoc run without UV (if venv active): `python main.py`
- Note: `pyproject.toml` targets Python 3.12 and uses NVIDIA’s extra index for `warp-lang`.

## Coding Style & Naming Conventions
- Python style: PEP 8, 4‑space indentation, 120‑col soft limit.
- Names: `snake_case` for functions/modules, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Prefer type hints and module‑level docstrings. Keep functions small and GPU kernels isolated in `*_kernels.py`.
- If using formatters, prefer `ruff` and `black` (keep diffs minimal if not configured).

## Testing Guidelines
- Tests are currently minimal. Add `tests/` with `test_*.py` using `pytest`.
- Run (once tests exist): `uv run pytest -q` and aim for coverage on new/changed code.
- Provide small mesh fixtures under `meshes/simple_test/` for deterministic tests.

## Commit & Pull Request Guidelines
- Commits: concise imperative subject (≤72 chars). Body explains motivation and approach.
  - Example: `mesh: add loader for CGAL tetra sets`.
- PRs must include: summary, rationale, before/after notes or screenshots (if rendering), and any performance impact.
- Link related issues. Keep PRs focused; avoid asset bloat—use Git LFS or external links for large binaries.

## Security & Configuration Tips
- Large assets: do not commit new heavy textures/meshes without LFS.
- Local dev sources: `tool.uv.sources` may point to local `newton`. Keep paths valid or comment them when not used.
- GPU/driver: NVIDIA Warp requires a compatible GPU/driver; fall back to CPU for headless tests.
