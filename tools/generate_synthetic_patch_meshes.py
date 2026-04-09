from __future__ import annotations

import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MESH_ROOT = REPO_ROOT / "meshes"

PATCH_BOUNDS = {
    "x_min": -0.75,
    "x_max": 0.75,
    "y": 3.70,
    "z_min": -4.80,
    "z_max": -3.20,
}

RESOLUTIONS = {
    "low": (6, 6),
    "mid": (12, 12),
    "high": (24, 24),
}


def _vertex_index(i: int, j: int, ny: int) -> int:
    return i * (ny + 1) + j


def _generate_vertices(nx: int, ny: int, irregular: bool) -> tuple[list[tuple[float, float, float]], list[tuple[float, float]]]:
    x_min = PATCH_BOUNDS["x_min"]
    x_max = PATCH_BOUNDS["x_max"]
    y = PATCH_BOUNDS["y"]
    z_min = PATCH_BOUNDS["z_min"]
    z_max = PATCH_BOUNDS["z_max"]

    dx = (x_max - x_min) / float(nx)
    dz = (z_max - z_min) / float(ny)

    vertices: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []

    for i in range(nx + 1):
        u = i / float(nx)
        x = x_min + (x_max - x_min) * u
        for j in range(ny + 1):
            v = j / float(ny)
            z = z_min + (z_max - z_min) * v

            if irregular and 0 < i < nx and 0 < j < ny:
                phase = float(i * 17 + j * 11)
                x += dx * 0.22 * math.sin(phase * 0.63)
                z += dz * 0.22 * math.cos(phase * 0.49)

            vertices.append((x, y, z))
            uvs.append((u, v))

    return vertices, uvs


def _generate_tris(nx: int, ny: int, irregular: bool) -> list[tuple[int, int, int]]:
    tris: list[tuple[int, int, int]] = []
    for i in range(nx):
        for j in range(ny):
            v00 = _vertex_index(i, j, ny)
            v10 = _vertex_index(i + 1, j, ny)
            v01 = _vertex_index(i, j + 1, ny)
            v11 = _vertex_index(i + 1, j + 1, ny)

            if irregular and ((i + j) % 2 == 1):
                tris.append((v00, v11, v10))
                tris.append((v00, v01, v11))
            else:
                tris.append((v00, v01, v10))
                tris.append((v10, v01, v11))
    return tris


def _generate_edges(tris: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    edge_set: set[tuple[int, int]] = set()
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (u, v) if u < v else (v, u)
            edge_set.add(edge)
    return sorted(edge_set)


def _write_rows(path: Path, rows: list[tuple[int | float, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(" ".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in row))
            f.write("\n")


def _write_empty(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def generate_patch_asset(name: str, nx: int, ny: int, irregular: bool) -> None:
    vertices, uvs = _generate_vertices(nx, ny, irregular)
    tris = _generate_tris(nx, ny, irregular)
    edges = _generate_edges(tris)

    asset_dir = MESH_ROOT / name
    _write_rows(asset_dir / "model.vertices", vertices)
    _write_rows(asset_dir / "model.edges", edges)
    _write_rows(asset_dir / "model.tris", tris)
    _write_rows(asset_dir / "model.uvs", uvs)
    _write_empty(asset_dir / "model.tetras")


def main() -> None:
    for resolution_name, (nx, ny) in RESOLUTIONS.items():
        generate_patch_asset(f"cloth_regular_{resolution_name}", nx, ny, irregular=False)
        generate_patch_asset(f"cloth_irregular_{resolution_name}", nx, ny, irregular=True)


if __name__ == "__main__":
    main()
