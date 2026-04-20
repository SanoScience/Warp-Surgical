from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np

from omnisurg.mesh.assets import TetMeshAsset, load_tet_asset

REQUIRED_MESH_FILES = (
    "model.vertices",
    "model.tetras",
    "model.edges",
    "model.tris",
)
OPTIONAL_MESH_FILES = ("model.uvs",)


@dataclass(frozen=True)
class VtkExportResult:
    primary_output: Path
    surface_output: Path | None
    mode: str


def load_tet_asset_from_dir(asset_dir: str | Path) -> TetMeshAsset:
    """Load a tet asset directly from a mesh folder such as `meshes/liver`."""

    asset_path = Path(asset_dir)
    if not asset_path.exists():
        raise FileNotFoundError(f"Mesh folder does not exist: {asset_path}")
    if not asset_path.is_dir():
        raise NotADirectoryError(f"Expected a mesh folder, got: {asset_path}")

    missing = [name for name in REQUIRED_MESH_FILES if not (asset_path / name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Mesh folder is missing required files: {missing_text}")

    return load_tet_asset(asset_path.name, mesh_dir=str(asset_path.parent))


def _build_point_data(asset: TetMeshAsset) -> dict[str, np.ndarray]:
    point_data: dict[str, np.ndarray] = {}
    if asset.uvs is not None:
        point_data["uv"] = np.asarray(asset.uvs, dtype=np.float32)
    return point_data


def build_volume_mesh(
    asset: TetMeshAsset,
    *,
    include_surface_cells: bool = False,
    include_edge_cells: bool = False,
) -> meshio.Mesh:
    """Build an unstructured VTK mesh from the volumetric asset."""

    cells: list[tuple[str, np.ndarray]] = []
    if asset.tet_indices.size:
        cells.append(("tetra", np.asarray(asset.tet_indices, dtype=np.int32)))
    if include_surface_cells and asset.surface_tri_indices.size:
        cells.append(("triangle", np.asarray(asset.surface_tri_indices, dtype=np.int32)))
    if include_edge_cells and asset.edge_indices.size:
        cells.append(("line", np.asarray(asset.edge_indices, dtype=np.int32)))

    if not cells:
        raise ValueError("Asset does not contain tetrahedra; use surface mode instead.")

    return meshio.Mesh(
        points=np.asarray(asset.rest_positions, dtype=np.float32),
        cells=cells,
        point_data=_build_point_data(asset),
    )


def build_surface_mesh(asset: TetMeshAsset) -> meshio.Mesh:
    """Build a polygonal VTK mesh from the asset surface."""

    cells: list[tuple[str, np.ndarray]] = []
    if asset.surface_tri_indices.size:
        cells.append(("triangle", np.asarray(asset.surface_tri_indices, dtype=np.int32)))
    elif asset.edge_indices.size:
        cells.append(("line", np.asarray(asset.edge_indices, dtype=np.int32)))
    else:
        raise ValueError("Asset does not contain surface triangles or edges to export.")

    return meshio.Mesh(
        points=np.asarray(asset.rest_positions, dtype=np.float32),
        cells=cells,
        point_data=_build_point_data(asset),
    )


def _resolve_primary_mode(asset: TetMeshAsset, mode: str) -> str:
    if mode == "auto":
        return "volume" if asset.tet_indices.size else "surface"
    if mode == "volume" and not asset.tet_indices.size:
        raise ValueError("Volume mode requires tetrahedra, but the asset has none.")
    if mode not in {"volume", "surface"}:
        raise ValueError(f"Unsupported export mode: {mode}")
    return mode


def _default_primary_output(asset_dir: Path, mode: str) -> Path:
    suffix = ".vtu" if mode == "volume" else ".vtp"
    return asset_dir.parent / f"{asset_dir.name}{suffix}"


def _default_surface_output(asset_dir: Path) -> Path:
    return asset_dir.parent / f"{asset_dir.name}_surface.vtp"


def export_asset_dir_to_vtk(
    asset_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    mode: str = "auto",
    write_surface: bool = False,
    surface_output_path: str | Path | None = None,
    include_surface_cells: bool = False,
    include_edge_cells: bool = False,
    binary: bool = True,
) -> VtkExportResult:
    """Export a mesh folder like `meshes/liver` to VTK using meshio."""

    asset_path = Path(asset_dir)
    asset = load_tet_asset_from_dir(asset_path)
    primary_mode = _resolve_primary_mode(asset, mode)

    primary_output = Path(output_path) if output_path is not None else _default_primary_output(asset_path, primary_mode)
    primary_output.parent.mkdir(parents=True, exist_ok=True)

    if primary_mode == "volume":
        mesh = build_volume_mesh(
            asset,
            include_surface_cells=include_surface_cells,
            include_edge_cells=include_edge_cells,
        )
    else:
        mesh = build_surface_mesh(asset)

    meshio.write(primary_output, mesh, binary=binary)

    resolved_surface_output: Path | None = None
    if primary_mode != "surface" and (write_surface or surface_output_path is not None):
        surface_mesh = build_surface_mesh(asset)
        resolved_surface_output = (
            Path(surface_output_path)
            if surface_output_path is not None
            else _default_surface_output(asset_path)
        )
        resolved_surface_output.parent.mkdir(parents=True, exist_ok=True)
        meshio.write(resolved_surface_output, surface_mesh, binary=binary)

    return VtkExportResult(
        primary_output=primary_output,
        surface_output=resolved_surface_output,
        mode=primary_mode,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an OmniSurg anatomy mesh folder into VTK using meshio.",
    )
    parser.add_argument(
        "asset_dir",
        help="Path to a mesh folder such as meshes/liver.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Primary VTK output path. Defaults to <asset_dir>.vtu or <asset_dir>.vtp.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "volume", "surface"),
        default="auto",
        help="Export tetra volume data, surface triangles, or choose automatically.",
    )
    parser.add_argument(
        "--write-surface",
        action="store_true",
        help="Also write a surface `.vtp` file when the primary output is volumetric.",
    )
    parser.add_argument(
        "--surface-output",
        help="Optional path for the surface `.vtp` sidecar file.",
    )
    parser.add_argument(
        "--include-surface-cells",
        action="store_true",
        help="Embed surface triangles into the primary volumetric output as triangle cells.",
    )
    parser.add_argument(
        "--include-edge-cells",
        action="store_true",
        help="Embed spring edges into the primary volumetric output as line cells.",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Write ASCII VTK instead of binary where supported by the format.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = export_asset_dir_to_vtk(
        args.asset_dir,
        output_path=args.output,
        mode=args.mode,
        write_surface=args.write_surface,
        surface_output_path=args.surface_output,
        include_surface_cells=args.include_surface_cells,
        include_edge_cells=args.include_edge_cells,
        binary=not args.ascii,
    )
    print(f"Wrote {result.mode} VTK to {result.primary_output}")
    if result.surface_output is not None:
        print(f"Wrote surface VTK to {result.surface_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
