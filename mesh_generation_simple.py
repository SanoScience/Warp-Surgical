"""Utilities to generate Warp-format meshes for the surgical simulation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

VIEWER_ROOT = Path(__file__).resolve().parent.parent / "warp-cgal" / "viewer"
if VIEWER_ROOT.exists():
    viewer_path = str(VIEWER_ROOT)
    if viewer_path not in sys.path:
        sys.path.insert(0, viewer_path)

try:
    from simulation_bridge import (
        CGALUnavailableError,
        detect_image_labels,
        generate_mesh_from_image,
    )

    VIEWER_AVAILABLE = True
except ImportError:
    CGALUnavailableError = RuntimeError  # type: ignore[assignment]
    detect_image_labels = None  # type: ignore[assignment]
    generate_mesh_from_image = None  # type: ignore[assignment]
    VIEWER_AVAILABLE = False


def _resolve_image_path(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.exists():
        return candidate
    if VIEWER_ROOT.exists():
        fallback = VIEWER_ROOT.parent / candidate.name
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Could not find image file: {candidate}")


class SimpleMeshGenerator:
    """Thin wrapper around the warp-cgal viewer integration APIs."""

    def __init__(self) -> None:
        self.warp_cgal_available = VIEWER_AVAILABLE

    def _ensure_available(self) -> None:
        if not self.warp_cgal_available or detect_image_labels is None or generate_mesh_from_image is None:
            raise CGALUnavailableError("warp-cgal viewer integration is not available")

    def detect_labels(self, image_path: str | Path) -> List[int]:
        """Return segmentation labels available in the medical volume."""
        self._ensure_available()
        resolved = _resolve_image_path(Path(image_path))
        return list(detect_image_labels(str(resolved)))  # type: ignore[misc]

    def generate(
        self,
        image_path: Path,
        output_path: Path,
        *,
        quality: str,
        cell_size: Optional[float],
        multi_label: bool,
        force: bool,
    ) -> dict:
        """Generate Warp-format mesh data from a medical image."""
        self._ensure_available()
        kwargs = {
            "quality": quality,
            "multi_label": multi_label,
            "force": force,
        }
        if cell_size is not None:
            kwargs["cell_size"] = cell_size
        return generate_mesh_from_image(  # type: ignore[misc]
            str(image_path),
            str(output_path),
            **kwargs,
        )


def ensure_meshes_ready(args) -> Optional[str]:
    """Generate meshes when requested, returning the path to the Warp mesh folder."""
    generator = SimpleMeshGenerator()
    if not generator.warp_cgal_available:
        print("warp-cgal integration unavailable; proceeding with existing meshes")
        return None

    source_arg = getattr(args, "mesh_source", "liver-sliced.inr")
    try:
        source_path = _resolve_image_path(Path(source_arg))
    except FileNotFoundError as exc:
        print(f"Mesh generation skipped: {exc}")
        return None

    output_root = Path(__file__).resolve().parent / "meshes" / "generated"
    output_root.mkdir(parents=True, exist_ok=True)

    base_name = source_path.stem.replace(" ", "_")
    quality_tag = getattr(args, "mesh_quality", "balanced")
    output_base = output_root / f"{base_name}_{quality_tag}"

    result = generator.generate(
        image_path=source_path,
        output_path=output_base,
        quality=quality_tag,
        cell_size=getattr(args, "cell_size", None),
        multi_label=getattr(args, "multi_label", False),
        force=getattr(args, "force_regenerate", False),
    )

    output_paths = result.get("output_paths", [])
    stats = result.get("statistics")
    if output_paths:
        print(f"Generated Warp mesh assets: {output_paths}")
    if stats is not None:
        print(f"Mesh statistics: {stats}")

    return str(output_base)
