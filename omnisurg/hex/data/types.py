# SPDX-License-Identifier: Apache-2.0
"""Public data contracts for OmniSurg Hex datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from omnisurg.hex.io.digimouse import DigimouseAtlas
from omnisurg.hex.materials import MaterialTable


@dataclass(frozen=True)
class PreparedVolume:
    """A simulation-ready labelled voxel volume.

    ``labels`` index into ``materials`` and must use ``0`` for background.
    ``voxel_size_m`` is isotropic and expressed in metres.
    """

    labels: np.ndarray
    voxel_size_m: float
    materials: MaterialTable
    class_map: Mapping[int, str]
    texture_rgb: np.ndarray | None = None
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = np.ascontiguousarray(self.labels)
        if labels.ndim != 3:
            raise ValueError(f"labels must be a 3D array, got shape={labels.shape}")
        if labels.size and int(labels.min()) < 0:
            raise ValueError("labels must be non-negative material ids")
        if labels.size and int(labels.max()) >= len(self.materials):
            raise ValueError(
                f"labels reference material {int(labels.max())}, but table has {len(self.materials)} entries"
            )
        object.__setattr__(self, "labels", labels)

        texture = self.texture_rgb
        if texture is not None:
            texture = np.ascontiguousarray(texture)
            if texture.ndim != 4 or texture.shape[-1] != 3 or texture.dtype != np.uint8:
                raise ValueError(
                    "texture_rgb must be a uint8 (nx, ny, nz, 3) array, "
                    f"got shape={texture.shape} dtype={texture.dtype}"
                )
            object.__setattr__(self, "texture_rgb", texture)

        if float(self.voxel_size_m) <= 0.0:
            raise ValueError(f"voxel_size_m must be positive, got {self.voxel_size_m}")
        object.__setattr__(self, "voxel_size_m", float(self.voxel_size_m))
        object.__setattr__(self, "origin", tuple(float(v) for v in self.origin))
        object.__setattr__(self, "class_map", {int(k): str(v) for k, v in self.class_map.items()})
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_hex_atlas(self) -> DigimouseAtlas:
        """Return the atlas shape expected by ``build_hex_particle_grid``."""

        return DigimouseAtlas(
            labels=np.ascontiguousarray(self.labels),
            voxel_size=float(self.voxel_size_m),
            materials=self.materials,
            origin=tuple(float(v) for v in self.origin),
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration shared by all dataset loaders."""

    dataset: str
    root: Path | None
    cache_dir: Path | None = None
    downsample: int | None = None
    target_voxel_mm: float | None = None
    pad: int = 1
    use_cache: bool = True
    rebuild_cache: bool = False
    load_texture: bool = True
    texture_path: Path | None = None
    class_map_path: Path | None = None
    material_overrides_path: Path | None = None
    visible_class_crop_settings_path: Path | None = None
    visible_class_crop_margin_voxels: int = 4
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", str(self.dataset))
        object.__setattr__(self, "root", None if self.root is None else Path(self.root))
        object.__setattr__(self, "cache_dir", None if self.cache_dir is None else Path(self.cache_dir))
        object.__setattr__(self, "texture_path", None if self.texture_path is None else Path(self.texture_path))
        object.__setattr__(
            self,
            "class_map_path",
            None if self.class_map_path is None else Path(self.class_map_path),
        )
        object.__setattr__(
            self,
            "material_overrides_path",
            None if self.material_overrides_path is None else Path(self.material_overrides_path),
        )
        object.__setattr__(
            self,
            "visible_class_crop_settings_path",
            None if self.visible_class_crop_settings_path is None else Path(self.visible_class_crop_settings_path),
        )
        if int(self.visible_class_crop_margin_voxels) < 0:
            raise ValueError(
                "visible_class_crop_margin_voxels must be >= 0, "
                f"got {self.visible_class_crop_margin_voxels}"
            )
        object.__setattr__(
            self,
            "visible_class_crop_margin_voxels",
            int(self.visible_class_crop_margin_voxels),
        )
        if self.downsample is not None and int(self.downsample) < 1:
            raise ValueError(f"downsample must be >= 1, got {self.downsample}")
        object.__setattr__(self, "downsample", None if self.downsample is None else int(self.downsample))
        if self.target_voxel_mm is not None and float(self.target_voxel_mm) <= 0.0:
            raise ValueError(f"target_voxel_mm must be positive, got {self.target_voxel_mm}")
        object.__setattr__(self, "pad", max(0, int(self.pad)))
        object.__setattr__(self, "options", dict(self.options))


@dataclass(frozen=True)
class CacheArtifact:
    """A cache bundle produced by a loader preprocess step."""

    path: Path
    manifest: Mapping[str, Any]


class DatasetLoader(Protocol):
    """Protocol implemented by dataset-specific normalizers."""

    name: str

    def can_load(self, path: str | Path | None) -> bool:
        """Return whether this loader can read ``path``."""
        ...

    def preprocess(self, config: PreprocessConfig) -> CacheArtifact:
        """Build or refresh a cache artifact and return its manifest."""
        ...

    def load(self, config: PreprocessConfig) -> PreparedVolume:
        """Load a normalized volume."""
        ...
