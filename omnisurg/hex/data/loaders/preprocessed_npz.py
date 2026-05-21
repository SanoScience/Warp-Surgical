# SPDX-License-Identifier: Apache-2.0
"""Loader for already-preprocessed ``.npz`` volumes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..crop import crop_aligned_texture_rgb
from ..materials import load_class_map, pad_labels, pad_texture_rgb, remap_labels_to_internal
from ..types import CacheArtifact, PreparedVolume, PreprocessConfig
from ._common import (
    add_crop_metadata,
    apply_visible_class_crop,
    cache_lookup,
    crop_cache_options,
    crop_cache_sources,
    prepare_materials,
    volume_from_cache,
    write_cache,
)


def _manifest_from_npz(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "manifest" not in data.files:
        return {}
    raw = data["manifest"].item()
    return json.loads(str(raw))


def _load_texture_path(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        texture = np.load(path)
    else:
        with np.load(path, allow_pickle=False) as data:
            key = "texture_rgb" if "texture_rgb" in data.files else "texture"
            texture = data[key]
    texture = np.ascontiguousarray(texture)
    if texture.ndim != 4 or texture.shape[-1] != 3 or texture.dtype != np.uint8:
        raise ValueError(f"texture must be uint8 (nx, ny, nz, 3), got {texture.shape} {texture.dtype}")
    return texture


class PreprocessedNpzLoader:
    name = "preprocessed"

    def can_load(self, path: str | Path | None) -> bool:
        return path is not None and Path(path).suffix.lower() == ".npz"

    def preprocess(self, config: PreprocessConfig) -> CacheArtifact:
        volume = self.load(config)
        manifest = volume.metadata.get("cache_manifest", {})
        return CacheArtifact(path=Path(manifest.get("path", config.root or "")), manifest=manifest)

    def load(self, config: PreprocessConfig) -> PreparedVolume:
        if config.root is None:
            raise ValueError("--data is required for the preprocessed loader")
        source = Path(config.root)
        with np.load(source, allow_pickle=False) as data:
            if "labels" not in data.files:
                raise ValueError(f"{source} does not contain a labels array")
            raw_labels = np.ascontiguousarray(data["labels"])
            manifest = _manifest_from_npz(data)
            texture = np.ascontiguousarray(data["texture_rgb"]) if "texture_rgb" in data.files else None
            if texture is None and "texture" in data.files:
                texture = np.ascontiguousarray(data["texture"])
            voxel_size_m = float(data["voxel_size_m"]) if "voxel_size_m" in data.files else None

        if config.texture_path is not None:
            texture = _load_texture_path(config.texture_path)
        elif not config.load_texture:
            texture = None

        file_class_map: dict[int, str] = {}
        if "class_map" in manifest:
            file_class_map = {int(k): str(v) for k, v in manifest["class_map"].items()}
        file_class_map.update(load_class_map(config.class_map_path))
        labels, class_map, raw_to_internal = remap_labels_to_internal(raw_labels, file_class_map)
        labels, crop = apply_visible_class_crop(labels, class_map, config)
        texture = crop_aligned_texture_rgb(texture, crop, texture_name="preprocessed texture")
        labels = pad_labels(labels, config.pad)
        texture = pad_texture_rgb(texture, config.pad)

        target_voxel_mm = (
            float(config.target_voxel_mm)
            if config.target_voxel_mm is not None
            else float(manifest.get("target_voxel_mm", (voxel_size_m or 0.005) * 1.0e3))
        )
        voxel_size_m = target_voxel_mm * 1.0e-3
        materials, ui_metadata, _overrides, material_hash = prepare_materials(class_map, config)
        options = {
            "source": str(source.resolve()),
            "pad": config.pad,
            "target_voxel_mm": target_voxel_mm,
            "texture": str(config.texture_path) if config.texture_path is not None else bool(texture is not None),
            **crop_cache_options(config),
        }
        source_files = [source] + ([] if config.texture_path is None else [config.texture_path]) + crop_cache_sources(config)
        cache_path, cache_key, cached = cache_lookup(
            config=config,
            dataset=self.name,
            sources=source_files,
            options=options,
            material_hash=material_hash,
        )
        if cached is not None:
            cached_labels, cached_texture, cached_manifest = cached
            return volume_from_cache(cached_labels, cached_texture, cached_manifest, config, cache_path=cache_path)

        metadata = {
            "dataset": self.name,
            "scene_label": source.name,
            **ui_metadata,
        }
        origin = add_crop_metadata(metadata, crop, voxel_size_m=voxel_size_m)
        if config.use_cache:
            artifact = write_cache(
                path=cache_path,
                dataset=self.name,
                cache_key=cache_key,
                class_map=class_map,
                raw_to_internal=raw_to_internal,
                source_shape=tuple(int(v) for v in raw_labels.shape),
                source_spacing_mm=None,
                target_voxel_mm=target_voxel_mm,
                config=config,
                preprocess_options=options,
                source_files=source_files,
                material_hash=material_hash,
                labels=labels,
                texture_rgb=texture,
                metadata=metadata,
            )
            metadata["cache_manifest"] = {"path": str(artifact.path), **dict(artifact.manifest)}
        return PreparedVolume(
            labels=labels,
            voxel_size_m=voxel_size_m,
            materials=materials,
            class_map=class_map,
            texture_rgb=texture,
            origin=origin,
            metadata=metadata,
        )
