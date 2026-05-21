# SPDX-License-Identifier: Apache-2.0
"""Synthetic block loader for tests and demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..materials import pad_labels, remap_labels_to_internal
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


class SyntheticBlockLoader:
    name = "synthetic"

    def can_load(self, path: str | Path | None) -> bool:
        return path is None or str(path).strip().lower() in {"", "synthetic", "synthetic_block"}

    def preprocess(self, config: PreprocessConfig) -> CacheArtifact:
        volume = self.load(config)
        manifest = volume.metadata.get("cache_manifest", {})
        path = Path(manifest.get("path", ""))
        return CacheArtifact(path=path, manifest=manifest)

    def load(self, config: PreprocessConfig) -> PreparedVolume:
        size = int(config.options.get("size", 8))
        voxel_m = float(config.options.get("voxel_m", 0.005))
        if config.target_voxel_mm is not None:
            voxel_m = float(config.target_voxel_mm) * 1.0e-3
        raw = np.ones((size, size, size), dtype=np.uint8)
        raw_class_map = {0: "background", 1: "synthetic_block"}
        labels, class_map, raw_to_internal = remap_labels_to_internal(raw, raw_class_map)
        labels, crop = apply_visible_class_crop(labels, class_map, config)
        labels = pad_labels(labels, config.pad)
        materials, ui_metadata, _overrides, material_hash = prepare_materials(class_map, config)
        options = {"size": size, "voxel_m": voxel_m, "pad": config.pad, **crop_cache_options(config)}
        cache_path, cache_key, cached = cache_lookup(
            config=config,
            dataset=self.name,
            sources=crop_cache_sources(config),
            options=options,
            material_hash=material_hash,
        )
        if cached is not None:
            cached_labels, cached_texture, manifest = cached
            return volume_from_cache(cached_labels, cached_texture, manifest, config, cache_path=cache_path)

        metadata = {
            "dataset": self.name,
            "scene_label": f"synthetic block {size}^3",
            **ui_metadata,
        }
        origin = add_crop_metadata(metadata, crop, voxel_size_m=voxel_m)
        if config.use_cache:
            artifact = write_cache(
                path=cache_path,
                dataset=self.name,
                cache_key=cache_key,
                class_map=class_map,
                raw_to_internal=raw_to_internal,
                source_shape=tuple(int(v) for v in raw.shape),
                source_spacing_mm=(voxel_m * 1.0e3,) * 3,
                target_voxel_mm=voxel_m * 1.0e3,
                config=config,
                preprocess_options=options,
                source_files=crop_cache_sources(config),
                material_hash=material_hash,
                labels=labels,
                texture_rgb=None,
                metadata=metadata,
            )
            metadata["cache_manifest"] = {"path": str(artifact.path), **dict(artifact.manifest)}
        return PreparedVolume(
            labels=labels,
            voxel_size_m=voxel_m,
            materials=materials,
            class_map=class_map,
            origin=origin,
            metadata=metadata,
        )
