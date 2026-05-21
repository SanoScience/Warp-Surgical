# SPDX-License-Identifier: Apache-2.0
"""AbdomenAtlas ``combined_labels.nii.gz`` loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..materials import ABDOMEN_ATLAS_CLASS_MAP, load_class_map, pad_labels, remap_labels_to_internal
from ..types import CacheArtifact, PreparedVolume, PreprocessConfig
from ._common import (
    add_crop_metadata,
    apply_visible_class_crop,
    cache_lookup,
    crop_cache_options,
    crop_cache_sources,
    prepare_materials,
    resample_nearest_to_isotropic,
    resolve_target_voxel_mm,
    volume_from_cache,
    write_cache,
)


class AbdomenAtlasLoader:
    name = "abdomen-atlas"

    def can_load(self, path: str | Path | None) -> bool:
        if path is None:
            return False
        p = Path(path)
        return p.name == "combined_labels.nii.gz" or (p.is_dir() and (p / "combined_labels.nii.gz").exists())

    def preprocess(self, config: PreprocessConfig) -> CacheArtifact:
        volume = self.load(config)
        manifest = volume.metadata.get("cache_manifest", {})
        return CacheArtifact(path=Path(manifest.get("path", "")), manifest=manifest)

    def load(self, config: PreprocessConfig) -> PreparedVolume:
        try:
            import nibabel as nib
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("AbdomenAtlas loading requires nibabel>=5") from exc

        source = Path(config.root or "AbdomenAtlas/combined_labels.nii.gz")
        if source.is_dir():
            source = source / "combined_labels.nii.gz"
        if not source.exists():
            raise FileNotFoundError(f"AbdomenAtlas label file not found: {source}")

        img = nib.load(str(source))
        source_spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
        target_voxel_mm = resolve_target_voxel_mm(config, source_spacing)
        class_map = dict(ABDOMEN_ATLAS_CLASS_MAP)
        class_map.update(load_class_map(config.class_map_path))
        _raw_materials, _raw_ui_metadata, _raw_overrides, cache_material_hash = prepare_materials(class_map, config)
        options = {
            "pad": config.pad,
            "target_voxel_mm": target_voxel_mm,
            "source_spacing_mm": source_spacing,
            "class_map_scope": "observed-v1",
            **crop_cache_options(config),
        }
        cache_path, cache_key, cached = cache_lookup(
            config=config,
            dataset=self.name,
            sources=[source] + crop_cache_sources(config),
            options=options,
            material_hash=cache_material_hash,
        )
        if cached is not None:
            labels, texture, manifest = cached
            return volume_from_cache(labels, texture, manifest, config, cache_path=cache_path)

        raw = np.asanyarray(img.dataobj)
        if not np.issubdtype(raw.dtype, np.integer):
            raw = np.rint(raw).astype(np.int32)
        else:
            raw = np.asarray(raw)
        raw = np.ascontiguousarray(raw)
        resampled = resample_nearest_to_isotropic(raw, source_spacing, target_voxel_mm)
        labels, internal_class_map, raw_to_internal = remap_labels_to_internal(resampled, class_map)
        labels, crop = apply_visible_class_crop(labels, internal_class_map, config)
        labels = pad_labels(labels, config.pad)
        materials, ui_metadata, _overrides, material_hash = prepare_materials(internal_class_map, config)

        metadata = {
            "dataset": self.name,
            "scene_label": f"AbdomenAtlas {source.name}",
            **ui_metadata,
        }
        origin = add_crop_metadata(metadata, crop, voxel_size_m=target_voxel_mm * 1.0e-3)
        if config.use_cache:
            artifact = write_cache(
                path=cache_path,
                dataset=self.name,
                cache_key=cache_key,
                class_map=internal_class_map,
                raw_to_internal=raw_to_internal,
                source_shape=tuple(int(v) for v in raw.shape),
                source_spacing_mm=source_spacing,
                target_voxel_mm=target_voxel_mm,
                config=config,
                preprocess_options=options,
                source_files=[source] + crop_cache_sources(config),
                material_hash=material_hash,
                labels=labels,
                texture_rgb=None,
                metadata=metadata,
            )
            metadata["cache_manifest"] = {"path": str(artifact.path), **dict(artifact.manifest)}
        return PreparedVolume(
            labels=labels,
            voxel_size_m=target_voxel_mm * 1.0e-3,
            materials=materials,
            class_map=internal_class_map,
            texture_rgb=None,
            origin=origin,
            metadata=metadata,
        )
