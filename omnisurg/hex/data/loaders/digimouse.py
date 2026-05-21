# SPDX-License-Identifier: Apache-2.0
"""Digimouse Analyze atlas loader for OmniSurg Hex."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from omnisurg.hex.io.digimouse import _downsample_majority, _read_analyze_header

from ..crop import crop_aligned_texture_rgb
from ..materials import (
    DIGIMOUSE_CLASS_MAP,
    DIGIMOUSE_MATERIAL_HINTS,
    load_class_map,
    pad_labels,
    remap_labels_to_internal,
)
from ..types import CacheArtifact, PreparedVolume, PreprocessConfig
from ._common import (
    add_crop_metadata,
    apply_visible_class_crop,
    cache_lookup,
    crop_cache_options,
    crop_cache_sources,
    prepare_materials,
    resolve_downsample,
    volume_from_cache,
    write_cache,
)


def _read_digimouse_class_map(path: Path) -> dict[int, str]:
    txt = path / "atlas_380x992x208.txt"
    if not txt.exists():
        return dict(DIGIMOUSE_CLASS_MAP)
    out = dict(DIGIMOUSE_CLASS_MAP)
    for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "-->" not in line or "+" in line:
            continue
        raw, name = line.split("-->", 1)
        try:
            out[int(raw.strip())] = name.strip().lower().replace(" ", "_")
        except ValueError:
            continue
    out.setdefault(0, "background")
    return out


def _load_texture(path: Path) -> np.ndarray:
    texture = np.load(path, mmap_mode=None)
    texture = np.ascontiguousarray(texture)
    if texture.ndim != 4 or texture.shape[-1] != 3 or texture.dtype != np.uint8:
        raise ValueError(f"Digimouse texture must be uint8 (nx, ny, nz, 3), got {texture.shape} {texture.dtype}")
    return texture


class DigimouseLoader:
    name = "digimouse"

    def can_load(self, path: str | Path | None) -> bool:
        if path is None:
            return False
        p = Path(path)
        return (p / "atlas_380x992x208.hdr").exists() or p.name.lower().endswith(".hdr")

    def preprocess(self, config: PreprocessConfig) -> CacheArtifact:
        volume = self.load(config)
        manifest = volume.metadata.get("cache_manifest", {})
        return CacheArtifact(path=Path(manifest.get("path", "")), manifest=manifest)

    def load(self, config: PreprocessConfig) -> PreparedVolume:
        atlas_dir = Path(config.root or "Digimouse/atlas/atlas")
        hdr = atlas_dir if atlas_dir.suffix.lower() == ".hdr" else atlas_dir / "atlas_380x992x208.hdr"
        atlas_dir = hdr.parent
        img = atlas_dir / "atlas_380x992x208.img"
        if not hdr.exists() or not img.exists():
            raise FileNotFoundError(f"Digimouse atlas not found under {atlas_dir}")

        (nx, ny, nz), spacing_mm, datatype = _read_analyze_header(hdr)
        if datatype != 2:
            raise ValueError(f"expected uint8 Analyze atlas (datatype=2), got datatype={datatype}")
        native_mm = float(spacing_mm[0])
        if max(abs(float(s) - native_mm) for s in spacing_mm) > 1.0e-5:
            raise ValueError(f"Digimouse loader expects isotropic voxels, got {spacing_mm}")
        downsample = resolve_downsample(config, native_mm, default=16)
        target_voxel_mm = native_mm * downsample

        class_map = _read_digimouse_class_map(atlas_dir)
        class_map.update(load_class_map(config.class_map_path))
        _raw_materials, _raw_ui_metadata, _raw_overrides, cache_material_hash = prepare_materials(
            class_map,
            config,
            defaults_by_name=DIGIMOUSE_MATERIAL_HINTS,
        )
        options = {
            "downsample": downsample,
            "pad": config.pad,
            "target_voxel_mm": target_voxel_mm,
            "load_texture": bool(config.load_texture),
            "texture_path": str(config.texture_path) if config.texture_path is not None else "auto",
            "class_map_scope": "observed-v1",
            **crop_cache_options(config),
        }
        texture_source = _resolve_texture_path(config, atlas_dir)
        sources = [hdr, img] + ([] if texture_source is None else [texture_source]) + crop_cache_sources(config)
        cache_path, cache_key, cached = cache_lookup(
            config=config,
            dataset=self.name,
            sources=sources,
            options=options,
            material_hash=cache_material_hash,
        )
        if cached is not None:
            labels, texture, manifest = cached
            return volume_from_cache(
                labels,
                texture,
                manifest,
                config,
                defaults_by_name=DIGIMOUSE_MATERIAL_HINTS,
                cache_path=cache_path,
            )

        raw = np.fromfile(img, dtype=np.uint8)
        if raw.size != nx * ny * nz:
            raise ValueError(f"atlas .img size {raw.size} != nx*ny*nz = {nx * ny * nz}")
        atlas = raw.reshape((nz, ny, nx)).transpose(2, 1, 0)
        downsampled_raw = _downsample_majority(atlas, downsample, n_labels=max(class_map) + 1)
        labels, internal_class_map, raw_to_internal = remap_labels_to_internal(downsampled_raw, class_map)
        labels, crop = apply_visible_class_crop(labels, internal_class_map, config)
        labels = pad_labels(labels, config.pad)
        texture = _load_texture(texture_source) if texture_source is not None else None
        texture = crop_aligned_texture_rgb(texture, crop, texture_name="Digimouse texture")
        materials, ui_metadata, _overrides, material_hash = prepare_materials(
            internal_class_map,
            config,
            defaults_by_name=DIGIMOUSE_MATERIAL_HINTS,
        )

        metadata = {
            "dataset": self.name,
            "scene_label": f"Digimouse --downsample {downsample}",
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
                source_shape=(nx, ny, nz),
                source_spacing_mm=tuple(float(v) for v in spacing_mm),
                target_voxel_mm=target_voxel_mm,
                config=config,
                preprocess_options=options,
                source_files=sources,
                material_hash=material_hash,
                labels=labels,
                texture_rgb=texture,
                metadata=metadata,
            )
            metadata["cache_manifest"] = {"path": str(artifact.path), **dict(artifact.manifest)}
        return PreparedVolume(
            labels=labels,
            voxel_size_m=target_voxel_mm * 1.0e-3,
            materials=materials,
            class_map=internal_class_map,
            texture_rgb=texture,
            origin=origin,
            metadata=metadata,
        )


def _resolve_texture_path(config: PreprocessConfig, atlas_dir: Path) -> Path | None:
    if not config.load_texture:
        return None
    if config.texture_path is not None:
        return config.texture_path
    candidates = [
        atlas_dir.parent.parent / "cryo_texture.npy",
        Path("Digimouse") / "cryo_texture.npy",
        Path("Digimouse") / "cryo_texture_half.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
