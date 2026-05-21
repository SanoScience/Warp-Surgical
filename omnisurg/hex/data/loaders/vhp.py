# SPDX-License-Identifier: Apache-2.0
"""VHP JPEG segmentation + cryosection loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from omnisurg.hex.io.vhp import (
    VHP_PALETTE,
    _downsample_majority,
    _load_cryo_jpegs,
    _load_slices,
    _match_palette,
    _stack_shape,
)

from ..crop import crop_aligned_texture_rgb
from ..materials import VHP_CLASS_MAP, load_class_map, pad_labels, pad_texture_rgb, remap_labels_to_internal
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


def _parse_vhp_palette_labels(seg_files: list[Path], *, downsample: int, report_every: int = 60) -> np.ndarray:
    palette_rgb = np.asarray([c for c, _ in VHP_PALETTE], dtype=np.int32)
    nx, ny, nz = _stack_shape(seg_files)
    labels_zyx = np.empty((nz, ny, nx), dtype=np.uint8)
    for i, sp in enumerate(seg_files):
        with Image.open(sp) as im:
            seg_rgb = np.asarray(im.convert("RGB"))
        if seg_rgb.shape != (ny, nx, 3):
            raise ValueError(f"slice {i}: seg {seg_rgb.shape} != ({ny},{nx},3)")
        labels_zyx[i] = _match_palette(seg_rgb, palette_rgb).astype(np.uint8)
        if report_every and ((i + 1) % report_every == 0 or i + 1 == nz):
            print(f"  [vhp labels] {i + 1}/{nz}  {sp.name}")
    labels_xyz = np.ascontiguousarray(np.transpose(labels_zyx, (2, 1, 0)))
    return _downsample_majority(labels_xyz, downsample, n_labels=len(VHP_PALETTE))


class VHPLoader:
    name = "vhp"

    def can_load(self, path: str | Path | None) -> bool:
        if path is None:
            return False
        p = Path(path)
        return p.is_dir() and any(p.glob("i*.jpg"))

    def preprocess(self, config: PreprocessConfig) -> CacheArtifact:
        volume = self.load(config)
        manifest = volume.metadata.get("cache_manifest", {})
        return CacheArtifact(path=Path(manifest.get("path", "")), manifest=manifest)

    def load(self, config: PreprocessConfig) -> PreparedVolume:
        root = Path(config.root or "VHP")
        downsample = resolve_downsample(config, native_voxel_mm=1.0, default=16)
        seg_files, cryo_files = _load_slices(root, require_cryo=bool(config.load_texture), verbose=True)
        seg_shape = _stack_shape(seg_files)
        if config.load_texture:
            cryo_shape = _stack_shape(cryo_files)
            if cryo_shape != seg_shape:
                raise ValueError(f"VHP slice dimensions differ: seg {seg_shape} vs cryo {cryo_shape}")

        class_map = dict(VHP_CLASS_MAP)
        class_map.update(load_class_map(config.class_map_path))
        _raw_materials, _raw_ui_metadata, _raw_overrides, cache_material_hash = prepare_materials(class_map, config)
        target_voxel_mm = float(config.target_voxel_mm) if config.target_voxel_mm is not None else float(downsample)
        options = {
            "downsample": downsample,
            "pad": config.pad,
            "target_voxel_mm": target_voxel_mm,
            "load_texture": bool(config.load_texture),
            "class_map_scope": "observed-v1",
            **crop_cache_options(config),
        }
        sources = seg_files + (cryo_files if config.load_texture else []) + crop_cache_sources(config)
        cache_path, cache_key, cached = cache_lookup(
            config=config,
            dataset=self.name,
            sources=sources,
            options=options,
            material_hash=cache_material_hash,
        )
        if cached is not None:
            labels, texture, manifest = cached
            return volume_from_cache(labels, texture, manifest, config, cache_path=cache_path)

        raw_labels = _parse_vhp_palette_labels(
            seg_files,
            downsample=downsample,
            report_every=int(config.options.get("report_every", 60)),
        )
        labels, internal_class_map, raw_to_internal = remap_labels_to_internal(raw_labels, class_map)
        labels, crop = apply_visible_class_crop(labels, internal_class_map, config)
        labels = pad_labels(labels, config.pad)
        materials, ui_metadata, _overrides, material_hash = prepare_materials(internal_class_map, config)
        texture = None
        if config.load_texture:
            texture = _load_cryo_jpegs(
                cryo_files,
                downsample=downsample,
                report_every=int(config.options.get("report_every", 60)),
            )
            texture = crop_aligned_texture_rgb(texture, crop, texture_name="VHP cryo texture")
            texture = pad_texture_rgb(texture, config.pad, scale=downsample)

        metadata = {
            "dataset": self.name,
            "scene_label": f"VHP --downsample {downsample}",
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
                source_shape=tuple(int(v) for v in seg_shape),
                source_spacing_mm=(1.0, 1.0, 1.0),
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
