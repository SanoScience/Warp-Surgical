# SPDX-License-Identifier: Apache-2.0
"""Shared loader helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..cache import (
    build_manifest,
    cache_path_for,
    default_cache_dir,
    make_cache_key,
    try_read_valid_cache,
    write_npz_atomic,
)
from ..crop import (
    VisibleClassCrop,
    crop_labels_to_visible_classes,
    visible_class_crop_options,
    visible_class_crop_sources,
)
from ..materials import generate_material_table, load_material_overrides, material_defaults_hash
from ..types import CacheArtifact, PreparedVolume, PreprocessConfig


def resolve_downsample(config: PreprocessConfig, native_voxel_mm: float, default: int) -> int:
    if config.downsample is not None:
        return max(1, int(config.downsample))
    if config.target_voxel_mm is not None:
        return max(1, int(round(float(config.target_voxel_mm) / float(native_voxel_mm))))
    return max(1, int(default))


def resolve_target_voxel_mm(config: PreprocessConfig, source_spacing_mm: tuple[float, float, float]) -> float:
    if config.target_voxel_mm is not None:
        return float(config.target_voxel_mm)
    return float(max(source_spacing_mm))


def resample_nearest_to_isotropic(
    labels: np.ndarray,
    spacing_mm: tuple[float, float, float],
    target_mm: float,
) -> np.ndarray:
    """Nearest-neighbor voxel-center resampling for categorical volumes."""

    spacing = np.asarray(spacing_mm, dtype=np.float64)
    target = float(target_mm)
    if np.allclose(spacing, target, rtol=0.0, atol=1.0e-6):
        return np.ascontiguousarray(labels)
    shape = np.asarray(labels.shape[:3], dtype=np.int64)
    extent = shape.astype(np.float64) * spacing
    out_shape = np.maximum(1, np.ceil(extent / target).astype(np.int64))
    axes: list[np.ndarray] = []
    for axis in range(3):
        centers = (np.arange(out_shape[axis], dtype=np.float64) + 0.5) * target
        src = np.rint((centers / spacing[axis]) - 0.5).astype(np.int64)
        axes.append(np.clip(src, 0, shape[axis] - 1))
    if labels.ndim == 3:
        return np.ascontiguousarray(labels[np.ix_(axes[0], axes[1], axes[2])])
    if labels.ndim == 4:
        return np.ascontiguousarray(labels[np.ix_(axes[0], axes[1], axes[2], np.arange(labels.shape[3]))])
    raise ValueError(f"expected a 3D or 4D volume, got shape={labels.shape}")


def prepare_materials(
    class_map: Mapping[int, str],
    config: PreprocessConfig,
    *,
    defaults_by_name: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Any, dict[str, list[Any]], dict[str, dict[str, Any]], str]:
    overrides = load_material_overrides(config.material_overrides_path)
    material_hash = material_defaults_hash(class_map, overrides, defaults_by_name)
    table, ui_metadata = generate_material_table(class_map, overrides=overrides, defaults_by_name=defaults_by_name)
    return table, ui_metadata, overrides, material_hash


def crop_cache_options(config: PreprocessConfig) -> dict[str, Any]:
    return visible_class_crop_options(
        config.visible_class_crop_settings_path,
        config.visible_class_crop_margin_voxels,
    )


def crop_cache_sources(config: PreprocessConfig) -> list[Path]:
    return visible_class_crop_sources(config.visible_class_crop_settings_path)


def apply_visible_class_crop(
    labels: np.ndarray,
    class_map: Mapping[int, str],
    config: PreprocessConfig,
) -> tuple[np.ndarray, VisibleClassCrop | None]:
    if config.visible_class_crop_settings_path is None:
        return np.ascontiguousarray(labels), None
    crop = crop_labels_to_visible_classes(
        labels,
        class_map,
        config.visible_class_crop_settings_path,
        margin_voxels=config.visible_class_crop_margin_voxels,
    )
    return crop.labels, crop


def crop_origin(crop: VisibleClassCrop | None, voxel_size_m: float) -> tuple[float, float, float]:
    if crop is None:
        return (0.0, 0.0, 0.0)
    return tuple(float(v) * float(voxel_size_m) for v in crop.crop_min)


def add_crop_metadata(
    metadata: dict[str, Any],
    crop: VisibleClassCrop | None,
    *,
    voxel_size_m: float,
) -> tuple[float, float, float]:
    origin = crop_origin(crop, voxel_size_m)
    if crop is not None:
        metadata["visible_class_crop"] = dict(crop.metadata)
        metadata["origin"] = [float(v) for v in origin]
    return origin


def cache_lookup(
    *,
    config: PreprocessConfig,
    dataset: str,
    sources: list[Path],
    options: Mapping[str, Any],
    material_hash: str,
) -> tuple[Path, str, tuple[np.ndarray, np.ndarray | None, dict[str, Any]] | None]:
    cache_dir = config.cache_dir or default_cache_dir(config.root, dataset)
    cache_key = make_cache_key(dataset=dataset, sources=sources, options=options, material_hash=material_hash)
    path = cache_path_for(cache_dir, dataset, cache_key)
    cached = None
    if config.use_cache and not config.rebuild_cache:
        cached = try_read_valid_cache(path, cache_key)
    return path, cache_key, cached


def write_cache(
    *,
    path: Path,
    dataset: str,
    cache_key: str,
    class_map: Mapping[int, str],
    raw_to_internal: Mapping[int, int],
    source_shape: tuple[int, int, int] | None,
    source_spacing_mm: tuple[float, float, float] | None,
    target_voxel_mm: float,
    config: PreprocessConfig,
    preprocess_options: Mapping[str, Any],
    source_files: list[Path],
    material_hash: str,
    labels: np.ndarray,
    texture_rgb: np.ndarray | None,
    metadata: Mapping[str, Any],
) -> CacheArtifact:
    manifest = build_manifest(
        dataset=dataset,
        cache_key=cache_key,
        class_map=class_map,
        raw_to_internal=raw_to_internal,
        source_shape=source_shape,
        source_spacing_mm=source_spacing_mm,
        target_voxel_mm=target_voxel_mm,
        pad=config.pad,
        preprocess_options=preprocess_options,
        source_files=source_files,
        material_defaults_hash=material_hash,
        metadata=metadata,
    )
    return write_npz_atomic(path, labels=labels, texture_rgb=texture_rgb, manifest=manifest)


def volume_from_cache(
    labels: np.ndarray,
    texture_rgb: np.ndarray | None,
    manifest: Mapping[str, Any],
    config: PreprocessConfig,
    *,
    defaults_by_name: Mapping[str, Mapping[str, Any]] | None = None,
    cache_path: Path | None = None,
) -> PreparedVolume:
    class_map = {int(k): str(v) for k, v in manifest["class_map"].items()}
    materials, ui_metadata, _overrides, _material_hash = prepare_materials(
        class_map,
        config,
        defaults_by_name=defaults_by_name,
    )
    metadata = dict(manifest.get("metadata", {}))
    metadata.update(ui_metadata)
    metadata.setdefault("dataset", manifest.get("dataset"))
    if cache_path is not None:
        metadata["cache_manifest"] = {"path": str(cache_path), **dict(manifest)}
    origin = tuple(float(v) for v in metadata.get("origin", (0.0, 0.0, 0.0)))
    return PreparedVolume(
        labels=labels,
        voxel_size_m=float(manifest["target_voxel_mm"]) * 1.0e-3,
        materials=materials,
        class_map=class_map,
        texture_rgb=texture_rgb,
        origin=origin,
        metadata=metadata,
    )
