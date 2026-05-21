# SPDX-License-Identifier: Apache-2.0
"""Visible-class driven label-volume cropping."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BACKGROUND_NAMES = {"background"}


@dataclass(frozen=True)
class VisibleClassSelection:
    """Classes selected from a segmentation-panel settings JSON."""

    ids: tuple[int, ...]
    names: tuple[str, ...]
    panel_entries: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class VisibleClassCrop:
    """Cropped label volume plus enough metadata to crop aligned textures."""

    labels: np.ndarray
    crop_slices: tuple[slice, slice, slice]
    crop_min: tuple[int, int, int]
    crop_max: tuple[int, int, int]
    original_shape: tuple[int, int, int]
    metadata: dict[str, Any]


def visible_class_crop_sources(settings_path: str | Path | None) -> list[Path]:
    """Return source files that affect a visible-class crop."""

    return [] if settings_path is None else [Path(settings_path)]


def visible_class_crop_options(settings_path: str | Path | None, margin_voxels: int) -> dict[str, Any]:
    """Return cache-key options for the visible-class crop."""

    if settings_path is None:
        return {}
    return {
        "visible_class_crop_settings_path": str(Path(settings_path)),
        "visible_class_crop_margin_voxels": int(margin_voxels),
        "visible_class_crop_selection": "name-first-v2",
    }


def select_visible_class_ids(
    settings_path: str | Path,
    class_map: Mapping[int, str],
) -> VisibleClassSelection:
    """Select visible non-background classes by material id or exact name."""

    path = Path(settings_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    materials = payload.get("materials") if isinstance(payload, Mapping) else None
    if not isinstance(materials, list):
        raise ValueError(f"{path} does not contain a materials list")

    normalized_class_map = {int(k): str(v) for k, v in class_map.items()}
    names_to_ids: dict[str, list[int]] = {}
    for class_id, name in normalized_class_map.items():
        names_to_ids.setdefault(name, []).append(class_id)

    visible_entries: list[Mapping[str, Any]] = []
    selected_ids: set[int] = set()
    for item in materials:
        if not isinstance(item, Mapping) or not bool(item.get("visible", False)):
            continue
        entry = dict(item)
        raw_id = _coerce_int(entry.get("id"))
        raw_name = entry.get("name")
        name = str(raw_name) if raw_name is not None else None
        if _is_background_entry(raw_id, name):
            continue
        visible_entries.append(entry)
        if name is not None:
            for matched_id in names_to_ids.get(name, []):
                if matched_id != 0:
                    selected_ids.add(int(matched_id))
        if raw_id is not None and raw_id in normalized_class_map and raw_id != 0:
            current_name = normalized_class_map[int(raw_id)]
            if name is None or name == current_name:
                selected_ids.add(int(raw_id))

    if not visible_entries:
        raise ValueError(f"{path} has no visible non-background classes")
    if not selected_ids:
        raise ValueError(f"visible classes in {path} do not match the loaded volume class map")

    ids = tuple(sorted(selected_ids))
    return VisibleClassSelection(
        ids=ids,
        names=tuple(normalized_class_map[class_id] for class_id in ids),
        panel_entries=tuple(visible_entries),
    )


def crop_labels_to_visible_classes(
    labels: np.ndarray,
    class_map: Mapping[int, str],
    settings_path: str | Path,
    *,
    margin_voxels: int = 4,
) -> VisibleClassCrop:
    """Crop to visible classes and keep only their Chebyshev-margin support."""

    margin = int(margin_voxels)
    if margin < 0:
        raise ValueError(f"visible-class crop margin must be >= 0, got {margin_voxels}")

    source = np.ascontiguousarray(labels)
    if source.ndim != 3:
        raise ValueError(f"labels must be 3D for visible-class crop, got shape={source.shape}")

    selection = select_visible_class_ids(settings_path, class_map)
    seed_mask = np.isin(source, np.asarray(selection.ids, dtype=np.int64))
    seed_count = int(np.count_nonzero(seed_mask))
    if seed_count == 0:
        names = ", ".join(selection.names)
        raise ValueError(f"visible crop classes are not present in loaded labels: {names}")

    seed_coords = np.argwhere(seed_mask)
    seed_min = seed_coords.min(axis=0)
    seed_max = seed_coords.max(axis=0) + 1
    shape = np.asarray(source.shape, dtype=np.int64)
    crop_min_arr = np.maximum(0, seed_min - margin)
    crop_max_arr = np.minimum(shape, seed_max + margin)
    crop_slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(crop_min_arr, crop_max_arr, strict=True))

    cropped_labels = np.ascontiguousarray(source[crop_slices])
    cropped_seed = np.ascontiguousarray(seed_mask[crop_slices])
    support = _dilate_chebyshev(cropped_seed, margin)
    zeroed_voxel_count = int(np.count_nonzero(cropped_labels[~support]))
    if zeroed_voxel_count:
        cropped_labels = cropped_labels.copy()
        cropped_labels[~support] = 0
    cropped_labels = np.ascontiguousarray(cropped_labels)
    kept_voxel_count = int(np.count_nonzero(cropped_labels))
    if kept_voxel_count == 0:
        raise ValueError("visible-class crop removed all occupied voxels")

    crop_min = tuple(int(v) for v in crop_min_arr.tolist())
    crop_max = tuple(int(v) for v in crop_max_arr.tolist())
    metadata = {
        "settings_path": str(Path(settings_path)),
        "margin_voxels": margin,
        "selected_ids": [int(v) for v in selection.ids],
        "selected_names": [str(v) for v in selection.names],
        "panel_visible_entries": [dict(item) for item in selection.panel_entries],
        "original_shape": [int(v) for v in source.shape],
        "crop_min": [int(v) for v in crop_min],
        "crop_max": [int(v) for v in crop_max],
        "crop_shape": [int(v) for v in cropped_labels.shape],
        "crop_slices": [[int(s.start), int(s.stop)] for s in crop_slices],
        "seed_voxel_count": seed_count,
        "support_voxel_count": int(np.count_nonzero(support)),
        "kept_voxel_count": kept_voxel_count,
        "zeroed_voxel_count": zeroed_voxel_count,
    }
    return VisibleClassCrop(
        labels=cropped_labels,
        crop_slices=crop_slices,  # type: ignore[arg-type]
        crop_min=crop_min,
        crop_max=crop_max,
        original_shape=tuple(int(v) for v in source.shape),
        metadata=metadata,
    )


def crop_aligned_texture_rgb(
    texture: np.ndarray | None,
    crop: VisibleClassCrop | None,
    *,
    texture_name: str = "texture",
) -> np.ndarray | None:
    """Apply a label crop to a 1:1 or integer-resolution RGB texture volume."""

    if texture is None:
        return None
    if crop is None:
        return np.ascontiguousarray(texture)
    source = np.ascontiguousarray(texture)
    if source.ndim != 4 or source.shape[-1] != 3 or source.dtype != np.uint8:
        raise ValueError(f"{texture_name} must be uint8 (nx, ny, nz, 3), got {source.shape} {source.dtype}")

    factors = _texture_label_factors(source.shape[:3], crop.original_shape, texture_name=texture_name)
    scaled_slices = tuple(
        slice(int(s.start) * factor, int(s.stop) * factor)
        for s, factor in zip(crop.crop_slices, factors, strict=True)
    )
    return np.ascontiguousarray(source[(*scaled_slices, slice(None))])


def _texture_label_factors(
    texture_shape: tuple[int, int, int],
    label_shape: tuple[int, int, int],
    *,
    texture_name: str,
) -> tuple[int, int, int]:
    factors: list[int] = []
    for _axis, (tex_dim, label_dim) in enumerate(zip(texture_shape, label_shape, strict=True)):
        if label_dim <= 0 or tex_dim < label_dim or tex_dim % label_dim != 0:
            raise ValueError(
                f"{texture_name} shape {tuple(texture_shape)} is not aligned to label shape {tuple(label_shape)} "
                "for visible-class cropping; disable texture or provide a 1:1/integer-resolution aligned texture"
            )
        factors.append(int(tex_dim // label_dim))
    return tuple(factors)  # type: ignore[return-value]


def _dilate_chebyshev(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ascontiguousarray(mask.astype(bool, copy=False))
    out = np.ascontiguousarray(mask.astype(bool, copy=False))
    for axis in range(3):
        out = _dilate_axis(out, axis, radius)
    return out


def _dilate_axis(mask: np.ndarray, axis: int, radius: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    size = int(mask.shape[axis])
    for offset in range(-int(radius), int(radius) + 1):
        src_start = max(0, -offset)
        src_stop = min(size, size - offset)
        if src_start >= src_stop:
            continue
        dst_start = max(0, offset)
        dst_stop = dst_start + (src_stop - src_start)
        src_slices = [slice(None)] * mask.ndim
        dst_slices = [slice(None)] * mask.ndim
        src_slices[axis] = slice(src_start, src_stop)
        dst_slices[axis] = slice(dst_start, dst_stop)
        out[tuple(dst_slices)] |= mask[tuple(src_slices)]
    return out


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_background_entry(class_id: int | None, name: str | None) -> bool:
    if class_id == 0:
        return True
    return name is not None and name.strip().lower() in BACKGROUND_NAMES
