# SPDX-License-Identifier: Apache-2.0
"""KiTS23 kidney-tumor NIfTI segmentation loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..materials import (
    KITS23_CLASS_MAP,
    KITS23_MATERIAL_HINTS,
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
    resample_nearest_to_isotropic,
    resolve_target_voxel_mm,
    volume_from_cache,
    write_cache,
)

_SEGMENTATION_FILENAMES = (
    "segmentation.nii.gz",
    "segmentation.nii",
    "labels.nii.gz",
    "labels.nii",
)


class Kits23Loader:
    name = "kits23"

    def can_load(self, path: str | Path | None) -> bool:
        if path is None:
            return False
        p = Path(path)
        if p.is_file():
            return _is_flat_case_file(p) or _is_case_segmentation_file(p)
        if not p.is_dir():
            return False
        return bool(_iter_case_sources(p))

    def preprocess(self, config: PreprocessConfig) -> CacheArtifact:
        volume = self.load(config)
        manifest = volume.metadata.get("cache_manifest", {})
        return CacheArtifact(path=Path(manifest.get("path", "")), manifest=manifest)

    def load(self, config: PreprocessConfig) -> PreparedVolume:
        try:
            import nibabel as nib
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("KiTS23 loading requires nibabel>=5") from exc

        source = _resolve_source(config.root, config.options.get("case"))
        img = nib.load(str(source))
        source_spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
        target_voxel_mm = resolve_target_voxel_mm(config, source_spacing)

        override_class_map = load_class_map(config.class_map_path)
        cache_class_map = dict(KITS23_CLASS_MAP)
        cache_class_map.update(override_class_map)
        _raw_materials, _raw_ui_metadata, _raw_overrides, cache_material_hash = prepare_materials(
            cache_class_map,
            config,
            defaults_by_name=KITS23_MATERIAL_HINTS,
        )
        case_id = _case_id(source)
        options = {
            "pad": config.pad,
            "target_voxel_mm": target_voxel_mm,
            "source_spacing_mm": source_spacing,
            "case_id": case_id,
            "class_map_scope": "kidney-tumor-observed-v1",
            **crop_cache_options(config),
        }
        source_files = [source, *crop_cache_sources(config)]
        cache_path, cache_key, cached = cache_lookup(
            config=config,
            dataset=self.name,
            sources=source_files,
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
                defaults_by_name=KITS23_MATERIAL_HINTS,
                cache_path=cache_path,
            )

        raw = np.asanyarray(img.dataobj)
        if raw.ndim != 3:
            raise ValueError(f"KiTS23 label file must be 3D, got shape={raw.shape} from {source}")
        if not np.issubdtype(raw.dtype, np.integer):
            raw = np.rint(raw).astype(np.int32)
        else:
            raw = np.asarray(raw)
        raw = np.ascontiguousarray(raw)

        class_map = _observed_class_map(raw, override_class_map)
        resampled = resample_nearest_to_isotropic(raw, source_spacing, target_voxel_mm)
        labels, internal_class_map, raw_to_internal = remap_labels_to_internal(resampled, class_map)
        labels, crop = apply_visible_class_crop(labels, internal_class_map, config)
        labels = pad_labels(labels, config.pad)
        materials, ui_metadata, _overrides, material_hash = prepare_materials(
            internal_class_map,
            config,
            defaults_by_name=KITS23_MATERIAL_HINTS,
        )

        metadata = {
            "dataset": self.name,
            "scene_label": f"KiTS23 {case_id}",
            "case_id": case_id,
            "source_file": str(source),
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
                source_files=source_files,
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


def _resolve_source(root: str | Path | None, case: Any = None) -> Path:
    if root is None:
        root = Path("Kits2023")
    p = Path(root)
    if p.is_file():
        if not (_is_flat_case_file(p) or _is_case_segmentation_file(p)):
            raise FileNotFoundError(f"KiTS23 file must be a case NIfTI label file: {p}")
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"KiTS23 root not found: {p}")

    if case is not None:
        selected = _case_source_from_option(p, case)
        if selected is not None:
            return selected
        raise FileNotFoundError(f"KiTS23 case {case!r} not found under {p}")

    sources = _iter_case_sources(p)
    if not sources:
        raise FileNotFoundError(f"KiTS23 root must contain case_*.nii.gz files or case directories: {p}")
    return sources[0]


def _case_source_from_option(root: Path, case: Any) -> Path | None:
    text = str(case).strip()
    if text == "":
        return None
    candidates: list[Path] = []
    raw_path = Path(text)
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(root / raw_path)
    if text.isdigit():
        stem = f"case_{int(text):05d}"
    else:
        stem = text
    if not stem.startswith("case_") and stem.isdigit():
        stem = f"case_{int(stem):05d}"
    if _nifti_suffix(Path(stem)) is None:
        candidates.extend((root / f"{stem}.nii.gz", root / f"{stem}.nii"))
    candidates.append(root / stem)

    for candidate in candidates:
        if candidate.is_file() and (_is_flat_case_file(candidate) or _is_case_segmentation_file(candidate)):
            return candidate
        if candidate.is_dir():
            nested = _find_case_dir_segmentation(candidate)
            if nested is not None:
                return nested
    return None


def _iter_case_sources(root: Path) -> list[Path]:
    sources: list[Path] = []
    for child in _iter_children(root):
        if child.is_file() and _is_flat_case_file(child):
            sources.append(child)
        elif child.is_dir():
            nested = _find_case_dir_segmentation(child)
            if nested is not None:
                sources.append(nested)
    root_segmentation = _find_case_dir_segmentation(root)
    if root_segmentation is not None:
        sources.append(root_segmentation)
    return sorted(set(sources), key=lambda path: str(path).lower())


def _iter_children(root: Path) -> list[Path]:
    try:
        return sorted(root.iterdir(), key=lambda path: path.name.lower())
    except OSError:
        return []


def _find_case_dir_segmentation(case_dir: Path) -> Path | None:
    for filename in _SEGMENTATION_FILENAMES:
        candidate = case_dir / filename
        if candidate.is_file():
            return candidate
    return None


def _observed_class_map(raw_labels: np.ndarray, override_class_map: dict[int, str]) -> dict[int, str]:
    class_map = dict(KITS23_CLASS_MAP)
    raw_ids = sorted(int(v) for v in np.unique(raw_labels).astype(np.int64).tolist())
    tumor_ids = [raw_id for raw_id in raw_ids if raw_id > 1]
    for idx, raw_id in enumerate(tumor_ids, start=1):
        class_map[raw_id] = "tumor" if idx == 1 else f"tumor_{idx}"
    class_map.update(override_class_map)
    return class_map


def _case_id(source: Path) -> str:
    if source.parent.name.startswith("case_") and _is_case_segmentation_file(source):
        return source.parent.name
    return source.name.removesuffix(".nii.gz").removesuffix(".nii")


def _is_flat_case_file(path: Path) -> bool:
    return path.name.startswith("case_") and _nifti_suffix(path) is not None


def _is_case_segmentation_file(path: Path) -> bool:
    return path.name in _SEGMENTATION_FILENAMES and path.parent.name.startswith("case_")


def _nifti_suffix(path: Path) -> str | None:
    if path.name.endswith(".nii.gz"):
        return ".nii.gz"
    if path.suffix == ".nii":
        return ".nii"
    return None
