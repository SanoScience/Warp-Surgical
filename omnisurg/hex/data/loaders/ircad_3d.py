# SPDX-License-Identifier: Apache-2.0
"""3Dircadb ``LABELLED_DICOM`` loader."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # optional dependency; only required when the 3Dircadb loader is used.
    import pydicom
    from pydicom.errors import InvalidDicomError
except ModuleNotFoundError:  # pragma: no cover - exercised by registry smoke tests without pydicom.
    pydicom = None

    class InvalidDicomError(Exception):
        pass

from ..crop import crop_aligned_texture_rgb
from ..materials import (
    THREE_DIRCADB_CLASS_MAP,
    THREE_DIRCADB_MATERIAL_HINTS,
    load_class_map,
    pad_labels,
    pad_texture_rgb,
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


@dataclass(frozen=True)
class _DicomSliceHeader:
    path: Path
    dataset: Any
    member: str | None = None

    @property
    def sort_label(self) -> str:
        if self.member is None:
            return str(self.path).lower()
        return f"{self.path}!{self.member}".lower()


@dataclass(frozen=True)
class _DicomSeries:
    source: Path
    slices: list[_DicomSliceHeader]
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]

    @property
    def files(self) -> list[Path]:
        if self.source.is_file():
            return [self.source]
        return [item.path for item in self.slices]


@dataclass(frozen=True)
class _TextureSource:
    source_type: str
    path: Path
    series: _DicomSeries | None = None

    @property
    def files(self) -> list[Path]:
        if self.source_type == "dicom" and self.series is not None:
            return self.series.files
        return [self.path]


class ThreeDircadbLoader:
    name = "3dircadb"

    def can_load(self, path: str | Path | None) -> bool:
        return _resolve_dataset_root(path) is not None

    def preprocess(self, config: PreprocessConfig) -> CacheArtifact:
        volume = self.load(config)
        manifest = volume.metadata.get("cache_manifest", {})
        return CacheArtifact(path=Path(manifest.get("path", "")), manifest=manifest)

    def load(self, config: PreprocessConfig) -> PreparedVolume:
        _require_pydicom()
        dataset_root = _resolve_dataset_root(config.root)
        if dataset_root is None:
            raise FileNotFoundError("3Dircadb root must contain a LABELLED_DICOM directory")

        labelled_source = _find_child_source(dataset_root, "LABELLED_DICOM")
        if labelled_source is None:
            raise FileNotFoundError(f"3Dircadb LABELLED_DICOM directory not found under {dataset_root}")

        label_series = _read_dicom_series(labelled_source)
        source_spacing = label_series.spacing_mm
        target_voxel_mm = resolve_target_voxel_mm(config, source_spacing)
        mask_source = _find_child_source(dataset_root, "MASKS_DICOM")
        texture_source = _resolve_texture_source(config, dataset_root)

        class_map = dict(THREE_DIRCADB_CLASS_MAP)
        override_class_map = load_class_map(config.class_map_path)
        class_map.update(override_class_map)
        _raw_materials, _raw_ui_metadata, _raw_overrides, cache_material_hash = prepare_materials(
            class_map,
            config,
            defaults_by_name=THREE_DIRCADB_MATERIAL_HINTS,
        )
        options = {
            "pad": config.pad,
            "target_voxel_mm": target_voxel_mm,
            "source_spacing_mm": source_spacing,
            "label_source": "LABELLED_DICOM",
            "load_texture": bool(config.load_texture),
            "texture_source": _texture_source_option(texture_source, config),
            "mask_class_map": mask_source is not None,
            "class_map_scope": "mask-derived-v1",
            **crop_cache_options(config),
        }
        sources = (
            label_series.files
            + ([] if mask_source is None else [mask_source])
            + ([] if texture_source is None else texture_source.files)
            + crop_cache_sources(config)
        )
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
                defaults_by_name=THREE_DIRCADB_MATERIAL_HINTS,
                cache_path=cache_path,
            )

        raw = _read_dicom_pixel_volume(label_series, apply_modality=False)
        if not np.issubdtype(raw.dtype, np.integer):
            raw = np.rint(raw).astype(np.int32)
        else:
            raw = np.asarray(raw)
        raw = np.ascontiguousarray(raw)
        class_map = dict(THREE_DIRCADB_CLASS_MAP)
        if mask_source is not None:
            class_map.update(_derive_class_map_from_masks(raw, mask_source))
        class_map.update(override_class_map)
        resampled = resample_nearest_to_isotropic(raw, source_spacing, target_voxel_mm)
        labels, internal_class_map, raw_to_internal = remap_labels_to_internal(resampled, class_map)
        labels, crop = apply_visible_class_crop(labels, internal_class_map, config)
        labels = pad_labels(labels, config.pad)

        texture = _load_texture(texture_source, target_voxel_mm)
        texture = crop_aligned_texture_rgb(texture, crop, texture_name="3Dircadb texture")
        texture = pad_texture_rgb(texture, config.pad)
        materials, ui_metadata, _overrides, material_hash = prepare_materials(
            internal_class_map,
            config,
            defaults_by_name=THREE_DIRCADB_MATERIAL_HINTS,
        )
        metadata = {
            "dataset": self.name,
            "scene_label": f"3Dircadb {dataset_root.name}",
            "source_root": str(dataset_root),
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
                source_shape=label_series.shape,
                source_spacing_mm=source_spacing,
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


def _require_pydicom() -> None:
    if pydicom is None:
        raise ModuleNotFoundError("3Dircadb loading requires the optional 'pydicom' package")


def _resolve_dataset_root(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    root = Path(path)
    if not root.exists() or not root.is_dir():
        return None
    if root.name.upper() == "LABELLED_DICOM":
        return root.parent
    if _find_child_source(root, "LABELLED_DICOM") is not None:
        return root

    matches: list[Path] = []
    for child in _iter_child_dirs(root):
        if _find_child_source(child, "LABELLED_DICOM") is not None:
            matches.append(child)
    if len(matches) == 1:
        return matches[0]
    return None


def _find_child_source(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_dir():
        return direct
    target = name.upper()
    for child in _iter_child_dirs(root):
        if child.name.upper() == target:
            return child
    zip_name = f"{name}.zip"
    direct_zip = root / zip_name
    if direct_zip.is_file():
        return direct_zip
    zip_target = zip_name.upper()
    try:
        for child in sorted((p for p in root.iterdir() if p.is_file()), key=lambda p: p.name.lower()):
            if child.name.upper() == zip_target:
                return child
    except OSError:
        return None
    return None


def _iter_child_dirs(root: Path) -> list[Path]:
    try:
        return sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.lower())
    except OSError:
        return []


def _read_dicom_series(source: Path) -> _DicomSeries:
    headers: list[_DicomSliceHeader] = []
    if source.is_dir():
        for path in sorted((p for p in source.rglob("*") if p.is_file()), key=lambda p: str(p).lower()):
            try:
                ds = pydicom.dcmread(str(path), force=True, stop_before_pixels=True)
            except (InvalidDicomError, OSError):
                continue
            if not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
                continue
            headers.append(_DicomSliceHeader(path=path, dataset=ds))
    elif source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            headers = _read_dicom_zip_headers(source, zf, [n for n in zf.namelist() if not n.endswith("/")])
    else:
        raise FileNotFoundError(f"DICOM source not found: {source}")

    return _dicom_series_from_headers(source, headers)


def _read_dicom_zip_headers(source: Path, zf: zipfile.ZipFile, members: list[str]) -> list[_DicomSliceHeader]:
    headers: list[_DicomSliceHeader] = []
    for member in sorted(members, key=str.lower):
        try:
            payload = zf.read(member)
            ds = pydicom.dcmread(io.BytesIO(payload), force=True, stop_before_pixels=True)
        except (InvalidDicomError, OSError, KeyError, zipfile.BadZipFile):
            continue
        if not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
            continue
        headers.append(_DicomSliceHeader(path=source, member=member, dataset=ds))
    return headers


def _dicom_series_from_headers(source: Path, headers: list[_DicomSliceHeader]) -> _DicomSeries:
    if not headers:
        raise FileNotFoundError(f"no readable DICOM image slices found under {source}")
    headers = _sort_dicom_slices(headers)
    first = headers[0].dataset
    rows = int(first.Rows)
    columns = int(first.Columns)
    for item in headers[1:]:
        if int(item.dataset.Rows) != rows or int(item.dataset.Columns) != columns:
            raise ValueError(f"DICOM slice dimensions differ in {source}")
    return _DicomSeries(
        source=source,
        slices=headers,
        shape=(rows, columns, len(headers)),
        spacing_mm=_spacing_mm(headers),
    )


def _sort_dicom_slices(headers: list[_DicomSliceHeader]) -> list[_DicomSliceHeader]:
    normal = _slice_normal(headers)
    position_keys = [_position_scalar(item.dataset, normal) for item in headers]
    if all(value is not None for value in position_keys):
        return [
            item
            for _, item in sorted(
                zip(position_keys, headers, strict=True),
                key=lambda pair: (float(pair[0]), _instance_number(pair[1].dataset) or 0, pair[1].sort_label),
            )
        ]

    instance_keys = [_instance_number(item.dataset) for item in headers]
    if all(value is not None for value in instance_keys):
        return sorted(headers, key=lambda item: (int(_instance_number(item.dataset) or 0), item.sort_label))
    return sorted(headers, key=lambda item: item.sort_label)


def _slice_normal(headers: list[_DicomSliceHeader]) -> np.ndarray:
    for item in headers:
        raw = getattr(item.dataset, "ImageOrientationPatient", None)
        if raw is None or len(raw) < 6:
            continue
        row = np.asarray([float(v) for v in raw[:3]], dtype=np.float64)
        col = np.asarray([float(v) for v in raw[3:6]], dtype=np.float64)
        normal = np.cross(row, col)
        norm = float(np.linalg.norm(normal))
        if norm > 0.0:
            return normal / norm
    return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)


def _position_scalar(ds: Any, normal: np.ndarray) -> float | None:
    raw = getattr(ds, "ImagePositionPatient", None)
    if raw is None or len(raw) < 3:
        return None
    pos = np.asarray([float(v) for v in raw[:3]], dtype=np.float64)
    return float(np.dot(pos, normal))


def _instance_number(ds: Any) -> int | None:
    if not hasattr(ds, "InstanceNumber"):
        return None
    try:
        return int(ds.InstanceNumber)
    except (TypeError, ValueError):
        return None


def _spacing_mm(headers: list[_DicomSliceHeader]) -> tuple[float, float, float]:
    first = headers[0].dataset
    pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    if len(pixel_spacing) < 2:
        row_spacing, column_spacing = 1.0, 1.0
    else:
        row_spacing, column_spacing = float(pixel_spacing[0]), float(pixel_spacing[1])

    normal = _slice_normal(headers)
    positions = [_position_scalar(item.dataset, normal) for item in headers]
    z_spacing = None
    if len(positions) > 1 and all(value is not None for value in positions):
        diffs = np.diff(np.asarray([float(value) for value in positions], dtype=np.float64))
        diffs = np.abs(diffs[np.abs(diffs) > 1.0e-6])
        if diffs.size:
            z_spacing = float(np.median(diffs))
    if z_spacing is None:
        for attr in ("SpacingBetweenSlices", "SliceThickness"):
            if hasattr(first, attr):
                value = float(getattr(first, attr))
                if value > 0.0:
                    z_spacing = value
                    break
    if z_spacing is None:
        z_spacing = 1.0
    return (row_spacing, column_spacing, z_spacing)


def _read_dicom_pixel_volume(series: _DicomSeries, *, apply_modality: bool) -> np.ndarray:
    arrays: list[np.ndarray] = []
    if series.source.is_file() and series.source.suffix.lower() == ".zip":
        with zipfile.ZipFile(series.source) as zf:
            for item in series.slices:
                if item.member is None:
                    raise ValueError(f"zip-backed DICOM slice is missing member name for {item.path}")
                ds = pydicom.dcmread(io.BytesIO(zf.read(item.member)), force=True)
                arrays.append(_dicom_slice_pixels(ds, item.sort_label, apply_modality=apply_modality))
    else:
        for item in series.slices:
            ds = pydicom.dcmread(str(item.path), force=True)
            arrays.append(_dicom_slice_pixels(ds, item.sort_label, apply_modality=apply_modality))
    return np.ascontiguousarray(np.stack(arrays, axis=-1))


def _dicom_slice_pixels(ds: Any, source_label: str, *, apply_modality: bool) -> np.ndarray:
    arr = np.asarray(ds.pixel_array)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"expected single-frame DICOM slice in {source_label}, got shape={arr.shape}")
    if apply_modality:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr.astype(np.float32) * slope + intercept
    return np.ascontiguousarray(arr)


def _derive_class_map_from_masks(raw_labels: np.ndarray, mask_source: Path) -> dict[int, str]:
    assignments: dict[int, tuple[int, int, str]] = {}
    for mask_name, series in _iter_mask_series(mask_source):
        class_name = _class_name_from_mask_name(mask_name)
        priority = _mask_name_priority(class_name)
        mask = _read_dicom_pixel_volume(series, apply_modality=False)
        if mask.shape != raw_labels.shape:
            raise ValueError(
                f"mask {mask_name!r} shape {mask.shape} does not match LABELLED_DICOM shape {raw_labels.shape}"
            )
        raw_ids, counts = np.unique(raw_labels[mask > 0], return_counts=True)
        if class_name == "skin" and raw_ids.size:
            nonzero = raw_ids != 0
            if np.any(nonzero):
                ids_nonzero = raw_ids[nonzero]
                counts_nonzero = counts[nonzero]
                keep = int(np.argmax(counts_nonzero))
                raw_ids = np.asarray([ids_nonzero[keep]])
                counts = np.asarray([counts_nonzero[keep]])
        for raw_id_value, count_value in zip(raw_ids.astype(np.int64).tolist(), counts.astype(np.int64).tolist(), strict=True):
            raw_id = int(raw_id_value)
            if raw_id == 0:
                continue
            count = int(count_value)
            previous = assignments.get(raw_id)
            if previous is None:
                assignments[raw_id] = (priority, count, class_name)
            elif priority > previous[0] and count >= max(1, int(previous[1] * 0.05)):
                assignments[raw_id] = (priority, count, class_name)
            elif priority == previous[0] and count > previous[1]:
                assignments[raw_id] = (priority, count, class_name)
    return {raw_id: name for raw_id, (_priority, _count, name) in sorted(assignments.items())}


def _iter_mask_series(mask_source: Path) -> list[tuple[str, _DicomSeries]]:
    if mask_source.is_dir():
        groups: list[tuple[str, _DicomSeries]] = []
        for directory in sorted((p for p in mask_source.rglob("*") if p.is_dir()), key=lambda p: str(p).lower()):
            try:
                has_files = any(child.is_file() for child in directory.iterdir())
            except OSError:
                continue
            if not has_files:
                continue
            try:
                groups.append((directory.name, _read_dicom_series(directory)))
            except FileNotFoundError:
                continue
        return groups

    if mask_source.is_file() and mask_source.suffix.lower() == ".zip":
        with zipfile.ZipFile(mask_source) as zf:
            members_by_name: dict[str, list[str]] = {}
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                parts = [part for part in member.split("/") if part]
                if len(parts) < 2:
                    continue
                members_by_name.setdefault(parts[-2], []).append(member)
            groups = []
            for name in sorted(members_by_name, key=str.lower):
                headers = _read_dicom_zip_headers(mask_source, zf, members_by_name[name])
                if headers:
                    groups.append((name, _dicom_series_from_headers(mask_source, headers)))
            return groups
    return []


def _class_name_from_mask_name(mask_name: str) -> str:
    compact = "".join(ch for ch in mask_name.lower() if ch.isalnum())
    if compact in {"leftkidney", "rightkidney", "kidney", "kidneys"}:
        return "kidneys"
    if compact in {"leftlung", "rightlung", "lung", "lungs"}:
        return "lungs"
    if compact in {"portalvein", "portalvenous"}:
        return "portal_vein"
    if compact in {"venoussystem", "venacava", "vena", "cava"}:
        return "venous_system"
    if compact.startswith("livertumor") or compact in {"tumor", "tumour", "hyperplasie", "hyperplasia"}:
        return "liver_tumor"
    if compact in {"liverkyst", "livercyst", "cyst"}:
        return "liver_cyst"
    if compact in {"aorta", "artery", "arteries"}:
        return "artery"
    if compact == "lymphnodes":
        return "lymph_nodes"
    if compact == "gallbladder":
        return "gall_bladder"
    return compact or "unknown"


def _mask_name_priority(class_name: str) -> int:
    if class_name == "skin":
        return 0
    if class_name == "liver":
        return 10
    if class_name in {"liver_tumor", "liver_cyst", "portal_vein", "venous_system"}:
        return 30
    return 20


def _resolve_texture_source(config: PreprocessConfig, dataset_root: Path) -> _TextureSource | None:
    if not config.load_texture:
        return None
    if config.texture_path is not None:
        path = Path(config.texture_path)
        if path.is_dir() or path.suffix.lower() == ".zip":
            return _TextureSource(source_type="dicom", path=path, series=_read_dicom_series(path))
        return _TextureSource(source_type="rgb", path=path)

    patient_source = _find_child_source(dataset_root, "PATIENT_DICOM")
    if patient_source is None:
        return None
    return _TextureSource(source_type="dicom", path=patient_source, series=_read_dicom_series(patient_source))


def _texture_source_option(texture_source: _TextureSource | None, config: PreprocessConfig) -> str | bool:
    if not config.load_texture:
        return False
    if texture_source is None:
        return "auto-missing"
    if config.texture_path is not None:
        return str(texture_source.path)
    return texture_source.source_type


def _load_texture(texture_source: _TextureSource | None, target_voxel_mm: float) -> np.ndarray | None:
    if texture_source is None:
        return None
    if texture_source.source_type == "rgb":
        texture = _load_rgb_texture_path(texture_source.path)
    else:
        if texture_source.series is None:
            return None
        ct = _read_dicom_pixel_volume(texture_source.series, apply_modality=True)
        ct = resample_nearest_to_isotropic(ct, texture_source.series.spacing_mm, target_voxel_mm)
        texture = _ct_to_rgb(ct)
    return np.ascontiguousarray(texture)


def _load_rgb_texture_path(path: Path) -> np.ndarray:
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


def _ct_to_rgb(volume: np.ndarray) -> np.ndarray:
    values = np.asarray(volume, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        gray = np.zeros(values.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(finite, [1.0, 99.0])
        if float(hi) <= float(lo):
            lo, hi = float(np.min(finite)), float(np.max(finite))
        if float(hi) <= float(lo):
            gray = np.zeros(values.shape, dtype=np.uint8)
        else:
            scaled = (np.clip(values, lo, hi) - lo) * (255.0 / (hi - lo))
            gray = np.ascontiguousarray(scaled.astype(np.uint8))
    return np.ascontiguousarray(np.repeat(gray[..., None], 3, axis=3))


ThreeDircadb1Loader = ThreeDircadbLoader
