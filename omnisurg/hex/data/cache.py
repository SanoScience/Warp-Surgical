# SPDX-License-Identifier: Apache-2.0
"""Uncompressed NPZ cache bundles for prepared OmniSurg volumes."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .materials import stable_hash
from .types import CacheArtifact

CACHE_SCHEMA_VERSION = 1


def source_fingerprint(path: str | Path) -> dict[str, Any]:
    """Return a cheap invalidation fingerprint for a source file."""

    p = Path(path)
    st = p.stat()
    return {
        "path": str(p.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def source_fingerprints(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    return [source_fingerprint(p) for p in paths]


def make_cache_key(
    *,
    dataset: str,
    sources: Iterable[str | Path],
    options: Mapping[str, Any],
    material_hash: str,
) -> str:
    """Build a stable cache key from source fingerprints and preprocessing options."""

    return stable_hash(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "dataset": dataset,
            "sources": source_fingerprints(sources),
            "options": dict(options),
            "material_hash": material_hash,
        }
    )


def default_cache_dir(root: str | Path | None, dataset: str) -> Path:
    """Pick a local cache directory near the source path when possible."""

    if root is None:
        return Path(".cache") / "omnisurg" / "hex" / dataset
    p = Path(root)
    base = p if p.is_dir() else p.parent
    return base / ".cache" / "omnisurg" / "hex" / dataset


def cache_path_for(cache_dir: str | Path, dataset: str, cache_key: str) -> Path:
    safe_dataset = dataset.replace("-", "_")
    return Path(cache_dir) / f"{safe_dataset}_{cache_key[:16]}.npz"


def build_manifest(
    *,
    dataset: str,
    cache_key: str,
    class_map: Mapping[int, str],
    raw_to_internal: Mapping[int, int],
    source_shape: tuple[int, int, int] | None,
    source_spacing_mm: tuple[float, float, float] | None,
    target_voxel_mm: float,
    pad: int,
    preprocess_options: Mapping[str, Any],
    source_files: Iterable[str | Path],
    material_defaults_hash: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the JSON manifest stored in every cache bundle."""

    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset": dataset,
        "cache_key": cache_key,
        "class_map": {str(int(k)): str(v) for k, v in sorted(class_map.items())},
        "raw_to_internal": {str(int(k)): int(v) for k, v in sorted(raw_to_internal.items())},
        "source_shape": None if source_shape is None else [int(v) for v in source_shape],
        "source_spacing_mm": None if source_spacing_mm is None else [float(v) for v in source_spacing_mm],
        "target_voxel_mm": float(target_voxel_mm),
        "pad": int(pad),
        "preprocess_options": dict(preprocess_options),
        "source_files": source_fingerprints(source_files),
        "material_defaults_hash": str(material_defaults_hash),
        "metadata": dict(metadata or {}),
    }


def write_npz_atomic(
    path: str | Path,
    *,
    labels: np.ndarray,
    manifest: Mapping[str, Any],
    texture_rgb: np.ndarray | None = None,
) -> CacheArtifact:
    """Atomically write an uncompressed ``.npz`` cache bundle."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        arrays: dict[str, Any] = {
            "labels": np.ascontiguousarray(labels),
            "manifest": np.asarray(json.dumps(dict(manifest), sort_keys=True)),
        }
        if texture_rgb is not None:
            arrays["texture_rgb"] = np.ascontiguousarray(texture_rgb)
        with tmp_path.open("wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp_path, target)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise
    return CacheArtifact(path=target, manifest=dict(manifest))


def read_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Read labels, optional texture, and manifest from a cache bundle."""

    with np.load(Path(path), allow_pickle=False) as data:
        labels = np.ascontiguousarray(data["labels"])
        texture = np.ascontiguousarray(data["texture_rgb"]) if "texture_rgb" in data.files else None
        manifest_raw = data["manifest"].item()
    manifest = json.loads(str(manifest_raw))
    return labels, texture, manifest


def try_read_valid_cache(path: str | Path, expected_cache_key: str) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]] | None:
    """Return cache contents when schema and key match, otherwise ``None``."""

    p = Path(path)
    if not p.exists():
        return None
    labels, texture, manifest = read_npz(p)
    if int(manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        return None
    if str(manifest.get("cache_key", "")) != str(expected_cache_key):
        return None
    return labels, texture, manifest
