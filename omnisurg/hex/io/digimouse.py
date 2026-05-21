# SPDX-License-Identifier: Apache-2.0
"""Loader for the Digimouse anatomical atlas.

The atlas is distributed as an Analyze 7.5 ``.hdr`` / ``.img`` pair.
On disk (``Digimouse/atlas/atlas/``) it is a 380 x 992 x 208 uint8 volume at
0.1 mm isotropic voxel spacing. Labels 0..21 name tissues; see the sibling
``atlas_380x992x208.txt`` for the mapping.
"""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..materials import MaterialTable, digimouse_material_table


@dataclass(frozen=True)
class DigimouseAtlas:
    """Parsed Digimouse atlas volume remapped to the canonical material palette.

    Attributes:
        labels: ``uint8[nx, ny, nz]`` material id per voxel, indexing into
            ``materials``. The 0 label means background (empty).
        voxel_size: isotropic voxel edge length in metres (0.1 mm = 1e-4 m for
            the native atlas; coarser after downsampling).
        materials: canonical palette the labels index into.
        origin: world-space position of the ``[0,0,0]`` voxel centre in metres.
    """

    labels: np.ndarray
    voxel_size: float
    materials: MaterialTable
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    metadata: dict = field(default_factory=dict)


def _read_analyze_header(hdr_path: Path) -> tuple[tuple[int, int, int], tuple[float, float, float], int]:
    """Parse the minimal Analyze 7.5 fields we care about.

    Returns:
        (dim_xyz, pixdim_xyz_mm, datatype_code)
    """
    data = hdr_path.read_bytes()
    if len(data) != 348:
        raise ValueError(f"expected 348-byte Analyze header, got {len(data)} in {hdr_path}")
    (sizeof_hdr,) = struct.unpack("<i", data[0:4])
    if sizeof_hdr != 348:
        raise ValueError(f"header sizeof_hdr = {sizeof_hdr}, expected 348 (little-endian)")
    dim = struct.unpack("<8h", data[40:56])
    if dim[0] < 3:
        raise ValueError(f"header declares {dim[0]} dimensions, expected >=3")
    datatype = struct.unpack("<h", data[70:72])[0]
    pixdim = struct.unpack("<8f", data[76:108])
    return (int(dim[1]), int(dim[2]), int(dim[3])), (pixdim[1], pixdim[2], pixdim[3]), datatype


def _downsample_majority(vol: np.ndarray, factor: int, n_labels: int) -> np.ndarray:
    """Block-majority downsample of a categorical label volume.

    Crops to a multiple of ``factor`` in each axis, then takes the most common
    label in each ``factor**3`` block via one-hot counts.
    """
    if factor <= 1:
        return vol
    sx, sy, sz = vol.shape
    sx -= sx % factor
    sy -= sy % factor
    sz -= sz % factor
    v = vol[:sx, :sy, :sz]
    v = v.reshape(sx // factor, factor, sy // factor, factor, sz // factor, factor)
    v = v.transpose(0, 2, 4, 1, 3, 5).reshape(-1, factor**3)

    counts = np.zeros((v.shape[0], n_labels), dtype=np.int32)
    for lbl in range(n_labels):
        counts[:, lbl] = (v == lbl).sum(axis=1)
    winners = counts.argmax(axis=1).astype(vol.dtype)
    return winners.reshape(sx // factor, sy // factor, sz // factor)


def _remap_hash() -> str:
    table, remap = digimouse_material_table()
    h = hashlib.sha1()
    h.update(remap.tobytes())
    for m in table.materials:
        h.update(m.name.encode())
        h.update(struct.pack("<i", int(m.id)))
    return h.hexdigest()[:8]


def _default_cache_path(atlas_dir: Path, downsample: int) -> Path:
    return atlas_dir / ".cache" / f"digimouse_ds{int(downsample)}_remap{_remap_hash()}.npz"


def _sources_newer_than(cache_path: Path, sources: list[Path]) -> bool:
    cache_mtime = cache_path.stat().st_mtime
    return any(p.stat().st_mtime > cache_mtime for p in sources)


def load_digimouse(
    atlas_dir: str | Path = "Digimouse/atlas/atlas",
    downsample: int = 8,
    crop: tuple[slice, slice, slice] | None = None,
    *,
    pad: int = 1,
    cache_path: str | Path | None = None,
    use_cache: bool = True,
    rebuild_cache: bool = False,
) -> DigimouseAtlas:
    """Load the Digimouse atlas and return it remapped to canonical materials.

    Args:
        atlas_dir: directory containing ``atlas_380x992x208.{hdr,img}``.
        downsample: integer block factor for majority-vote downsampling. 1
            keeps the native 380x992x208 grid; 8 (the default) yields a
            47x124x26 grid (~150k occupied voxels) which matches the paper's
            100k-particle regime.
        crop: optional per-axis slice applied **before** downsampling, in
            native voxel coordinates. Use to select the torso only. The
            cache is bypassed when ``crop`` is set since the cache key does
            not encode the slice.
        pad: empty-voxel margin added to every face of the downsampled
            volume so Marching Cubes sees an exterior boundary where the
            atlas touches a grid edge (e.g. Z=max for the native mouse
            pose). Default 1 voxel; pass 0 to disable. Applied after the
            cache round-trip, so changing this does not invalidate cached
            files.
        cache_path: ``.npz`` path to memoize the downsampled labels in.
            ``None`` picks ``<atlas_dir>/.cache/digimouse_ds<downsample>_remap<hash>.npz``.
            The remap-hash suffix guarantees stale caches are ignored when
            the canonical material palette changes.
        use_cache: when ``True`` (default), read from ``cache_path`` if it
            exists and is newer than the source ``.hdr``/``.img``. Set
            ``False`` to force a full re-parse without touching the cache
            file.
        rebuild_cache: when ``True``, re-parse the atlas and overwrite
            ``cache_path`` even if a valid cache exists.

    Returns:
        ``DigimouseAtlas`` with labels already mapped into ``MaterialTable``
        indices (not raw atlas labels).
    """
    atlas_dir = Path(atlas_dir)
    hdr = atlas_dir / "atlas_380x992x208.hdr"
    img = atlas_dir / "atlas_380x992x208.img"
    if not hdr.exists() or not img.exists():
        raise FileNotFoundError(f"Digimouse atlas not found under {atlas_dir}")

    table, remap = digimouse_material_table()

    can_use_cache = use_cache and crop is None
    cache = Path(cache_path) if cache_path is not None else _default_cache_path(atlas_dir, downsample)

    downsampled: np.ndarray | None = None
    voxel_size_m: float | None = None

    if can_use_cache and not rebuild_cache and cache.exists():
        if _sources_newer_than(cache, [hdr, img]):
            print(f"[digimouse] cache {cache.name} stale vs source atlas; rebuilding")
        else:
            t0 = time.perf_counter()
            with np.load(cache) as data:
                downsampled = np.ascontiguousarray(data["labels"])
                voxel_size_m = float(data["voxel_size"])
            print(f"[digimouse] loaded cache {cache} in {time.perf_counter() - t0:.2f}s")

    if downsampled is None or voxel_size_m is None:
        t0 = time.perf_counter()
        (nx, ny, nz), (dx, dy, dz), datatype = _read_analyze_header(hdr)
        if datatype != 2:
            raise ValueError(f"expected uint8 (datatype=2) atlas, got datatype={datatype}")
        if abs(dx - dy) > 1e-5 or abs(dx - dz) > 1e-5:
            raise ValueError(f"non-isotropic voxel ({dx}, {dy}, {dz}); loader assumes isotropic")

        raw = np.fromfile(img, dtype=np.uint8)
        if raw.size != nx * ny * nz:
            raise ValueError(f"atlas .img size {raw.size} != nx*ny*nz = {nx * ny * nz}")
        # Analyze stores voxels in fortran / column-major order; reshape accordingly.
        atlas = raw.reshape((nz, ny, nx)).transpose(2, 1, 0)

        if crop is not None:
            atlas = atlas[crop]

        if atlas.max() >= remap.size:
            raise ValueError(f"atlas label {atlas.max()} exceeds remap table size {remap.size}")

        mapped = remap[atlas]
        downsampled = _downsample_majority(mapped, downsample, n_labels=len(table))
        voxel_size_m = dx * 1e-3 * downsample  # Analyze pixdim is in mm; convert to metres.
        print(f"[digimouse] parsed atlas in {time.perf_counter() - t0:.2f}s")

        if can_use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, labels=downsampled, voxel_size=np.float32(voxel_size_m))
            size_mb = cache.stat().st_size / 1e6
            print(f"[digimouse] wrote cache {cache} ({size_mb:.1f} MB)")

    if pad > 0:
        downsampled = np.pad(downsampled, int(pad), mode="constant", constant_values=0)
    return DigimouseAtlas(labels=downsampled, voxel_size=voxel_size_m, materials=table)
