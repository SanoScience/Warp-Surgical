# SPDX-License-Identifier: Apache-2.0
"""Loader for the Voxel-Man / Visible-Human-style VHP dataset.

The folder layout is the one used by the reference UFRGS cutting code
(``EfficientPBDCutting/ufrgs/springGrid.h``):

* ``iNNNN.jpg`` — RGB segmentation slice N (one per atlas Z index). Pixel
  colours encode tissue class via a fixed palette.
* ``t (N).jpg`` — RGB cryosection slice N, co-registered to the same grid.

Palette definition comes from ``springGrid.h:getMaterial()``. Pixel
(0, 0, 0) is the JPEG canvas outside the body and is mapped to
``BACKGROUND`` here — the reference code aliased it with ``bone`` and
relied on a boundary rewrite, which we do not replicate.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import numpy as np
import warp as wp
from PIL import Image

from ..materials import (
    ARTERY,
    BACKGROUND,
    DEFAULT_MATERIALS,
    FAT,
    HEART,
    LIVER,
    LUNG,
    MUSCLE,
    STOMACH,
    MaterialTable,
)
from .digimouse import DigimouseAtlas

# Palette copied verbatim from EfficientPBDCutting/ufrgs/springGrid.h
# (getMaterial, lines 207-224). Second element is the canonical material id
# in omnisurg.hex.materials.DEFAULT_MATERIALS.
VHP_PALETTE: tuple[tuple[tuple[int, int, int], int], ...] = (
    ((0, 0, 0), BACKGROUND.id),
    ((254, 254, 254), FAT.id),
    ((254, 0, 0), MUSCLE.id),
    ((0, 255, 0), LIVER.id),
    ((152, 102, 153), HEART.id),
    ((51, 102, 103), STOMACH.id),
    ((255, 255, 0), LUNG.id),
    ((154, 154, 0), LUNG.id),
    ((0, 0, 254), ARTERY.id),
    ((0, 0, 152), ARTERY.id),
    ((0, 153, 203), ARTERY.id),
    ((0, 0, 102), ARTERY.id),
    ((0, 52, 102), ARTERY.id),
)


def _list_indexed(root: Path, pattern: re.Pattern[str]) -> list[Path]:
    hits: list[tuple[int, Path]] = []
    for p in root.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if m is None:
            continue
        hits.append((int(m.group(1)), p))
    hits.sort(key=lambda it: it[0])
    return [p for _, p in hits]


def _match_palette(slice_rgb: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    # slice_rgb: (H, W, 3) uint8 ; palette_rgb: (K, 3) float32
    diff = slice_rgb.astype(np.int32)[:, :, None, :] - palette_rgb[None, None, :, :].astype(np.int32)
    dist_sq = (diff * diff).sum(axis=-1)
    return np.argmin(dist_sq, axis=-1).astype(np.int32)


def _downsample_majority(vol: np.ndarray, factor: int, n_labels: int) -> np.ndarray:
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


def _load_slices(
    root: Path,
    *,
    seg_prefix: str = "i",
    cryo_prefix: str = "t",
    require_cryo: bool = True,
    verbose: bool = True,
) -> tuple[list[Path], list[Path]]:
    seg_re = re.compile(rf"^{re.escape(seg_prefix)}(\d+)\.jpg$", re.IGNORECASE)
    # Reference cryo names have the form `t (N).jpg` with a space.
    cryo_re = re.compile(rf"^{re.escape(cryo_prefix)}\s*\(?(\d+)\)?\.jpg$", re.IGNORECASE)
    seg = _list_indexed(root, seg_re)
    cryo = _list_indexed(root, cryo_re)
    if not seg:
        raise FileNotFoundError(f"no segmentation slices matching {seg_re.pattern} under {root}")
    if require_cryo and not cryo:
        raise FileNotFoundError(f"no cryo slices matching {cryo_re.pattern} under {root}")
    if require_cryo and len(seg) != len(cryo):
        raise ValueError(f"VHP slice counts differ: {len(seg)} seg vs {len(cryo)} cryo under {root}")
    if verbose:
        if require_cryo:
            print(f"[vhp] {len(seg)} seg + cryo slices under {root}")
        else:
            print(f"[vhp] {len(seg)} seg slices under {root}")
    return seg, cryo


def _palette_hash() -> str:
    h = hashlib.sha1()
    for (r, g, b), mid in VHP_PALETTE:
        h.update(bytes((r & 0xFF, g & 0xFF, b & 0xFF, mid & 0xFF)))
    return h.hexdigest()[:8]


def _default_cache_path(root: Path, downsample: int) -> Path:
    return root / ".cache" / f"vhp_ds{int(downsample)}_palette{_palette_hash()}.npz"


def _stack_shape(files: list[Path]) -> tuple[int, int, int]:
    with Image.open(files[0]) as im0:
        nx, ny = im0.size
    return int(nx), int(ny), len(files)


def _cropped_shape(shape_xyz: tuple[int, int, int], factor: int) -> tuple[int, int, int]:
    factor = max(1, int(factor))
    sx, sy, sz = shape_xyz
    sx -= sx % factor
    sy -= sy % factor
    sz -= sz % factor
    return sx, sy, sz


def _downsampled_shape(shape_xyz: tuple[int, int, int], factor: int) -> tuple[int, int, int]:
    factor = max(1, int(factor))
    sx, sy, sz = _cropped_shape(shape_xyz, factor)
    return sx // factor, sy // factor, sz // factor


def _crop_xyz_to_factor(vol: np.ndarray, factor: int) -> np.ndarray:
    sx, sy, sz = _cropped_shape(tuple(int(v) for v in vol.shape[:3]), factor)
    return np.ascontiguousarray(vol[:sx, :sy, :sz, ...])


def _parse_segmentation_jpegs(
    seg_files: list[Path],
    *,
    downsample: int,
    report_every: int,
) -> np.ndarray:
    palette_rgb = np.asarray([c for c, _ in VHP_PALETTE], dtype=np.int32)
    palette_to_mat = np.asarray([mid for _, mid in VHP_PALETTE], dtype=np.int32)

    nx, ny, nz = _stack_shape(seg_files)

    labels_zyx = np.empty((nz, ny, nx), dtype=np.uint8)

    for i, sp in enumerate(seg_files):
        with Image.open(sp) as im:
            seg_rgb = np.asarray(im.convert("RGB"))
        if seg_rgb.shape != (ny, nx, 3):
            raise ValueError(f"slice {i}: seg {seg_rgb.shape} != ({ny},{nx},3)")
        palette_idx = _match_palette(seg_rgb, palette_rgb)
        labels_zyx[i] = palette_to_mat[palette_idx].astype(np.uint8)
        if report_every and ((i + 1) % report_every == 0 or i + 1 == nz):
            print(f"  [vhp labels] {i + 1}/{nz}  {sp.name}")

    # (nz, ny, nx) -> (nx, ny, nz), matching the DigimouseAtlas convention.
    labels_xyz = np.ascontiguousarray(np.transpose(labels_zyx, (2, 1, 0)))

    n_labels = len(DEFAULT_MATERIALS)
    return _downsample_majority(labels_xyz, downsample, n_labels=n_labels)


def _load_cryo_jpegs(
    cryo_files: list[Path],
    *,
    downsample: int,
    report_every: int,
) -> np.ndarray:
    nx, ny, nz = _stack_shape(cryo_files)
    cryo_zyx = np.empty((nz, ny, nx, 3), dtype=np.uint8)

    for i, cp in enumerate(cryo_files):
        with Image.open(cp) as im:
            cryo_rgb = np.asarray(im.convert("RGB"))
        if cryo_rgb.shape != (ny, nx, 3):
            raise ValueError(f"slice {i}: cryo {cryo_rgb.shape} != ({ny},{nx},3)")
        cryo_zyx[i] = cryo_rgb
        if report_every and ((i + 1) % report_every == 0 or i + 1 == nz):
            print(f"  [vhp cryo] {i + 1}/{nz}  {cp.name}")

    # Keep the RGB cryosection at source-pixel resolution. We crop only the
    # edge remainder that the label downsampling also discards, so normalized
    # texture coordinates span the same anatomical extent as the simulation grid.
    cryo_xyz = np.ascontiguousarray(np.transpose(cryo_zyx, (2, 1, 0, 3)))
    return _crop_xyz_to_factor(cryo_xyz, downsample)


def _sources_newer_than(cache_path: Path, sources: list[Path]) -> bool:
    cache_mtime = cache_path.stat().st_mtime
    return any(p.stat().st_mtime > cache_mtime for p in sources)


def load_vhp(
    root: str | Path = "VHP",
    *,
    downsample: int = 16,
    voxel_size: float = 1.0e-3,
    seg_prefix: str = "i",
    cryo_prefix: str = "t",
    report_every: int = 60,
    cache_path: str | Path | None = None,
    use_cache: bool = True,
    rebuild_cache: bool = False,
    pad: int = 1,
    load_cryo: bool = True,
) -> tuple[DigimouseAtlas, np.ndarray | None]:
    """Load the VHP (Voxel-Man / Visible-Human) folder.

    Args:
        root: directory containing ``iNNNN.jpg`` segmentation slices and
            ``t (N).jpg`` cryosection slices.
        downsample: integer block factor applied uniformly in x/y/z to the
            segmentation labels after per-pixel palette matching. 1 keeps the
            native grid (large: 800x520x465 ≈ 193 MB of labels); 16 yields a
            ~50x32x29 simulation grid. The returned cryosection RGB volume is
            kept at source-pixel resolution, cropped to the same
            downsample-aligned extent.
        voxel_size: native isotropic voxel edge length in metres. Voxel-Man
            slices are ~1 mm isotropic; this becomes
            ``voxel_size * downsample`` after loading.
        seg_prefix: filename prefix of segmentation slices (default ``i``).
        cryo_prefix: filename prefix of cryosection slices (default ``t``).
        report_every: print progress every N slices (0 to silence).
        cache_path: ``.npz`` path to memoize the downsampled labels in.
            ``None`` picks ``<root>/.cache/vhp_ds<downsample>_palette<hash>.npz``.
            The palette-hash suffix guarantees stale caches are ignored when
            :data:`VHP_PALETTE` changes.
        use_cache: when ``True`` (default), read labels from ``cache_path`` if
            it exists and is newer than every segmentation JPEG. Set ``False``
            to force a full segmentation re-parse without touching the cache
            file. Cryo RGB slices are decoded from their source JPEGs whenever
            ``load_cryo`` is true so they stay at native resolution.
        rebuild_cache: when ``True``, re-parse the JPEGs and overwrite
            ``cache_path`` even if a valid cache exists.
        pad: empty-voxel margin added to every face of the downsampled
            volume so Marching Cubes sees an exterior boundary where the
            body is flush against the atlas (head/feet slabs). Default 1
            voxel. The cryo RGB volume receives matching black padding at
            native-pixel scale (``pad * downsample``) when ``load_cryo`` is
            true. Applied after the cache round-trip, so changing this does
            not invalidate cached files.
        load_cryo: when ``True`` (default), load and return the native-resolution
            cryosection RGB volume. Set ``False`` when only labels are needed
            (for example with ``--seg-as-cryo`` diagnostics).

    Returns:
        ``(atlas, cryo_host)`` where ``atlas`` is a ``DigimouseAtlas``
        carrying the downsampled label volume with canonical material ids.
        ``cryo_host`` is a source-resolution ``(nx, ny, nz, 3)`` uint8 volume
        from the cryosection slices, cropped and padded to align with the atlas
        UV domain, or ``None`` when ``load_cryo=False``. Pass ``cryo_host`` to
        :func:`load_vhp_cryo_device` to build a sampling-ready ``wp.array3d``.
    """
    root = Path(root)
    seg_files, cryo_files = _load_slices(
        root,
        seg_prefix=seg_prefix,
        cryo_prefix=cryo_prefix,
        require_cryo=load_cryo,
    )
    seg_shape = _stack_shape(seg_files)
    if load_cryo:
        cryo_shape = _stack_shape(cryo_files)
        if cryo_shape != seg_shape:
            raise ValueError(f"VHP slice dimensions differ: seg {seg_shape} vs cryo {cryo_shape}")

    cache = Path(cache_path) if cache_path is not None else _default_cache_path(root, downsample)
    labels_ds: np.ndarray | None = None

    if use_cache and not rebuild_cache and cache.exists():
        if _sources_newer_than(cache, seg_files):
            print(f"[vhp] cache {cache.name} stale vs segmentation slices; rebuilding")
        else:
            t0 = time.perf_counter()
            with np.load(cache) as data:
                cached_labels = np.ascontiguousarray(data["labels"])
            expected_labels_shape = _downsampled_shape(seg_shape, downsample)
            if tuple(int(v) for v in cached_labels.shape) == expected_labels_shape:
                labels_ds = cached_labels
                print(f"[vhp] loaded label cache {cache} in {time.perf_counter() - t0:.2f}s")
            else:
                print(
                    f"[vhp] cache {cache.name} shape {cached_labels.shape} "
                    f"!= expected {expected_labels_shape}; rebuilding"
                )

    if labels_ds is None:
        t0 = time.perf_counter()
        labels_ds = _parse_segmentation_jpegs(
            seg_files,
            downsample=downsample,
            report_every=report_every,
        )
        print(f"[vhp] parsed {len(seg_files)} segmentation slices in {time.perf_counter() - t0:.1f}s")
        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, labels=labels_ds)
            size_mb = cache.stat().st_size / 1e6
            print(f"[vhp] wrote label cache {cache} ({size_mb:.1f} MB)")

    cryo_host: np.ndarray | None = None
    if load_cryo:
        t0 = time.perf_counter()
        cryo_host = _load_cryo_jpegs(
            cryo_files,
            downsample=downsample,
            report_every=report_every,
        )
        print(f"[vhp] loaded {len(cryo_files)} full-res cryo slices in {time.perf_counter() - t0:.1f}s")

    if pad > 0:
        p = int(pad)
        labels_ds = np.pad(labels_ds, p, mode="constant", constant_values=0)
        if cryo_host is not None:
            cryo_pad = p * max(1, int(downsample))
            cryo_host = np.pad(
                cryo_host,
                ((cryo_pad, cryo_pad), (cryo_pad, cryo_pad), (cryo_pad, cryo_pad), (0, 0)),
                mode="constant",
                constant_values=0,
            )

    table = MaterialTable(DEFAULT_MATERIALS)
    atlas = DigimouseAtlas(
        labels=labels_ds,
        voxel_size=voxel_size * max(1, int(downsample)),
        materials=table,
    )
    return atlas, cryo_host


def load_vhp_cryo_device(
    cryo_host: np.ndarray,
    *,
    device: str | wp.context.Device | None = None,
) -> wp.array:
    """Upload a VHP cryosection host volume to a Warp device as ``vec3`` in [0, 1]."""
    if cryo_host.ndim != 4 or cryo_host.shape[-1] != 3 or cryo_host.dtype != np.uint8:
        raise ValueError(
            f"expected (nx, ny, nz, 3) uint8 cryo volume, got shape={cryo_host.shape} dtype={cryo_host.dtype}"
        )
    as_float = np.ascontiguousarray(cryo_host).astype(np.float32) / 255.0
    return wp.array(as_float, dtype=wp.vec3, device=device)


def seg_as_cryo(atlas: DigimouseAtlas) -> np.ndarray:
    """Build a ``(nx, ny, nz, 3)`` uint8 volume where each voxel holds the
    material-palette colour of its label.

    Use as a drop-in replacement for the cryo texture when diagnosing
    texture-sampling orientation: anything that shows up as the wrong colour
    on the rendered MC surface proves that sampling is misaligned with the
    label volume it was derived from.
    """
    palette = (
        (np.asarray([m.color for m in atlas.materials.materials], dtype=np.float32) * 255.0)
        .clip(0.0, 255.0)
        .astype(np.uint8)
    )
    return np.ascontiguousarray(palette[atlas.labels])
