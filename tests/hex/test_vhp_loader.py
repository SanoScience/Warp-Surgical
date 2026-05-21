# SPDX-License-Identifier: Apache-2.0
"""Focused checks for the VHP JPEG loader."""

from __future__ import annotations

import numpy as np
from PIL import Image

from omnisurg.hex.io.vhp import load_vhp


def _write_jpeg(path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb, mode="RGB").save(path, quality=100, subsampling=0)


def test_vhp_downsamples_labels_but_keeps_cryo_at_source_resolution(tmp_path):
    height, width, slices = 4, 5, 3
    seg_rgb = np.full((height, width, 3), (254, 254, 254), dtype=np.uint8)

    for z in range(slices):
        cryo_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        cryo_rgb[..., 0] = z * 40
        cryo_rgb[..., 1] = np.arange(width, dtype=np.uint8)[None, :]
        cryo_rgb[..., 2] = np.arange(height, dtype=np.uint8)[:, None]
        _write_jpeg(tmp_path / f"i{z + 1:04d}.jpg", seg_rgb)
        _write_jpeg(tmp_path / f"t ({z + 1}).jpg", cryo_rgb)

    atlas, cryo = load_vhp(tmp_path, downsample=2, pad=1, use_cache=False, report_every=0)

    assert atlas.labels.shape == (4, 4, 3)
    assert cryo is not None
    assert cryo.shape == (8, 8, 6, 3)
    assert np.all(cryo[:2] == 0)
    assert np.all(cryo[:, :2] == 0)
    assert np.all(cryo[:, :, :2] == 0)


def test_vhp_can_load_labels_without_cryo_slices(tmp_path):
    height, width, slices = 4, 4, 2
    seg_rgb = np.full((height, width, 3), (254, 254, 254), dtype=np.uint8)
    for z in range(slices):
        _write_jpeg(tmp_path / f"i{z + 1:04d}.jpg", seg_rgb)

    atlas, cryo = load_vhp(
        tmp_path,
        downsample=2,
        pad=0,
        use_cache=False,
        report_every=0,
        load_cryo=False,
    )

    assert atlas.labels.shape == (2, 2, 1)
    assert cryo is None
