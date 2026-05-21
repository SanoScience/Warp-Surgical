# SPDX-License-Identifier: Apache-2.0
"""Loader for the packed cryosection 3D texture.

Pairs with ``tools/build_cryo_texture.py``. The on-disk file is a plain
``numpy.save`` of a ``(nx, ny, nz, 3)`` ``uint8`` volume - indexed the same
way as the Digimouse atlas so a per-voxel ``(x, y, z)`` reads the tissue
colour at that anatomical location.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp


def load_cryo_texture(
    path: str | Path = "Digimouse/cryo_texture.npy",
    *,
    mmap: bool = True,
    device: str | wp.context.Device | None = None,
) -> tuple[np.ndarray, wp.array]:
    """Load the packed cryosection volume from disk.

    Args:
        path: ``.npy`` file produced by ``tools/build_cryo_texture.py``.
        mmap: Memory-map the file instead of reading it into memory. On
            disk it is ~235 MB; mmap keeps RAM footprint low until the
            Warp array upload actually touches pages.
        device: Warp device for the GPU copy.

    Returns:
        ``(host_volume, device_volume)`` where ``host_volume`` is the
        ``(nx, ny, nz, 3)`` uint8 array (mmap or not) and
        ``device_volume`` is a ``wp.array3d(dtype=wp.vec3)`` on ``device``
        with values in ``[0, 1]`` ready to be sampled by Warp kernels.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python tools/build_cryo_texture.py` first"
        )
    host = np.load(path, mmap_mode="r" if mmap else None)
    if host.ndim != 4 or host.shape[-1] != 3 or host.dtype != np.uint8:
        raise ValueError(
            f"expected (nx, ny, nz, 3) uint8 volume, got shape={host.shape} dtype={host.dtype}"
        )

    # Convert to float [0, 1] vec3 on the target device. The volume is
    # small enough that a one-shot upload is cleaner than streaming tiles.
    as_float = (np.ascontiguousarray(host).astype(np.float32) / 255.0)
    device_vol = wp.array(as_float, dtype=wp.vec3, device=device)
    return host, device_vol
