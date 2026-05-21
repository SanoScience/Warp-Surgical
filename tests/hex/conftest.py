from __future__ import annotations

import os
from pathlib import Path


def _configure_warp_paths() -> None:
    root = Path(__file__).resolve().parent.parent
    cache_dir = root / ".warp-cache"
    temp_dir = root / ".warp-temp"
    cache_dir.mkdir(exist_ok=True)
    temp_dir.mkdir(exist_ok=True)

    os.environ.setdefault("WARP_CACHE_PATH", str(cache_dir))
    os.environ.setdefault("TEMP", str(temp_dir))
    os.environ.setdefault("TMP", str(temp_dir))


_configure_warp_paths()

import warp as wp

wp.config.use_precompiled_headers = False
