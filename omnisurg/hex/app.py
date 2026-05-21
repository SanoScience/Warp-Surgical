# SPDX-License-Identifier: Apache-2.0
"""Application orchestration for the packaged hex-grid runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .data.types import PreparedVolume


@dataclass
class OmniSurgHexApp:
    """Run the Newton/Warp hex-grid viewer from a prepared volume."""

    volume: PreparedVolume

    def run(self, argv: Sequence[str] | None = None) -> int:
        from . import app_runtime

        old_volume = app_runtime._OMNISURG_PREPARED_VOLUME
        old_texture = app_runtime._OMNISURG_PREPARED_TEXTURE_RGB
        app_runtime._OMNISURG_PREPARED_VOLUME = self.volume
        app_runtime._OMNISURG_PREPARED_TEXTURE_RGB = self.volume.texture_rgb
        try:
            return int(app_runtime.main(list(argv or ())))
        finally:
            app_runtime._OMNISURG_PREPARED_VOLUME = old_volume
            app_runtime._OMNISURG_PREPARED_TEXTURE_RGB = old_texture
