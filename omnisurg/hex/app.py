# SPDX-License-Identifier: Apache-2.0
"""Application orchestration for the packaged corner-grid runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .data.types import PreparedVolume


@dataclass
class OmniSurgHexApp:
    """Run the Newton/Warp corner-grid viewer from a prepared volume."""

    volume: PreparedVolume

    def run(self, argv: Sequence[str] | None = None) -> int:
        from . import _legacy_corner_app

        old_volume = _legacy_corner_app._OMNISURG_PREPARED_VOLUME
        old_texture = _legacy_corner_app._OMNISURG_PREPARED_TEXTURE_RGB
        _legacy_corner_app._OMNISURG_PREPARED_VOLUME = self.volume
        _legacy_corner_app._OMNISURG_PREPARED_TEXTURE_RGB = self.volume.texture_rgb
        try:
            return int(_legacy_corner_app.main(list(argv or ())))
        finally:
            _legacy_corner_app._OMNISURG_PREPARED_VOLUME = old_volume
            _legacy_corner_app._OMNISURG_PREPARED_TEXTURE_RGB = old_texture
