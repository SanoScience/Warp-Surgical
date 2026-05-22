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

        return int(app_runtime.run_prepared_volume(self.volume, argv))
