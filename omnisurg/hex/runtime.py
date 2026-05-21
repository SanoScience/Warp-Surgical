# SPDX-License-Identifier: Apache-2.0
"""Runtime entry points for OmniSurg Hex."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .app import OmniSurgHexApp
from .data.types import PreparedVolume


@dataclass
class HexAppLauncher:
    """Launch the packaged hex app.

    This is intentionally not a per-frame runtime. The current hex solver is
    still hosted by the integrated corner-grid app, so callers should treat this
    as a process-local app launcher until the legacy loop is decomposed.
    """

    volume: PreparedVolume
    argv: Sequence[str] = field(default_factory=tuple)
    return_code: int | None = None
    _closed: bool = False

    def run(self, argv: Sequence[str] | None = None) -> int:
        args = list(self.argv if argv is None else argv)
        self.return_code = int(OmniSurgHexApp(self.volume).run(args))
        self._closed = True
        return self.return_code

    def run_exit_after_init(self) -> int:
        args = list(self.argv)
        if "--exit-after-init" not in args:
            args.append("--exit-after-init")
        return self.run(args)

    def close(self) -> None:
        self._closed = True


__all__ = ["HexAppLauncher", "OmniSurgHexApp"]
