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

    This is intentionally a process-local app launcher while the integrated
    viewer loop continues to be split into smaller runtime modules.
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


@dataclass
class HexRuntime:
    """Small runtime facade used by tests and app orchestration."""

    launcher: HexAppLauncher
    _running: bool = False

    def poll_input(self) -> None:
        return None

    def step(self) -> None:
        if not self._running:
            self._running = True
            self.launcher.run()
            self._running = False

    def render(self) -> None:
        return None

    def is_running(self) -> bool:
        return bool(self._running)

    def pace(self) -> None:
        return None

    def close(self) -> None:
        self._running = False
        self.launcher.close()


__all__ = ["HexAppLauncher", "HexRuntime", "OmniSurgHexApp"]
