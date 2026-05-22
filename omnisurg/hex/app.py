# SPDX-License-Identifier: Apache-2.0
"""Application orchestration for the packaged hex-grid runtime."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from .data.types import PreparedVolume

_HEX_RUNTIME_DRIVER_ARG = "--hex-runtime-driver"


def _runtime_driver_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omnisurg hex", add_help=False, allow_abbrev=False)
    parser.add_argument(
        _HEX_RUNTIME_DRIVER_ARG,
        choices=("app", "session"),
        default="app",
        help=argparse.SUPPRESS,
    )
    return parser


def _split_runtime_driver(argv: Sequence[str] | None) -> tuple[str, Sequence[str] | None]:
    if argv is None:
        return "app", None
    if not any(arg == _HEX_RUNTIME_DRIVER_ARG or arg.startswith(f"{_HEX_RUNTIME_DRIVER_ARG}=") for arg in argv):
        return "app", argv

    args = list(argv)
    namespace, runtime_args = _runtime_driver_parser().parse_known_args(args)
    return str(namespace.hex_runtime_driver), runtime_args


@dataclass
class OmniSurgHexApp:
    """Run the Newton/Warp hex-grid viewer from a prepared volume."""

    volume: PreparedVolume

    def run(self, argv: Sequence[str] | None = None) -> int:
        driver, runtime_args = _split_runtime_driver(argv)

        if driver == "session":
            from .runtime import HexAppLauncher

            return int(HexAppLauncher(self.volume, runtime_args or ()).run())

        from . import app_runtime

        return int(app_runtime.run_prepared_volume(self.volume, runtime_args))
