# SPDX-License-Identifier: Apache-2.0
"""Internal resource ownership helpers for hex runtime drivers."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

import newton

from .haptic import FallbackInput, HapticUnavailable, InputPose, open_haptic_inputs, open_minimou_inputs
from .runtime_config import INSTRUMENT_COUNT


class HexRuntimeResourceOwner:
    """Own closeable startup resources shared by the hex runtime drivers."""

    def __init__(self, *, input_devices: Sequence[Any] | None = None, render_bridge: Any | None = None) -> None:
        self.input_devices = list(input_devices or ())
        self.render_bridge = render_bridge
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        input_devices = list(self.input_devices)
        render_bridge = self.render_bridge
        self.input_devices = []
        self.render_bridge = None

        first_error: BaseException | None = None
        first_traceback = None

        for input_device in reversed(input_devices):
            close = getattr(input_device, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:  # noqa: BLE001
                if first_error is None:
                    first_error = exc
                    first_traceback = exc.__traceback__

        if render_bridge is not None:
            close = getattr(render_bridge, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001
                    if first_error is None:
                        first_error = exc
                        first_traceback = exc.__traceback__

        if first_error is not None:
            raise first_error.with_traceback(first_traceback)


def _close_runtime_resources_for_exception(owner: HexRuntimeResourceOwner) -> None:
    """Close owned runtime resources during exception unwinds without raising."""

    try:
        owner.close()
    except BaseException as exc:  # noqa: BLE001
        print(
            "[hex runtime] resource cleanup failed during exception unwind: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def _fallback_instrument_inputs(count: int) -> list[FallbackInput]:
    return [FallbackInput(InputPose(position=(0.0, 0.0, 0.0), valid=True)) for _ in range(count)]


def _close_inputs(input_devices: Sequence[Any]) -> None:
    first_error: BaseException | None = None
    first_traceback = None
    for input_device in reversed(input_devices):
        close = getattr(input_device, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception as exc:  # noqa: BLE001
            if first_error is None:
                first_error = exc
                first_traceback = exc.__traceback__
    if first_error is not None:
        raise first_error.with_traceback(first_traceback)


def open_hex_instrument_inputs(args, *, expected_count: int | None = None) -> list[Any]:
    """Open the configured instrument inputs and validate the expected count."""

    backend = str(args.input_backend)
    if backend == "off":
        return []

    count = INSTRUMENT_COUNT if expected_count is None else int(expected_count)
    if backend == "fallback":
        input_devices: list[Any] = _fallback_instrument_inputs(count)
    elif backend == "minimou":
        input_devices = open_minimou_inputs(count=count)
    elif backend == "openhaptics":
        input_devices = open_haptic_inputs([str(args.device_name), str(args.left_device_name)])
    else:
        raise ValueError(f"unknown input backend {backend!r}")

    if expected_count is not None and len(input_devices) != int(expected_count):
        try:
            _close_inputs(input_devices)
        except Exception as exc:  # noqa: BLE001
            raise HapticUnavailable(f"expected {expected_count} devices, got {len(input_devices)}") from exc
        raise HapticUnavailable(f"expected {expected_count} devices, got {len(input_devices)}")

    return input_devices


def build_hex_usd_viewer(path: str, frame_loop_config):
    """Build a USD viewer with the runtime frame-loop limit."""

    return newton.viewer.ViewerUSD(path, num_frames=frame_loop_config.max_frames)
