# SPDX-License-Identifier: Apache-2.0
"""Internal resource ownership helpers for hex runtime drivers."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

import newton

from omnisurg.input import factory as input_factory
from omnisurg.input.sources import ControllerSample, InputSource

from .haptic import InputPose
from .runtime_config import INSTRUMENT_COUNT


class HapticUnavailable(RuntimeError):
    """Raised when configured hex instrument input cannot be opened."""


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

    roles = ("right", "left")[: INSTRUMENT_COUNT if expected_count is None else int(expected_count)]
    configs: list[input_factory.RoleInputConfig] = []
    for role in roles:
        backend = input_factory.normalize_input_backend(
            getattr(args, f"{role}_input_backend", getattr(args, "input_backend", "off"))
        )
        configs.append(
            input_factory.RoleInputConfig(
                role=role,
                backend=backend,
                device_name=getattr(
                    args,
                    f"{role}_device_name",
                    getattr(args, "device_name", None) if role == "right" else None,
                ),
                device_index=getattr(args, f"{role}_device_index", 0 if role == "right" else 1),
                replay_path=getattr(args, f"{role}_replay", None),
                force_feedback=False,
            )
        )

    if all(config.backend == "off" for config in configs):
        return []

    try:
        result = input_factory.open_input_sources(configs, require_all=True)
    except input_factory.InputOpenError as exc:
        raise HapticUnavailable(str(exc)) from exc

    input_devices: list[Any] = []
    for config in configs:
        if config.backend == "off":
            input_devices.append(_OffInputPoseAdapter())
        elif config.role in result.sources:
            input_devices.append(_ControllerSampleInputPoseAdapter(config.role, result.sources[config.role]))

    expected_active_count = sum(1 for config in configs if config.backend != "off")
    if expected_count is not None and expected_active_count == int(expected_count) and len(input_devices) != int(expected_count):
        try:
            _close_inputs(input_devices)
        except Exception as exc:  # noqa: BLE001
            raise HapticUnavailable(f"expected {expected_count} devices, got {len(input_devices)}") from exc
        raise HapticUnavailable(f"expected {expected_count} devices, got {len(input_devices)}")

    return input_devices


class _ControllerSampleInputPoseAdapter:
    """Hex compatibility layer from canonical ControllerSample sources to InputPose."""

    def __init__(self, role: str, source: InputSource) -> None:
        self.role = role
        self._source = source

    def poll(self) -> InputPose:
        sample = ControllerSample.from_sample_dict(self._source.poll())
        if not sample.valid:
            return InputPose(valid=False)

        position = (
            (0.0, 0.0, 0.0)
            if sample.position is None
            else tuple(float(value) for value in sample.position[:3])
        )
        quaternion = (
            (0.0, 0.0, 0.0, 1.0)
            if sample.rotation is None
            else tuple(float(value) for value in sample.rotation[:4])
        )
        return InputPose(
            position=position,
            quaternion=quaternion,
            button1=bool(sample.button),
            button2=bool(sample.button2),
            tool_pos=float(sample.tool_scalar),
            grip=float(sample.grip),
            handle_pos=float(sample.handle_pos),
            handle_active=bool(sample.handle_active),
            valid=True,
        )

    def close(self) -> None:
        self._source.close()

    def angles_degrees(self):
        ctrl = getattr(self._source, "_ctrl", None)
        getter = getattr(ctrl, "angles_degrees", None)
        if callable(getter):
            return getter()
        return None


class _OffInputPoseAdapter:
    def poll(self) -> InputPose:
        return InputPose(valid=False)

    def close(self) -> None:
        pass


def build_hex_usd_viewer(path: str, frame_loop_config):
    """Build a USD viewer with the runtime frame-loop limit."""

    return newton.viewer.ViewerUSD(path, num_frames=frame_loop_config.max_frames)
