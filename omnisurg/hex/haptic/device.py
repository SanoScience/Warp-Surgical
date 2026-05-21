# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper over pyopenhaptics that exposes 6-DOF input only.

The device is polled at ~1 kHz by the OpenHaptics async scheduler; this
module snapshots the latest pose into a thread-safe :class:`InputPose`
record that the simulation loop reads from at frame rate. No force output
is sent.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

from .frames import matrix_to_quaternion, minimou_orientation_to_quaternion, minimou_position_to_adapter


class HapticUnavailable(RuntimeError):
    """Raised when the OpenHaptics SDK is not installed or the device is absent."""


@dataclass
class InputPose:
    """Latest snapshot of the device.

    Attributes:
        position: tool tip position in device-space millimetres (OpenHaptics
            returns mm). Callers apply whatever world-space scaling they want.
        quaternion: tool orientation as ``(x, y, z, w)``.
        button1: first front button state.
        button2: second front button state (None if the device only has one).
        tool_pos: MiniMou tool-position scalar used as the cut trigger.
        valid: False until the scheduler has produced at least one update.
    """

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    button1: bool = False
    button2: bool = False
    tool_pos: float = 0.0
    valid: bool = False


class HapticInput:
    """Phantom Omni driver providing pose + button state only.

    Construction loads ``HD.dll``/``libHD.so`` via ctypes and starts the
    OpenHaptics async scheduler. ``poll()`` returns the latest snapshot
    written by the scheduler thread. ``close()`` stops the scheduler and
    releases the device. Use as a context manager to guarantee cleanup.
    """

    def __init__(self, device_name: str = "Default Device", *, start_scheduler: bool = True):
        # The pyopenhaptics submodule loads HD.dll at import time; catch that
        # here so the caller can fall back gracefully on machines without the
        # OpenHaptics SDK or the device itself.
        try:
            from .pyopenhaptics import hd as _hd  # noqa: PLC0415
            from .pyopenhaptics.hd_callback import hd_callback  # noqa: PLC0415
            from .pyopenhaptics.hd_define import HD_BAD_HANDLE  # noqa: PLC0415
            from .pyopenhaptics.hd_device import HapticDevice  # noqa: PLC0415
        except (OSError, ImportError) as exc:
            raise HapticUnavailable(f"OpenHaptics SDK unavailable: {exc}") from exc

        self._hd = _hd
        self._pose = InputPose()
        self._lock = threading.Lock()
        self._closed = False

        try:
            self._device = HapticDevice(
                device_name=device_name,
                scheduler_type="async",
                auto_start_scheduler=False,
                enable_force_output=False,
            )
        except Exception as exc:
            raise HapticUnavailable(f"device init failed: {exc}") from exc
        if getattr(self._device, "id", HD_BAD_HANDLE) == HD_BAD_HANDLE:
            raise HapticUnavailable(f'device "{device_name}" not found')

        # The async callback runs at device frequency (~1 kHz on a Phantom
        # Omni). It only snapshots this device's state - no force is sent.
        @hd_callback(device_id=self._device.id)
        def _update():
            try:
                transform = _hd.get_transform()
                buttons = _hd.get_buttons()
                px = float(transform[3][0])
                py = float(transform[3][1])
                pz = float(transform[3][2])
                quat = matrix_to_quaternion(
                    (
                        (float(transform[0][0]), float(transform[0][1]), float(transform[0][2])),
                        (float(transform[1][0]), float(transform[1][1]), float(transform[1][2])),
                        (float(transform[2][0]), float(transform[2][1]), float(transform[2][2])),
                    )
                )
                b1 = bool(buttons & 0x01)
                b2 = bool(buttons & 0x02)
                with self._lock:
                    self._pose.position = (px, py, pz)
                    self._pose.quaternion = quat
                    self._pose.button1 = b1
                    self._pose.button2 = b2
                    self._pose.tool_pos = 0.0
                    self._pose.valid = True
            except Exception:
                # Swallow per-frame errors so the scheduler thread does not
                # crash the process; the last good snapshot remains visible.
                pass

        self._callback_ref = _update  # keep ctypes callback alive
        try:
            self._device.scheduler(_update, "async")
            if start_scheduler:
                HapticDevice.start_scheduler()
        except Exception as exc:
            self._device.close()
            raise HapticUnavailable(f"device scheduler failed: {exc}") from exc

    def poll(self) -> InputPose:
        """Return a copy of the latest snapshot."""
        with self._lock:
            return InputPose(
                position=self._pose.position,
                quaternion=self._pose.quaternion,
                button1=self._pose.button1,
                button2=self._pose.button2,
                tool_pos=self._pose.tool_pos,
                valid=self._pose.valid,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._device.close()
        except Exception:
            pass

    def __enter__(self) -> HapticInput:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def open_haptic_inputs(device_names: Sequence[str]) -> list[HapticInput]:
    """Open several OpenHaptics devices before starting the shared scheduler."""
    inputs: list[HapticInput] = []
    try:
        for name in device_names:
            inputs.append(HapticInput(device_name=name, start_scheduler=False))

        if inputs:
            # All devices are initialized and callbacks are scheduled; now start
            # the global HD scheduler once, matching the dual-device C samples.
            inputs[0]._device.start_scheduler()
        return inputs
    except Exception:
        for input_device in reversed(inputs):
            input_device.close()
        raise


class MiniMouInput:
    """Follou MiniMou adapter exposing the same pose API as HapticInput."""

    def __init__(self, controller):
        self._controller = controller
        self._angles_degrees: tuple[float, float, float] | None = None
        self._closed = False

    @classmethod
    def discover(cls, count: int = 1) -> list[MiniMouInput]:
        try:
            from follou.devices.minimou import MiniMou  # noqa: PLC0415
            from follou.manager import DeviceManager  # noqa: PLC0415
        except (OSError, ImportError, FileNotFoundError) as exc:
            raise HapticUnavailable(f"Follou MiniMou support unavailable: {exc}") from exc

        try:
            manager = DeviceManager()
        except Exception as exc:
            raise HapticUnavailable(f"Follou device discovery failed: {exc}") from exc

        inputs: list[MiniMouInput] = []
        for idx in range(count):
            controller = manager.get_device_controller(MiniMou, idx)
            if controller is None:
                break
            inputs.append(cls(controller))

        if len(inputs) < count:
            for input_device in reversed(inputs):
                input_device.close()
            raise HapticUnavailable(f"found {len(inputs)} MiniMou device(s), need {count}")
        return inputs

    def poll(self) -> InputPose:
        try:
            self._controller.perform_update()
            pos = self._controller.get_position()
            angles = (
                float(self._controller.get_rot_angle()),
                float(self._controller.get_pitch_angle()),
                float(self._controller.get_yaw_angle()),
            )
            orientation = self._controller.get_orientation()
            tool_pos = float(getattr(self._controller, "get_tool_pos", lambda: 0.0)())

            self._angles_degrees = angles
            return InputPose(
                position=minimou_position_to_adapter(pos),
                quaternion=minimou_orientation_to_quaternion(orientation),
                button1=tool_pos < 0.1,
                tool_pos=tool_pos,
                valid=True,
            )
        except Exception:
            return InputPose(valid=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._controller.close()
        except Exception:
            pass

    def __enter__(self) -> MiniMouInput:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def angles_degrees(self) -> tuple[float, float, float] | None:
        """Return MiniMou joint angles as (rot, pitch, yaw)."""
        if self._angles_degrees is not None:
            return self._angles_degrees
        try:
            return (
                float(self._controller.get_rot_angle()),
                float(self._controller.get_pitch_angle()),
                float(self._controller.get_yaw_angle()),
            )
        except Exception:
            return None


def open_minimou_inputs(count: int = 1) -> list[MiniMouInput]:
    """Discover Follou MiniMou devices and expose them as InputPose sources."""
    return MiniMouInput.discover(count=count)


@dataclass
class FallbackInput:
    """Fixed-pose input used when no haptic device is present.

    Callers can write to :attr:`pose` (e.g. from a keyboard handler) to drive
    the tool without hardware. The interface matches :class:`HapticInput`.
    """

    pose: InputPose = field(default_factory=lambda: InputPose(valid=True))

    def poll(self) -> InputPose:
        return InputPose(
            position=self.pose.position,
            quaternion=self.pose.quaternion,
            button1=self.pose.button1,
            button2=self.pose.button2,
            tool_pos=self.pose.tool_pos,
            valid=True,
        )

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    def __enter__(self) -> FallbackInput:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        return None
