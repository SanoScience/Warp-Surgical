# SPDX-License-Identifier: Apache-2.0
"""Hex compatibility adapters for canonical OmniSurg input sources."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


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
        grip: normalized handle/tool closure in [0, 1], where 1 is closed.
        handle_pos: raw MiniMou handle opening value when available.
        handle_active: MiniMou handle activity bit when available.
        valid: False until the scheduler has produced at least one update.
    """

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    button1: bool = False
    button2: bool = False
    tool_pos: float = 0.0
    grip: float = 0.0
    handle_pos: float = 0.0
    handle_active: bool = False
    valid: bool = False


class HapticInput:
    """Compatibility adapter over the canonical OpenHaptics source."""

    def __init__(self, device_name: str = "Default Device", *, start_scheduler: bool = True):
        del start_scheduler
        try:
            from omnisurg.input.sources import LiveHapticSource  # noqa: PLC0415

            self._source = LiveHapticSource(device_name=device_name, force_feedback=False)
        except Exception as exc:  # noqa: BLE001
            raise HapticUnavailable(f"device init failed: {exc}") from exc
        self._closed = False

    def poll(self) -> InputPose:
        return _controller_sample_dict_to_input_pose(self._source.poll())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._source.close()

    def __enter__(self) -> HapticInput:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def open_haptic_inputs(device_names: Sequence[str]) -> list[HapticInput]:
    """Open several pose-only OpenHaptics devices through the canonical input stack."""
    inputs: list[HapticInput] = []
    try:
        for name in device_names:
            inputs.append(HapticInput(device_name=name, start_scheduler=False))
        return inputs
    except Exception:
        for input_device in reversed(inputs):
            input_device.close()
        raise


class MiniMouInput:
    """Compatibility adapter exposing canonical MiniMou samples as InputPose."""

    def __init__(self, controller):
        self._controller = controller
        self._sample_state = None
        self._angles_degrees: tuple[float, float, float] | None = None
        self._closed = False

    @classmethod
    def discover(cls, count: int = 1) -> list[MiniMouInput]:
        try:
            from omnisurg.input.follou import MiniMouController  # noqa: PLC0415
        except Exception as exc:
            raise HapticUnavailable(f"Follou MiniMou support unavailable: {exc}") from exc

        inputs: list[MiniMouInput] = []
        try:
            for idx in range(count):
                inputs.append(cls(MiniMouController(device_index=idx)))
        except Exception as exc:
            for input_device in reversed(inputs):
                input_device.close()
            raise HapticUnavailable(f"found {len(inputs)} MiniMou device(s), need {count}: {exc}") from exc
        return inputs

    def poll(self) -> InputPose:
        try:
            if hasattr(self._controller, "poll"):
                sample = dict(self._controller.poll())
                angles = getattr(self._controller, "angles_degrees", lambda: None)()
            else:
                from omnisurg.input.follou import _MiniMouSampleState, poll_minimou_controller  # noqa: PLC0415

                if self._sample_state is None:
                    self._sample_state = _MiniMouSampleState()
                sample = poll_minimou_controller(self._controller, self._sample_state)
                angles = sample.pop("_angles_degrees", None)
            self._angles_degrees = angles
            return _controller_sample_dict_to_input_pose(sample)
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


def _controller_sample_dict_to_input_pose(sample_dict: dict | None) -> InputPose:
    from omnisurg.input.sources import ControllerSample  # noqa: PLC0415

    sample = ControllerSample.from_sample_dict(sample_dict)
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
            grip=self.pose.grip,
            handle_pos=self.pose.handle_pos,
            handle_active=self.pose.handle_active,
            valid=True,
        )

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    def __enter__(self) -> FallbackInput:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        return None
