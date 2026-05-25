from __future__ import annotations

import math
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The follou package is vendored at <repo-root>/follou (sibling of omnisurg/).
FOLLOU_ROOT = Path(__file__).resolve().parents[2] / "follou"

_manager_lock = threading.Lock()
_manager_entry: "_ManagerEntry | None" = None


@dataclass
class _ManagerEntry:
    manager: object
    refcount: int = 0


@dataclass
class _MiniMouSampleState:
    handle_min: float | None = None
    handle_max: float | None = None
    tool_min: float | None = None
    tool_max: float | None = None


def ensure_follou_importable():
    if FOLLOU_ROOT.exists():
        repo_root = str(FOLLOU_ROOT.parent)
    else:
        repo_root = ""

    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from follou.devices.minimou import MiniMou
        from follou.manager import DeviceManager
    except Exception as exc:
        location = f'"{FOLLOU_ROOT}"' if FOLLOU_ROOT.exists() else "installed package or repo-local ./follou"
        raise RuntimeError(f"Failed to import Follou SDK from {location}: {exc}") from exc

    return DeviceManager, MiniMou


_ensure_follou_importable = ensure_follou_importable


def acquire_manager():
    DeviceManager, MiniMou = ensure_follou_importable()
    global _manager_entry
    with _manager_lock:
        if _manager_entry is None:
            _manager_entry = _ManagerEntry(manager=DeviceManager())
        _manager_entry.refcount += 1
        return _manager_entry.manager, MiniMou


def release_manager():
    global _manager_entry
    with _manager_lock:
        if _manager_entry is None:
            return

        _manager_entry.refcount -= 1
        if _manager_entry.refcount > 0:
            return

        for device in getattr(_manager_entry.manager, "devices", []):
            try:
                device.close()
            except Exception:
                pass
        _manager_entry = None


class MiniMouController:
    def __init__(self, *, device_index: int = 0, scale: float = 1.0):
        self._manager, self._mini_mou_cls = acquire_manager()
        self.device_index = int(device_index)
        self.scale = float(scale)
        try:
            self._controller = self._manager.get_device_controller(self._mini_mou_cls, count=self.device_index)
        except TypeError as exc:
            if "count" not in str(exc):
                raise
            self._controller = self._manager.get_device_controller(self._mini_mou_cls, self.device_index)
        self._closed = False
        self._sample_state = _MiniMouSampleState()
        self._angles_degrees: tuple[float, float, float] | None = None

        if self._controller is None:
            available = [type(device).__name__ for device in getattr(self._manager, "devices", [])]
            release_manager()
            raise RuntimeError(
                f"MiniMou device index {self.device_index} not available; discovered devices: {available or ['none']}"
            )

    def poll(self) -> dict:
        if self._closed:
            return {}

        sample = poll_minimou_controller(self._controller, self._sample_state, scale=self.scale)
        self._angles_degrees = sample.pop("_angles_degrees", None)
        return sample

    def close(self):
        if self._closed:
            return
        self._closed = True
        release_manager()

    def angles_degrees(self) -> tuple[float, float, float] | None:
        if self._angles_degrees is not None:
            return self._angles_degrees
        return _read_angles_degrees(self._controller)


def poll_minimou_controller(
    controller,
    state: _MiniMouSampleState | None = None,
    *,
    scale: float = 1.0,
) -> dict:
    state = state or _MiniMouSampleState()
    controller.perform_update()
    position = np.asarray(controller.get_position()[:3], dtype=np.float32) * float(scale)
    # MiniMou X motion is mirrored relative to the OmniSurg scene axes.
    position[0] *= -1.0
    rotation = _normalize_quaternion(controller.get_orientation())
    tool_pos = _read_float(getattr(controller, "get_tool_pos", lambda: 0.0), 0.0)
    handle_pos = _read_float(getattr(controller, "get_handle_opening_value", lambda: tool_pos), tool_pos)
    handle_active = bool(_read_float(getattr(controller, "get_handle_activity", lambda: 0.0), 0.0))
    grip = max(
        _closing_grip(handle_pos, state, "handle_min", "handle_max"),
        _closing_grip(tool_pos, state, "tool_min", "tool_max"),
    )
    button = handle_active or grip >= 0.5

    return {
        "position": position,
        "rotation": rotation,
        "button": button,
        "grip": grip,
        "tool_scalar": tool_pos,
        "handle_pos": handle_pos,
        "handle_active": handle_active,
        "valid": True,
        "_angles_degrees": _read_angles_degrees(controller),
    }


def _read_float(reader, default: float) -> float:
    try:
        value = float(reader())
    except Exception:
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return value


def _read_angles_degrees(controller) -> tuple[float, float, float] | None:
    try:
        return (
            float(controller.get_rot_angle()),
            float(controller.get_pitch_angle()),
            float(controller.get_yaw_angle()),
        )
    except Exception:
        return None


def _closing_grip(value: float, state: _MiniMouSampleState, min_attr: str, max_attr: str) -> float:
    if not math.isfinite(value):
        return 0.0

    current_min = getattr(state, min_attr)
    current_max = getattr(state, max_attr)
    if current_min is None or current_max is None:
        setattr(state, min_attr, float(value))
        setattr(state, max_attr, float(value))
        return 0.0

    current_min = min(float(current_min), float(value))
    current_max = max(float(current_max), float(value))
    setattr(state, min_attr, current_min)
    setattr(state, max_attr, current_max)
    span = current_max - current_min
    if span < 1.0e-4:
        return 0.0

    opening = (float(value) - current_min) / span
    # MiniMou handle/tool values behave like opening signals, so lower values mean more closed.
    return float(np.clip(1.0 - opening, 0.0, 1.0))


def _normalize_quaternion(quaternion) -> np.ndarray:
    values = np.asarray([float(value) for value in tuple(quaternion)[:4]], dtype=np.float32)
    if values.shape[0] != 4:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    norm = float(np.linalg.norm(values))
    if norm < 1.0e-8 or not math.isfinite(norm):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return values / norm
