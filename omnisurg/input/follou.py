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


def _ensure_follou_importable():
    if not FOLLOU_ROOT.exists():
        raise RuntimeError(f'Follou package not found at "{FOLLOU_ROOT}"')

    repo_root = str(FOLLOU_ROOT.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from follou.devices.minimou import MiniMou
        from follou.manager import DeviceManager
    except Exception as exc:
        raise RuntimeError(f'Failed to import Follou SDK from "{FOLLOU_ROOT}": {exc}') from exc

    return DeviceManager, MiniMou


def acquire_manager():
    DeviceManager, MiniMou = _ensure_follou_importable()
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
        self._controller = self._manager.get_device_controller(self._mini_mou_cls, count=self.device_index)
        self._closed = False
        self._tool_min: float | None = None
        self._tool_max: float | None = None

        if self._controller is None:
            available = [type(device).__name__ for device in getattr(self._manager, "devices", [])]
            release_manager()
            raise RuntimeError(
                f"MiniMou device index {self.device_index} not available; discovered devices: {available or ['none']}"
            )

    def poll(self) -> dict:
        if self._closed:
            return {}

        self._controller.perform_update()
        position = np.asarray(self._controller.get_position()[:3], dtype=np.float32) * self.scale
        # MiniMou X motion is mirrored relative to the OmniSurg scene axes.
        position[0] *= -1.0
        rotation = _axis_angle_to_quaternion(self._controller.get_orientation())
        tool_pos = float(self._controller.get_tool_pos())
        grip = self._tool_position_to_grip(tool_pos)
        button = grip >= 0.5

        return {
            "position": position,
            "rotation": rotation,
            "button": button,
            "grip": grip,
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        release_manager()

    def _tool_position_to_grip(self, tool_pos: float) -> float:
        if not math.isfinite(tool_pos):
            return 0.0

        if self._tool_min is None or self._tool_max is None:
            self._tool_min = tool_pos
            self._tool_max = tool_pos
            return 0.0

        self._tool_min = min(self._tool_min, tool_pos)
        self._tool_max = max(self._tool_max, tool_pos)
        span = self._tool_max - self._tool_min
        if span < 1.0e-4:
            return 0.0

        normalized = (tool_pos - self._tool_min) / span
        normalized = min(1.0, max(0.0, normalized))
        # MiniMou tool position behaves like an opening signal, so lower values mean more closed.
        return 1.0 - normalized


def _axis_angle_to_quaternion(orientation) -> np.ndarray:
    axis_x, axis_y, axis_z, angle = [float(value) for value in orientation]
    axis = np.array([axis_x, axis_y, axis_z], dtype=np.float32)
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-8 or not math.isfinite(angle):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    axis /= norm
    half_angle = 0.5 * angle
    sin_half = math.sin(half_angle)
    return np.array(
        [
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            math.cos(half_angle),
        ],
        dtype=np.float32,
    )
