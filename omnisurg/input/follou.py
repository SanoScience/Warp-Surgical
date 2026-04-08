from __future__ import annotations

import math
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_FOLLOU_ROOT = Path(r"G:\warp\python_device_manager")

_manager_lock = threading.Lock()
_manager_entries: dict[Path, "_ManagerEntry"] = {}


@dataclass
class _ManagerEntry:
    manager: object
    refcount: int = 0


def _normalize_root(root: str | Path | None) -> Path:
    candidate = Path(root) if root is not None else DEFAULT_FOLLOU_ROOT
    return candidate.expanduser().resolve()


def _ensure_follou_importable(root: str | Path | None):
    package_root = _normalize_root(root)
    src_root = package_root / "src"
    if not src_root.exists():
        raise RuntimeError(f'Follou source path not found: "{src_root}"')

    src_str = str(src_root)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    try:
        from follou.devices.minimou import MiniMou
        from follou.manager import DeviceManager
    except Exception as exc:
        raise RuntimeError(f'Failed to import Follou SDK from "{src_root}": {exc}') from exc

    return package_root, DeviceManager, MiniMou


def acquire_manager(root: str | Path | None):
    package_root, DeviceManager, MiniMou = _ensure_follou_importable(root)
    with _manager_lock:
        entry = _manager_entries.get(package_root)
        if entry is None:
            entry = _ManagerEntry(manager=DeviceManager())
            _manager_entries[package_root] = entry
        entry.refcount += 1
        return package_root, entry.manager, MiniMou


def release_manager(root: str | Path | None):
    package_root = _normalize_root(root)
    with _manager_lock:
        entry = _manager_entries.get(package_root)
        if entry is None:
            return

        entry.refcount -= 1
        if entry.refcount > 0:
            return

        for device in getattr(entry.manager, "devices", []):
            try:
                device.close()
            except Exception:
                pass
        _manager_entries.pop(package_root, None)


class MiniMouController:
    def __init__(self, *, root: str | Path | None = None, device_index: int = 0, scale: float = 1.0):
        self.root, self._manager, self._mini_mou_cls = acquire_manager(root)
        self.device_index = int(device_index)
        self.scale = float(scale)
        self._controller = self._manager.get_device_controller(self._mini_mou_cls, count=self.device_index)
        self._closed = False
        self._tool_min: float | None = None
        self._tool_max: float | None = None

        if self._controller is None:
            available = [type(device).__name__ for device in getattr(self._manager, "devices", [])]
            release_manager(self.root)
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
        release_manager(self.root)

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
