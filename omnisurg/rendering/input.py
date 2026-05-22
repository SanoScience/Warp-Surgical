from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class ViewportInputAdapter:
    """Renderer-agnostic mouse/key state fed by RenderBridge callbacks."""

    def __init__(self, bridge: Any):
        self.bridge = bridge
        self.cursor_xy: tuple[float, float] | None = None
        self.buttons = 0
        self.modifiers = 0
        self._keys: dict[int, bool] = {}

    def is_key_down(self, symbol: int) -> bool:
        symbol = int(symbol)
        if bool(self._keys.get(symbol, False)):
            return True
        method = getattr(self.bridge, "is_key_down", None)
        return bool(method(symbol)) if callable(method) else False

    def is_ui_capturing(self) -> bool:
        method = getattr(self.bridge, "is_ui_capturing", None)
        return bool(method()) if callable(method) else False

    def screen_to_world_ray(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        origin, direction = self.bridge.screen_to_world_ray(float(x), float(y))
        origin = np.asarray(origin, dtype=np.float32).reshape(3)
        direction = np.asarray(direction, dtype=np.float32).reshape(3)
        length = float(np.linalg.norm(direction))
        if length > 1.0e-8:
            direction /= length
        return origin, direction

    def wrap_key_press(self, callback: Callable[..., Any] | None = None):
        def _on_key_press(symbol, modifiers):
            self._keys[int(symbol)] = True
            self.modifiers = int(modifiers)
            if callback is not None:
                return callback(symbol, modifiers)
            return None

        return _on_key_press

    def wrap_key_release(self, callback: Callable[..., Any] | None = None):
        def _on_key_release(symbol, modifiers):
            self._keys[int(symbol)] = False
            self.modifiers = int(modifiers)
            if callback is not None:
                return callback(symbol, modifiers)
            return None

        return _on_key_release

    def wrap_mouse_motion(self, callback: Callable[..., Any] | None = None):
        def _on_mouse_motion(x, y, dx, dy):
            self.cursor_xy = (float(x), float(y))
            if callback is not None:
                return callback(x, y, dx, dy)
            return None

        return _on_mouse_motion

    def wrap_mouse_press(self, callback: Callable[..., Any] | None = None):
        def _on_mouse_press(x, y, button, modifiers):
            self.cursor_xy = (float(x), float(y))
            self.buttons |= int(button)
            self.modifiers = int(modifiers)
            if callback is not None:
                return callback(x, y, button, modifiers)
            return None

        return _on_mouse_press

    def wrap_mouse_drag(self, callback: Callable[..., Any] | None = None):
        def _on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            self.cursor_xy = (float(x), float(y))
            self.buttons = int(buttons)
            self.modifiers = int(modifiers)
            if callback is not None:
                return callback(x, y, dx, dy, buttons, modifiers)
            return None

        return _on_mouse_drag

    def wrap_mouse_release(self, callback: Callable[..., Any] | None = None):
        def _on_mouse_release(x, y, button, modifiers):
            self.cursor_xy = (float(x), float(y))
            self.buttons &= ~int(button)
            self.modifiers = int(modifiers)
            if callback is not None:
                return callback(x, y, button, modifiers)
            return None

        return _on_mouse_release
