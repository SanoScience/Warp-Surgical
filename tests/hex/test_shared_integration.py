from __future__ import annotations

import numpy as np

from omnisurg.input.sources import ControllerSample, device_position_to_world, sample_position_to_world
from omnisurg.rendering.bridge import RenderBridge


class _FakeRenderer:
    def __init__(self):
        self.callbacks = {}
        self.cryo_calls = []
        self.environment = {}

    def register_mouse_motion(self, callback):
        self.callbacks["motion"] = callback

    def register_mouse_press(self, callback):
        self.callbacks["press"] = callback

    def register_mouse_drag(self, callback):
        self.callbacks["drag"] = callback

    def register_mouse_release(self, callback):
        self.callbacks["release"] = callback

    def screen_to_world_ray(self, x, y):
        return np.asarray((x, y, 1.0), dtype=np.float32), np.asarray((0.0, 0.0, -1.0), dtype=np.float32)

    def draw_cryo_surface(self, *args, **kwargs):
        self.cryo_calls.append((args, kwargs))
        return "drawn"

    def set_environment_path(self, path):
        self.environment["path"] = path


def _bridge_for(renderer):
    bridge = RenderBridge.__new__(RenderBridge)
    bridge._backend = "fake"
    bridge._renderer = renderer
    return bridge


def test_render_bridge_forwards_hex_mouse_ray_cryo_and_environment_controls():
    renderer = _FakeRenderer()
    bridge = _bridge_for(renderer)

    bridge.set_mouse_callbacks(on_motion=lambda *args: None, on_press=lambda *args: None)
    origin, direction = bridge.screen_to_world_ray(4.0, 5.0)
    result = bridge.draw_cryo_surface("frame", hidden=False)
    bridge.set_environment_path("studio.hdr")

    assert set(renderer.callbacks) == {"motion", "press"}
    assert np.allclose(origin, (4.0, 5.0, 1.0))
    assert np.allclose(direction, (0.0, 0.0, -1.0))
    assert result == "drawn"
    assert renderer.cryo_calls == [(("frame",), {"hidden": False})]
    assert renderer.environment["path"] == "studio.hdr"


def test_controller_sample_accepts_tool_position_and_frame_conversion():
    sample = ControllerSample.from_sample_dict(
        {
            "position": [1.0, 2.0, 3.0],
            "tool_pos": [4.0, 5.0, 6.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
        }
    )
    converted = sample_position_to_world(sample, position_offset=(1.0, 0.0, -1.0), position_scale=0.5)

    assert sample.active
    assert np.allclose(device_position_to_world(sample.position, position_offset=(1, 0, -1), position_scale=0.5), (1, 1, 1))
    assert np.allclose(converted.position, (1.0, 1.0, 1.0))
    assert np.allclose(converted.tool_pos, (2.5, 2.5, 2.5))
