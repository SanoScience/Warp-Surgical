from __future__ import annotations

import numpy as np
import pytest

from omnisurg.input.factory import InputOpenError, RoleInputConfig, normalize_input_backend, open_input_sources
from omnisurg.input.sources import ControllerSample, FallbackInputSource, ReplayInputSource


def test_controller_sample_accepts_extended_fields():
    sample = ControllerSample.from_sample_dict(
        {
            "position": [1.0, 2.0, 3.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "button1": True,
            "button2": True,
            "grip": 0.25,
            "tool_scalar": 0.75,
            "handle_pos": 12.0,
            "handle_active": True,
            "valid": True,
        }
    )

    assert sample.active
    assert sample.button
    assert sample.button2
    assert sample.grip == 0.25
    assert sample.tool_scalar == 0.75
    assert sample.handle_pos == 12.0
    assert sample.handle_active


def test_input_backend_normalizes_legacy_none_alias():
    assert normalize_input_backend("none") == "off"
    assert normalize_input_backend("off") == "off"


def test_input_factory_dispatches_supported_backends(monkeypatch, tmp_path):
    calls = []

    class FakeHapticSource:
        def __init__(self, *, device_name, scale=1.0, force_feedback=True):
            calls.append(("openhaptics", device_name, scale, force_feedback))

        def poll(self):
            return {}

        def close(self):
            pass

    class FakeMiniMouSource:
        def __init__(self, *, device_index, scale=1.0):
            calls.append(("minimou", device_index, scale))

        def poll(self):
            return {}

        def close(self):
            pass

    monkeypatch.setattr("omnisurg.haptics.LiveHapticSource", FakeHapticSource)
    monkeypatch.setattr("omnisurg.haptics.LiveMiniMouSource", FakeMiniMouSource)

    replay_path = tmp_path / "trace.npy"
    np.save(replay_path, np.zeros((1, 7), dtype=np.float32))

    result = open_input_sources(
        [
            RoleInputConfig("right", "off"),
            RoleInputConfig("fallback", "fallback"),
            RoleInputConfig("haptic", "openhaptics", device_name="Right Device", force_feedback=True),
            RoleInputConfig("mini", "minimou", device_index=3),
            RoleInputConfig("replay", "replay", replay_path=replay_path),
        ],
        require_all=True,
    )

    assert isinstance(result.sources["fallback"], FallbackInputSource)
    assert isinstance(result.sources["haptic"], FakeHapticSource)
    assert isinstance(result.sources["mini"], FakeMiniMouSource)
    assert isinstance(result.sources["replay"], ReplayInputSource)
    assert calls == [
        ("openhaptics", "Right Device", 1.0, True),
        ("minimou", 3, 1.0),
    ]


def test_input_factory_closes_partial_sources_in_reverse_order(monkeypatch):
    close_order = []

    class FakeHapticSource:
        def __init__(self, *, device_name, scale=1.0, force_feedback=True):
            del scale, force_feedback
            self.device_name = device_name
            if device_name == "broken":
                raise RuntimeError("device missing")

        def poll(self):
            return {}

        def close(self):
            close_order.append(self.device_name)

    monkeypatch.setattr("omnisurg.haptics.LiveHapticSource", FakeHapticSource)

    with pytest.raises(InputOpenError, match="device missing"):
        open_input_sources(
            [
                RoleInputConfig("right", "openhaptics", device_name="first"),
                RoleInputConfig("left", "openhaptics", device_name="second"),
                RoleInputConfig("left", "openhaptics", device_name="broken"),
            ],
            require_all=True,
        )

    assert close_order == ["second", "first"]


def test_fallback_input_source_polls_valid_zero_pose():
    sample = ControllerSample.from_sample_dict(FallbackInputSource().poll())

    assert sample.active
    np.testing.assert_allclose(sample.position, np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(sample.rotation, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
