from __future__ import annotations

from types import SimpleNamespace

import pytest

from omnisurg.hex import runtime_resources
from omnisurg.hex.haptic import HapticUnavailable


class FakeCloseable:
    def __init__(self, name: str, order: list[str], *, error: Exception | None = None) -> None:
        self.name = name
        self.order = order
        self.error = error
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        self.order.append(self.name)
        if self.error is not None:
            raise self.error


def _args(backend: str, **overrides):
    values = {
        "input_backend": backend,
        "device_name": "Right Device",
        "left_device_name": "Left Device",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_resource_owner_closes_inputs_in_reverse_order_before_render_bridge():
    order: list[str] = []
    devices = [FakeCloseable("input0", order), FakeCloseable("input1", order)]
    render_bridge = FakeCloseable("render", order)

    runtime_resources.HexRuntimeResourceOwner(input_devices=devices, render_bridge=render_bridge).close()

    assert order == ["input1", "input0", "render"]


def test_resource_owner_close_is_idempotent_and_clears_references():
    order: list[str] = []
    device = FakeCloseable("input", order)
    render_bridge = FakeCloseable("render", order)
    owner = runtime_resources.HexRuntimeResourceOwner(input_devices=[device], render_bridge=render_bridge)

    owner.close()
    owner.close()

    assert order == ["input", "render"]
    assert device.close_count == 1
    assert render_bridge.close_count == 1
    assert owner.input_devices == []
    assert owner.render_bridge is None


def test_resource_owner_attempts_all_closes_and_reraises_first_error():
    order: list[str] = []
    first_error = RuntimeError("first")
    devices = [
        FakeCloseable("input0", order),
        FakeCloseable("input1", order, error=first_error),
    ]
    render_bridge = FakeCloseable("render", order, error=RuntimeError("second"))
    owner = runtime_resources.HexRuntimeResourceOwner(input_devices=devices, render_bridge=render_bridge)

    with pytest.raises(RuntimeError, match="first") as exc_info:
        owner.close()

    assert exc_info.value is first_error
    assert order == ["input1", "input0", "render"]
    assert owner.input_devices == []
    assert owner.render_bridge is None


def test_open_hex_instrument_inputs_dispatches_supported_backends(monkeypatch):
    calls = []

    def fake_minimou_inputs(*, count):
        calls.append(("minimou", count))
        return [object() for _ in range(count)]

    def fake_haptic_inputs(device_names):
        calls.append(("openhaptics", tuple(device_names)))
        return [object(), object()]

    monkeypatch.setattr(runtime_resources, "open_minimou_inputs", fake_minimou_inputs)
    monkeypatch.setattr(runtime_resources, "open_haptic_inputs", fake_haptic_inputs)

    fallback_inputs = runtime_resources.open_hex_instrument_inputs(_args("fallback"), expected_count=2)
    minimou_inputs = runtime_resources.open_hex_instrument_inputs(_args("minimou"), expected_count=2)
    haptic_inputs = runtime_resources.open_hex_instrument_inputs(_args("openhaptics"), expected_count=2)

    assert len(fallback_inputs) == 2
    assert len(minimou_inputs) == 2
    assert len(haptic_inputs) == 2
    assert calls == [
        ("minimou", 2),
        ("openhaptics", ("Right Device", "Left Device")),
    ]


def test_open_hex_instrument_inputs_off_returns_no_devices():
    assert runtime_resources.open_hex_instrument_inputs(_args("off"), expected_count=2) == []


def test_open_hex_instrument_inputs_count_mismatch_closes_devices_and_raises(monkeypatch):
    order: list[str] = []
    opened = [FakeCloseable("only", order)]

    def fake_haptic_inputs(device_names):
        del device_names
        return opened

    monkeypatch.setattr(runtime_resources, "open_haptic_inputs", fake_haptic_inputs)

    with pytest.raises(HapticUnavailable, match="expected 2 devices, got 1"):
        runtime_resources.open_hex_instrument_inputs(_args("openhaptics"), expected_count=2)

    assert order == ["only"]


def test_build_hex_usd_viewer_passes_resolved_num_frames(monkeypatch):
    calls = []

    class FakeUsdViewer:
        def __init__(self, path, num_frames=None):
            calls.append((path, num_frames))

    monkeypatch.setattr(runtime_resources.newton.viewer, "ViewerUSD", FakeUsdViewer)

    viewer = runtime_resources.build_hex_usd_viewer("hex.usd", SimpleNamespace(max_frames=4))

    assert isinstance(viewer, FakeUsdViewer)
    assert calls == [("hex.usd", 4)]
