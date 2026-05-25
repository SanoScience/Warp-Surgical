from __future__ import annotations

from types import SimpleNamespace

import pytest

from omnisurg.hex import runtime_resources


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


def test_exception_cleanup_attempts_all_closes_and_suppresses_errors():
    order: list[str] = []
    devices = [
        FakeCloseable("input0", order),
        FakeCloseable("input1", order, error=RuntimeError("input close failed")),
    ]
    render_bridge = FakeCloseable("render", order, error=RuntimeError("render close failed"))
    owner = runtime_resources.HexRuntimeResourceOwner(input_devices=devices, render_bridge=render_bridge)

    runtime_resources._close_runtime_resources_for_exception(owner)

    assert order == ["input1", "input0", "render"]
    assert owner.input_devices == []
    assert owner.render_bridge is None


def test_exception_cleanup_reports_cleanup_failures_to_stderr(capsys):
    order: list[str] = []
    owner = runtime_resources.HexRuntimeResourceOwner(
        input_devices=[FakeCloseable("input", order, error=RuntimeError("input close failed"))],
    )

    runtime_resources._close_runtime_resources_for_exception(owner)

    captured = capsys.readouterr()
    assert "resource cleanup failed during exception unwind" in captured.err
    assert "RuntimeError: input close failed" in captured.err


def test_open_hex_instrument_inputs_dispatches_supported_backends(monkeypatch):
    calls = []

    class FakeInputSource:
        def poll(self):
            return {
                "position": (1.0, 2.0, 3.0),
                "rotation": (0.0, 0.0, 0.0, 1.0),
                "button": True,
                "button2": True,
                "tool_scalar": 0.75,
                "grip": 0.25,
                "handle_pos": 0.5,
                "handle_active": True,
            }

        def close(self):
            pass

    def fake_open_input_sources(configs, *, require_all=False):
        assert require_all
        config_list = list(configs)
        calls.append(
            [
                (config.role, config.backend, config.device_name, config.device_index, config.force_feedback)
                for config in config_list
            ]
        )
        return runtime_resources.input_factory.InputOpenResult(
            sources={config.role: FakeInputSource() for config in config_list if config.backend != "off"}
        )

    monkeypatch.setattr(runtime_resources.input_factory, "open_input_sources", fake_open_input_sources)

    fallback_inputs = runtime_resources.open_hex_instrument_inputs(_args("fallback"), expected_count=2)
    minimou_inputs = runtime_resources.open_hex_instrument_inputs(_args("minimou"), expected_count=2)
    haptic_inputs = runtime_resources.open_hex_instrument_inputs(_args("openhaptics"), expected_count=2)

    assert len(fallback_inputs) == 2
    assert len(minimou_inputs) == 2
    assert len(haptic_inputs) == 2
    pose = fallback_inputs[0].poll()
    assert pose.valid
    assert pose.position == (1.0, 2.0, 3.0)
    assert pose.button1
    assert pose.button2
    assert pose.tool_pos == 0.75
    assert pose.grip == 0.25
    assert pose.handle_pos == 0.5
    assert pose.handle_active
    assert calls == [
        [
            ("right", "fallback", "Right Device", 0, False),
            ("left", "fallback", "Left Device", 1, False),
        ],
        [
            ("right", "minimou", "Right Device", 0, False),
            ("left", "minimou", "Left Device", 1, False),
        ],
        [
            ("right", "openhaptics", "Right Device", 0, False),
            ("left", "openhaptics", "Left Device", 1, False),
        ],
    ]


def test_open_hex_instrument_inputs_off_returns_no_devices():
    assert runtime_resources.open_hex_instrument_inputs(_args("off"), expected_count=2) == []


def test_open_hex_instrument_inputs_count_mismatch_closes_devices_and_raises(monkeypatch):
    order: list[str] = []

    class FakeHapticSource(FakeCloseable):
        def __init__(self, *, device_name, scale=1.0, force_feedback=True):
            del scale, force_feedback
            if device_name != "Right Device":
                raise RuntimeError("missing left")
            super().__init__("right", order)

        def poll(self):
            return {"position": (0.0, 0.0, 0.0), "rotation": (0.0, 0.0, 0.0, 1.0)}

    monkeypatch.setattr("omnisurg.haptics.LiveHapticSource", FakeHapticSource)

    with pytest.raises(runtime_resources.HapticUnavailable, match="left .*missing left"):
        runtime_resources.open_hex_instrument_inputs(_args("openhaptics"), expected_count=2)

    assert order == ["right"]


def test_build_hex_usd_viewer_passes_resolved_num_frames(monkeypatch):
    calls = []

    class FakeUsdViewer:
        def __init__(self, path, num_frames=None):
            calls.append((path, num_frames))

    monkeypatch.setattr(runtime_resources.newton.viewer, "ViewerUSD", FakeUsdViewer)

    viewer = runtime_resources.build_hex_usd_viewer("hex.usd", SimpleNamespace(max_frames=4))

    assert isinstance(viewer, FakeUsdViewer)
    assert calls == [("hex.usd", 4)]
