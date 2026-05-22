from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from omnisurg.hex import app_runtime
from omnisurg.hex.app import OmniSurgHexApp
from omnisurg.hex.cli import build_parser
from omnisurg.hex.data.types import PreparedVolume
from omnisurg.hex.materials import DEFAULT_MATERIALS, MaterialTable
from omnisurg.hex import runtime as hex_runtime
from omnisurg.hex.runtime import HexAppLauncher, HexRuntime


def _make_volume() -> PreparedVolume:
    return PreparedVolume(
        labels=np.ones((2, 2, 2), dtype=np.uint8),
        voxel_size_m=0.005,
        materials=MaterialTable(DEFAULT_MATERIALS),
        class_map={0: "background", 1: "synthetic_block"},
        texture_rgb=np.zeros((2, 2, 2, 3), dtype=np.uint8),
    )


def test_unified_hex_parser_prog():
    parser = build_parser()

    assert parser.prog == "omnisurg hex"


def test_app_run_passes_volume_and_argv_to_prepared_entrypoint(monkeypatch):
    volume = _make_volume()
    argv = ("--viewer", "headless", "--frames", "1")
    calls = []

    def fake_run_prepared_volume(passed_volume, passed_argv=None):
        legacy_prefix = "_OMNISURG_" + "PREPARED"
        assert not any(name.startswith(legacy_prefix) for name in vars(app_runtime))
        calls.append((passed_volume, passed_argv))
        return 17

    monkeypatch.setattr(app_runtime, "run_prepared_volume", fake_run_prepared_volume)

    assert OmniSurgHexApp(volume).run(argv) == 17
    assert len(calls) == 1
    assert calls[0][0] is volume
    assert calls[0][1] is argv


@pytest.mark.parametrize(
    "argv",
    [
        ("--viewer", "headless", "--hex-runtime-driver", "app", "--frames", "1"),
        ("--viewer", "headless", "--hex-runtime-driver=app", "--frames", "1"),
    ],
)
def test_app_runtime_driver_selector_is_stripped_before_prepared_entrypoint(monkeypatch, argv):
    volume = _make_volume()
    calls = []

    def fake_run_prepared_volume(passed_volume, passed_argv=None):
        calls.append((passed_volume, passed_argv))
        return 19

    monkeypatch.setattr(app_runtime, "run_prepared_volume", fake_run_prepared_volume)

    assert OmniSurgHexApp(volume).run(argv) == 19
    assert calls == [(volume, ["--viewer", "headless", "--frames", "1"])]


@pytest.mark.parametrize(
    "argv",
    [
        ("--viewer", "headless", "--hex-runtime-driver", "session", "--frames", "1"),
        ("--viewer", "headless", "--hex-runtime-driver=session", "--frames", "1"),
    ],
)
def test_session_runtime_driver_routes_through_launcher(monkeypatch, argv):
    volume = _make_volume()
    calls = []

    def fail_run_prepared_volume(passed_volume, passed_argv=None):
        raise AssertionError("session driver should not call app_runtime.run_prepared_volume")

    class FakeHexAppLauncher:
        def __init__(self, passed_volume, passed_argv):
            calls.append(("init", passed_volume, passed_argv))

        def run(self):
            calls.append(("run",))
            return 23

    monkeypatch.setattr(app_runtime, "run_prepared_volume", fail_run_prepared_volume)
    monkeypatch.setattr(hex_runtime, "HexAppLauncher", FakeHexAppLauncher)

    assert OmniSurgHexApp(volume).run(argv) == 23
    assert calls == [
        ("init", volume, ["--viewer", "headless", "--frames", "1"]),
        ("run",),
    ]


def test_invalid_runtime_driver_selector_exits_with_argparse_error(capsys):
    with pytest.raises(SystemExit) as exc_info:
        OmniSurgHexApp(_make_volume()).run(("--hex-runtime-driver", "bogus"))

    assert exc_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_hex_help_does_not_advertise_runtime_driver_selector():
    result = subprocess.run(
        [sys.executable, "-m", "omnisurg", "hex", "--help"],
        check=False,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "hex-runtime-driver" not in result.stdout
    assert "hex-runtime-driver" not in result.stderr


def test_python_module_hex_synthetic_headless_exit_after_init():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnisurg",
            "hex",
            "--dataset",
            "synthetic",
            "--size",
            "2",
            "--texture",
            "off",
            "--viewer",
            "headless",
            "--input-backend",
            "off",
            "--exit-after-init",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "[startup]" in result.stdout


def test_python_hex_module_synthetic_headless_exit_after_init():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnisurg.hex",
            "--dataset",
            "synthetic",
            "--size",
            "2",
            "--texture",
            "off",
            "--viewer",
            "headless",
            "--input-backend",
            "off",
            "--exit-after-init",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "[startup]" in result.stdout


def test_hex_runtime_headless_lifecycle_does_not_launch_app_loop(monkeypatch):
    volume = PreparedVolume(
        labels=np.ones((2, 2, 2), dtype=np.uint8),
        voxel_size_m=0.005,
        materials=MaterialTable(DEFAULT_MATERIALS),
        class_map={0: "background", 1: "synthetic_block"},
    )
    launcher = HexAppLauncher(
        volume,
        (
            "--viewer",
            "headless",
            "--input-backend",
            "off",
            "--frames",
            "1",
            "--cryo-renderer",
            "off",
            "--no-gl-interop",
        ),
    )

    def fail_run(argv=None):
        raise AssertionError("HexRuntime.step() should not launch the full app loop")

    monkeypatch.setattr(launcher, "run", fail_run)

    runtime = HexRuntime(launcher)
    runtime.init()
    assert runtime.is_running()

    runtime.poll_input()
    runtime.step()
    runtime.render()
    runtime.pace()
    runtime.close()

    assert not runtime.is_running()
    assert runtime.return_code == 0


def test_hex_runtime_backend_lifecycles_do_not_launch_app_loop(monkeypatch, tmp_path):
    volume = PreparedVolume(
        labels=np.ones((2, 2, 2), dtype=np.uint8),
        voxel_size_m=0.005,
        materials=MaterialTable(DEFAULT_MATERIALS),
        class_map={0: "background", 1: "synthetic_block"},
    )
    created_backends: list[str] = []

    class FakeBridge:
        def __init__(self, viewer_config, model, device):
            del model, device
            self.backend = str(viewer_config.backend)
            self.show_particles = False
            created_backends.append(self.backend)

        @classmethod
        def wrap_existing(cls, viewer, *, backend=None, device=None):
            del viewer, device
            obj = cls.__new__(cls)
            obj.backend = str(backend)
            obj.show_particles = False
            created_backends.append(obj.backend)
            return obj

        def set_camera(self, *args, **kwargs):
            pass

        def begin_frame(self, time):
            pass

        def log_state(self, state):
            pass

        def end_frame(self):
            pass

        def is_running(self):
            return True

        def close(self):
            pass

    class FakeUsdViewer:
        def __init__(self, path, num_frames=None):
            self.path = path
            self.num_frames = num_frames

        def set_model(self, model):
            self.model = model

    monkeypatch.setattr(hex_runtime, "RenderBridge", FakeBridge)
    import newton

    monkeypatch.setattr(newton.viewer, "ViewerUSD", FakeUsdViewer)

    cases = [
        ("gl", ("--viewer", "gl")),
        ("slang", ("--viewer", "slang")),
        ("usd", ("--viewer", "headless", "--usd", str(tmp_path / "hex.usd"))),
    ]
    for expected_backend, backend_args in cases:
        launcher = HexAppLauncher(
            volume,
            (
                *backend_args,
                "--input-backend",
                "off",
                "--frames",
                "1",
                "--cryo-renderer",
                "off",
                "--no-gl-interop",
            ),
        )

        def fail_run(argv=None):
            raise AssertionError("HexRuntime should not launch the retired app loop")

        monkeypatch.setattr(launcher, "run", fail_run)

        runtime = HexRuntime(launcher)
        runtime.init()
        runtime.poll_input()
        runtime.step()
        runtime.render()
        runtime.pace()
        runtime.close()

        assert runtime.return_code == 0
        assert expected_backend in created_backends


def test_removed_spring_flags_are_rejected():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnisurg",
            "hex",
            "--dataset",
            "synthetic",
            "--size",
            "1",
            "--texture",
            "off",
            "--viewer",
            "headless",
            "--input-backend",
            "off",
            "--exit-after-init",
            "--edge-ke",
            "1.0",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert "unrecognized arguments: --edge-ke" in result.stderr
