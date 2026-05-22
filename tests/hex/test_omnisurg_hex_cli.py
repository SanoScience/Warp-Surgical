from __future__ import annotations

import subprocess
import sys

import numpy as np

from omnisurg.hex.cli import build_parser
from omnisurg.hex.data.types import PreparedVolume
from omnisurg.hex.materials import DEFAULT_MATERIALS, MaterialTable
from omnisurg.hex import runtime as hex_runtime
from omnisurg.hex.runtime import HexAppLauncher, HexRuntime


def test_unified_hex_parser_prog():
    parser = build_parser()

    assert parser.prog == "omnisurg hex"


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
