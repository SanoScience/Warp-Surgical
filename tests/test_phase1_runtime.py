import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WARP_CACHE_DIR = REPO_ROOT / ".warp_cache"
TEST_TMP_DIR = REPO_ROOT / ".tmp_tests"
WARP_CACHE_DIR.mkdir(exist_ok=True)
TEST_TMP_DIR.mkdir(exist_ok=True)
os.environ["WARP_CACHE_PATH"] = str(WARP_CACHE_DIR)

import warp as wp

from omnisurg import BoundsConfig, HapticConfig, Runtime, SceneConfig, SimulationConfig, ViewerConfig
import omnisurg.main as omnisurg_main
from omnisurg.assets import load_scene_asset, load_tet_asset
from omnisurg.haptics import BimanualReplayRig, ReplayInputSource
from omnisurg.instruments.grasper import load_kinematic_grasper
from omnisurg.rendering.bridge import RenderBridge
from omnisurg.input.follou import MiniMouController
from omnisurg.input.sources import LiveMiniMouSource
from omnisurg.scene_builder import build_scene


def _live_args(**overrides):
    values = {
        "right_input_backend": "openhaptics",
        "left_input_backend": "openhaptics",
        "right_device_name": "Default Device",
        "left_device_name": "Left Device",
        "right_device_index": 0,
        "left_device_index": 1,
        "follou_root": r"G:\warp\python_device_manager",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestPhaseRuntime(unittest.TestCase):
    def test_cli_help(self):
        env = os.environ.copy()
        env["WARP_CACHE_PATH"] = str(WARP_CACHE_DIR)
        result = subprocess.run(
            [sys.executable, "-m", "omnisurg", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("OmniSurg Phase 2", result.stdout)
        self.assertIn("headless", result.stdout)
        self.assertIn("--scene", result.stdout)
        self.assertIn("--no-textures", result.stdout)
        self.assertIn("--left-replay", result.stdout)
        self.assertIn("--left-device-name", result.stdout)
        self.assertIn("--right-input-backend", result.stdout)
        self.assertIn("--follou-root", result.stdout)

    def test_live_input_rig_keeps_running_when_one_device_fails(self):
        class FakeSource:
            def __init__(self, device_name: str):
                self.device_name = device_name
                self.closed = False

            def poll(self):
                return {}

            def close(self):
                self.closed = True

        def build_source(*, device_name: str, scale: float = 1.0):
            if device_name == "Broken Left":
                raise AttributeError("decode")
            return FakeSource(device_name)

        args = _live_args(left_device_name="Broken Left")
        rig = None
        with mock.patch("omnisurg.haptics.LiveHapticSource", side_effect=build_source):
            rig = omnisurg_main._build_live_input_rig(args)

        try:
            self.assertIsNotNone(rig)
            self.assertEqual(set(rig._sources.keys()), {"right"})
        finally:
            if rig is not None:
                rig.close()

    def test_live_input_rig_returns_none_when_all_devices_fail(self):
        args = _live_args(right_device_name="Missing Right", left_device_name="Missing Left")

        with mock.patch("omnisurg.haptics.LiveHapticSource", side_effect=RuntimeError("device init failed")):
            rig = omnisurg_main._build_live_input_rig(args)

        self.assertIsNone(rig)

    def test_live_input_rig_uses_left_name_alias(self):
        attempted_names = []

        class FakeSource:
            def __init__(self, device_name: str):
                self.device_name = device_name

            def poll(self):
                return {}

            def close(self):
                pass

        def build_source(*, device_name: str, scale: float = 1.0):
            attempted_names.append(device_name)
            if device_name in {"Missing Right", "Device Left"}:
                raise RuntimeError("alias required")
            return FakeSource(device_name)

        args = _live_args(right_device_name="Missing Right", left_device_name="Device Left")
        rig = None
        with mock.patch("omnisurg.haptics.LiveHapticSource", side_effect=build_source):
            rig = omnisurg_main._build_live_input_rig(args)

        try:
            self.assertIsNotNone(rig)
            self.assertEqual(set(rig._sources.keys()), {"left"})
            self.assertEqual(attempted_names, ["Missing Right", "Device Left", "Left Device"])
        finally:
            if rig is not None:
                rig.close()

    def test_live_input_rig_builds_minimou_source(self):
        class FakeSource:
            def __init__(self, root: str, device_index: int):
                self.root = root
                self.device_index = device_index

            def poll(self):
                return {}

            def close(self):
                pass

        def build_source(*, root: str, device_index: int, scale: float = 1.0):
            return FakeSource(root, device_index)

        args = _live_args(
            right_input_backend="minimou",
            left_input_backend="none",
            right_device_index=2,
            follou_root=r"G:\warp\python_device_manager",
        )
        rig = None
        with mock.patch("omnisurg.haptics.LiveMiniMouSource", side_effect=build_source):
            rig = omnisurg_main._build_live_input_rig(args)

        try:
            self.assertIsNotNone(rig)
            self.assertEqual(set(rig._sources.keys()), {"right"})
            self.assertEqual(rig._sources["right"].device_index, 2)
        finally:
            if rig is not None:
                rig.close()

    def test_live_minimou_source_polls_controller_samples(self):
        class FakeController:
            def __init__(self, *, root=None, device_index=0, scale=1.0):
                self.root = root
                self.device_index = device_index
                self.scale = scale
                self.closed = False

            def poll(self):
                return {
                    "position": [1.0, 2.0, 3.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "button": True,
                    "grip": 0.25,
                }

            def close(self):
                self.closed = True

        source = None
        with mock.patch("omnisurg.input.follou.MiniMouController", FakeController):
            source = LiveMiniMouSource(root=r"G:\warp\python_device_manager", device_index=1)
            sample = source.poll()
            np.testing.assert_allclose(sample["position"], np.array([1.0, 2.0, 3.0], dtype=np.float32))
            np.testing.assert_allclose(sample["rotation"], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            self.assertTrue(sample["button"])
            self.assertAlmostEqual(sample["grip"], 0.25)
            source.close()
            self.assertIsNone(source._ctrl)

    def test_minimou_controller_flips_x_position_and_maps_grip(self):
        class FakeDevice:
            def __init__(self):
                self._tool_positions = iter((10.0, 30.0, 15.0))

            def perform_update(self):
                pass

            def get_position(self):
                return (1.0, 2.0, 3.0, 1.0)

            def get_orientation(self):
                return (0.0, 0.0, 1.0, 0.0)

            def get_tool_pos(self):
                return next(self._tool_positions)

            def close(self):
                pass

        class FakeManager:
            def __init__(self):
                self.devices = [FakeDevice()]

            def get_device_controller(self, cls, count=0):
                return self.devices[count]

        controller = None
        with mock.patch("omnisurg.input.follou.acquire_manager", return_value=(Path(r"G:\warp\python_device_manager"), FakeManager(), object())):
            with mock.patch("omnisurg.input.follou.release_manager"):
                controller = MiniMouController(root=r"G:\warp\python_device_manager", device_index=0, scale=1.0)
                sample = controller.poll()
                np.testing.assert_allclose(sample["position"], np.array([-1.0, 2.0, 3.0], dtype=np.float32))
                np.testing.assert_allclose(sample["rotation"], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
                self.assertFalse(sample["button"])
                self.assertAlmostEqual(sample["grip"], 0.0)
                controller.poll()
                sample = controller.poll()
                self.assertTrue(sample["button"])
                self.assertAlmostEqual(sample["grip"], 0.75)
                controller.close()

    def test_replay_button_support(self):
        replay_path = TEST_TMP_DIR / "trace_button.npy"
        np.save(replay_path, np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=np.float32))
        try:
            source = ReplayInputSource(str(replay_path))
            sample = source.poll()
            self.assertTrue(sample["button"])
            self.assertEqual(sample["rotation"].shape[0], 4)
        finally:
            if replay_path.exists():
                replay_path.unlink()

    def test_replay_grip_support(self):
        replay_path = TEST_TMP_DIR / "trace_grip.npy"
        np.save(replay_path, np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.25]], dtype=np.float32))
        try:
            source = ReplayInputSource(str(replay_path))
            sample = source.poll()
            self.assertFalse(sample["button"])
            self.assertAlmostEqual(sample["grip"], 0.25)
        finally:
            if replay_path.exists():
                replay_path.unlink()

    def test_bimanual_replay_rig_support(self):
        right_path = TEST_TMP_DIR / "trace_right.npy"
        left_path = TEST_TMP_DIR / "trace_left.npy"
        np.save(right_path, np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=np.float32))
        np.save(left_path, np.array([[50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32))
        rig = None
        try:
            rig = BimanualReplayRig(right_path=str(right_path), left_path=str(left_path))
            frame = rig.poll()
            self.assertIn("right", frame)
            self.assertIn("left", frame)
            self.assertTrue(frame["right"].button)
            self.assertAlmostEqual(float(frame["left"].position[0]), 50.0)
        finally:
            if rig is not None:
                rig.close()
            if right_path.exists():
                right_path.unlink()
            if left_path.exists():
                left_path.unlink()

    def test_asset_loading_liver_counts(self):
        asset = load_tet_asset("liver")
        self.assertEqual(int(asset.rest_positions.shape[0]), 1986)
        self.assertEqual(int(asset.tet_indices.shape[0]), 6653)
        self.assertEqual(int(asset.edge_indices.shape[0]), 10622)
        self.assertEqual(int(asset.surface_tri_indices.shape[0]), 3968)
        self.assertTrue(asset.uvs is not None)

    def test_multi_organ_asset_loading(self):
        asset = load_scene_asset(SceneConfig(scene_preset="chole"))
        self.assertEqual(int(asset.rest_positions.shape[0]), 5612)
        self.assertEqual(int(asset.tet_indices.shape[0]), 18823)
        self.assertEqual(int(asset.edge_indices.shape[0]), 30038)
        self.assertEqual(int(asset.surface_tri_indices.shape[0]), 11212)
        self.assertEqual(len(asset.connectors), 1479)
        self.assertEqual(asset.mesh_ranges["fat"].vertex_start, 1986)
        self.assertEqual(asset.mesh_ranges["gallbladder"].vertex_start, 4674)
        self.assertEqual(asset.uvs.shape, (5612, 2))

    def test_scene_construction_single_asset(self):
        with wp.ScopedDevice("cpu"):
            asset = load_tet_asset("liver")
            scene = build_scene(asset, SceneConfig(), HapticConfig(), wp.get_device())

        self.assertEqual(scene.model.particle_count, 1986)
        self.assertEqual(scene.model.spring_count, 10622)
        self.assertEqual(scene.model.tri_count, 3968)
        self.assertEqual(scene.tetrahedra_wp.shape[0], 6653)
        self.assertEqual(len(scene.model.tri_points_connectors), 0)
        self.assertGreaterEqual(scene.haptic_proxy.body_id, 0)
        self.assertEqual(scene.haptic_proxy.radius, 0.1)
        self.assertIsNotNone(scene.uvs)
        self.assertEqual(scene.uvs.shape[0], 1986)

    def test_scene_construction_multi_organ(self):
        with wp.ScopedDevice("cpu"):
            asset = load_scene_asset(SceneConfig(scene_preset="chole"))
            scene = build_scene(asset, SceneConfig(scene_preset="chole"), HapticConfig(), wp.get_device())

        self.assertEqual(scene.model.particle_count, 5612)
        self.assertEqual(scene.model.spring_count, 30038)
        self.assertEqual(scene.model.tri_count, 11212)
        self.assertEqual(scene.tetrahedra_wp.shape[0], 18823)
        self.assertEqual(len(scene.model.tri_points_connectors), 1479)
        self.assertEqual(scene.mesh_ranges["liver"].tet_count, 6653)
        self.assertEqual(scene.mesh_ranges["fat"].tet_count, 9102)
        self.assertEqual(scene.mesh_ranges["gallbladder"].tet_count, 3068)
        self.assertIsNotNone(scene.uvs)
        self.assertEqual(scene.uvs.shape[0], 5612)

    def test_grasper_asset_loading(self):
        with wp.ScopedDevice("cpu"):
            grasper = load_kinematic_grasper(REPO_ROOT / "meshes" / "pgrasp.usdc", wp.get_device())

        self.assertIsNotNone(grasper)
        self.assertEqual(len(grasper.pieces), 4)
        self.assertEqual(len(grasper.sphere_chains), 2)
        self.assertTrue(any(piece.jaw_sign > 0.0 for piece in grasper.pieces))
        self.assertTrue(any(piece.jaw_sign < 0.0 for piece in grasper.pieces))

    def test_runtime_texture_toggle(self):
        runtime = None
        try:
            with wp.ScopedDevice("cpu"):
                runtime = Runtime(
                    SimulationConfig(substeps=1, fps=20, constraint_iterations=1),
                    SceneConfig(scene_preset="chole"),
                    HapticConfig(),
                    ViewerConfig(backend="headless"),
                    BoundsConfig(),
                )
                self.assertTrue(runtime.textures_enabled)
                runtime.toggle_textures()
                self.assertFalse(runtime.textures_enabled)
                runtime.toggle_textures()
                self.assertTrue(runtime.textures_enabled)
        finally:
            if runtime is not None:
                runtime.close()

    def test_render_bridge_normalizes_scalar_point_inputs(self):
        class FakeRenderer:
            def __init__(self):
                self.calls = []

            def log_points(self, name, points, radii, colors):
                self.calls.append((name, points, radii, colors))

        with wp.ScopedDevice("cpu"):
            bridge = RenderBridge.__new__(RenderBridge)
            bridge._backend = "gl"
            bridge._renderer = FakeRenderer()
            bridge.gpu = type("GpuStub", (), {"device": wp.get_device()})()
            bridge._point_radii = {}
            bridge._point_colors = {}
            points = wp.zeros(1, dtype=wp.vec3, device=wp.get_device())

            bridge.draw_points(
                "left_controller_root",
                points,
                0.025,
                (0.18, 0.78, 1.0),
            )

            self.assertEqual(len(bridge._renderer.calls), 1)
            _name, _points, radii, colors = bridge._renderer.calls[0]
            np.testing.assert_allclose(radii.numpy(), np.array([0.025], dtype=np.float32))
            np.testing.assert_allclose(colors.numpy(), np.array([[0.18, 0.78, 1.0]], dtype=np.float32))

    def test_headless_replay_runtime_smoke_single_asset(self):
        trace = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 0.5, -0.5, 0.0, 0.0, 0.0, 1.0, 0.0],
                [2.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        replay_path = TEST_TMP_DIR / "trace_single.npy"
        np.save(replay_path, trace)

        source = None
        runtime = None
        try:
            with wp.ScopedDevice("cpu"):
                runtime = Runtime(
                    SimulationConfig(substeps=2, fps=30, constraint_iterations=1),
                    SceneConfig(asset_name="liver"),
                    HapticConfig(),
                    ViewerConfig(backend="headless", textures_enabled=False),
                    BoundsConfig(),
                )
                source = ReplayInputSource(str(replay_path))

                for _ in range(trace.shape[0]):
                    runtime.poll_input(source)
                    runtime.step()
                    runtime.render()

                self.assertTrue(runtime.is_running())
                self.assertFalse(runtime.textures_enabled)
                self.assertTrue(Path(runtime.mesh_textures["liver"]).exists())
                self.assertIsNotNone(runtime.graspers["right"])
                runtime.close()
                self.assertFalse(runtime.is_running())
        finally:
            if source is not None:
                source.close()
            if runtime is not None:
                runtime.close()
            if replay_path.exists():
                replay_path.unlink()

    def test_headless_replay_runtime_smoke_multi_organ(self):
        trace = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.5, 0.5, -0.5, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        replay_path = TEST_TMP_DIR / "trace_chole.npy"
        np.save(replay_path, trace)

        source = None
        runtime = None
        try:
            with wp.ScopedDevice("cpu"):
                runtime = Runtime(
                    SimulationConfig(substeps=1, fps=20, constraint_iterations=1),
                    SceneConfig(scene_preset="chole"),
                    HapticConfig(),
                    ViewerConfig(backend="headless"),
                    BoundsConfig(),
                )
                source = ReplayInputSource(str(replay_path))

                for _ in range(trace.shape[0]):
                    runtime.poll_input(source)
                    runtime.step()
                    runtime.render()

                self.assertTrue(runtime.is_running())
                self.assertEqual(len(runtime.model.tri_points_connectors), 1479)
                for name in ("liver", "fat", "gallbladder"):
                    self.assertTrue(Path(runtime.mesh_textures[name]).exists())
        finally:
            if source is not None:
                source.close()
            if runtime is not None:
                runtime.close()
            if replay_path.exists():
                replay_path.unlink()

    def test_grasper_follows_haptic_and_closes_on_button(self):
        trace = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        replay_path = TEST_TMP_DIR / "trace_grasper.npy"
        np.save(replay_path, trace)

        source = None
        runtime = None
        try:
            with wp.ScopedDevice("cpu"):
                runtime = Runtime(
                    SimulationConfig(substeps=2, fps=30, constraint_iterations=1),
                    SceneConfig(scene_preset="chole"),
                    HapticConfig(),
                    ViewerConfig(backend="headless"),
                    BoundsConfig(),
                )
                source = ReplayInputSource(str(replay_path))
                initial_angle = runtime.graspers["right"].jaw_angle

                for _ in range(trace.shape[0]):
                    runtime.poll_input(source)
                    runtime.step()
                    runtime.render()

                self.assertLess(runtime.graspers["right"].jaw_angle, initial_angle)
                left_chain = next(chain for chain in runtime.graspers["right"].sphere_chains if chain.jaw_sign > 0.0)
                left_points = left_chain.world_points.numpy()
                self.assertEqual(left_points.shape[0], 16)
                self.assertAlmostEqual(float(left_points[0][0]), 0.0, delta=0.25)
                self.assertAlmostEqual(float(left_points[0][1]), 0.85, delta=0.35)
                self.assertAlmostEqual(float(left_points[0][2]), -3.4, delta=0.7)
        finally:
            if source is not None:
                source.close()
            if runtime is not None:
                runtime.close()
            if replay_path.exists():
                replay_path.unlink()

    def test_grasper_tracks_partial_grip_signal(self):
        trace = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.75],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.75],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.75],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.75],
            ],
            dtype=np.float32,
        )

        replay_path = TEST_TMP_DIR / "trace_grip_partial.npy"
        np.save(replay_path, trace)

        source = None
        runtime = None
        try:
            with wp.ScopedDevice("cpu"):
                runtime = Runtime(
                    SimulationConfig(substeps=2, fps=30, constraint_iterations=1),
                    SceneConfig(scene_preset="chole"),
                    HapticConfig(),
                    ViewerConfig(backend="headless"),
                    BoundsConfig(),
                )
                source = ReplayInputSource(str(replay_path))
                initial_angle = runtime.graspers["right"].jaw_angle

                for _ in range(trace.shape[0]):
                    runtime.poll_input(source)
                    runtime.step()
                    runtime.render()

                self.assertLess(runtime.graspers["right"].jaw_angle, initial_angle)
                self.assertGreater(runtime.graspers["right"].jaw_angle, runtime.graspers["right"].jaw_closed_angle)
        finally:
            if source is not None:
                source.close()
            if runtime is not None:
                runtime.close()
            if replay_path.exists():
                replay_path.unlink()

    def test_grasper_root_position_interpolates_across_substeps(self):
        trace = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        replay_path = TEST_TMP_DIR / "trace_grasper_interp.npy"
        np.save(replay_path, trace)

        source = None
        runtime = None
        try:
            with wp.ScopedDevice("cpu"):
                runtime = Runtime(
                    SimulationConfig(substeps=4, fps=30, constraint_iterations=1),
                    SceneConfig(scene_preset="chole"),
                    HapticConfig(),
                    ViewerConfig(backend="headless"),
                    BoundsConfig(),
                )
                source = ReplayInputSource(str(replay_path))

                runtime.poll_input(source)
                runtime.step()

                runtime.poll_input(source)
                runtime.step()

                grasper = runtime.graspers["right"]
                self.assertIsNotNone(grasper)
                np.testing.assert_allclose(
                    grasper.root_position_target.numpy()[0],
                    np.array([1.0, 1.0, -4.0], dtype=np.float32),
                    atol=1.0e-4,
                )
                np.testing.assert_allclose(
                    grasper.root_position.numpy()[0],
                    np.array([0.75, 1.0, -4.0], dtype=np.float32),
                    atol=1.0e-4,
                )
        finally:
            if source is not None:
                source.close()
            if runtime is not None:
                runtime.close()
            if replay_path.exists():
                replay_path.unlink()

    def test_bimanual_graspers_follow_independent_controllers(self):
        right_trace = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        left_trace = np.array(
            [
                [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        right_path = TEST_TMP_DIR / "trace_right_bimanual.npy"
        left_path = TEST_TMP_DIR / "trace_left_bimanual.npy"
        np.save(right_path, right_trace)
        np.save(left_path, left_trace)

        rig = None
        runtime = None
        try:
            with wp.ScopedDevice("cpu"):
                runtime = Runtime(
                    SimulationConfig(substeps=2, fps=30, constraint_iterations=1),
                    SceneConfig(scene_preset="chole"),
                    HapticConfig(),
                    ViewerConfig(backend="headless"),
                    BoundsConfig(),
                )
                rig = BimanualReplayRig(right_path=str(right_path), left_path=str(left_path))

                for _ in range(right_trace.shape[0]):
                    runtime.poll_input(rig)
                    runtime.step()
                    runtime.render()

                self.assertTrue(runtime.controller_states["right"].active)
                self.assertTrue(runtime.controller_states["left"].active)
                self.assertLess(runtime.graspers["right"].jaw_angle, runtime.graspers["left"].jaw_angle)

                right_chain = next(chain for chain in runtime.graspers["right"].sphere_chains if chain.jaw_sign > 0.0)
                left_chain = next(chain for chain in runtime.graspers["left"].sphere_chains if chain.jaw_sign > 0.0)
                right_points = right_chain.world_points.numpy()
                left_points = left_chain.world_points.numpy()
                self.assertEqual(right_points.shape[0], 16)
                self.assertEqual(left_points.shape[0], 16)
                self.assertGreater(float(left_points[0][0]) - float(right_points[0][0]), 0.75)
        finally:
            if rig is not None:
                rig.close()
            if runtime is not None:
                runtime.close()
            if right_path.exists():
                right_path.unlink()
            if left_path.exists():
                left_path.unlink()


if __name__ == "__main__":
    unittest.main()

