import os
import shutil
import subprocess
import sys
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import meshio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WARP_CACHE_DIR = REPO_ROOT / ".warp_cache"
TEST_TMP_DIR = REPO_ROOT / ".tmp_tests"
WARP_CACHE_DIR.mkdir(exist_ok=True)
TEST_TMP_DIR.mkdir(exist_ok=True)
os.environ["WARP_CACHE_PATH"] = str(WARP_CACHE_DIR)

import warp as wp

from omnisurg import BoundsConfig, HapticConfig, Runtime, SceneConfig, SimulationConfig, ViewerConfig
import omnisurg.haptic_feedback as haptic_feedback_module
import omnisurg.main as omnisurg_main
import omnisurg.input.haptic_collision as haptic_collision_module
import omnisurg.physics.collision as collision_module
import omnisurg.runtime as runtime_module
from omnisurg.assets import load_scene_asset, load_tet_asset
from omnisurg.haptics import BimanualReplayRig, ReplayInputSource
from omnisurg.instruments.grasper import load_kinematic_grasper
from omnisurg.input.sources import MultiSourceRig
from omnisurg.rendering.bridge import RenderBridge
from omnisurg.rendering.input import ViewportInputAdapter
from omnisurg.rendering.slang import (
    LENS_DIRT_TEXTURE_LABELS,
    PostProcessParams,
    SlangRenderer,
    TISSUE_DEBUG_MODE_LABELS,
    TissueMaterialParams,
    _SlangImmediateUi,
)
from omnisurg.input.follou import MiniMouController
from omnisurg.input.sources import LiveHapticSource, LiveMiniMouSource
from omnisurg.mesh.vtk_export import export_asset_dir_to_vtk
from omnisurg.scene_builder import build_scene


SYNTHETIC_PATCH_ASSETS = {
    "cloth_regular_low": {"vertices": 49, "edges": 120, "tris": 72},
    "cloth_regular_mid": {"vertices": 169, "edges": 456, "tris": 288},
    "cloth_regular_high": {"vertices": 625, "edges": 1776, "tris": 1152},
    "cloth_irregular_low": {"vertices": 49, "edges": 120, "tris": 72},
    "cloth_irregular_mid": {"vertices": 169, "edges": 456, "tris": 288},
    "cloth_irregular_high": {"vertices": 625, "edges": 1776, "tris": 1152},
}


def _live_args(**overrides):
    values = {
        "right_input_backend": "openhaptics",
        "left_input_backend": "openhaptics",
        "right_device_name": "Default Device",
        "left_device_name": "Left Device",
        "right_device_index": 0,
        "left_device_index": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_haptic_runtime_stub(preset_dir: Path):
    runtime = runtime_module.Runtime.__new__(runtime_module.Runtime)
    runtime.controller_states = {
        runtime_module.PRIMARY_CONTROLLER_ID: SimpleNamespace(
            sample_present=True,
            scaled_position=np.zeros(3, dtype=np.float32),
        )
    }
    runtime.sim_config = SimpleNamespace(frame_dt=0.1)
    with mock.patch.object(runtime_module, "HAPTIC_PRESET_DIR", preset_dir):
        runtime_module.Runtime._initialize_haptic_feedback(runtime)
    return runtime


def _prepare_test_subdir(name: str) -> Path:
    path = TEST_TMP_DIR / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


class _FakeSlangUiWidget:
    def __init__(self, parent=None):
        self.parent = parent
        self.visible = True


class _FakeSlangUiText(_FakeSlangUiWidget):
    def __init__(self, parent, text=""):
        super().__init__(parent)
        self.text = text


class _FakeSlangUiGroup(_FakeSlangUiWidget):
    def __init__(self, parent, label=""):
        super().__init__(parent)
        self.label = label


class _FakeSlangUiButton(_FakeSlangUiWidget):
    def __init__(self, parent, label="", callback=None):
        super().__init__(parent)
        self.label = label
        self.callback = callback


class _FakeSlangUiValue(_FakeSlangUiWidget):
    def __init__(self, parent, label="", value=None, callback=None):
        super().__init__(parent)
        self.label = label
        self.value = value
        self.callback = callback


class _FakeSlangUiSlider(_FakeSlangUiValue):
    def __init__(self, parent, label="", value=0.0, callback=None, min=0.0, max=1.0, format="%.3f"):
        super().__init__(parent, label, value, callback)
        self.min = min
        self.max = max
        self.format = format


class _FakeSlangUiCombo(_FakeSlangUiValue):
    def __init__(self, parent, label="", value=0, callback=None, items=None):
        super().__init__(parent, label, value, callback)
        self.items = [] if items is None else list(items)


class _FakeSlangUiWindow(_FakeSlangUiWidget):
    def __init__(self, parent, title="", position=None, size=None):
        super().__init__(parent)
        self.title = title
        self.position = position
        self.size = size


class _FakeSlangUiModule:
    def __init__(self):
        self.created = defaultdict(int)

    def Text(self, parent, text=""):
        self.created["Text"] += 1
        return _FakeSlangUiText(parent, text)

    def Group(self, parent, label=""):
        self.created["Group"] += 1
        return _FakeSlangUiGroup(parent, label)

    def Window(self, parent, title="", position=None, size=None):
        self.created["Window"] += 1
        return _FakeSlangUiWindow(parent, title, position, size)

    def CheckBox(self, parent, label="", value=False, callback=None):
        self.created["CheckBox"] += 1
        return _FakeSlangUiValue(parent, label, value, callback)

    def SliderFloat(self, parent, label="", value=0.0, callback=None, min=0.0, max=1.0, format="%.3f"):
        self.created["SliderFloat"] += 1
        return _FakeSlangUiSlider(parent, label, value, callback, min, max, format)

    def SliderInt(self, parent, label="", value=0, callback=None, min=0, max=1):
        self.created["SliderInt"] += 1
        return _FakeSlangUiSlider(parent, label, value, callback, min, max, "%d")

    def Button(self, parent, label="", callback=None):
        self.created["Button"] += 1
        return _FakeSlangUiButton(parent, label, callback)

    def ComboBox(self, parent, label="", value=0, callback=None, items=None):
        self.created["ComboBox"] += 1
        return _FakeSlangUiCombo(parent, label, value, callback, items)

    def InputText(self, parent, label="", value="", callback=None):
        self.created["InputText"] += 1
        return _FakeSlangUiValue(parent, label, value, callback)


class _FakeTissueDebugComboUi:
    def __init__(self, combo_result=(False, 0), button_result=False):
        self.combo_result = combo_result
        self.button_result = button_result
        self.combo_calls = []
        self.texts = []
        self.buttons = []

    def combo(self, label, value, items):
        self.combo_calls.append((label, value, tuple(items)))
        return self.combo_result

    def text(self, value):
        self.texts.append(str(value))

    def button(self, label):
        self.buttons.append(label)
        return self.button_result


class _FakeTissueDebugSliderUi:
    def __init__(self, slider_result=(False, 0), button_result=False):
        self.slider_result = slider_result
        self.button_result = button_result
        self.slider_calls = []
        self.texts = []
        self.buttons = []

    def slider_int(self, label, value, min_value, max_value):
        self.slider_calls.append((label, value, min_value, max_value))
        return self.slider_result

    def text(self, value):
        self.texts.append(str(value))

    def button(self, label):
        self.buttons.append(label)
        return self.button_result


class _FakeTissueMaterialUi:
    def __init__(self, slider_results=None, checkbox_results=None):
        self.slider_results = {} if slider_results is None else dict(slider_results)
        self.checkbox_results = {} if checkbox_results is None else dict(checkbox_results)
        self.slider_calls = []
        self.checkbox_calls = []
        self.combo_calls = []
        self.texts = []
        self.buttons = []

    def combo(self, label, value, items):
        self.combo_calls.append((label, value, tuple(items)))
        return False, value

    def slider_float(self, label, value, min_value, max_value, fmt):
        self.slider_calls.append((label, value, min_value, max_value, fmt))
        if label in self.slider_results:
            return True, self.slider_results[label]
        return False, value

    def checkbox(self, label, value):
        self.checkbox_calls.append((label, value))
        if label in self.checkbox_results:
            return True, self.checkbox_results[label]
        return False, value

    def text(self, value):
        self.texts.append(str(value))

    def button(self, label):
        self.buttons.append(label)
        return False


class _FakePostProcessUi:
    def __init__(self, checkbox_results=None, slider_results=None, combo_results=None):
        self.checkbox_results = {} if checkbox_results is None else dict(checkbox_results)
        self.slider_results = {} if slider_results is None else dict(slider_results)
        self.combo_results = {} if combo_results is None else dict(combo_results)
        self.checkbox_calls = []
        self.slider_calls = []
        self.combo_calls = []
        self.texts = []

    def checkbox(self, label, value):
        self.checkbox_calls.append((label, value))
        if label in self.checkbox_results:
            return True, self.checkbox_results[label]
        return False, value

    def slider_float(self, label, value, min_value, max_value, fmt):
        self.slider_calls.append((label, value, min_value, max_value, fmt))
        if label in self.slider_results:
            return True, self.slider_results[label]
        return False, value

    def combo(self, label, value, items):
        self.combo_calls.append((label, value, tuple(items)))
        if label in self.combo_results:
            return True, self.combo_results[label]
        return False, value

    def text(self, value):
        self.texts.append(str(value))


class TestPhaseRuntime(unittest.TestCase):
    def _make_slang_camera_renderer(self):
        renderer = SlangRenderer.__new__(SlangRenderer)
        renderer._camera_world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        renderer._camera_scene_radius = 1.0
        renderer._camera_default_pos = np.array([0.0, 0.0, 3.0], dtype=np.float32)
        renderer._camera_default_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        renderer._camera_key_state = set()
        renderer._camera_fast_modifier = False
        renderer._camera_slow_modifier = False
        renderer._camera_mouse_action = None
        renderer._camera_mouse_pos = None
        renderer._set_camera_look_at(renderer._camera_default_pos, renderer._camera_default_target)
        return renderer

    def _slang_key_event(self, key, *, press=False, release=False, repeat=False):
        return SimpleNamespace(
            key=SimpleNamespace(name=key),
            is_key_press=lambda: press,
            is_key_release=lambda: release,
            is_key_repeat=lambda: repeat,
        )

    def _slang_mouse_event(
        self,
        event_type,
        *,
        pos=(0.0, 0.0),
        button=None,
        scroll=(0.0, 0.0),
    ):
        return SimpleNamespace(
            pos=SimpleNamespace(x=pos[0], y=pos[1]),
            button=SimpleNamespace(name=button) if button is not None else None,
            scroll=SimpleNamespace(x=scroll[0], y=scroll[1]),
            is_button_down=lambda: event_type == "down",
            is_button_up=lambda: event_type == "up",
            is_move=lambda: event_type == "move",
            is_scroll=lambda: event_type == "scroll",
        )

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
            def __init__(self, device_index: int):
                self.device_index = device_index

            def poll(self):
                return {}

            def close(self):
                pass

        def build_source(*, device_index: int, scale: float = 1.0):
            return FakeSource(device_index)

        args = _live_args(
            right_input_backend="minimou",
            left_input_backend="none",
            right_device_index=2,
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

    def test_live_input_rig_preserves_minimou_position_and_maps_rotation_to_tet_frame(self):
        from omnisurg.hex.haptic import axis_degrees_to_quaternion, quat_to_matrix

        class FakeSource:
            def __init__(self, *, device_index: int, scale: float = 1.0):
                del device_index, scale

            def poll(self):
                return {
                    "position": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                    "rotation": np.array(axis_degrees_to_quaternion(0.0, 1.0, 0.0, 30.0), dtype=np.float32),
                    "valid": True,
                }

            def close(self):
                pass

        args = _live_args(
            right_input_backend="minimou",
            left_input_backend="none",
            right_device_index=2,
        )
        rig = None
        with mock.patch("omnisurg.haptics.LiveMiniMouSource", FakeSource):
            rig = omnisurg_main._build_live_input_rig(args)

        try:
            self.assertIsNotNone(rig)
            sample = rig.poll()["right"]
            np.testing.assert_allclose(sample.position, np.array([1.0, 2.0, 3.0], dtype=np.float32))
            expected_rotation = axis_degrees_to_quaternion(0.0, 0.0, -1.0, 30.0)
            np.testing.assert_allclose(quat_to_matrix(sample.rotation), quat_to_matrix(expected_rotation), atol=1e-6)
        finally:
            if rig is not None:
                rig.close()

    def test_live_minimou_source_polls_controller_samples(self):
        class FakeController:
            def __init__(self, *, device_index=0, scale=1.0):
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
            source = LiveMiniMouSource(device_index=1)
            sample = source.poll()
            np.testing.assert_allclose(sample["position"], np.array([1.0, 2.0, 3.0], dtype=np.float32))
            np.testing.assert_allclose(sample["rotation"], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            self.assertTrue(sample["button"])
            self.assertAlmostEqual(sample["grip"], 0.25)
            source.close()
            self.assertIsNone(source._ctrl)

    def test_live_haptic_source_prefers_single_snapshot_poll(self):
        class FakeController:
            def __init__(self):
                self.calls = []

            def poll_state(self):
                self.calls.append("poll_state")
                return {
                    "position": [1.0, 2.0, 3.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "button": True,
                }

        source = LiveHapticSource.__new__(LiveHapticSource)
        source._ctrl = FakeController()
        source._reported_failure = False
        source._device_name = "test-device"

        sample = source.poll()

        np.testing.assert_allclose(sample["position"], np.array([1.0, 2.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(sample["rotation"], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        self.assertTrue(sample["button"])
        self.assertEqual(source._ctrl.calls, ["poll_state"])

    def test_minimou_controller_flips_x_position_and_maps_grip(self):
        class FakeDevice:
            def __init__(self):
                self._tool_positions = iter((10.0, 30.0, 15.0))

            def perform_update(self):
                pass

            def get_position(self):
                return (1.0, 2.0, 3.0, 1.0)

            def get_orientation(self):
                return (0.0, 0.0, 0.0, 1.0)

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
        with mock.patch("omnisurg.input.follou.acquire_manager", return_value=(FakeManager(), object())):
            with mock.patch("omnisurg.input.follou.release_manager"):
                controller = MiniMouController(device_index=0, scale=1.0)
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

    def test_vtk_export_writes_liver_volume_with_uvs(self):
        output_dir = _prepare_test_subdir("vtk_export_liver")
        output_path = output_dir / "liver.vtu"

        result = export_asset_dir_to_vtk(REPO_ROOT / "meshes" / "liver", output_path=output_path)
        mesh = meshio.read(result.primary_output)
        asset = load_tet_asset("liver")

        self.assertEqual(result.mode, "volume")
        self.assertEqual(result.primary_output, output_path)
        self.assertTrue(result.primary_output.exists())
        self.assertIsNone(result.surface_output)
        self.assertEqual(mesh.points.shape, asset.rest_positions.shape)
        self.assertIn("tetra", mesh.cells_dict)
        self.assertEqual(mesh.cells_dict["tetra"].shape, asset.tet_indices.shape)
        self.assertIn("uv", mesh.point_data)
        self.assertEqual(mesh.point_data["uv"].shape, asset.uvs.shape)

    def test_vtk_export_auto_falls_back_to_surface_for_patch_meshes(self):
        output_dir = _prepare_test_subdir("vtk_export_surface_only")
        output_path = output_dir / "cloth_regular_low.vtp"

        result = export_asset_dir_to_vtk(
            REPO_ROOT / "meshes" / "cloth_regular_low",
            output_path=output_path,
        )
        mesh = meshio.read(result.primary_output)
        asset = load_tet_asset("cloth_regular_low")

        self.assertEqual(result.mode, "surface")
        self.assertEqual(result.primary_output, output_path)
        self.assertTrue(result.primary_output.exists())
        self.assertIn("triangle", mesh.cells_dict)
        self.assertEqual(mesh.cells_dict["triangle"].shape, asset.surface_tri_indices.shape)
        self.assertNotIn("tetra", mesh.cells_dict)

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

    def test_synthetic_patch_asset_loading(self):
        for asset_name, expected in SYNTHETIC_PATCH_ASSETS.items():
            with self.subTest(asset=asset_name):
                asset = load_tet_asset(asset_name)
                self.assertEqual(int(asset.rest_positions.shape[0]), expected["vertices"])
                self.assertEqual(int(asset.edge_indices.shape[0]), expected["edges"])
                self.assertEqual(int(asset.surface_tri_indices.shape[0]), expected["tris"])
                self.assertEqual(int(asset.tet_indices.shape[0]), 0)
                self.assertIsNotNone(asset.uvs)
                self.assertEqual(asset.uvs.shape[0], expected["vertices"])

    def test_synthetic_patch_assets_are_horizontal_and_above_ground(self):
        for asset_name in ("cloth_regular_low", "cloth_irregular_mid"):
            with self.subTest(asset=asset_name):
                asset = load_tet_asset(asset_name)
                scene_config = SceneConfig(asset_name=asset_name)

                self.assertLess(float(np.ptp(asset.rest_positions[:, 1])), 1.0e-6)
                self.assertGreater(float(np.ptp(asset.rest_positions[:, 0])), 0.0)
                self.assertGreater(float(np.ptp(asset.rest_positions[:, 2])), 0.0)

                translated_y = asset.rest_positions[:, 1] + float(scene_config.translation[1])
                self.assertGreater(float(np.min(translated_y)), 0.5)

    def test_synthetic_patch_assets_wind_upward(self):
        for asset_name in ("cloth_regular_low", "cloth_irregular_mid"):
            with self.subTest(asset=asset_name):
                asset = load_tet_asset(asset_name)
                tri = asset.surface_tri_indices[0]
                p0 = asset.rest_positions[tri[0]]
                p1 = asset.rest_positions[tri[1]]
                p2 = asset.rest_positions[tri[2]]
                normal = np.cross(p1 - p0, p2 - p0)
                self.assertGreater(float(normal[1]), 0.0)

    def test_synthetic_patch_scene_pins_only_four_corners(self):
        expected_corner_ids = {0, 6, 42, 48}
        for asset_name in ("cloth_regular_low", "cloth_irregular_low"):
            with self.subTest(asset=asset_name):
                scene_config = SceneConfig(asset_name=asset_name)
                self.assertIsNone(scene_config.pin_center)
                self.assertFalse(scene_config.show_grasper_mesh)
                self.assertFalse(scene_config.enable_grasper_collisions)

                with wp.ScopedDevice("cpu"):
                    asset = load_tet_asset(asset_name)
                    scene = build_scene(asset, scene_config, HapticConfig(), wp.get_device())

                inv_masses = scene.model.particle_inv_mass.numpy()
                pinned_ids = {idx for idx, inv_mass in enumerate(inv_masses) if float(inv_mass) == 0.0}
                self.assertEqual(pinned_ids, expected_corner_ids)

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

    def test_scene_construction_synthetic_patch_assets(self):
        for asset_name, expected in (
            ("cloth_regular_low", SYNTHETIC_PATCH_ASSETS["cloth_regular_low"]),
            ("cloth_irregular_mid", SYNTHETIC_PATCH_ASSETS["cloth_irregular_mid"]),
        ):
            with self.subTest(asset=asset_name):
                with wp.ScopedDevice("cpu"):
                    asset = load_tet_asset(asset_name)
                    scene = build_scene(asset, SceneConfig(asset_name=asset_name), HapticConfig(), wp.get_device())

                self.assertEqual(scene.model.particle_count, expected["vertices"])
                self.assertEqual(scene.model.spring_count, expected["edges"])
                self.assertEqual(scene.model.tri_count, expected["tris"])
                self.assertEqual(scene.tetrahedra_wp.shape[0], 0)
                self.assertEqual(len(scene.model.tri_points_connectors), 0)
                self.assertIn(asset_name, scene.surface_meshes)
                self.assertEqual(scene.surface_meshes[asset_name].shape[0], expected["tris"] * 3)

    def test_runtime_synthetic_patch_uses_haptic_proxy_only(self):
        runtime = None
        try:
            with wp.ScopedDevice("cpu"):
                runtime = Runtime(
                    SimulationConfig(substeps=1, fps=20, constraint_iterations=1),
                    SceneConfig(asset_name="cloth_regular_low"),
                    HapticConfig(),
                    ViewerConfig(backend="headless", textures_enabled=False),
                    BoundsConfig(),
                )

                self.assertFalse(runtime.show_grasper_mesh)
                self.assertFalse(runtime.enable_grasper_collisions)
                self.assertTrue(runtime.haptic_collision_system.enabled)
                self.assertFalse(runtime.haptic_truncation_system.enabled)
                self.assertFalse(runtime.grasper_collision_system.enabled)
                self.assertFalse(runtime.grasper_truncation_system.enabled)
                self.assertAlmostEqual(runtime.haptic_visual_radius, 0.0625)

                runtime._pending_grasper_collision_mode = "truncation"
                self.assertTrue(runtime._has_pending_solver_changes())
                self.assertTrue(runtime._pending_graph_rebuild_needed())

                runtime._apply_pending_solver_settings()

                self.assertFalse(runtime.enable_grasper_collisions)
                self.assertFalse(runtime.haptic_collision_system.enabled)
                self.assertTrue(runtime.haptic_truncation_system.enabled)
                self.assertFalse(runtime.grasper_collision_system.enabled)
                self.assertFalse(runtime.grasper_truncation_system.enabled)

                runtime._pending_enable_grasper_collisions = True
                runtime._pending_grasper_collision_mode = "hybrid"
                runtime._apply_pending_solver_settings()

                self.assertTrue(runtime.enable_grasper_collisions)
                self.assertTrue(runtime.haptic_collision_system.enabled)
                self.assertTrue(runtime.haptic_truncation_system.enabled)
                self.assertTrue(runtime.grasper_collision_system.enabled)
                self.assertTrue(runtime.grasper_truncation_system.enabled)
        finally:
            if runtime is not None:
                runtime.close()

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
                self.assertEqual(len(runtime.vertex_colors), runtime.model.particle_count)
                np.testing.assert_allclose(
                    runtime.vertex_colors.numpy(),
                    np.zeros((runtime.model.particle_count, 4), dtype=np.float32),
                )
                runtime.toggle_textures()
                self.assertFalse(runtime.textures_enabled)
                runtime.toggle_textures()
                self.assertTrue(runtime.textures_enabled)
        finally:
            if runtime is not None:
                runtime.close()

    def test_tissue_blend_debug_channels_fill_vertex_colors(self):
        with wp.ScopedDevice("cpu"):
            runtime = runtime_module.Runtime.__new__(runtime_module.Runtime)
            runtime.device = wp.get_device()
            runtime.vertex_colors = wp.zeros(4, dtype=wp.vec4f, device=runtime.device)
            runtime.tissue_blend_damage = 0.0
            runtime.tissue_blend_coag = 0.0
            runtime.tissue_blend_blood = 0.0
            runtime._tissue_blend_dirty = False

            runtime._set_tissue_blend_channel("tissue_blend_damage", 0.25)
            runtime._set_tissue_blend_channel("tissue_blend_coag", 0.50)
            runtime._set_tissue_blend_channel("tissue_blend_blood", 0.75)
            runtime._sync_tissue_blend_vertex_colors()

            expected = np.tile(np.array([0.25, 0.50, 0.75, 0.0], dtype=np.float32), (4, 1))
            np.testing.assert_allclose(runtime.vertex_colors.numpy(), expected)
            self.assertFalse(runtime._tissue_blend_dirty)

            runtime._reset_tissue_blend_channels()
            runtime._sync_tissue_blend_vertex_colors()

            np.testing.assert_allclose(runtime.vertex_colors.numpy(), np.zeros((4, 4), dtype=np.float32))

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

    def test_render_bridge_haptic_sphere_accepts_scalar_radius(self):
        class FakeRenderer:
            def __init__(self):
                self.calls = []

            def log_points(self, name, points, radii, colors):
                self.calls.append((name, points, radii, colors))

        with wp.ScopedDevice("cpu"):
            bridge = RenderBridge.__new__(RenderBridge)
            bridge._backend = "gl"
            bridge._renderer = FakeRenderer()
            bridge.gpu = type(
                "GpuStub",
                (),
                {
                    "device": wp.get_device(),
                    "haptic_radius": wp.array([0.025], dtype=wp.float32, device=wp.get_device()),
                    "haptic_color": wp.array([[0.8, 0.2, 0.2]], dtype=wp.vec3f, device=wp.get_device()),
                },
            )()
            bridge._point_radii = {}
            bridge._point_colors = {}
            points = wp.zeros(1, dtype=wp.vec3, device=wp.get_device())

            bridge.draw_haptic_sphere(points, radius=0.0625)

            self.assertEqual(len(bridge._renderer.calls), 1)
            _name, _points, radii, colors = bridge._renderer.calls[0]
            np.testing.assert_allclose(radii.numpy(), np.array([0.0625], dtype=np.float32))
            np.testing.assert_allclose(colors.numpy(), np.array([[0.8, 0.2, 0.2]], dtype=np.float32))

    def test_render_bridge_forwards_vertex_colors_to_slang(self):
        class FakeSlangRenderer:
            def __init__(self):
                self.calls = []

            def draw_mesh(self, **kwargs):
                self.calls.append(kwargs)

        with wp.ScopedDevice("cpu"):
            bridge = RenderBridge.__new__(RenderBridge)
            bridge._backend = "slang"
            bridge._renderer = FakeSlangRenderer()
            bridge._mesh_created = set()
            points = wp.zeros(3, dtype=wp.vec3f, device=wp.get_device())
            indices = wp.array([0, 1, 2], dtype=wp.int32, device=wp.get_device())
            vertex_colors = wp.zeros(3, dtype=wp.vec4f, device=wp.get_device())

            bridge.draw_mesh(
                "tissue",
                points,
                indices,
                texture="textures/tissue/diffuse-base.png",
                vertex_colors=vertex_colors,
            )

            self.assertEqual(len(bridge._renderer.calls), 1)
            self.assertIs(bridge._renderer.calls[0]["vertex_colors"], vertex_colors)

    def test_render_bridge_accepts_vertex_colors_on_gl_path(self):
        class FakeGlRenderer:
            def __init__(self):
                self.mesh_calls = []
                self.instance_calls = []

            def log_mesh(self, name, points, indices, uvs, texture, hidden):
                self.mesh_calls.append((name, points, indices, uvs, texture, hidden))

            def log_instances(self, *args, **kwargs):
                self.instance_calls.append((args, kwargs))

        with wp.ScopedDevice("cpu"):
            bridge = RenderBridge.__new__(RenderBridge)
            bridge._backend = "gl"
            bridge._renderer = FakeGlRenderer()
            bridge._instance_colors = {}
            bridge._mesh_instance_state = {}
            bridge._mesh_created = set()
            bridge.gpu = SimpleNamespace(
                device=wp.get_device(),
                white_color=wp.array([wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3, device=wp.get_device()),
                default_material=wp.array([wp.vec4(0.5, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=wp.get_device()),
                textured_material=wp.array([wp.vec4(0.5, 0.0, 0.0, 1.0)], dtype=wp.vec4, device=wp.get_device()),
                identity_xform=wp.array([wp.transform()], dtype=wp.transformf, device=wp.get_device()),
                unit_scale=wp.array([1.0, 1.0, 1.0], dtype=wp.vec3, device=wp.get_device()),
            )
            points = wp.zeros(3, dtype=wp.vec3f, device=wp.get_device())
            indices = wp.array([0, 1, 2], dtype=wp.int32, device=wp.get_device())
            vertex_colors = wp.zeros(3, dtype=wp.vec4f, device=wp.get_device())

            bridge.draw_mesh(
                "flat_mesh",
                points,
                indices,
                color=(0.1, 0.2, 0.3),
                vertex_colors=vertex_colors,
            )

            self.assertEqual(len(bridge._renderer.mesh_calls), 1)
            self.assertEqual(len(bridge._renderer.instance_calls), 1)

    def test_render_bridge_forwards_tissue_material_params_to_slang(self):
        class FakeSlangRenderer:
            def __init__(self):
                self.params = []

            def set_tissue_material_params(self, **params):
                self.params.append(params)

        bridge = RenderBridge.__new__(RenderBridge)
        bridge._backend = "slang"
        bridge._renderer = FakeSlangRenderer()

        bridge.set_tissue_material_params(debug_mode=2, specular_scale=0.6)

        self.assertEqual(bridge._renderer.params, [{"debug_mode": 2, "specular_scale": 0.6}])

    def test_render_bridge_ignores_tissue_material_params_on_non_slang(self):
        class FakeGlRenderer:
            def set_tissue_material_params(self, **params):
                raise AssertionError("non-Slang renderers should not receive tissue material params")

        bridge = RenderBridge.__new__(RenderBridge)
        bridge._backend = "gl"
        bridge._renderer = FakeGlRenderer()

        bridge.set_tissue_material_params(debug_mode=2, specular_scale=0.6)

    def test_render_bridge_forwards_postprocess_params_to_slang(self):
        class FakeSlangRenderer:
            def __init__(self):
                self.params = []

            def set_postprocess_params(self, **params):
                self.params.append(params)

        bridge = RenderBridge.__new__(RenderBridge)
        bridge._backend = "slang"
        bridge._renderer = FakeSlangRenderer()

        bridge.set_postprocess_params(enabled=False, exposure=1.4)

        self.assertEqual(bridge._renderer.params, [{"enabled": False, "exposure": 1.4}])

    def test_render_bridge_ignores_postprocess_params_on_non_slang(self):
        class FakeGlRenderer:
            def set_postprocess_params(self, **params):
                raise AssertionError("non-Slang renderers should not receive postprocess params")

        bridge = RenderBridge.__new__(RenderBridge)
        bridge._backend = "gl"
        bridge._renderer = FakeGlRenderer()

        bridge.set_postprocess_params(enabled=False, exposure=1.4)

    def test_slang_renderer_tissue_material_params_are_clamped_and_bound(self):
        class FakeSpy:
            def __init__(self):
                self.cursor = SimpleNamespace()

            def ShaderCursor(self, shader_object):
                del shader_object
                return self.cursor

            def float3(self, *values):
                return tuple(values)

            def float4(self, *values):
                return tuple(values)

        def layer_set(prefix):
            return SimpleNamespace(
                base=f"{prefix}-base",
                damage=f"{prefix}-damage",
                coag=f"{prefix}-coag",
                blood=f"{prefix}-blood",
            )

        renderer = SlangRenderer.__new__(SlangRenderer)
        renderer._tissue_material_params = TissueMaterialParams()
        renderer._spy = FakeSpy()
        renderer._surface_texture = SimpleNamespace(width=1600, height=800)
        renderer._camera_pos = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        renderer._camera_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        renderer._camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        renderer._camera_forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        renderer._camera_inv_tan_half_fovy = 1.25
        renderer._camera_near = 0.01
        renderer._camera_far = 10.0
        material = SimpleNamespace(
            valid=True,
            diffuse=layer_set("diffuse"),
            normal=layer_set("normal"),
            spec=layer_set("spec"),
            layer_masks=layer_set("mask"),
            blood_mask_texture="blood-mask",
            heat_mask_texture="heat-mask",
            sampler="sampler",
        )

        renderer.set_tissue_material_params(
            debug_mode=999,
            normal_strength=1.2,
            specular_scale=0.7,
            roughness_bias=0.25,
            ambient=0.11,
            rim_strength=0.33,
            wetness=1.2,
            wet_spec_scale=5.0,
            wet_roughness=0.0,
            subsurface_color=(2.0, -1.0, 0.5),
            subsurface_strength=0.4,
            blood_wetness=3.0,
            specular_aa_enabled=0,
            specular_aa_strength=3.0,
            specular_aa_min_roughness=1.0,
        )
        renderer._set_shader_uniforms("shader-object", (1.0, 1.0, 1.0, 1.0), material)

        cursor = renderer._spy.cursor
        self.assertEqual(cursor.debug_mode, len(TISSUE_DEBUG_MODE_LABELS) - 1)
        self.assertEqual(cursor.subsurface_color, (1.0, 0.0, 0.5))
        self.assertEqual(cursor.diffuse_blood_tex, "diffuse-blood")
        self.assertEqual(cursor.normal_coag_tex, "normal-coag")
        self.assertEqual(cursor.spec_damage_tex, "spec-damage")
        self.assertEqual(cursor.layer_mask_blood_tex, "mask-blood")
        self.assertEqual(cursor.blood_mask_tex, "blood-mask")
        self.assertEqual(cursor.heat_mask_tex, "heat-mask")
        self.assertEqual(cursor.material_sampler, "sampler")
        self.assertEqual(cursor.normal_strength, 1.2)
        self.assertEqual(cursor.specular_scale, 0.7)
        self.assertEqual(cursor.roughness_bias, 0.25)
        self.assertEqual(cursor.ambient, 0.11)
        self.assertEqual(cursor.rim_strength, 0.33)
        self.assertEqual(cursor.blend_damage, 1.0)
        self.assertEqual(cursor.blend_coag, 1.0)
        self.assertEqual(cursor.blend_blood, 1.0)
        self.assertEqual(cursor.wetness, 1.0)
        self.assertEqual(cursor.wet_spec_scale, 4.0)
        self.assertEqual(cursor.wet_roughness, 0.02)
        self.assertEqual(cursor.subsurface_strength, 0.4)
        self.assertEqual(cursor.blood_wetness, 2.0)
        self.assertEqual(cursor.specular_aa_enabled, 0)
        self.assertEqual(cursor.specular_aa_strength, 2.0)
        self.assertEqual(cursor.specular_aa_min_roughness, 0.25)

    def test_slang_renderer_clamps_wet_tissue_material_params_to_ui_ranges(self):
        renderer = SlangRenderer.__new__(SlangRenderer)
        renderer._tissue_material_params = TissueMaterialParams()

        renderer.set_tissue_material_params(
            wetness=-1.0,
            wet_spec_scale=-1.0,
            wet_roughness=-1.0,
            blood_wetness=-1.0,
            specular_aa_strength=-1.0,
            specular_aa_min_roughness=-1.0,
        )

        params = renderer._tissue_material_params
        self.assertEqual(params.wetness, 0.0)
        self.assertEqual(params.wet_spec_scale, 0.0)
        self.assertEqual(params.wet_roughness, 0.02)
        self.assertEqual(params.blood_wetness, 0.0)
        self.assertEqual(params.specular_aa_strength, 0.0)
        self.assertEqual(params.specular_aa_min_roughness, 0.0)

        renderer.set_tissue_material_params(
            wetness=2.0,
            wet_spec_scale=8.0,
            wet_roughness=1.0,
            blood_wetness=4.0,
            specular_aa_strength=4.0,
            specular_aa_min_roughness=1.0,
        )

        self.assertEqual(params.wetness, 1.0)
        self.assertEqual(params.wet_spec_scale, 4.0)
        self.assertEqual(params.wet_roughness, 0.6)
        self.assertEqual(params.blood_wetness, 2.0)
        self.assertEqual(params.specular_aa_strength, 2.0)
        self.assertEqual(params.specular_aa_min_roughness, 0.25)

    def test_slang_renderer_postprocess_params_are_clamped_and_bound(self):
        class FakeSpy:
            def __init__(self):
                self.cursor = SimpleNamespace()

            def ShaderCursor(self, shader_object):
                del shader_object
                return self.cursor

            def float2(self, *values):
                return tuple(values)

            def float3(self, *values):
                return tuple(values)

        renderer = SlangRenderer.__new__(SlangRenderer)
        renderer._postprocess_params = PostProcessParams()
        renderer._spy = FakeSpy()
        renderer._scene_color_texture = "scene-color"
        renderer._depth_texture = "depth"
        renderer._resolved_bloom_texture = "bloom"
        renderer._bloom_down_textures = ["bloom-global"]
        renderer._resolved_ao_texture = "ao"
        renderer._lens_dirt_textures = {3: "lens-dirt-3"}
        renderer._auto_exposure_textures = ["auto-previous", "auto-current"]
        renderer._auto_exposure_index = 1
        renderer._auto_exposure_initialized = True
        renderer._frame_id = 42
        renderer._post_sampler = "sampler"

        renderer.set_postprocess_params(
            enabled=0,
            exposure=12.0,
            white_balance=(8.0, -1.0, 0.5),
            auto_exposure_enabled=0,
            auto_exposure_target_luma=3.0,
            auto_exposure_min=0.0,
            auto_exposure_max=9.0,
            auto_exposure_speed=99.0,
            auto_exposure_highlight_weight=9.0,
            ao_enabled=0,
            ao_intensity=9.0,
            ao_radius=1.0,
            ao_bias=1.0,
            ao_power=9.0,
            fxaa_enabled=0,
            fxaa_subpix=9.0,
            fxaa_edge_threshold=9.0,
            fxaa_edge_threshold_min=9.0,
            bloom_enabled=0,
            bloom_threshold=12.0,
            bloom_intensity=2.0,
            bloom_radius=20.0,
            lens_dirt_enabled=0,
            lens_dirt_texture_index=99,
            lens_dirt_intensity=12.0,
            lens_dirt_threshold=2.0,
            lens_dirt_base_opacity=2.0,
            lens_dirt_global_drive=8.0,
            lens_dirt_mask_gamma=0.0,
            lens_distortion_enabled=0,
            lens_distortion_strength=1.0,
            lens_distortion_zoom=3.0,
            chromatic_aberration_enabled=0,
            chromatic_aberration_strength=99.0,
            sensor_noise_enabled=0,
            sensor_noise_strength=1.0,
            sensor_noise_shadow_boost=9.0,
            color_grade_enabled=0,
            color_saturation=9.0,
            color_contrast=9.0,
            color_gamma=0.0,
            color_warmth=9.0,
            vignette_strength=2.0,
            vignette_radius=-1.0,
            scope_radius=3.0,
            scope_softness=0.0,
        )
        renderer._set_postprocess_uniforms("shader-object", 1600, 800)

        cursor = renderer._spy.cursor
        self.assertEqual(cursor.scene_color_tex, "scene-color")
        self.assertEqual(cursor.depth_tex, "depth")
        self.assertEqual(cursor.bloom_tex, "bloom")
        self.assertEqual(cursor.bloom_global_tex, "bloom-global")
        self.assertEqual(cursor.ao_tex, "ao")
        self.assertEqual(cursor.lens_dirt_tex, "lens-dirt-3")
        self.assertEqual(cursor.auto_exposure_tex, "auto-current")
        self.assertEqual(cursor.post_sampler, "sampler")
        self.assertEqual(cursor.output_size, (1600.0, 800.0))
        self.assertEqual(cursor.frame_index, 42)
        self.assertEqual(cursor.postprocess_enabled, 0)
        self.assertEqual(cursor.exposure, 8.0)
        self.assertEqual(cursor.white_balance, (4.0, 0.0, 0.5))
        self.assertEqual(cursor.auto_exposure_enabled, 0)
        self.assertEqual(cursor.auto_exposure_target_luma, 1.0)
        self.assertEqual(cursor.auto_exposure_min, 0.05)
        self.assertEqual(cursor.auto_exposure_max, 4.0)
        self.assertEqual(cursor.auto_exposure_speed, 12.0)
        self.assertEqual(cursor.auto_exposure_highlight_weight, 4.0)
        self.assertEqual(cursor.ao_enabled, 0)
        self.assertEqual(cursor.ao_intensity, 4.0)
        self.assertEqual(cursor.bloom_enabled, 0)
        self.assertEqual(cursor.bloom_threshold, 1.0)
        self.assertEqual(cursor.bloom_intensity, 2.0)
        self.assertEqual(cursor.bloom_radius, 20.0)
        self.assertEqual(cursor.lens_dirt_enabled, 0)
        self.assertEqual(renderer._postprocess_params.lens_dirt_texture_index, 3)
        self.assertEqual(cursor.lens_dirt_intensity, 10.0)
        self.assertEqual(cursor.lens_dirt_threshold, 1.0)
        self.assertEqual(cursor.lens_dirt_base_opacity, 1.0)
        self.assertEqual(cursor.lens_dirt_global_drive, 4.0)
        self.assertEqual(cursor.lens_dirt_mask_gamma, 0.25)
        self.assertEqual(cursor.lens_distortion_enabled, 0)
        self.assertEqual(cursor.lens_distortion_strength, 0.35)
        self.assertEqual(cursor.lens_distortion_zoom, 1.2)
        self.assertEqual(cursor.chromatic_aberration_enabled, 0)
        self.assertEqual(cursor.chromatic_aberration_strength, 4.0)
        self.assertEqual(cursor.sensor_noise_enabled, 0)
        self.assertEqual(cursor.sensor_noise_strength, 0.10)
        self.assertEqual(cursor.sensor_noise_shadow_boost, 4.0)
        self.assertEqual(cursor.color_grade_enabled, 0)
        self.assertEqual(cursor.color_saturation, 2.0)
        self.assertEqual(cursor.color_contrast, 1.5)
        self.assertEqual(cursor.color_gamma, 0.5)
        self.assertEqual(cursor.color_warmth, 1.0)
        self.assertEqual(cursor.vignette_strength, 1.0)
        self.assertEqual(cursor.vignette_radius, 0.0)
        self.assertEqual(cursor.scope_radius, 1.5)
        self.assertEqual(cursor.scope_softness, 0.001)

        renderer._manual_srgb_encode = True
        renderer._set_fxaa_uniforms("shader-object", "post-color", 1600, 800)
        self.assertEqual(cursor.source_tex, "post-color")
        self.assertEqual(cursor.fxaa_subpix, 1.0)
        self.assertEqual(cursor.fxaa_edge_threshold, 0.333)
        self.assertEqual(cursor.fxaa_edge_threshold_min, 0.0833)
        renderer._set_present_uniforms("shader-object", "fxaa-color", 1600, 800)
        self.assertEqual(cursor.source_tex, "fxaa-color")
        self.assertEqual(cursor.manual_srgb_encode, 1)
        self.assertEqual(cursor.sensor_noise_enabled, 0)

    def test_slang_renderer_rejects_unknown_postprocess_params(self):
        renderer = SlangRenderer.__new__(SlangRenderer)
        renderer._postprocess_params = PostProcessParams()

        with self.assertRaises(ValueError):
            renderer.set_postprocess_params(bloom_strength=1.0)

    def test_slang_renderer_bloom_level_sizes_are_bounded(self):
        renderer = SlangRenderer.__new__(SlangRenderer)

        self.assertEqual(
            renderer._bloom_level_sizes(1600, 1000),
            [(800, 500), (400, 250), (200, 125), (100, 63), (50, 32)],
        )
        self.assertEqual(renderer._bloom_level_sizes(1, 1), [(1, 1)])
        self.assertEqual(renderer._bloom_level_sizes(3, 2), [(2, 1), (1, 1)])

    def test_slang_renderer_resize_invalidates_bloom_textures(self):
        class FakeDevice:
            def __init__(self):
                self.waited = False

            def wait(self):
                self.waited = True

        class FakeSurface:
            def __init__(self):
                self.configured = None
                self.unconfigured = False

            def configure(self, **kwargs):
                self.configured = kwargs

            def unconfigure(self):
                self.unconfigured = True

        renderer = SlangRenderer.__new__(SlangRenderer)
        renderer._pending_resize = (800, 600)
        renderer._device = FakeDevice()
        renderer._surface = FakeSurface()
        renderer._vsync = False
        renderer._spy = SimpleNamespace(
            Viewport=SimpleNamespace(from_size=lambda width, height: ("viewport", width, height)),
            ScissorRect=SimpleNamespace(from_size=lambda width, height: ("scissor", width, height)),
        )
        renderer._scene_color_texture = "scene"
        renderer._depth_texture = "depth"
        renderer._post_color_texture = "post"
        renderer._fxaa_texture = "fxaa"
        renderer._post_texture_size = (1, 1)
        renderer._ao_texture = "ao"
        renderer._ao_blur_texture = "ao-blur"
        renderer._ao_texture_size = (1, 1)
        renderer._resolved_ao_texture = "ao-resolved"
        renderer._bloom_down_textures = ["down"]
        renderer._bloom_up_textures = ["up"]
        renderer._bloom_texture_size = (1, 1)
        renderer._resolved_bloom_texture = "resolved"
        renderer._auto_exposure_textures = ["auto-a", "auto-b"]
        renderer._auto_exposure_index = 1
        renderer._auto_exposure_initialized = True

        renderer._apply_pending_resize()

        self.assertTrue(renderer._device.waited)
        self.assertEqual(renderer._surface.configured, {"width": 800, "height": 600, "vsync": False})
        self.assertIsNone(renderer._scene_color_texture)
        self.assertIsNone(renderer._depth_texture)
        self.assertIsNone(renderer._post_color_texture)
        self.assertIsNone(renderer._fxaa_texture)
        self.assertIsNone(renderer._post_texture_size)
        self.assertIsNone(renderer._ao_texture)
        self.assertIsNone(renderer._ao_blur_texture)
        self.assertIsNone(renderer._ao_texture_size)
        self.assertIsNone(renderer._resolved_ao_texture)
        self.assertEqual(renderer._bloom_down_textures, [])
        self.assertEqual(renderer._bloom_up_textures, [])
        self.assertIsNone(renderer._bloom_texture_size)
        self.assertIsNone(renderer._resolved_bloom_texture)
        self.assertEqual(renderer._auto_exposure_textures, [])
        self.assertEqual(renderer._auto_exposure_index, 0)
        self.assertFalse(renderer._auto_exposure_initialized)

    def test_slang_renderer_camera_orbit_pan_and_dolly_keep_valid_basis(self):
        renderer = self._make_slang_camera_renderer()
        initial_pos = renderer._camera_pos.copy()

        renderer._orbit_camera(40.0, -20.0)

        self.assertFalse(np.allclose(renderer._camera_pos, initial_pos))
        self.assertAlmostEqual(float(np.linalg.norm(renderer._camera_forward)), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(renderer._camera_right)), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(renderer._camera_up)), 1.0, places=5)

        orbit_offset = renderer._camera_pos - renderer._camera_target
        orbit_distance = float(np.linalg.norm(orbit_offset))
        target_before_pan = renderer._camera_target.copy()
        renderer._pan_camera(10.0, -5.0)

        self.assertFalse(np.allclose(renderer._camera_target, target_before_pan))
        np.testing.assert_allclose(
            renderer._camera_pos - renderer._camera_target,
            orbit_offset,
            rtol=1.0e-5,
            atol=1.0e-5,
        )

        renderer._dolly_camera(1.0)
        self.assertLess(float(np.linalg.norm(renderer._camera_pos - renderer._camera_target)), orbit_distance)

    def test_slang_renderer_camera_keyboard_controls_move_and_reset(self):
        renderer = self._make_slang_camera_renderer()
        initial_pos = renderer._camera_pos.copy()
        initial_target = renderer._camera_target.copy()

        renderer._on_camera_keyboard_event(self._slang_key_event("w", press=True))
        renderer._apply_keyboard_camera_motion(0.5)

        self.assertIn("w", renderer._camera_key_state)
        self.assertLess(float(renderer._camera_pos[2]), float(initial_pos[2]))
        self.assertLess(float(renderer._camera_target[2]), float(initial_target[2]))

        renderer._on_camera_keyboard_event(self._slang_key_event("w", release=True))
        self.assertNotIn("w", renderer._camera_key_state)

        renderer._on_camera_keyboard_event(self._slang_key_event("left_shift", press=True))
        self.assertTrue(renderer._camera_fast_modifier)
        renderer._on_camera_keyboard_event(self._slang_key_event("left_shift", release=True))
        self.assertFalse(renderer._camera_fast_modifier)

        renderer._on_camera_keyboard_event(self._slang_key_event("home", press=True))
        np.testing.assert_allclose(renderer._camera_pos, renderer._camera_default_pos, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(
            renderer._camera_target,
            renderer._camera_default_target,
            rtol=1.0e-6,
            atol=1.0e-6,
        )

    def test_slang_renderer_maps_cut_modifier_keys_to_pyglet_symbols(self):
        try:
            import pyglet
        except Exception as exc:  # pragma: no cover - pyglet is available in CI
            self.skipTest(f"pyglet unavailable: {exc}")

        renderer = self._make_slang_camera_renderer()

        self.assertEqual(renderer._pyglet_symbol_from_slang_key(SimpleNamespace(name="left_control")), pyglet.window.key.LCTRL)
        self.assertEqual(renderer._pyglet_symbol_from_slang_key(SimpleNamespace(name="left_ctrl")), pyglet.window.key.LCTRL)
        self.assertEqual(renderer._pyglet_symbol_from_slang_key(SimpleNamespace(name="control")), pyglet.window.key.LCTRL)
        self.assertEqual(renderer._pyglet_symbol_from_slang_key(SimpleNamespace(name="left_alt")), pyglet.window.key.LALT)
        self.assertEqual(renderer._pyglet_symbol_from_slang_key(SimpleNamespace(name="alt")), pyglet.window.key.LALT)

    def test_viewport_input_adapter_reads_slang_modifier_state_when_ui_captures_press(self):
        try:
            import pyglet
        except Exception as exc:  # pragma: no cover - pyglet is available in CI
            self.skipTest(f"pyglet unavailable: {exc}")

        renderer = self._make_slang_camera_renderer()
        renderer._ui_enabled = True
        renderer._ui_context = SimpleNamespace(handle_keyboard_event=lambda _event: True)
        renderer._key_handler = {}
        renderer._on_key_press_callback = lambda *_args: self.fail("captured key press should not reach scene callback")
        renderer._on_key_release_callback = None
        renderer._paused = True
        renderer.close = lambda: None
        renderer._warnings = set()
        renderer._warn_once = lambda *_args, **_kwargs: None

        bridge = RenderBridge.__new__(RenderBridge)
        bridge._backend = "slang"
        bridge._renderer = renderer
        bridge._viewer_renderer = lambda: None
        adapter = ViewportInputAdapter(bridge)

        renderer._on_keyboard_event(self._slang_key_event("left_control", press=True))

        self.assertTrue(renderer.is_key_down(pyglet.window.key.LCTRL))
        self.assertTrue(bridge.is_key_down(pyglet.window.key.LCTRL))
        self.assertTrue(adapter.is_key_down(pyglet.window.key.LCTRL))

    def test_slang_renderer_camera_mouse_controls_respect_capture(self):
        renderer = self._make_slang_camera_renderer()

        renderer._on_camera_mouse_event(self._slang_mouse_event("down", pos=(100.0, 100.0), button="left"))
        renderer._on_camera_mouse_event(self._slang_mouse_event("move", pos=(130.0, 90.0)))
        self.assertFalse(np.allclose(renderer._camera_pos, renderer._camera_default_pos))
        renderer._on_camera_mouse_event(self._slang_mouse_event("up", pos=(130.0, 90.0), button="left"))
        self.assertIsNone(renderer._camera_mouse_action)

        target_before_pan = renderer._camera_target.copy()
        renderer._on_camera_mouse_event(self._slang_mouse_event("down", pos=(40.0, 40.0), button="right"))
        renderer._on_camera_mouse_event(self._slang_mouse_event("move", pos=(55.0, 30.0)))
        self.assertFalse(np.allclose(renderer._camera_target, target_before_pan))

        distance_before_scroll = float(np.linalg.norm(renderer._camera_pos - renderer._camera_target))
        renderer._on_camera_mouse_event(self._slang_mouse_event("scroll", scroll=(0.0, 1.0)))
        self.assertLess(
            float(np.linalg.norm(renderer._camera_pos - renderer._camera_target)),
            distance_before_scroll,
        )

        captured_renderer = self._make_slang_camera_renderer()
        captured_renderer._on_camera_mouse_event(
            self._slang_mouse_event("down", pos=(100.0, 100.0), button="left"),
            captured=True,
        )
        self.assertIsNone(captured_renderer._camera_mouse_action)

    def test_slang_renderer_left_drag_orbits_with_scene_mouse_callbacks(self):
        renderer = self._make_slang_camera_renderer()
        calls = []
        renderer._ui_enabled = False
        renderer._ui_context = None
        renderer._ui_mouse_active = False
        renderer._ui_capturing = False
        renderer._mouse_buttons = 0
        renderer._mouse_pos = None
        renderer._on_mouse_motion_callback = None
        renderer._on_mouse_press_callback = lambda *args: calls.append(("press", args))
        renderer._on_mouse_drag_callback = lambda *args: calls.append(("drag", args))
        renderer._on_mouse_release_callback = None

        initial_pos = renderer._camera_pos.copy()

        renderer._on_mouse_event(self._slang_mouse_event("down", pos=(100.0, 100.0), button="left"))
        renderer._on_mouse_event(self._slang_mouse_event("move", pos=(130.0, 90.0)))

        self.assertFalse(np.allclose(renderer._camera_pos, initial_pos))
        self.assertTrue(any(name == "press" for name, _args in calls))
        self.assertTrue(any(name == "drag" for name, _args in calls))

    def test_runtime_tissue_material_param_sync_updates_renderer(self):
        class FakeRenderer:
            def __init__(self):
                self.params = []

            def set_tissue_material_params(self, **params):
                self.params.append(params)

        runtime = Runtime.__new__(Runtime)
        runtime.renderer = FakeRenderer()
        runtime.tissue_debug_mode = 0
        runtime.tissue_normal_strength = 0.65
        runtime.tissue_specular_scale = 0.35
        runtime.tissue_roughness_bias = 0.45
        runtime.tissue_ambient = 0.22
        runtime.tissue_rim_strength = 0.12
        runtime.tissue_wetness = 0.0
        runtime.tissue_wet_spec_scale = 1.0
        runtime.tissue_wet_roughness = 0.18
        runtime.tissue_subsurface_color = (0.8, 0.22, 0.16)
        runtime.tissue_subsurface_strength = 0.0
        runtime.tissue_blood_wetness = 1.0
        runtime.tissue_specular_aa_enabled = True
        runtime.tissue_specular_aa_strength = 0.35
        runtime.tissue_specular_aa_min_roughness = 0.04

        runtime._sync_tissue_material_params()

        self.assertEqual(runtime.renderer.params[-1]["debug_mode"], 0)
        self.assertEqual(runtime.renderer.params[-1]["normal_strength"], 0.65)
        self.assertEqual(runtime.renderer.params[-1]["subsurface_color"], (0.8, 0.22, 0.16))
        self.assertEqual(runtime.renderer.params[-1]["specular_aa_enabled"], True)
        self.assertEqual(runtime.renderer.params[-1]["specular_aa_strength"], 0.35)

        runtime._set_tissue_debug_mode(999)
        self.assertEqual(runtime.tissue_debug_mode, len(TISSUE_DEBUG_MODE_LABELS) - 1)
        self.assertEqual(runtime.renderer.params[-1]["debug_mode"], len(TISSUE_DEBUG_MODE_LABELS) - 1)

        runtime._set_tissue_material_float("tissue_specular_scale", 4.0, 0.0, 2.0)
        self.assertEqual(runtime.tissue_specular_scale, 2.0)
        self.assertEqual(runtime.renderer.params[-1]["specular_scale"], 2.0)

    def test_runtime_postprocess_param_sync_updates_renderer(self):
        class FakeRenderer:
            def __init__(self):
                self.params = []

            def set_postprocess_params(self, **params):
                self.params.append(params)

        runtime = Runtime.__new__(Runtime)
        runtime.renderer = FakeRenderer()
        runtime.postprocess_enabled = True
        runtime.postprocess_exposure = 1.0
        runtime.postprocess_white_balance = (1.0, 1.0, 1.0)
        runtime.postprocess_auto_exposure_enabled = False
        runtime.postprocess_auto_exposure_target_luma = 0.35
        runtime.postprocess_auto_exposure_min = 0.35
        runtime.postprocess_auto_exposure_max = 1.8
        runtime.postprocess_auto_exposure_speed = 4.0
        runtime.postprocess_auto_exposure_highlight_weight = 0.8
        runtime.postprocess_ao_enabled = True
        runtime.postprocess_ao_intensity = 1.6
        runtime.postprocess_ao_radius = 0.22
        runtime.postprocess_ao_bias = 0.004
        runtime.postprocess_ao_power = 1.6
        runtime.postprocess_fxaa_enabled = True
        runtime.postprocess_fxaa_subpix = 0.75
        runtime.postprocess_fxaa_edge_threshold = 0.125
        runtime.postprocess_fxaa_edge_threshold_min = 0.0312
        runtime.postprocess_bloom_enabled = True
        runtime.postprocess_bloom_threshold = 1.0
        runtime.postprocess_bloom_intensity = 0.12
        runtime.postprocess_bloom_radius = 3.0
        runtime.postprocess_lens_dirt_enabled = True
        runtime.postprocess_lens_dirt_texture_index = 2
        runtime.postprocess_lens_dirt_intensity = 0.25
        runtime.postprocess_lens_dirt_threshold = 0.35
        runtime.postprocess_lens_dirt_base_opacity = 0.05
        runtime.postprocess_lens_dirt_global_drive = 1.0
        runtime.postprocess_lens_dirt_mask_gamma = 0.6
        runtime.postprocess_lens_distortion_enabled = True
        runtime.postprocess_lens_distortion_strength = 0.08
        runtime.postprocess_lens_distortion_zoom = 1.04
        runtime.postprocess_chromatic_aberration_enabled = True
        runtime.postprocess_chromatic_aberration_strength = 0.6
        runtime.postprocess_sensor_noise_enabled = True
        runtime.postprocess_sensor_noise_strength = 0.008
        runtime.postprocess_sensor_noise_shadow_boost = 1.5
        runtime.postprocess_color_grade_enabled = True
        runtime.postprocess_color_saturation = 1.0
        runtime.postprocess_color_contrast = 1.0
        runtime.postprocess_color_gamma = 1.0
        runtime.postprocess_color_warmth = 0.0
        runtime.postprocess_vignette_strength = 0.45
        runtime.postprocess_vignette_radius = 0.78
        runtime.postprocess_scope_radius = 0.965
        runtime.postprocess_scope_softness = 0.035

        runtime._sync_postprocess_params()

        self.assertEqual(runtime.renderer.params[-1]["enabled"], True)
        self.assertEqual(runtime.renderer.params[-1]["exposure"], 1.0)
        self.assertEqual(runtime.renderer.params[-1]["white_balance"], (1.0, 1.0, 1.0))
        self.assertEqual(runtime.renderer.params[-1]["auto_exposure_enabled"], False)
        self.assertEqual(runtime.renderer.params[-1]["auto_exposure_target_luma"], 0.35)
        self.assertEqual(runtime.renderer.params[-1]["auto_exposure_highlight_weight"], 0.8)
        self.assertEqual(runtime.renderer.params[-1]["ao_enabled"], True)
        self.assertEqual(runtime.renderer.params[-1]["ao_intensity"], 1.6)
        self.assertEqual(runtime.renderer.params[-1]["fxaa_enabled"], True)
        self.assertEqual(runtime.renderer.params[-1]["fxaa_subpix"], 0.75)
        self.assertEqual(runtime.renderer.params[-1]["bloom_enabled"], True)
        self.assertEqual(runtime.renderer.params[-1]["bloom_threshold"], 1.0)
        self.assertEqual(runtime.renderer.params[-1]["lens_dirt_enabled"], True)
        self.assertEqual(runtime.renderer.params[-1]["lens_dirt_texture_index"], 2)
        self.assertEqual(runtime.renderer.params[-1]["lens_dirt_intensity"], 0.25)
        self.assertEqual(runtime.renderer.params[-1]["lens_dirt_base_opacity"], 0.05)
        self.assertEqual(runtime.renderer.params[-1]["lens_dirt_global_drive"], 1.0)
        self.assertEqual(runtime.renderer.params[-1]["lens_dirt_mask_gamma"], 0.6)
        self.assertEqual(runtime.renderer.params[-1]["lens_distortion_enabled"], True)
        self.assertEqual(runtime.renderer.params[-1]["lens_distortion_strength"], 0.08)
        self.assertEqual(runtime.renderer.params[-1]["chromatic_aberration_enabled"], True)
        self.assertEqual(runtime.renderer.params[-1]["sensor_noise_enabled"], True)
        self.assertEqual(runtime.renderer.params[-1]["color_grade_enabled"], True)

        runtime._set_postprocess_lens_dirt_texture(99)
        self.assertEqual(runtime.postprocess_lens_dirt_texture_index, len(LENS_DIRT_TEXTURE_LABELS) - 1)
        self.assertEqual(
            runtime.renderer.params[-1]["lens_dirt_texture_index"],
            len(LENS_DIRT_TEXTURE_LABELS) - 1,
        )

        runtime._set_postprocess_float("postprocess_exposure", 12.0, 0.0, 8.0)
        self.assertEqual(runtime.postprocess_exposure, 8.0)
        self.assertEqual(runtime.renderer.params[-1]["exposure"], 8.0)

        runtime._set_postprocess_white_balance_channel(1, 5.0)
        self.assertEqual(runtime.postprocess_white_balance, (1.0, 4.0, 1.0))
        self.assertEqual(runtime.renderer.params[-1]["white_balance"], (1.0, 4.0, 1.0))

    def test_runtime_postprocess_ui_exposes_optics_controls_and_syncs_renderer(self):
        class FakeRenderer:
            def __init__(self):
                self.params = []

            def set_postprocess_params(self, **params):
                self.params.append(params)

        runtime = Runtime.__new__(Runtime)
        runtime.renderer = FakeRenderer()
        runtime.postprocess_enabled = True
        runtime.postprocess_exposure = 1.0
        runtime.postprocess_white_balance = (1.0, 1.0, 1.0)
        runtime.postprocess_auto_exposure_enabled = False
        runtime.postprocess_auto_exposure_target_luma = 0.35
        runtime.postprocess_auto_exposure_min = 0.35
        runtime.postprocess_auto_exposure_max = 1.8
        runtime.postprocess_auto_exposure_speed = 4.0
        runtime.postprocess_auto_exposure_highlight_weight = 0.8
        runtime.postprocess_ao_enabled = True
        runtime.postprocess_ao_intensity = 1.6
        runtime.postprocess_ao_radius = 0.22
        runtime.postprocess_ao_bias = 0.004
        runtime.postprocess_ao_power = 1.6
        runtime.postprocess_fxaa_enabled = True
        runtime.postprocess_fxaa_subpix = 0.75
        runtime.postprocess_fxaa_edge_threshold = 0.125
        runtime.postprocess_fxaa_edge_threshold_min = 0.0312
        runtime.postprocess_bloom_enabled = True
        runtime.postprocess_bloom_threshold = 1.0
        runtime.postprocess_bloom_intensity = 0.12
        runtime.postprocess_bloom_radius = 3.0
        runtime.postprocess_lens_dirt_enabled = True
        runtime.postprocess_lens_dirt_texture_index = 0
        runtime.postprocess_lens_dirt_intensity = 0.25
        runtime.postprocess_lens_dirt_threshold = 0.35
        runtime.postprocess_lens_dirt_base_opacity = 0.05
        runtime.postprocess_lens_dirt_global_drive = 1.0
        runtime.postprocess_lens_dirt_mask_gamma = 0.6
        runtime.postprocess_lens_distortion_enabled = True
        runtime.postprocess_lens_distortion_strength = 0.08
        runtime.postprocess_lens_distortion_zoom = 1.04
        runtime.postprocess_chromatic_aberration_enabled = True
        runtime.postprocess_chromatic_aberration_strength = 0.6
        runtime.postprocess_sensor_noise_enabled = True
        runtime.postprocess_sensor_noise_strength = 0.008
        runtime.postprocess_sensor_noise_shadow_boost = 1.5
        runtime.postprocess_color_grade_enabled = True
        runtime.postprocess_color_saturation = 1.0
        runtime.postprocess_color_contrast = 1.0
        runtime.postprocess_color_gamma = 1.0
        runtime.postprocess_color_warmth = 0.0
        runtime.postprocess_vignette_strength = 0.45
        runtime.postprocess_vignette_radius = 0.78
        runtime.postprocess_scope_radius = 0.965
        runtime.postprocess_scope_softness = 0.035

        ui = _FakePostProcessUi(
            checkbox_results={
                "Postprocess": False,
                "Auto Exposure": True,
                "Ambient Occlusion": False,
                "FXAA": False,
                "Bloom": False,
                "Lens Dirt": False,
                "Lens Distortion": False,
                "Chromatic Aberration": False,
                "Color Grade": False,
                "Sensor Noise": False,
            },
            slider_results={
                "Exposure": 9.0,
                "Auto Exposure Target": 3.0,
                "Auto Exposure Min": 0.0,
                "Auto Exposure Max": 9.0,
                "Auto Exposure Speed": 99.0,
                "Auto Exposure Highlight Weight": 9.0,
                "AO Intensity": 9.0,
                "AO Radius": 1.0,
                "AO Bias": 1.0,
                "AO Power": 9.0,
                "FXAA Subpix": 9.0,
                "FXAA Edge Threshold": 9.0,
                "FXAA Edge Threshold Min": 9.0,
                "Bloom Threshold": 9.0,
                "Bloom Intensity": 2.0,
                "Bloom Radius": 20.0,
                "Lens Dirt Intensity": 12.0,
                "Lens Dirt Threshold": 2.0,
                "Lens Dirt Base Opacity": 2.0,
                "Lens Dirt Global Drive": 9.0,
                "Lens Dirt Mask Gamma": 0.0,
                "Lens Distortion Strength": 1.0,
                "Lens Distortion Zoom": 3.0,
                "Chromatic Aberration Strength": 9.0,
                "Color Saturation": 9.0,
                "Color Contrast": 9.0,
                "Color Gamma": 0.0,
                "Color Warmth": 9.0,
                "Sensor Noise Strength": 1.0,
                "Sensor Noise Shadow Boost": 9.0,
                "White Balance R": -1.0,
                "White Balance G": 3.0,
                "White Balance B": 5.0,
                "Vignette Strength": 2.0,
                "Vignette Radius": -1.0,
                "Scope Radius": 2.0,
                "Scope Softness": 0.0,
            },
            combo_results={"Lens Dirt Texture": 3},
        )

        runtime._render_postprocess_ui(ui)

        calls_by_label = {call[0]: call for call in ui.slider_calls}
        self.assertIn("Postprocessing", ui.texts)
        self.assertEqual(
            ui.checkbox_calls,
            [
                ("Postprocess", True),
                ("Auto Exposure", False),
                ("Ambient Occlusion", True),
                ("FXAA", True),
                ("Bloom", True),
                ("Lens Dirt", True),
                ("Lens Distortion", True),
                ("Chromatic Aberration", True),
                ("Color Grade", True),
                ("Sensor Noise", True),
            ],
        )
        self.assertEqual(ui.combo_calls, [("Lens Dirt Texture", 0, LENS_DIRT_TEXTURE_LABELS)])
        self.assertEqual(calls_by_label["Exposure"][2:], (0.0, 8.0, "%.2f"))
        self.assertEqual(calls_by_label["Auto Exposure Target"][2:], (0.05, 1.0, "%.2f"))
        self.assertEqual(calls_by_label["Auto Exposure Min"][2:], (0.05, 2.0, "%.2f"))
        self.assertEqual(calls_by_label["Auto Exposure Max"][2:], (0.1, 4.0, "%.2f"))
        self.assertEqual(calls_by_label["Auto Exposure Speed"][2:], (0.1, 12.0, "%.1f"))
        self.assertEqual(calls_by_label["Auto Exposure Highlight Weight"][2:], (0.0, 4.0, "%.2f"))
        self.assertEqual(calls_by_label["AO Intensity"][2:], (0.0, 4.0, "%.2f"))
        self.assertEqual(calls_by_label["AO Radius"][2:], (0.0, 0.30, "%.3f"))
        self.assertEqual(calls_by_label["AO Bias"][2:], (0.0, 0.03, "%.4f"))
        self.assertEqual(calls_by_label["AO Power"][2:], (0.25, 4.0, "%.2f"))
        self.assertEqual(calls_by_label["FXAA Subpix"][2:], (0.0, 1.0, "%.2f"))
        self.assertEqual(calls_by_label["FXAA Edge Threshold"][2:], (0.0312, 0.333, "%.4f"))
        self.assertEqual(calls_by_label["FXAA Edge Threshold Min"][2:], (0.0, 0.0833, "%.4f"))
        self.assertEqual(calls_by_label["Bloom Threshold"][2:], (0.0, 1.0, "%.2f"))
        self.assertEqual(calls_by_label["Bloom Intensity"][2:], (0.0, 10.0, "%.2f"))
        self.assertEqual(calls_by_label["Bloom Radius"][2:], (0.0, 64.0, "%.1f px"))
        self.assertEqual(calls_by_label["Lens Dirt Intensity"][2:], (0.0, 10.0, "%.2f"))
        self.assertEqual(calls_by_label["Lens Dirt Threshold"][2:], (0.0, 1.0, "%.2f"))
        self.assertEqual(calls_by_label["Lens Dirt Base Opacity"][2:], (0.0, 1.0, "%.2f"))
        self.assertEqual(calls_by_label["Lens Dirt Global Drive"][2:], (0.0, 4.0, "%.2f"))
        self.assertEqual(calls_by_label["Lens Dirt Mask Gamma"][2:], (0.25, 2.0, "%.2f"))
        self.assertEqual(calls_by_label["Lens Distortion Strength"][2:], (-0.35, 0.35, "%.3f"))
        self.assertEqual(calls_by_label["Lens Distortion Zoom"][2:], (0.8, 1.2, "%.2f"))
        self.assertEqual(calls_by_label["Chromatic Aberration Strength"][2:], (0.0, 4.0, "%.2f px"))
        self.assertEqual(calls_by_label["Color Saturation"][2:], (0.0, 2.0, "%.2f"))
        self.assertEqual(calls_by_label["Color Contrast"][2:], (0.5, 1.5, "%.2f"))
        self.assertEqual(calls_by_label["Color Gamma"][2:], (0.5, 2.0, "%.2f"))
        self.assertEqual(calls_by_label["Color Warmth"][2:], (-1.0, 1.0, "%.2f"))
        self.assertEqual(calls_by_label["Sensor Noise Strength"][2:], (0.0, 0.10, "%.3f"))
        self.assertEqual(calls_by_label["Sensor Noise Shadow Boost"][2:], (0.0, 4.0, "%.2f"))
        self.assertEqual(calls_by_label["White Balance R"][2:], (0.0, 4.0, "%.2f"))
        self.assertEqual(calls_by_label["Vignette Strength"][2:], (0.0, 1.0, "%.2f"))
        self.assertEqual(calls_by_label["Vignette Radius"][2:], (0.0, 1.5, "%.2f"))
        self.assertEqual(calls_by_label["Scope Radius"][2:], (0.0, 1.5, "%.3f"))
        self.assertEqual(calls_by_label["Scope Softness"][2:], (0.001, 0.5, "%.3f"))
        self.assertFalse(runtime.postprocess_enabled)
        self.assertEqual(runtime.postprocess_exposure, 8.0)
        self.assertEqual(runtime.postprocess_white_balance, (0.0, 3.0, 4.0))
        self.assertTrue(runtime.postprocess_auto_exposure_enabled)
        self.assertEqual(runtime.postprocess_auto_exposure_target_luma, 1.0)
        self.assertEqual(runtime.postprocess_auto_exposure_min, 0.05)
        self.assertEqual(runtime.postprocess_auto_exposure_max, 4.0)
        self.assertEqual(runtime.postprocess_auto_exposure_speed, 12.0)
        self.assertEqual(runtime.postprocess_auto_exposure_highlight_weight, 4.0)
        self.assertFalse(runtime.postprocess_ao_enabled)
        self.assertEqual(runtime.postprocess_ao_intensity, 4.0)
        self.assertEqual(runtime.postprocess_ao_radius, 0.30)
        self.assertEqual(runtime.postprocess_ao_bias, 0.03)
        self.assertEqual(runtime.postprocess_ao_power, 4.0)
        self.assertFalse(runtime.postprocess_fxaa_enabled)
        self.assertEqual(runtime.postprocess_fxaa_subpix, 1.0)
        self.assertEqual(runtime.postprocess_fxaa_edge_threshold, 0.333)
        self.assertEqual(runtime.postprocess_fxaa_edge_threshold_min, 0.0833)
        self.assertFalse(runtime.postprocess_bloom_enabled)
        self.assertEqual(runtime.postprocess_bloom_threshold, 1.0)
        self.assertEqual(runtime.postprocess_bloom_intensity, 2.0)
        self.assertEqual(runtime.postprocess_bloom_radius, 20.0)
        self.assertFalse(runtime.postprocess_lens_dirt_enabled)
        self.assertEqual(runtime.postprocess_lens_dirt_texture_index, 3)
        self.assertEqual(runtime.postprocess_lens_dirt_intensity, 10.0)
        self.assertEqual(runtime.postprocess_lens_dirt_threshold, 1.0)
        self.assertEqual(runtime.postprocess_lens_dirt_base_opacity, 1.0)
        self.assertEqual(runtime.postprocess_lens_dirt_global_drive, 4.0)
        self.assertEqual(runtime.postprocess_lens_dirt_mask_gamma, 0.25)
        self.assertFalse(runtime.postprocess_lens_distortion_enabled)
        self.assertEqual(runtime.postprocess_lens_distortion_strength, 0.35)
        self.assertEqual(runtime.postprocess_lens_distortion_zoom, 1.2)
        self.assertFalse(runtime.postprocess_chromatic_aberration_enabled)
        self.assertEqual(runtime.postprocess_chromatic_aberration_strength, 4.0)
        self.assertFalse(runtime.postprocess_color_grade_enabled)
        self.assertEqual(runtime.postprocess_color_saturation, 2.0)
        self.assertEqual(runtime.postprocess_color_contrast, 1.5)
        self.assertEqual(runtime.postprocess_color_gamma, 0.5)
        self.assertEqual(runtime.postprocess_color_warmth, 1.0)
        self.assertFalse(runtime.postprocess_sensor_noise_enabled)
        self.assertEqual(runtime.postprocess_sensor_noise_strength, 0.10)
        self.assertEqual(runtime.postprocess_sensor_noise_shadow_boost, 4.0)
        self.assertEqual(runtime.postprocess_vignette_strength, 1.0)
        self.assertEqual(runtime.postprocess_vignette_radius, 0.0)
        self.assertEqual(runtime.postprocess_scope_radius, 1.5)
        self.assertEqual(runtime.postprocess_scope_softness, 0.001)
        self.assertEqual(runtime.renderer.params[-1]["scope_softness"], 0.001)
        self.assertEqual(runtime.renderer.params[-1]["white_balance"], (0.0, 3.0, 4.0))
        self.assertEqual(runtime.renderer.params[-1]["auto_exposure_enabled"], True)
        self.assertEqual(runtime.renderer.params[-1]["auto_exposure_target_luma"], 1.0)
        self.assertEqual(runtime.renderer.params[-1]["auto_exposure_highlight_weight"], 4.0)
        self.assertEqual(runtime.renderer.params[-1]["ao_enabled"], False)
        self.assertEqual(runtime.renderer.params[-1]["ao_radius"], 0.30)
        self.assertEqual(runtime.renderer.params[-1]["fxaa_enabled"], False)
        self.assertEqual(runtime.renderer.params[-1]["fxaa_edge_threshold"], 0.333)
        self.assertEqual(runtime.renderer.params[-1]["bloom_radius"], 20.0)
        self.assertEqual(runtime.renderer.params[-1]["lens_dirt_texture_index"], 3)
        self.assertEqual(runtime.renderer.params[-1]["lens_dirt_threshold"], 1.0)
        self.assertEqual(runtime.renderer.params[-1]["lens_dirt_base_opacity"], 1.0)
        self.assertEqual(runtime.renderer.params[-1]["lens_dirt_global_drive"], 4.0)
        self.assertEqual(runtime.renderer.params[-1]["lens_dirt_mask_gamma"], 0.25)
        self.assertEqual(runtime.renderer.params[-1]["lens_distortion_enabled"], False)
        self.assertEqual(runtime.renderer.params[-1]["lens_distortion_strength"], 0.35)
        self.assertEqual(runtime.renderer.params[-1]["chromatic_aberration_strength"], 4.0)
        self.assertEqual(runtime.renderer.params[-1]["sensor_noise_strength"], 0.10)
        self.assertEqual(runtime.renderer.params[-1]["color_warmth"], 1.0)

    def test_runtime_tissue_material_ui_exposes_wet_sliders_and_syncs_renderer(self):
        class FakeRenderer:
            def __init__(self):
                self.params = []

            def set_tissue_material_params(self, **params):
                self.params.append(params)

        runtime = Runtime.__new__(Runtime)
        runtime.renderer = FakeRenderer()
        runtime.tissue_debug_mode = 0
        runtime.tissue_normal_strength = 0.65
        runtime.tissue_specular_scale = 0.35
        runtime.tissue_roughness_bias = 0.45
        runtime.tissue_ambient = 0.22
        runtime.tissue_rim_strength = 0.12
        runtime.tissue_wetness = 0.0
        runtime.tissue_wet_spec_scale = 1.0
        runtime.tissue_wet_roughness = 0.18
        runtime.tissue_subsurface_color = (0.8, 0.22, 0.16)
        runtime.tissue_subsurface_strength = 0.0
        runtime.tissue_blood_wetness = 1.0
        runtime.tissue_specular_aa_enabled = True
        runtime.tissue_specular_aa_strength = 0.35
        runtime.tissue_specular_aa_min_roughness = 0.04

        ui = _FakeTissueMaterialUi(
            slider_results={
                "Wetness": 0.4,
                "Wet Spec Scale": 4.5,
                "Wet Roughness": 0.01,
                "Blood Wetness": 3.0,
                "Specular AA Strength": 3.0,
                "Specular AA Min Roughness": 1.0,
            },
            checkbox_results={"Specular AA": False},
        )

        runtime._render_tissue_material_ui(ui)

        calls_by_label = {call[0]: call for call in ui.slider_calls}
        self.assertIn("Tissue Material", ui.texts)
        self.assertEqual(calls_by_label["Wetness"][2:], (0.0, 1.0, "%.2f"))
        self.assertEqual(calls_by_label["Wet Spec Scale"][2:], (0.0, 4.0, "%.2f"))
        self.assertEqual(calls_by_label["Wet Roughness"][2:], (0.02, 0.6, "%.2f"))
        self.assertEqual(calls_by_label["Blood Wetness"][2:], (0.0, 2.0, "%.2f"))
        self.assertEqual(ui.checkbox_calls, [("Specular AA", True)])
        self.assertEqual(calls_by_label["Specular AA Strength"][2:], (0.0, 2.0, "%.2f"))
        self.assertEqual(calls_by_label["Specular AA Min Roughness"][2:], (0.0, 0.25, "%.2f"))
        self.assertEqual(runtime.tissue_wetness, 0.4)
        self.assertEqual(runtime.tissue_wet_spec_scale, 4.0)
        self.assertEqual(runtime.tissue_wet_roughness, 0.02)
        self.assertEqual(runtime.tissue_blood_wetness, 2.0)
        self.assertFalse(runtime.tissue_specular_aa_enabled)
        self.assertEqual(runtime.tissue_specular_aa_strength, 2.0)
        self.assertEqual(runtime.tissue_specular_aa_min_roughness, 0.25)
        self.assertEqual(runtime.renderer.params[-1]["wetness"], 0.4)
        self.assertEqual(runtime.renderer.params[-1]["wet_spec_scale"], 4.0)
        self.assertEqual(runtime.renderer.params[-1]["wet_roughness"], 0.02)
        self.assertEqual(runtime.renderer.params[-1]["blood_wetness"], 2.0)
        self.assertEqual(runtime.renderer.params[-1]["specular_aa_enabled"], False)
        self.assertEqual(runtime.renderer.params[-1]["specular_aa_strength"], 2.0)

    def test_runtime_tissue_debug_ui_uses_named_combo_and_reset_button(self):
        class FakeRenderer:
            def __init__(self):
                self.params = []

            def set_tissue_material_params(self, **params):
                self.params.append(params)

        runtime = Runtime.__new__(Runtime)
        runtime.renderer = FakeRenderer()
        runtime.tissue_debug_mode = 0
        runtime.tissue_normal_strength = 0.65
        runtime.tissue_specular_scale = 0.35
        runtime.tissue_roughness_bias = 0.45
        runtime.tissue_ambient = 0.22
        runtime.tissue_rim_strength = 0.12
        runtime.tissue_wetness = 0.0
        runtime.tissue_wet_spec_scale = 1.0
        runtime.tissue_wet_roughness = 0.18
        runtime.tissue_subsurface_color = (0.8, 0.22, 0.16)
        runtime.tissue_subsurface_strength = 0.0
        runtime.tissue_blood_wetness = 1.0

        ui = _FakeTissueDebugComboUi(combo_result=(True, 3), button_result=False)
        runtime._render_tissue_debug_ui(ui)

        self.assertEqual(runtime.tissue_debug_mode, 3)
        self.assertEqual(runtime.renderer.params[-1]["debug_mode"], 3)
        self.assertEqual(ui.combo_calls, [("Tissue Debug View", 0, TISSUE_DEBUG_MODE_LABELS)])
        self.assertIn("Active: Blended Diffuse", ui.texts)
        self.assertEqual(ui.buttons, ["Show Final Tissue"])

        ui = _FakeTissueDebugComboUi(combo_result=(False, 3), button_result=True)
        runtime._render_tissue_debug_ui(ui)

        self.assertEqual(runtime.tissue_debug_mode, 0)
        self.assertEqual(runtime.renderer.params[-1]["debug_mode"], 0)

    def test_runtime_tissue_debug_ui_falls_back_to_slider(self):
        class FakeRenderer:
            def __init__(self):
                self.params = []

            def set_tissue_material_params(self, **params):
                self.params.append(params)

        runtime = Runtime.__new__(Runtime)
        runtime.renderer = FakeRenderer()
        runtime.tissue_debug_mode = 0
        runtime.tissue_normal_strength = 0.65
        runtime.tissue_specular_scale = 0.35
        runtime.tissue_roughness_bias = 0.45
        runtime.tissue_ambient = 0.22
        runtime.tissue_rim_strength = 0.12
        runtime.tissue_wetness = 0.0
        runtime.tissue_wet_spec_scale = 1.0
        runtime.tissue_wet_roughness = 0.18
        runtime.tissue_subsurface_color = (0.8, 0.22, 0.16)
        runtime.tissue_subsurface_strength = 0.0
        runtime.tissue_blood_wetness = 1.0

        ui = _FakeTissueDebugSliderUi(slider_result=(True, 5), button_result=False)
        runtime._render_tissue_debug_ui(ui)

        self.assertEqual(runtime.tissue_debug_mode, 5)
        self.assertEqual(runtime.renderer.params[-1]["debug_mode"], 5)
        self.assertEqual(ui.slider_calls, [("Tissue Debug View", 0, 0, len(TISSUE_DEBUG_MODE_LABELS) - 1)])
        self.assertIn("Active: Spec/Roughness", ui.texts)

    def test_slang_tissue_material_paths_derive_all_layers(self):
        texture_dir = _prepare_test_subdir("slang_material_paths")
        base = texture_dir / "diffuse-base.png"
        renderer = SlangRenderer.__new__(SlangRenderer)

        paths = renderer._material_paths(str(base))
        resolved = base.resolve(strict=False)

        self.assertEqual(
            paths.diffuse.as_tuple(),
            (
                resolved,
                resolved.with_name("diffuse-damage.png"),
                resolved.with_name("diffuse-coag.png"),
                resolved.with_name("diffuse-blood.png"),
            ),
        )
        self.assertEqual(
            paths.normal.as_tuple(),
            (
                resolved.with_name("normal-base.png"),
                resolved.with_name("normal-damage.png"),
                resolved.with_name("normal-coag.png"),
                resolved.with_name("normal-blood.png"),
            ),
        )
        self.assertEqual(
            paths.spec.as_tuple(),
            (
                resolved.with_name("spec-base.png"),
                resolved.with_name("spec-damage.png"),
                resolved.with_name("spec-coag.png"),
                resolved.with_name("spec-blood.png"),
            ),
        )
        self.assertEqual(len(renderer._material_cache_key(paths)), 12)

    def test_slang_tissue_material_defaults_missing_optional_layers(self):
        texture_dir = _prepare_test_subdir("slang_material_defaults")
        base = texture_dir / "diffuse-base.png"
        base.write_bytes(b"placeholder")
        renderer = SlangRenderer.__new__(SlangRenderer)
        renderer._materials = {}
        renderer._default_textures = {}
        renderer._warnings = set()
        renderer._material_sampler = "sampler"
        renderer._linear_wrap_sampler = lambda: "sampler"
        renderer._load_texture_data = lambda path: np.zeros((1, 1, 4), dtype=np.uint8)
        renderer._create_texture_from_data = lambda label, data, srgb: f"{label}:{srgb}"

        resource = renderer._material_resource("organ", str(base))

        self.assertTrue(resource.valid)
        self.assertEqual(resource.diffuse.base, "organ-diffuse-base:True")
        self.assertEqual(resource.diffuse.damage, resource.diffuse.base)
        self.assertEqual(resource.diffuse.coag, resource.diffuse.base)
        self.assertEqual(resource.diffuse.blood, resource.diffuse.base)
        self.assertEqual(resource.normal.base, "omnisurg-default-normal:False")
        self.assertEqual(resource.normal.damage, resource.normal.base)
        self.assertEqual(resource.normal.coag, resource.normal.base)
        self.assertEqual(resource.normal.blood, resource.normal.base)
        self.assertEqual(resource.spec.base, "omnisurg-default-spec:False")
        self.assertEqual(resource.spec.damage, resource.spec.base)
        self.assertEqual(resource.spec.coag, resource.spec.base)
        self.assertEqual(resource.spec.blood, resource.spec.base)
        self.assertEqual(resource.layer_masks.damage, "omnisurg-default-mask-white:False")
        self.assertEqual(resource.blood_mask_texture, "omnisurg-default-blood-mask:False")
        self.assertEqual(resource.heat_mask_texture, "omnisurg-default-heat-mask:False")

        missing = texture_dir / "missing-diffuse-base.png"
        missing_resource = renderer._material_resource("missing", str(missing))
        self.assertFalse(missing_resource.valid)

    def test_slang_tissue_shader_declares_color_and_layer_bindings(self):
        shader_source = (REPO_ROOT / "omnisurg" / "rendering" / "slang_shaders" / "omnisurg_tissue.slang").read_text()
        renderer_source = (REPO_ROOT / "omnisurg" / "rendering" / "slang.py").read_text()

        self.assertIn("float4 color : COLOR", shader_source)
        for kind in ("diffuse", "normal", "spec"):
            for layer in ("base", "damage", "coag", "blood"):
                self.assertIn(f"Texture2D<float4> {kind}_{layer}_tex", shader_source)
        self.assertIn("uniform int debug_mode", shader_source)
        for uniform in (
            "wetness",
            "wet_spec_scale",
            "wet_roughness",
            "subsurface_color",
            "subsurface_strength",
            "blood_wetness",
            "specular_aa_enabled",
            "specular_aa_strength",
            "specular_aa_min_roughness",
        ):
            self.assertIn(uniform, shader_source)
        for debug_symbol in (
            "DEBUG_VERTEX_BLEND_RGB",
            "DEBUG_MASKED_LAYER_WEIGHTS",
            "DEBUG_BLENDED_DIFFUSE",
            "DEBUG_BLENDED_NORMAL",
            "DEBUG_SPEC_ROUGHNESS",
            "DEBUG_HEAT_BLOOD_MASKS",
        ):
            self.assertIn(debug_symbol, shader_source)
        self.assertIn('"semantic_name": "COLOR"', renderer_source)
        self.assertIn("TISSUE_DEBUG_MODE_LABELS", renderer_source)
        self.assertEqual(len(TISSUE_DEBUG_MODE_LABELS), 7)
        self.assertIn("rgba32_float", renderer_source)
        self.assertIn("float dry_roughness = saturate(roughness_bias + (1.0 - spec_mask) * 0.35)", shader_source)
        self.assertIn("float apply_specular_aa", shader_source)
        self.assertIn("ddx(normal)", shader_source)
        self.assertIn("dry_roughness = apply_specular_aa(dry_roughness, normal)", shader_source)
        self.assertIn("float dry_spec = pow(saturate(dot(normal, half_dir)), dry_shininess)", shader_source)
        self.assertIn("float blood_film = saturate(max(blood_blend, blood_mask) * blood_wetness)", shader_source)
        self.assertIn("float wet_film = saturate(wetness + blood_film)", shader_source)
        self.assertIn("float wet_roughness_aa = apply_specular_aa(saturate(wet_roughness), normal)", shader_source)
        self.assertIn("float wet_shininess = lerp(160.0, 32.0, wet_roughness_aa)", shader_source)
        self.assertIn("* wet_spec_scale", shader_source)
        self.assertIn("float film_spec = dry_spec * (1.0 - wet_film) + wet_spec", shader_source)
        self.assertIn("float3 wet_base_color = lerp(base_color, base_color * 0.72, wet_film * 0.35)", shader_source)
        self.assertIn("float3 subsurface_light = subsurface_color * backscatter * saturate(subsurface_strength)", shader_source)
        self.assertIn("+ subsurface_light", shader_source)
        self.assertIn("float3(spec_mask, dry_roughness, wet_film)", shader_source)
        self.assertIn("wet_base_color * diffuse_light", shader_source)
        self.assertIn("+ light_key_color * film_spec", shader_source)

    def test_slang_postprocess_shader_declares_tonemap_and_scope_bindings(self):
        shader_source = (REPO_ROOT / "omnisurg" / "rendering" / "slang_shaders" / "omnisurg_post.slang").read_text()
        bloom_source = (REPO_ROOT / "omnisurg" / "rendering" / "slang_shaders" / "omnisurg_bloom.slang").read_text()
        exposure_source = (REPO_ROOT / "omnisurg" / "rendering" / "slang_shaders" / "omnisurg_exposure.slang").read_text()
        ao_source = (REPO_ROOT / "omnisurg" / "rendering" / "slang_shaders" / "omnisurg_ao.slang").read_text()
        fxaa_source = (REPO_ROOT / "omnisurg" / "rendering" / "slang_shaders" / "omnisurg_fxaa.slang").read_text()
        present_source = (REPO_ROOT / "omnisurg" / "rendering" / "slang_shaders" / "omnisurg_present.slang").read_text()
        mesh_source = (REPO_ROOT / "omnisurg" / "rendering" / "slang_shaders" / "omnisurg_mesh.slang").read_text()
        tissue_source = (REPO_ROOT / "omnisurg" / "rendering" / "slang_shaders" / "omnisurg_tissue.slang").read_text()
        renderer_source = (REPO_ROOT / "omnisurg" / "rendering" / "slang.py").read_text()

        self.assertIn("uint vertex_id : SV_VertexID", shader_source)
        self.assertIn("Texture2D<float4> scene_color_tex", shader_source)
        self.assertIn("Texture2D<float> depth_tex", shader_source)
        self.assertIn("Texture2D<float4> bloom_tex", shader_source)
        self.assertIn("Texture2D<float4> bloom_global_tex", shader_source)
        self.assertIn("Texture2D<float4> ao_tex", shader_source)
        self.assertIn("Texture2D<float4> lens_dirt_tex", shader_source)
        self.assertIn("Texture2D<float4> auto_exposure_tex", shader_source)
        self.assertIn("SamplerState post_sampler", shader_source)
        for uniform in (
            "frame_index",
            "postprocess_enabled",
            "exposure",
            "white_balance",
            "auto_exposure_enabled",
            "auto_exposure_target_luma",
            "auto_exposure_min",
            "auto_exposure_max",
            "auto_exposure_speed",
            "auto_exposure_highlight_weight",
            "ao_enabled",
            "ao_intensity",
            "bloom_enabled",
            "bloom_threshold",
            "bloom_intensity",
            "bloom_radius",
            "lens_dirt_enabled",
            "lens_dirt_intensity",
            "lens_dirt_threshold",
            "lens_dirt_base_opacity",
            "lens_dirt_global_drive",
            "lens_dirt_mask_gamma",
            "lens_distortion_enabled",
            "lens_distortion_strength",
            "lens_distortion_zoom",
            "chromatic_aberration_enabled",
            "chromatic_aberration_strength",
            "sensor_noise_enabled",
            "sensor_noise_strength",
            "sensor_noise_shadow_boost",
            "color_grade_enabled",
            "color_saturation",
            "color_contrast",
            "color_gamma",
            "color_warmth",
            "vignette_strength",
            "vignette_radius",
            "scope_radius",
            "scope_softness",
        ):
            self.assertIn(uniform, shader_source)
        self.assertIn("float3 aces_tonemap", shader_source)
        self.assertIn("bloom_tex.Sample", shader_source)
        self.assertIn("float3 apply_lens_dirt", shader_source)
        self.assertIn("float2 distort_uv", shader_source)
        self.assertIn("float3 sample_hdr_with_chromatic_aberration", shader_source)
        self.assertIn("float3 apply_color_grade", shader_source)
        self.assertIn("auto_exposure_tex.Sample", shader_source)
        self.assertIn("ao_tex.Sample", shader_source)
        self.assertIn("float3 bloom_drive = bloom_signal", shader_source)
        self.assertIn("float global_bloom_luminance", shader_source)
        self.assertIn("lens_dirt_base_opacity + driven_opacity", shader_source)
        self.assertIn("lens_dirt_global_drive", shader_source)
        self.assertIn("pow(saturate(lens_dirt_tex.Sample", shader_source)
        self.assertLess(shader_source.index("sample_hdr_with_chromatic_aberration"), shader_source.index("aces_tonemap(color)"))
        self.assertLess(shader_source.index("ao_tex.Sample"), shader_source.index("aces_tonemap(color)"))
        self.assertLess(shader_source.index("aces_tonemap(color)"), shader_source.index("apply_lens_dirt(color, input.uv, bloom_drive, global_drive)"))
        self.assertLess(shader_source.index("apply_lens_dirt(color, input.uv, bloom_drive, global_drive)"), shader_source.index("apply_color_grade(color)"))
        self.assertIn("prefilter_downsample_fragment", bloom_source)
        self.assertIn("downsample_fragment", bloom_source)
        self.assertIn("upsample_fragment", bloom_source)
        self.assertIn("float karis_weight", bloom_source)
        self.assertIn("float3 downsample_13_tap", bloom_source)
        self.assertIn("float3 tent_upsample", bloom_source)
        self.assertIn("Texture2D<float4> base_tex", bloom_source)
        self.assertIn("adapt_exposure_fragment", exposure_source)
        self.assertIn("Texture2D<float4> previous_exposure_tex", exposure_source)
        self.assertIn("auto_exposure_target_luma", exposure_source)
        self.assertIn("sampled_highlight_luminance", exposure_source)
        self.assertIn("ao_fragment", ao_source)
        self.assertIn("blur_fragment", ao_source)
        self.assertIn("Texture2D<float> depth_tex", ao_source)
        self.assertIn("reconstruct_view_position", ao_source)
        self.assertIn("reconstruct_normal", ao_source)
        self.assertIn("0.5 * ao_radius * camera_inv_tan_half_fovy", ao_source)
        self.assertIn("projected_radius = min(projected_radius, 0.06)", ao_source)
        self.assertIn("float recess_depth", ao_source)
        self.assertIn("occlusion = saturate(occlusion / 6.0)", ao_source)
        self.assertIn("source_ao_tex.Sample", ao_source)
        self.assertIn("fxaa_edge_threshold", fxaa_source)
        self.assertIn("luma_max - luma_min", fxaa_source)
        self.assertIn("manual_srgb_encode", present_source)
        self.assertIn("float3 apply_sensor_noise", present_source)
        self.assertIn("linear_to_srgb", present_source)
        self.assertIn("float endoscope_vignette", shader_source)
        self.assertIn("float scope_mask", shader_source)
        self.assertIn("scene_color_tex.Sample", shader_source)
        self.assertIn("omnisurg_post.slang", renderer_source)
        self.assertIn("omnisurg_bloom.slang", renderer_source)
        self.assertIn("omnisurg_exposure.slang", renderer_source)
        self.assertIn("omnisurg_ao.slang", renderer_source)
        self.assertIn("omnisurg_fxaa.slang", renderer_source)
        self.assertIn("omnisurg_present.slang", renderer_source)
        self.assertIn("PostProcessParams", renderer_source)
        self.assertIn("LENS_DIRT_TEXTURE_PATHS", renderer_source)
        self.assertIn("LENS_DIRT_TEXTURE_LABELS", renderer_source)
        self.assertIn("DEFAULT_LENS_DIRT_PATH", renderer_source)
        self.assertIn("LensDirt{index:02d}.png", renderer_source)
        self.assertIn("lens_dirt_texture_index", renderer_source)
        self.assertIn("TextureUsage.render_target | self._spy.TextureUsage.shader_resource", renderer_source)
        self.assertIn("rgba16_float", renderer_source)
        self.assertIn("self._post_pipeline", renderer_source)
        self.assertIn("self._bloom_prefilter_pipeline", renderer_source)
        self.assertIn("self._auto_exposure_pipeline", renderer_source)
        self.assertIn("self._ao_pipeline", renderer_source)
        self.assertIn("self._fxaa_pipeline", renderer_source)
        self.assertIn("self._present_pipeline", renderer_source)
        self.assertIn("_manual_srgb_encode", renderer_source)
        self.assertIn("input_layout=None", renderer_source)
        self.assertIn("max(color, float3(0.0))", mesh_source)
        self.assertIn("max(color, float3(0.0))", tissue_source)

    def test_slang_immediate_ui_reuses_button_until_click_is_reported(self):
        fake_sui = _FakeSlangUiModule()
        screen = _FakeSlangUiWidget()
        parent = _FakeSlangUiWidget(screen)
        adapter = _SlangImmediateUi(SimpleNamespace(float2=lambda x, y: (x, y)), fake_sui, screen)

        adapter.begin_frame()
        adapter.reset(parent, 640, 480)
        self.assertFalse(adapter.button("Apply"))
        adapter.finish_frame()
        self.assertEqual(fake_sui.created["Button"], 1)

        record = next(record for record in adapter._records.values() if record.widget.__class__ is _FakeSlangUiButton)
        record.widget.callback()

        adapter.begin_frame()
        adapter.reset(parent, 640, 480)
        self.assertTrue(adapter.button("Apply"))
        adapter.finish_frame()

        adapter.begin_frame()
        adapter.reset(parent, 640, 480)
        self.assertFalse(adapter.button("Apply"))
        adapter.finish_frame()
        self.assertEqual(fake_sui.created["Button"], 1)

    def test_slang_immediate_ui_reuses_value_widgets_for_drag_and_release(self):
        fake_sui = _FakeSlangUiModule()
        screen = _FakeSlangUiWidget()
        parent = _FakeSlangUiWidget(screen)
        adapter = _SlangImmediateUi(SimpleNamespace(float2=lambda x, y: (x, y)), fake_sui, screen)

        adapter.begin_frame()
        adapter.reset(parent, 640, 480)
        changed, value = adapter.slider_float("Damage", 0.0, 0.0, 1.0, "%.2f")
        self.assertFalse(changed)
        self.assertEqual(value, 0.0)
        adapter.finish_frame()

        record = next(record for record in adapter._records.values() if record.widget.__class__ is _FakeSlangUiSlider)
        first_widget = record.widget
        first_widget.callback(0.4)

        adapter.begin_frame()
        adapter.reset(parent, 640, 480)
        changed, value = adapter.slider_float("Damage", 0.0, 0.0, 1.0, "%.2f")
        self.assertTrue(changed)
        self.assertEqual(value, 0.4)
        self.assertIs(record.widget, first_widget)
        adapter.finish_frame()
        self.assertEqual(fake_sui.created["SliderFloat"], 1)

        first_widget.callback(0.8)
        adapter.begin_frame()
        adapter.reset(parent, 640, 480)
        changed, value = adapter.slider_float("Damage", 0.4, 0.0, 1.0, "%.2f")
        self.assertTrue(changed)
        self.assertEqual(value, 0.8)
        adapter.finish_frame()
        self.assertEqual(fake_sui.created["SliderFloat"], 1)

    def test_slang_immediate_ui_reuses_combo_for_debug_views(self):
        fake_sui = _FakeSlangUiModule()
        screen = _FakeSlangUiWidget()
        parent = _FakeSlangUiWidget(screen)
        adapter = _SlangImmediateUi(SimpleNamespace(float2=lambda x, y: (x, y)), fake_sui, screen)

        adapter.begin_frame()
        adapter.reset(parent, 640, 480)
        changed, value = adapter.combo("Tissue Debug View", 0, TISSUE_DEBUG_MODE_LABELS)
        self.assertFalse(changed)
        self.assertEqual(value, 0)
        adapter.finish_frame()

        record = next(record for record in adapter._records.values() if record.widget.__class__ is _FakeSlangUiCombo)
        first_widget = record.widget
        self.assertEqual(first_widget.items, list(TISSUE_DEBUG_MODE_LABELS))
        first_widget.callback(4)

        adapter.begin_frame()
        adapter.reset(parent, 640, 480)
        changed, value = adapter.combo("Tissue Debug View", 0, TISSUE_DEBUG_MODE_LABELS)
        self.assertTrue(changed)
        self.assertEqual(value, 4)
        self.assertIs(record.widget, first_widget)
        adapter.finish_frame()
        self.assertEqual(fake_sui.created["ComboBox"], 1)

    def test_slang_immediate_ui_reuses_checkbox_and_hides_unused_widgets(self):
        fake_sui = _FakeSlangUiModule()
        screen = _FakeSlangUiWidget()
        parent = _FakeSlangUiWidget(screen)
        adapter = _SlangImmediateUi(SimpleNamespace(float2=lambda x, y: (x, y)), fake_sui, screen)

        adapter.begin_frame()
        adapter.reset(parent, 640, 480)
        adapter.text("Visible")
        changed, value = adapter.checkbox("Enabled", False)
        self.assertFalse(changed)
        self.assertFalse(value)
        adapter.finish_frame()

        checkbox_record = next(record for record in adapter._records.values() if record.widget.__class__ is _FakeSlangUiValue)
        text_record = next(record for record in adapter._records.values() if record.widget.__class__ is _FakeSlangUiText)
        checkbox_record.widget.callback(True)

        adapter.begin_frame()
        adapter.reset(parent, 640, 480)
        changed, value = adapter.checkbox("Enabled", False)
        self.assertTrue(changed)
        self.assertTrue(value)
        adapter.finish_frame()

        self.assertEqual(fake_sui.created["CheckBox"], 1)
        self.assertFalse(text_record.widget.visible)

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

    def test_headless_runtime_smoke_synthetic_patch_assets(self):
        for asset_name in ("cloth_regular_low", "cloth_irregular_low"):
            runtime = None
            try:
                with self.subTest(asset=asset_name):
                    with wp.ScopedDevice("cpu"):
                        runtime = Runtime(
                            SimulationConfig(substeps=2, fps=30, constraint_iterations=1),
                            SceneConfig(asset_name=asset_name),
                            HapticConfig(),
                            ViewerConfig(backend="headless", textures_enabled=False),
                            BoundsConfig(),
                        )

                        runtime.step()
                        runtime.render()

                        self.assertTrue(runtime.is_running())
                        self.assertEqual(runtime.model.particle_count, SYNTHETIC_PATCH_ASSETS[asset_name]["vertices"])
                        self.assertFalse(runtime.mesh_textures)
            finally:
                if runtime is not None:
                    runtime.close()

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
                self.assertEqual(left_points.shape[0], runtime_module.GRASPER_JAW_SPHERE_COUNT)
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

    def test_grasper_truncation_kernel_behavior(self):
        with wp.ScopedDevice("cpu"):
            particle_flags = wp.array([int(runtime_module.ParticleFlags.ACTIVE)], dtype=wp.int32)
            surface_vertex_ids = wp.array([0], dtype=wp.int32)
            sphere_radii = wp.array([1.0], dtype=wp.float32)

            def run_kernel(base_position, displacement, *, safety=0.9, prev_center=None, curr_center=None, motion_samples=1):
                base_positions = wp.array([base_position], dtype=wp.vec3f)
                displacement_in = wp.array([displacement], dtype=wp.vec3f)
                sphere_centers_prev = wp.array([prev_center or [0.0, 0.0, 0.0]], dtype=wp.vec3f)
                sphere_centers = wp.array([curr_center or [0.0, 0.0, 0.0]], dtype=wp.vec3f)
                truncation_t = wp.array([1.0], dtype=wp.float32)
                wp.launch(
                    kernel=runtime_module.compute_vertex_sphere_truncation_factors,
                    dim=motion_samples,
                    inputs=[
                        particle_flags,
                        base_positions,
                        surface_vertex_ids,
                        displacement_in,
                        sphere_centers_prev,
                        sphere_centers,
                        sphere_radii,
                        1,
                        motion_samples,
                        0.0,
                        safety,
                    ],
                    outputs=[truncation_t],
                    device=wp.get_device(),
                )
                return float(truncation_t.numpy()[0])

            t_cross = run_kernel([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], safety=0.9)
            t_cross_low_safety = run_kernel([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], safety=0.5)
            t_away = run_kernel([1.1, 0.0, 0.0], [1.0, 0.0, 0.0], safety=0.9)

            self.assertGreater(t_cross, 0.0)
            self.assertLess(t_cross, 1.0)
            self.assertLess(t_cross_low_safety, t_cross)
            self.assertAlmostEqual(t_away, 1.0, places=6)

    def test_grasper_prediction_truncation_shortens_swept_displacement(self):
        with wp.ScopedDevice("cpu"):
            active_flag = wp.array([int(runtime_module.ParticleFlags.ACTIVE)], dtype=wp.int32)
            model = SimpleNamespace(
                particle_count=1,
                tri_count=1,
                particle_flags=active_flag,
                device=wp.get_device(),
            )
            surface_vertex_ids = wp.array([0], dtype=wp.int32)

            def build_system(motion_samples: int):
                chain = SimpleNamespace(
                    world_points_prev=wp.array([[-1.0, 0.0, 0.0]], dtype=wp.vec3f),
                    world_points=wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3f),
                    radii=wp.array([0.05], dtype=wp.float32),
                )
                system = runtime_module.GrasperSphereTruncationSystem(
                    graspers={"right": SimpleNamespace(sphere_chains=[chain])},
                    surface_vertex_ids=surface_vertex_ids,
                    motion_samples=motion_samples,
                    contact_margin=0.0,
                    safety=0.9,
                    truncate_prediction=True,
                )
                system.initialize(model)
                return system

            base_positions = wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3f)
            no_sweep_displacement = wp.array([[-0.98, 0.0, 0.0]], dtype=wp.vec3f)
            sweep_displacement = wp.array([[-0.98, 0.0, 0.0]], dtype=wp.vec3f)

            build_system(1).truncate_prediction(model, None, None, base_positions, no_sweep_displacement, 1.0 / 120.0)
            build_system(4).truncate_prediction(model, None, None, base_positions, sweep_displacement, 1.0 / 120.0)

            self.assertAlmostEqual(float(no_sweep_displacement.numpy()[0][0]), -0.98, places=5)
            self.assertGreater(float(sweep_displacement.numpy()[0][0]), -0.98)

    def test_grasper_iteration_truncation_reduces_elastic_delta(self):
        with wp.ScopedDevice("cpu"):
            active_flag = wp.array([int(runtime_module.ParticleFlags.ACTIVE)], dtype=wp.int32)
            model = SimpleNamespace(
                particle_count=1,
                tri_count=1,
                particle_flags=active_flag,
                device=wp.get_device(),
            )
            chain = SimpleNamespace(
                world_points_prev=wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3f),
                world_points=wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3f),
                radii=wp.array([1.0], dtype=wp.float32),
            )
            system = runtime_module.GrasperSphereTruncationSystem(
                graspers={"right": SimpleNamespace(sphere_chains=[chain])},
                surface_vertex_ids=wp.array([0], dtype=wp.int32),
                motion_samples=4,
                contact_margin=0.0,
                safety=0.9,
                truncate_prediction=True,
            )
            system.initialize(model)

            particle_q = wp.array([[2.0, 0.0, 0.0]], dtype=wp.vec3f)
            particle_qd = wp.zeros(1, dtype=wp.vec3f)
            particle_deltas = wp.array([[-2.0, 0.0, 0.0]], dtype=wp.vec3f)

            system.truncate_deltas(model, None, None, particle_q, particle_qd, particle_deltas, 1.0 / 120.0, 0)

            self.assertGreater(float(particle_deltas.numpy()[0][0]), -2.0)

    def test_haptic_iteration_truncation_reduces_elastic_delta(self):
        with wp.ScopedDevice("cpu"):
            active_flag = wp.array([int(runtime_module.ParticleFlags.ACTIVE)], dtype=wp.int32)
            model = SimpleNamespace(
                particle_count=1,
                tri_count=1,
                particle_flags=active_flag,
                device=wp.get_device(),
            )
            proxy = SimpleNamespace(
                center_scaled_prev=wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3f),
                center_scaled=wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3f),
                radius=1.0,
            )
            system = haptic_collision_module.HapticSphereTruncationSystem(
                proxy,
                surface_vertex_ids=wp.array([0], dtype=wp.int32),
                motion_samples=4,
                contact_margin=0.0,
                safety=0.9,
                truncate_prediction=True,
            )
            system.initialize(model)

            particle_q = wp.array([[2.0, 0.0, 0.0]], dtype=wp.vec3f)
            particle_qd = wp.zeros(1, dtype=wp.vec3f)
            particle_deltas = wp.array([[-2.0, 0.0, 0.0]], dtype=wp.vec3f)

            system.truncate_deltas(model, None, None, particle_q, particle_qd, particle_deltas, 1.0 / 120.0, 0)

            self.assertGreater(float(particle_deltas.numpy()[0][0]), -2.0)

    def test_runtime_grasper_collision_mode_wiring_and_pending_settings(self):
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

                self.assertEqual(runtime.grasper_collision_mode, "projection")
                self.assertTrue(runtime.haptic_collision_system.enabled)
                self.assertFalse(runtime.haptic_truncation_system.enabled)
                self.assertTrue(runtime.grasper_collision_system.enabled)
                self.assertFalse(runtime.grasper_truncation_system.enabled)

                runtime._pending_grasper_collision_mode = "hybrid"
                runtime._pending_grasper_collision_motion_samples = 6
                runtime._pending_grasper_collision_margin = 0.005
                runtime._pending_grasper_truncation_safety = 0.75
                runtime._pending_grasper_truncate_prediction = False

                self.assertTrue(runtime._has_pending_solver_changes())
                self.assertTrue(runtime._pending_graph_rebuild_needed())

                runtime._apply_pending_solver_settings()

                self.assertEqual(runtime.grasper_collision_mode, "hybrid")
                self.assertEqual(runtime.sim_config.grasper_collision_mode, "hybrid")
                self.assertEqual(runtime.grasper_collision_motion_samples, 6)
                self.assertAlmostEqual(runtime.grasper_collision_margin, 0.005)
                self.assertAlmostEqual(runtime.grasper_truncation_safety, 0.75)
                self.assertFalse(runtime.grasper_truncate_prediction)
                self.assertTrue(runtime.haptic_collision_system.enabled)
                self.assertTrue(runtime.haptic_truncation_system.enabled)
                self.assertEqual(runtime.haptic_truncation_system.motion_samples, 6)
                self.assertAlmostEqual(runtime.haptic_truncation_system.contact_margin, 0.005)
                self.assertAlmostEqual(runtime.haptic_truncation_system.safety, 0.75)
                self.assertFalse(runtime.haptic_truncation_system.truncate_prediction_enabled)
                self.assertTrue(runtime.grasper_collision_system.enabled)
                self.assertTrue(runtime.grasper_truncation_system.enabled)
                self.assertEqual(runtime.grasper_collision_system.motion_samples, 6)
                self.assertAlmostEqual(runtime.grasper_collision_system.contact_margin, 0.005)
                self.assertEqual(runtime.grasper_truncation_system.motion_samples, 6)
                self.assertAlmostEqual(runtime.grasper_truncation_system.contact_margin, 0.005)
                self.assertAlmostEqual(runtime.grasper_truncation_system.safety, 0.75)
                self.assertFalse(runtime.grasper_truncation_system.truncate_prediction_enabled)
                self.assertFalse(runtime._has_pending_solver_changes())

                runtime._pending_grasper_collision_mode = "truncation"
                runtime._apply_pending_solver_settings()

                self.assertEqual(runtime.grasper_collision_mode, "truncation")
                self.assertFalse(runtime.haptic_collision_system.enabled)
                self.assertTrue(runtime.haptic_truncation_system.enabled)
                self.assertFalse(runtime.grasper_collision_system.enabled)
                self.assertTrue(runtime.grasper_truncation_system.enabled)
        finally:
            if runtime is not None:
                runtime.close()

    def test_grasper_sweep_sampling_catches_crossing_contact(self):
        with wp.ScopedDevice("cpu"):
            positions = wp.array(
                [
                    [-0.5, -0.5, 0.0],
                    [0.5, -0.5, 0.0],
                    [0.0, 0.5, 0.0],
                ],
                dtype=wp.vec3f,
            )
            velocities = wp.zeros(3, dtype=wp.vec3f)
            inv_masses = wp.array([1.0, 1.0, 1.0], dtype=wp.float32)
            tri_indices = wp.array([[0, 1, 2]], dtype=wp.int32, ndim=2)
            sphere_centers_prev = wp.array([[0.0, 0.0, -1.0]], dtype=wp.vec3f)
            sphere_centers = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3f)
            sphere_radii = wp.array([0.1], dtype=wp.float32)

            def run_sweep(sample_count: int):
                delta_accumulator = wp.zeros(3, dtype=wp.vec3f)
                delta_counter = wp.zeros(3, dtype=wp.int32)
                reaction_accumulator = wp.zeros(1, dtype=wp.vec3f)
                reaction_counter = wp.zeros(1, dtype=wp.int32)
                wp.launch(
                    kernel=runtime_module.collide_triangles_vs_spheres,
                    dim=sample_count,
                    inputs=[
                        positions,
                        velocities,
                        inv_masses,
                        tri_indices,
                        sphere_centers_prev,
                        sphere_centers,
                        sphere_radii,
                        1,
                        sample_count,
                        0.0,
                        0.0,
                        1.0 / 120.0,
                        0.0,
                    ],
                    outputs=[delta_accumulator, delta_counter, reaction_accumulator, reaction_counter],
                    device=wp.get_device(),
                )
                return int(delta_counter.numpy().sum())

            self.assertEqual(run_sweep(1), 0)
            self.assertGreater(run_sweep(runtime_module.GRASPER_COLLISION_SWEEP_SAMPLES), 0)

    def test_grasper_collision_fallback_is_winding_invariant(self):
        with wp.ScopedDevice("cpu"):
            positions = wp.array(
                [
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=wp.vec3f,
            )
            velocities = wp.zeros(3, dtype=wp.vec3f)
            inv_masses = wp.array([1.0, 1.0, 1.0], dtype=wp.float32)
            sphere_centers_prev = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3f)
            sphere_centers = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3f)
            sphere_radii = wp.array([0.1], dtype=wp.float32)

            def collide(tri):
                delta_accumulator = wp.zeros(3, dtype=wp.vec3f)
                delta_counter = wp.zeros(3, dtype=wp.int32)
                reaction_accumulator = wp.zeros(1, dtype=wp.vec3f)
                reaction_counter = wp.zeros(1, dtype=wp.int32)
                wp.launch(
                    kernel=runtime_module.collide_triangles_vs_spheres,
                    dim=1,
                    inputs=[
                        positions,
                        velocities,
                        inv_masses,
                        wp.array([tri], dtype=wp.int32, ndim=2),
                        sphere_centers_prev,
                        sphere_centers,
                        sphere_radii,
                        1,
                        1,
                        0.0,
                        0.0,
                        1.0 / 120.0,
                        0.0,
                    ],
                    outputs=[delta_accumulator, delta_counter, reaction_accumulator, reaction_counter],
                    device=wp.get_device(),
                )
                return delta_accumulator.numpy(), delta_counter.numpy()

            delta_a, count_a = collide([0, 1, 2])
            delta_b, count_b = collide([0, 2, 1])
            np.testing.assert_allclose(delta_a, delta_b, atol=1.0e-6)
            np.testing.assert_array_equal(count_a, count_b)

    def test_haptic_collision_fallback_is_winding_invariant(self):
        with wp.ScopedDevice("cpu"):
            positions = wp.array(
                [
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=wp.vec3f,
            )
            velocities = wp.array(
                [
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, -1.0],
                ],
                dtype=wp.vec3f,
            )
            inv_masses = wp.array([1.0, 1.0, 1.0], dtype=wp.float32)
            sphere_center = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3f)

            def collide(tri):
                delta_accumulator = wp.zeros(3, dtype=wp.vec3f)
                delta_counter = wp.zeros(3, dtype=wp.int32)
                wp.launch(
                    kernel=collision_module.collide_triangles_vs_sphere,
                    dim=1,
                    inputs=[
                        positions,
                        velocities,
                        inv_masses,
                        wp.array([tri], dtype=wp.int32, ndim=2),
                        sphere_center,
                        0.1,
                        1.0,
                        0.0,
                        1.0 / 120.0,
                        0.0,
                    ],
                    outputs=[delta_accumulator, delta_counter],
                    device=wp.get_device(),
                )
                return delta_accumulator.numpy(), delta_counter.numpy()

            delta_a, count_a = collide([0, 1, 2])
            delta_b, count_b = collide([0, 2, 1])
            np.testing.assert_allclose(delta_a, delta_b, atol=1.0e-6)
            np.testing.assert_array_equal(count_a, count_b)

    def test_haptic_collision_system_accumulates_opposite_reaction(self):
        with wp.ScopedDevice("cpu"):
            proxy = SimpleNamespace(
                center_scaled=wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3f),
                radius=0.1,
                max_tri_extent=2.0,
            )
            system = haptic_collision_module.HapticSphereCollisionSystem(proxy)
            model = SimpleNamespace(
                particle_count=3,
                tri_count=1,
                particle_inv_mass=wp.array([1.0, 1.0, 1.0], dtype=wp.float32),
                tri_indices=wp.array([[0, 1, 2]], dtype=wp.int32, ndim=2),
                device=wp.get_device(),
            )
            system.initialize(model)

            particle_q = wp.array(
                [
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=wp.vec3f,
            )
            particle_qd = wp.array(
                [
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, -1.0],
                ],
                dtype=wp.vec3f,
            )
            particle_deltas = wp.zeros(3, dtype=wp.vec3f)

            system.clear_reaction()
            system.solve_constraints(
                model,
                None,
                None,
                particle_q,
                particle_qd,
                particle_deltas,
                None,
                None,
                None,
                1.0 / 120.0,
                0,
            )

            reaction, count = system.get_reaction_average()
            particle_delta_sum = particle_deltas.numpy().sum(axis=0)

            self.assertEqual(count, 1)
            np.testing.assert_allclose(reaction, -particle_delta_sum, atol=1.0e-6)

    def test_haptic_collision_system_reaction_is_zero_without_contact(self):
        with wp.ScopedDevice("cpu"):
            proxy = SimpleNamespace(
                center_scaled=wp.array([[0.0, 0.0, 2.0]], dtype=wp.vec3f),
                radius=0.1,
                max_tri_extent=2.0,
            )
            system = haptic_collision_module.HapticSphereCollisionSystem(proxy)
            model = SimpleNamespace(
                particle_count=3,
                tri_count=1,
                particle_inv_mass=wp.array([1.0, 1.0, 1.0], dtype=wp.float32),
                tri_indices=wp.array([[0, 1, 2]], dtype=wp.int32, ndim=2),
                device=wp.get_device(),
            )
            system.initialize(model)

            particle_q = wp.array(
                [
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=wp.vec3f,
            )
            particle_qd = wp.zeros(3, dtype=wp.vec3f)
            particle_deltas = wp.zeros(3, dtype=wp.vec3f)

            system.clear_reaction()
            system.solve_constraints(
                model,
                None,
                None,
                particle_q,
                particle_qd,
                particle_deltas,
                None,
                None,
                None,
                1.0 / 120.0,
                0,
            )

            reaction, count = system.get_reaction_average()

            self.assertEqual(count, 0)
            np.testing.assert_allclose(reaction, np.zeros(3, dtype=np.float32), atol=1.0e-6)

    def test_grasper_collision_system_accumulates_opposite_reaction(self):
        with wp.ScopedDevice("cpu"):
            chain = SimpleNamespace(
                world_points_prev=wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3f),
                world_points=wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3f),
                base_radii=wp.array([0.1], dtype=wp.float32),
                radii=wp.array([0.1], dtype=wp.float32),
            )
            system = runtime_module.GrasperSphereCollisionSystem(
                graspers={"right": SimpleNamespace(sphere_chains=[chain])},
                proxy=SimpleNamespace(max_tri_extent=2.0),
                motion_samples=1,
                contact_margin=0.0,
            )
            model = SimpleNamespace(
                particle_count=3,
                tri_count=1,
                particle_inv_mass=wp.array([1.0, 1.0, 1.0], dtype=wp.float32),
                tri_indices=wp.array([[0, 1, 2]], dtype=wp.int32, ndim=2),
                device=wp.get_device(),
            )
            system.initialize(model)

            particle_q = wp.array(
                [
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=wp.vec3f,
            )
            particle_qd = wp.zeros(3, dtype=wp.vec3f)
            particle_deltas = wp.zeros(3, dtype=wp.vec3f)

            system.clear_reaction()
            system.solve_constraints(
                model,
                None,
                None,
                particle_q,
                particle_qd,
                particle_deltas,
                None,
                None,
                None,
                1.0 / 120.0,
                0,
            )

            reaction, count = system.get_reaction_average("right")
            particle_delta_sum = particle_deltas.numpy().sum(axis=0)

            self.assertEqual(count, 1)
            np.testing.assert_allclose(reaction, -particle_delta_sum, atol=1.0e-6)

    def test_grasper_collision_system_keeps_controller_reactions_separate(self):
        with wp.ScopedDevice("cpu"):
            right_chain = SimpleNamespace(
                world_points_prev=wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3f),
                world_points=wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3f),
                base_radii=wp.array([0.1], dtype=wp.float32),
                radii=wp.array([0.1], dtype=wp.float32),
            )
            left_chain = SimpleNamespace(
                world_points_prev=wp.array([[0.0, 0.0, 2.0]], dtype=wp.vec3f),
                world_points=wp.array([[0.0, 0.0, 2.0]], dtype=wp.vec3f),
                base_radii=wp.array([0.1], dtype=wp.float32),
                radii=wp.array([0.1], dtype=wp.float32),
            )
            system = runtime_module.GrasperSphereCollisionSystem(
                graspers={
                    "right": SimpleNamespace(sphere_chains=[right_chain]),
                    "left": SimpleNamespace(sphere_chains=[left_chain]),
                },
                proxy=SimpleNamespace(max_tri_extent=2.0),
                motion_samples=1,
                contact_margin=0.0,
            )
            model = SimpleNamespace(
                particle_count=3,
                tri_count=1,
                particle_inv_mass=wp.array([1.0, 1.0, 1.0], dtype=wp.float32),
                tri_indices=wp.array([[0, 1, 2]], dtype=wp.int32, ndim=2),
                device=wp.get_device(),
            )
            system.initialize(model)

            particle_q = wp.array(
                [
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=wp.vec3f,
            )
            particle_qd = wp.zeros(3, dtype=wp.vec3f)
            particle_deltas = wp.zeros(3, dtype=wp.vec3f)

            system.clear_reaction()
            system.solve_constraints(
                model,
                None,
                None,
                particle_q,
                particle_qd,
                particle_deltas,
                None,
                None,
                None,
                1.0 / 120.0,
                0,
            )

            right_reaction, right_count = system.get_reaction_average("right")
            left_reaction, left_count = system.get_reaction_average("left")

            self.assertEqual(right_count, 1)
            self.assertEqual(left_count, 0)
            self.assertGreater(np.linalg.norm(right_reaction), 0.0)
            np.testing.assert_allclose(left_reaction, np.zeros(3, dtype=np.float32), atol=1.0e-6)

    def test_multi_source_rig_forwards_force_commands_and_ignores_unsupported_sources(self):
        class FakeSource:
            def __init__(self, force_capable: bool):
                self.force_capable = force_capable
                self.forces = []

            def poll(self):
                return {}

            def set_force(self, force_xyz):
                self.forces.append(np.asarray(force_xyz, dtype=np.float32).copy())

            def supports_force_feedback(self):
                return self.force_capable

            def close(self):
                pass

        right = FakeSource(force_capable=True)
        left = FakeSource(force_capable=False)
        rig = MultiSourceRig({"right": right, "left": left})

        rig.set_force_commands({"right": np.array([0.01, 0.0, 0.0], dtype=np.float32)})

        self.assertTrue(rig.supports_force_feedback())
        self.assertTrue(rig.supports_force_feedback("right"))
        self.assertFalse(rig.supports_force_feedback("left"))
        np.testing.assert_allclose(right.forces[0], np.array([0.01, 0.0, 0.0], dtype=np.float32), atol=1.0e-6)
        np.testing.assert_allclose(left.forces[0], np.zeros(3, dtype=np.float32), atol=1.0e-6)

    def test_main_force_dispatch_helpers_forward_and_zero_commands(self):
        class FakeRig:
            def __init__(self):
                self.commands = []

            def set_force_commands(self, force_commands):
                copied = {
                    controller_id: np.asarray(force, dtype=np.float32).copy()
                    for controller_id, force in force_commands.items()
                }
                self.commands.append(copied)

        rig = FakeRig()
        omnisurg_main._dispatch_haptic_force_commands(
            rig,
            {"right": np.array([0.02, 0.0, 0.0], dtype=np.float32)},
        )
        omnisurg_main._zero_haptic_force_commands(rig)

        np.testing.assert_allclose(rig.commands[0]["right"], np.array([0.02, 0.0, 0.0], dtype=np.float32), atol=1.0e-6)
        np.testing.assert_allclose(rig.commands[1]["right"], np.zeros(3, dtype=np.float32), atol=1.0e-6)
        np.testing.assert_allclose(rig.commands[1]["left"], np.zeros(3, dtype=np.float32), atol=1.0e-6)

    def test_runtime_haptic_force_feedback_zeroes_without_contact_or_when_disabled(self):
        runtime = _make_haptic_runtime_stub(_prepare_test_subdir("haptic_force_zero"))
        runtime._haptic_feedback_available = True

        runtime_module.Runtime._update_haptic_force_feedback(runtime, np.array([1.0, 0.0, 0.0], dtype=np.float32), 0)
        np.testing.assert_allclose(runtime.haptic_feedback_diagnostics.final_force, np.zeros(3, dtype=np.float32), atol=1.0e-6)

        runtime.haptic_feedback_settings.enabled = False
        runtime_module.Runtime._update_haptic_force_feedback(runtime, np.array([1.0, 0.0, 0.0], dtype=np.float32), 1)
        np.testing.assert_allclose(runtime.haptic_feedback_diagnostics.final_force, np.zeros(3, dtype=np.float32), atol=1.0e-6)

    def test_runtime_combined_haptic_reaction_uses_right_grasper_only(self):
        runtime = runtime_module.Runtime.__new__(runtime_module.Runtime)
        runtime.haptic_collision_system = SimpleNamespace(
            get_reaction_average=lambda: (np.array([1.0, 0.0, 0.0], dtype=np.float32), 2)
        )
        runtime.grasper_collision_system = SimpleNamespace(
            get_reaction_average=lambda controller_id: (
                (np.array([4.0, 0.0, 0.0], dtype=np.float32), 1)
                if controller_id == runtime_module.PRIMARY_CONTROLLER_ID
                else (np.array([99.0, 0.0, 0.0], dtype=np.float32), 1)
            )
        )

        reaction, count = runtime_module.Runtime._get_combined_haptic_reaction(runtime)

        self.assertEqual(count, 3)
        np.testing.assert_allclose(reaction, np.array([2.0, 0.0, 0.0], dtype=np.float32), atol=1.0e-6)

    def test_runtime_haptic_force_feedback_filter_pipeline_order(self):
        runtime = _make_haptic_runtime_stub(_prepare_test_subdir("haptic_force_filter"))
        runtime._haptic_feedback_available = True
        runtime.controller_states[runtime_module.PRIMARY_CONTROLLER_ID].scaled_position = np.zeros(3, dtype=np.float32)
        runtime.haptic_feedback_settings = haptic_feedback_module.HapticFeedbackSettings(
            enabled=True,
            reaction_scale=1.0,
            proxy_follow=1.0,
            max_proxy_offset=1.0,
            spring_k=1.0,
            damper_b=0.0,
            deadband=0.05,
            lowpass_alpha=0.5,
            max_force=0.1,
            slew_rate_limit=0.6,
        )

        runtime_module.Runtime._update_haptic_force_feedback(runtime, np.array([1.0, 0.0, 0.0], dtype=np.float32), 1)

        np.testing.assert_allclose(runtime.haptic_feedback_diagnostics.raw_force, np.array([1.0, 0.0, 0.0], dtype=np.float32), atol=1.0e-6)
        np.testing.assert_allclose(runtime.haptic_feedback_diagnostics.filtered_force, np.array([0.5, 0.0, 0.0], dtype=np.float32), atol=1.0e-6)
        np.testing.assert_allclose(runtime.haptic_feedback_diagnostics.final_force, np.array([0.06, 0.0, 0.0], dtype=np.float32), atol=1.0e-6)
        self.assertTrue(runtime.haptic_feedback_diagnostics.clamp_active)
        self.assertTrue(runtime.haptic_feedback_diagnostics.slew_active)
        np.testing.assert_allclose(runtime.get_haptic_force_commands()["right"], np.array([0.06, 0.0, 0.0], dtype=np.float32), atol=1.0e-6)

    def test_haptic_preset_round_trip_and_scan(self):
        preset_dir = _prepare_test_subdir("haptic_preset_roundtrip")
        preset_path = preset_dir / "custom.json"
        settings = haptic_feedback_module.HapticFeedbackSettings(
            reaction_scale=1.5,
            proxy_follow=0.4,
            spring_k=4.0,
            damper_b=0.12,
        )

        haptic_feedback_module.save_haptic_preset(preset_path, "Custom Tune", settings)
        presets, errors = haptic_feedback_module.scan_haptic_presets(preset_dir)
        loaded = haptic_feedback_module.load_haptic_preset(preset_path)

        self.assertEqual(errors, [])
        self.assertEqual(len(presets), 1)
        self.assertEqual(loaded.name, "Custom Tune")
        self.assertTrue(haptic_feedback_module.settings_almost_equal(loaded.settings, settings))

    def test_runtime_haptic_preset_dirty_flag_and_invalid_json_handling(self):
        preset_dir = _prepare_test_subdir("haptic_preset_invalid_json")
        runtime = _make_haptic_runtime_stub(preset_dir)
        preset_path = preset_dir / "loaded.json"
        haptic_feedback_module.save_haptic_preset(
            preset_path,
            "Loaded Tune",
            haptic_feedback_module.HapticFeedbackSettings(spring_k=5.0),
        )

        runtime._refresh_haptic_presets(show_status=False)
        self.assertTrue(runtime._load_haptic_preset_path(preset_path, show_status=False))
        self.assertFalse(runtime.haptic_feedback_diagnostics.dirty)

        runtime.haptic_feedback_settings.spring_k += 0.5
        runtime._sync_haptic_feedback_metadata()
        self.assertTrue(runtime.haptic_feedback_diagnostics.dirty)

        invalid_path = preset_dir / "broken.json"
        invalid_path.write_text("{not valid json", encoding="utf-8")
        previous_settings = haptic_feedback_module.copy_feedback_settings(runtime.haptic_feedback_settings)

        self.assertFalse(runtime._load_haptic_preset_path(invalid_path, show_status=False))
        self.assertTrue(haptic_feedback_module.settings_almost_equal(runtime.haptic_feedback_settings, previous_settings))
        self.assertTrue(runtime.haptic_feedback_diagnostics.status_is_error)
        self.assertIn("not valid JSON", runtime.haptic_feedback_diagnostics.status_message)

    def test_runtime_haptic_preset_methods_smoke_without_viewer(self):
        preset_dir = _prepare_test_subdir("haptic_preset_smoke")
        runtime = _make_haptic_runtime_stub(preset_dir)

        runtime.haptic_feedback_settings.reaction_scale = 1.25
        runtime.haptic_feedback_settings.spring_k = 4.5
        runtime._sync_haptic_feedback_metadata()

        self.assertTrue(runtime._save_haptic_preset_as_new("Smoke Tune"))
        self.assertGreaterEqual(len(runtime._haptic_presets), 1)
        self.assertEqual(runtime.haptic_feedback_diagnostics.current_preset_name, "Smoke Tune")
        self.assertFalse(runtime.haptic_feedback_diagnostics.dirty)

        runtime.haptic_feedback_settings.spring_k = 6.0
        runtime._sync_haptic_feedback_metadata()
        self.assertTrue(runtime.haptic_feedback_diagnostics.dirty)

        self.assertTrue(runtime._save_haptic_selected_preset())
        self.assertFalse(runtime.haptic_feedback_diagnostics.dirty)
        self.assertFalse(runtime.haptic_feedback_diagnostics.status_is_error)

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
                self.assertEqual(right_points.shape[0], runtime_module.GRASPER_JAW_SPHERE_COUNT)
                self.assertEqual(left_points.shape[0], runtime_module.GRASPER_JAW_SPHERE_COUNT)
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
