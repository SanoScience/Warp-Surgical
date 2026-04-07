import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WARP_CACHE_DIR = REPO_ROOT / ".warp_cache"
TEST_TMP_DIR = REPO_ROOT / ".tmp_tests"
WARP_CACHE_DIR.mkdir(exist_ok=True)
TEST_TMP_DIR.mkdir(exist_ok=True)
os.environ["WARP_CACHE_PATH"] = str(WARP_CACHE_DIR)

import warp as wp

from omnisurg import BoundsConfig, HapticConfig, Runtime, SceneConfig, SimulationConfig, ViewerConfig
from omnisurg.assets import load_tet_asset
from omnisurg.haptics import ReplayInputSource
from omnisurg.scene_builder import build_scene


class TestPhase1Runtime(unittest.TestCase):
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
        self.assertIn("OmniSurg Phase 1", result.stdout)
        self.assertIn("headless", result.stdout)

    def test_asset_loading_liver_counts(self):
        asset = load_tet_asset("liver")
        self.assertEqual(int(asset.rest_positions.shape[0]), 1986)
        self.assertEqual(int(asset.tet_indices.shape[0]), 6653)
        self.assertEqual(int(asset.edge_indices.shape[0]), 10622)
        self.assertEqual(int(asset.surface_tri_indices.shape[0]), 3968)
        self.assertTrue(asset.uvs is not None)

    def test_scene_construction(self):
        with wp.ScopedDevice("cpu"):
            asset = load_tet_asset("liver")
            scene = build_scene(asset, SceneConfig(), HapticConfig(), wp.get_device())

        self.assertEqual(scene.model.particle_count, 1986)
        self.assertEqual(scene.model.spring_count, 10622)
        self.assertEqual(scene.model.tri_count, 3968)
        self.assertEqual(scene.tetrahedra_wp.shape[0], 6653)
        self.assertGreaterEqual(scene.haptic_proxy.body_id, 0)
        self.assertEqual(scene.haptic_proxy.radius, 0.1)

    def test_headless_replay_runtime_smoke(self):
        trace = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.5, -0.5, 0.0, 0.0, 0.0, 1.0],
                [2.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        replay_path = TEST_TMP_DIR / "trace.npy"
        np.save(replay_path, trace)

        try:
            with wp.ScopedDevice("cpu"):
                runtime = Runtime(
                    SimulationConfig(substeps=2, fps=30, constraint_iterations=1),
                    SceneConfig(asset_name="liver"),
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
                source.close()
                runtime.close()
                self.assertFalse(runtime.is_running())
        finally:
            if replay_path.exists():
                replay_path.unlink()


if __name__ == "__main__":
    unittest.main()
