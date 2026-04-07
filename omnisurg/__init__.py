import os
from pathlib import Path

_cache_dir = Path(__file__).resolve().parents[1] / ".warp_cache"
if "WARP_CACHE_PATH" not in os.environ:
    _cache_dir.mkdir(exist_ok=True)
    os.environ["WARP_CACHE_PATH"] = str(_cache_dir)

from omnisurg.config import BoundsConfig, HapticConfig, SceneConfig, SimulationConfig, ViewerConfig
from omnisurg.input.sources import InputRig, InputSource
from omnisurg.mesh.assets import TetMeshAsset, load_tet_asset
from omnisurg.mesh.scene import SceneData, build_scene
from omnisurg.runtime import Runtime

__all__ = [
    "BoundsConfig",
    "HapticConfig",
    "InputRig",
    "InputSource",
    "Runtime",
    "SceneConfig",
    "SceneData",
    "SimulationConfig",
    "TetMeshAsset",
    "ViewerConfig",
    "build_scene",
    "load_tet_asset",
]
