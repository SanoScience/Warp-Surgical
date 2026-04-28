import os
import sys
from pathlib import Path

_cache_dir = Path(__file__).resolve().parents[1] / ".warp_cache"
if "WARP_CACHE_PATH" not in os.environ:
    _cache_dir.mkdir(exist_ok=True)
    os.environ["WARP_CACHE_PATH"] = str(_cache_dir)

# Keep stdout clean when invoked as `python -m omnisurg.haptic_bench` so
# `--json` pipes cleanly through `jq` / `python -c`. At the moment this
# file runs, sys.argv[0] is literally "-m" and __main__ has no spec yet,
# so we inspect runpy's call stack for the target module name.
def _is_bench_invocation() -> bool:
    if any("haptic_bench" in a for a in sys.argv):
        return True
    main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if main_spec is not None and "haptic_bench" in getattr(main_spec, "name", ""):
        return True
    import inspect
    for frame_info in inspect.stack():
        locals_ = frame_info.frame.f_locals
        mod_name = locals_.get("mod_name") or locals_.get("alter_argv")
        if isinstance(mod_name, str) and "haptic_bench" in mod_name:
            return True
        main_globals = locals_.get("main_globals")
        if isinstance(main_globals, dict):
            spec = main_globals.get("__spec__")
            if spec is not None and "haptic_bench" in getattr(spec, "name", ""):
                return True
    return False


if _is_bench_invocation():
    import warp as _wp
    _wp.config.quiet = True

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
