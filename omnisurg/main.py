import argparse
import sys

import numpy as np
import warp as wp


DEFAULT_FOLLOU_ROOT = r"G:\warp\python_device_manager"
LIVE_INPUT_BACKENDS = ["openhaptics", "minimou", "none"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="OmniSurg Phase 2 - soft-body simulation with configurable live input backends",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default=None, help="Warp device override")
    parser.add_argument(
        "--viewer",
        type=str,
        default="gl",
        choices=["surgsim", "gl", "rtx", "headless"],
        help="Viewer backend",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="chole",
        choices=["single", "chole"],
        help="Scene preset",
    )
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help="Path to right-controller .npy haptic replay trace",
    )
    parser.add_argument(
        "--left-replay",
        type=str,
        default=None,
        help="Path to left-controller .npy haptic replay trace",
    )
    parser.add_argument(
        "--right-input-backend",
        type=str,
        default="openhaptics",
        choices=LIVE_INPUT_BACKENDS,
        help="Live input backend for the right controller",
    )
    parser.add_argument(
        "--left-input-backend",
        type=str,
        default="openhaptics",
        choices=LIVE_INPUT_BACKENDS,
        help="Live input backend for the left controller",
    )
    parser.add_argument(
        "--right-device-name",
        type=str,
        default="Default Device",
        help="OpenHaptics device name for the right controller",
    )
    parser.add_argument(
        "--left-device-name",
        type=str,
        default="Left Device",
        help="OpenHaptics device name for the left controller",
    )
    parser.add_argument(
        "--right-device-index",
        type=int,
        default=0,
        help="Device index for backends that enumerate identical devices such as MiniMou",
    )
    parser.add_argument(
        "--left-device-index",
        type=int,
        default=1,
        help="Device index for backends that enumerate identical devices such as MiniMou",
    )
    parser.add_argument(
        "--follou-root",
        type=str,
        default=DEFAULT_FOLLOU_ROOT,
        help="Root folder of the local python_device_manager / Follou SDK checkout",
    )
    parser.add_argument(
        "--asset",
        type=str,
        default="liver",
        help="Single-asset mesh name used when --scene=single",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=0,
        help="Max frames to run (0 = unlimited)",
    )
    parser.add_argument("--no-vsync", action="store_true", help="Disable vsync for profiling")
    parser.add_argument(
        "--no-textures",
        action="store_true",
        help="Start with organ textures disabled",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["quality", "balanced", "performance"],
        help="Simulation preset (overrides default substeps/fps)",
    )
    return parser.parse_args()


def _device_name_candidates(device_name: str) -> tuple[str, ...]:
    aliases = {
        "Device Left": "Left Device",
        "Left Device": "Device Left",
    }
    candidates = [device_name]
    alias = aliases.get(device_name)
    if alias is not None and alias not in candidates:
        candidates.append(alias)
    return tuple(candidates)


def _build_openhaptics_source(controller_id: str, requested_name: str):
    from omnisurg.haptics import LiveHapticSource

    last_exc = None
    for candidate_name in _device_name_candidates(requested_name):
        try:
            source = LiveHapticSource(device_name=candidate_name)
            if candidate_name != requested_name:
                print(
                    f'{controller_id} device name fallback: using "{candidate_name}" instead of "{requested_name}"'
                )
            return source, candidate_name
        except Exception as exc:
            last_exc = exc

    assert last_exc is not None
    raise last_exc


class _SampleTransformSource:
    def __init__(self, source, *, position_transform=None, rotation_transform=None):
        self._source = source
        self._position_transform = position_transform
        self._rotation_transform = rotation_transform

    def __getattr__(self, name: str):
        return getattr(self._source, name)

    def poll(self):
        sample = self._source.poll()
        if not sample:
            return sample

        transformed = dict(sample)
        position = transformed.get("position")
        if position is not None and self._position_transform is not None:
            transformed["position"] = self._position_transform(np.asarray(position, dtype=np.float32))

        rotation = transformed.get("rotation")
        if rotation is not None and self._rotation_transform is not None:
            transformed["rotation"] = self._rotation_transform(np.asarray(rotation, dtype=np.float32))

        return transformed

    def close(self):
        self._source.close()


def _reflect_minimou_quaternion_x(rotation: np.ndarray) -> np.ndarray:
    corrected = np.asarray(rotation, dtype=np.float32).copy()
    if corrected.shape[0] >= 1:
        corrected[0] *= -1.0
    return corrected


def _build_minimou_source(device_index: int, follou_root: str):
    from omnisurg.haptics import LiveMiniMouSource

    source = LiveMiniMouSource(root=follou_root, device_index=device_index)
    source = _SampleTransformSource(source, rotation_transform=_reflect_minimou_quaternion_x)
    descriptor = f"MiniMou[{device_index}]"
    controller = getattr(getattr(source, "_ctrl", None), "_controller", None)
    device_id = getattr(controller, "device_id", None)
    if device_id is not None:
        descriptor = f"{descriptor}/id={device_id}"
    return source, descriptor


def _build_live_input_rig(args):
    from omnisurg.haptics import MultiSourceRig

    sources = {}
    active_names = {}
    failures = []
    for controller_id in ("right", "left"):
        backend = getattr(args, f"{controller_id}_input_backend")
        if backend == "none":
            continue

        try:
            if backend == "openhaptics":
                requested_name = getattr(args, f"{controller_id}_device_name")
                source, descriptor = _build_openhaptics_source(controller_id, requested_name)
            elif backend == "minimou":
                device_index = getattr(args, f"{controller_id}_device_index")
                source, descriptor = _build_minimou_source(device_index, args.follou_root)
            else:
                raise RuntimeError(f"Unsupported input backend: {backend}")
        except Exception as exc:
            if backend == "openhaptics":
                requested_name = getattr(args, f"{controller_id}_device_name")
                failures.append(f"{controller_id} ({backend}:{requested_name}): {exc}")
            elif backend == "minimou":
                device_index = getattr(args, f"{controller_id}_device_index")
                failures.append(f"{controller_id} ({backend}#{device_index}): {exc}")
            else:
                failures.append(f"{controller_id} ({backend}): {exc}")
            continue

        sources[controller_id] = source
        active_names[controller_id] = f"{backend}:{descriptor}"

    if failures:
        print("Unavailable live devices: " + "; ".join(failures))

    if not sources:
        return None

    print(
        "Using live input devices: "
        + ", ".join(f"{controller_id}={descriptor}" for controller_id, descriptor in active_names.items())
    )
    return MultiSourceRig(sources)


def _dispatch_haptic_force_commands(input_rig, force_commands):
    if input_rig is None:
        return

    dispatch = getattr(input_rig, "set_force_commands", None)
    if callable(dispatch):
        dispatch(force_commands)
        return

    set_force = getattr(input_rig, "set_force", None)
    if callable(set_force):
        right_force = force_commands.get("right", np.zeros(3, dtype=np.float32))
        set_force(right_force)


def _zero_haptic_force_commands(input_rig):
    zero = np.zeros(3, dtype=np.float32)
    _dispatch_haptic_force_commands(
        input_rig,
        {
            "right": zero,
            "left": zero.copy(),
        },
    )


def main():
    args = parse_args()

    print(f"Python {sys.version}")
    print(
        f"OmniSurg Phase 2 - scene={args.scene}, asset={args.asset}, viewer={args.viewer}, "
        f"textures={'off' if args.no_textures else 'on'}"
    )

    from omnisurg.config import BoundsConfig, HapticConfig, SceneConfig, SimulationConfig, SIMULATION_PRESETS, ViewerConfig
    from omnisurg.haptics import BimanualReplayRig
    from omnisurg.runtime import Runtime

    if args.preset:
        sim_config = SIMULATION_PRESETS[args.preset]
    else:
        sim_config = SimulationConfig()
    scene_config = SceneConfig(scene_preset=args.scene, asset_name=args.asset)
    haptic_config = HapticConfig()
    viewer_config = ViewerConfig(
        backend=args.viewer,
        vsync=not args.no_vsync,
        textures_enabled=not args.no_textures,
    )
    bounds_config = BoundsConfig()

    input_rig = None
    if args.replay or args.left_replay:
        input_rig = BimanualReplayRig(right_path=args.replay, left_path=args.left_replay)
        replay_desc = []
        if args.replay:
            replay_desc.append(f"right={args.replay}")
        if args.left_replay:
            replay_desc.append(f"left={args.left_replay}")
        print("Using replay input: " + ", ".join(replay_desc))
    elif args.viewer == "headless":
        print("Headless mode: running without live haptic input")
    else:
        input_rig = _build_live_input_rig(args)
        if input_rig is None:
            print("Haptic devices not available, running without input")

    if args.viewer in {"gl", "surgsim"}:
        print("Press T in the viewer to toggle textures")

    with wp.ScopedDevice(args.device):
        rt = Runtime(sim_config, scene_config, haptic_config, viewer_config, bounds_config)

        frame = 0
        while rt.is_running():
            if input_rig:
                rt.poll_input(input_rig)
            rt.step()
            if input_rig:
                _dispatch_haptic_force_commands(input_rig, rt.get_haptic_force_commands())
            rt.render()
            rt.pace()

            frame += 1
            if args.num_frames > 0 and frame >= args.num_frames:
                break

        if input_rig:
            _zero_haptic_force_commands(input_rig)
            input_rig.close()
        rt.close()


if __name__ == "__main__":
    main()
