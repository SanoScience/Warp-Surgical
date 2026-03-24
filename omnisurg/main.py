import argparse
import sys

import warp as wp


def parse_args():
    parser = argparse.ArgumentParser(
        description="OmniSurg Phase 1 — minimal soft-body simulation with haptic input",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Warp device override",
    )
    parser.add_argument(
        "--viewer",
        type=str,
        default="gl",
        choices=["surgsim", "gl", "rtx"],
        help="Viewer backend",
    )
    parser.add_argument(
        "--replay", type=str, default=None,
        help="Path to .npy haptic replay trace (N,7) for deterministic testing",
    )
    parser.add_argument(
        "--asset", type=str, default="liver",
        help="Mesh asset name (subdirectory of meshes/)",
    )
    parser.add_argument(
        "--num_frames", type=int, default=0,
        help="Max frames to run (0 = unlimited)",
    )
    parser.add_argument(
        "--no-vsync", action="store_true",
        help="Disable vsync for profiling",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["quality", "balanced", "performance"],
        help="Simulation preset (overrides default substeps/fps)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Python {sys.version}")
    print(f"OmniSurg Phase 1 — asset={args.asset}, viewer={args.viewer}")

    from omnisurg.config import (
        BoundsConfig,
        HapticConfig,
        SceneConfig,
        SimulationConfig,
        SIMULATION_PRESETS,
        ViewerConfig,
    )
    from omnisurg.runtime import Runtime

    if args.preset:
        sim_config = SIMULATION_PRESETS[args.preset]
    else:
        sim_config = SimulationConfig()
    scene_config = SceneConfig(asset_name=args.asset)
    haptic_config = HapticConfig()
    viewer_config = ViewerConfig(backend=args.viewer, vsync=not args.no_vsync)
    bounds_config = BoundsConfig()

    source = None
    if args.replay:
        from omnisurg.haptics import ReplayInputSource

        source = ReplayInputSource(args.replay)
        print(f"Using replay input: {args.replay}")
    else:
        try:
            from omnisurg.haptics import LiveHapticSource

            source = LiveHapticSource()
            print("Using live haptic device")
        except (ImportError, OSError) as e:
            print(f"Haptic device not available ({e}), running without input")

    with wp.ScopedDevice(args.device):
        rt = Runtime(sim_config, scene_config, haptic_config, viewer_config, bounds_config)

        frame = 0
        while rt.is_running():
            if source:
                rt.poll_input(source)
            rt.step()
            rt.render()
            rt.pace()

            frame += 1
            if args.num_frames > 0 and frame >= args.num_frames:
                break

        if source:
            source.close()
        rt.close()


if __name__ == "__main__":
    main()
