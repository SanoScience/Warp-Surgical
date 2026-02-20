import argparse
import sys


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="output.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument("--usd", action="store_true", help="Render to USD instead of OpenGL.")
    parser.add_argument("--ovrtx", action="store_true", help="Render with ovrtx RTX ray tracing.")
    parser.add_argument("--isaacsim", action="store_true", help="Render with Isaac Sim RTX renderer.")
    parser.add_argument("--isaacsim_headless", action="store_true", help="Run Isaac Sim in headless mode.")
    parser.add_argument("--isaacsim_renderer", type=str, default="RayTracedLighting",
                        help="Isaac Sim renderer: RayTracedLighting or PathTracing.")

    return parser.parse_known_args()[0]


def run_simulation(args, simulation_app=None):
    """Run the main simulation loop."""
    # Imports are deferred so that when --isaacsim is used, SimulationApp is
    # already running before any pxr / Omniverse module is loaded.
    import warp as wp
    from warp_simulation import WarpSim

    print("Python version:", sys.version)

    use_isaacsim = args.isaacsim
    use_opengl = not args.usd and not args.ovrtx and not use_isaacsim

    # Initialize simulation
    sim = WarpSim(
        stage_path=args.stage_path,
        num_frames=args.num_frames,
        use_opengl=use_opengl,
        use_ovrtx=args.ovrtx,
        use_isaacsim=use_isaacsim,
        simulation_app=simulation_app,
    )

    if args.usd:
        # Offline rendering mode
        for _ in range(args.num_frames):
            haptic_pos = [0.0, 0.0, 0.0]
            haptic_rot = [0.0, 0.0, 0.0, 1.0]
            sim.update_haptic_position(haptic_pos)
            sim.update_haptic_rotation(haptic_rot)

            sim.step()
            sim.render()
    else:
        # Real-time interactive mode
        while sim.is_running():
            haptic_pos = [0.0, 0.0, 0.0]
            haptic_rot = [0.0, 0.0, 0.0, 1.0]
            sim.update_haptic_position(haptic_pos)
            sim.update_haptic_rotation(haptic_rot)

            sim.step()
            sim.render()

    # Save results
    sim.save()


def main():
    """Main entry point."""
    args = parse_arguments()

    simulation_app = None

    if args.isaacsim:
        # SimulationApp MUST be created before importing warp / pxr modules.
        from isaacsim import SimulationApp

        config = {
            "headless": args.isaacsim_headless,
            "renderer": args.isaacsim_renderer,
        }
        simulation_app = SimulationApp(config)

    import warp as wp

    with wp.ScopedDevice(args.device):
        run_simulation(args, simulation_app=simulation_app)

if __name__ == "__main__":
    main()
