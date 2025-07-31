import argparse
import warp as wp

from haptic_device import HapticController
from warp_simulation import WarpSim

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
    
    return parser.parse_known_args()[0]

def run_simulation(args):
    """Run the main simulation loop."""
    # Initialize haptic controller
    haptic_controller = HapticController(scale=1.0)
    
    # Initialize simulation
    sim = WarpSim(
        stage_path=args.stage_path, 
        num_frames=args.num_frames, 
        use_opengl=not args.usd
    )
    
    if args.usd:
        # Offline rendering mode
        for _ in range(args.num_frames):
            # Update haptic position
            haptic_pos = haptic_controller.get_scaled_position()
            sim.update_haptic_position(haptic_pos)
            
            sim.step()
            sim.render()
    else:
        # Real-time interactive mode
        while sim.is_running():
            # Get current haptic device position
            haptic_pos = haptic_controller.get_scaled_position()
            sim.update_haptic_position(haptic_pos)
            
            # Advance simulation
            sim.step()
            sim.render()
    
    # Save results
    sim.save()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    with wp.ScopedDevice(args.device):
        run_simulation(args)

if __name__ == "__main__":
    main()
