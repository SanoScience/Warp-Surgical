"""
Main entry point for the surgical simulation.

This module provides the command-line interface and main simulation loop,
following Newton's Example class pattern for organization.
"""

import argparse
import sys
import warp as wp

from config import SimulationConfig, create_default_config, create_fast_config
from haptic_device import HapticController
from warp_simulation import WarpSim


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Surgical simulation using Warp physics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Device and output settings
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the default Warp device."
    )
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="output.usd",
        help="Path to the output USD file."
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=300,
        help="Total number of frames for offline rendering."
    )
    parser.add_argument(
        "--usd",
        action="store_true",
        help="Render to USD instead of OpenGL."
    )

    # Simulation quality settings
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast simulation settings (fewer substeps)."
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=None,
        help="Override number of simulation substeps."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override target FPS."
    )

    return parser.parse_known_args()[0]


def create_config_from_args(args) -> SimulationConfig:
    """Create simulation config from command line arguments."""
    if args.fast:
        config = create_fast_config()
    else:
        config = create_default_config()

    # Override specific values if provided
    if args.substeps is not None:
        config.substeps = args.substeps
    if args.fps is not None:
        config.fps = args.fps

    return config


class SurgicalSimulation:
    """Main simulation class following Newton's Example pattern.

    This class coordinates the simulation components:
    - Physics simulation (WarpSim)
    - Haptic input (HapticController)
    - Rendering (OpenGL or USD)

    Args:
        config: Simulation configuration.
        stage_path: Path for USD output.
        num_frames: Number of frames for offline mode.
        use_opengl: Use OpenGL (True) or USD (False) renderer.
    """

    def __init__(
        self,
        config: SimulationConfig,
        stage_path: str = "output.usd",
        num_frames: int = 300,
        use_opengl: bool = True
    ):
        self.config = config
        self.num_frames = num_frames
        self.use_opengl = use_opengl

        print("Python version:", sys.version)
        print(f"Simulation config: {config.substeps} substeps at {config.fps} FPS")

        # Initialize haptic controller
        self.haptic = HapticController(scale=1.0)

        # Initialize simulation with config
        self.sim = WarpSim(
            stage_path=stage_path,
            num_frames=num_frames,
            use_opengl=use_opengl,
            config=config
        )

    def update_haptic(self):
        """Update haptic device state."""
        haptic_pos = self.haptic.get_scaled_position()
        haptic_rot = self.haptic.get_rotation()
        self.sim.update_haptic_position(haptic_pos)
        self.sim.update_haptic_rotation(haptic_rot)

    def step(self):
        """Advance simulation by one frame."""
        self.update_haptic()
        self.sim.step()

    def render(self):
        """Render current simulation state."""
        self.sim.render()

    def is_running(self) -> bool:
        """Check if simulation should continue."""
        return self.sim.is_running()

    def save(self):
        """Save simulation results."""
        self.sim.save()

    def run_offline(self):
        """Run offline rendering mode (USD output)."""
        print(f"Running offline simulation for {self.num_frames} frames...")
        for frame in range(self.num_frames):
            self.step()
            self.render()
            if frame % 100 == 0:
                print(f"Frame {frame}/{self.num_frames}")
        self.save()
        print("Offline rendering complete.")

    def run_interactive(self):
        """Run interactive mode (real-time OpenGL)."""
        print("Running interactive simulation...")
        while self.is_running():
            self.step()
            self.render()
        self.save()
        print("Simulation ended.")


def run_simulation(args):
    """Run the main simulation loop."""
    # Create config from arguments
    config = create_config_from_args(args)

    # Create and run simulation
    simulation = SurgicalSimulation(
        config=config,
        stage_path=args.stage_path,
        num_frames=args.num_frames,
        use_opengl=not args.usd
    )

    if args.usd:
        simulation.run_offline()
    else:
        simulation.run_interactive()


def main():
    """Main entry point."""
    args = parse_arguments()

    with wp.ScopedDevice(args.device):
        run_simulation(args)


if __name__ == "__main__":
    main()
