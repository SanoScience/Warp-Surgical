import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import warp as wp

from haptic_device import HapticController
from mesh_loader import WarpMeshConfig
from warp_simulation import WarpSim

_CRITERIA_ARG_MAP = {
    "mesh_facet_angle": "facet_angle",
    "mesh_facet_size": "facet_size",
    "mesh_facet_distance": "facet_distance",
    "mesh_cell_radius_edge_ratio": "cell_radius_edge_ratio",
    "mesh_cell_size": "cell_size",
}


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

    parser.add_argument("--warp-mesh-path", type=str, default=None,
                        help="Path to a WarpSim-format mesh directory to load at startup.")
    parser.add_argument("--warp-mesh-multi", action="store_true",
                        help="Treat the warp mesh path as a base name for multi-label folders.")
    parser.add_argument("--warp-cgal-root", type=str, default=None,
                        help="Path to the warp-cgal repository root (defaults to ../warp-cgal).")
    parser.add_argument("--subdomain-map", type=str, default=None,
                        help="JSON object or path mapping subdomain names to label IDs (e.g. '{\"liver\":255}').")
    parser.add_argument("--remesh-image", type=str, default=None,
                        help="Segmented image to remesh before simulation using warp-cgal.")
    parser.add_argument("--remesh-output", type=str, default=None,
                        help="Directory to store the remeshed warp-format output.")
    parser.add_argument("--mesh-facet-angle", type=float, default=None,
                        help="CGAL facet angle criteria when remeshing from images.")
    parser.add_argument("--mesh-facet-size", type=float, default=None,
                        help="CGAL facet size criteria when remeshing from images.")
    parser.add_argument("--mesh-facet-distance", type=float, default=None,
                        help="CGAL facet distance criteria when remeshing from images.")
    parser.add_argument("--mesh-cell-radius-edge-ratio", type=float, default=None,
                        help="CGAL cell radius-edge ratio when remeshing from images.")
    parser.add_argument("--mesh-cell-size", type=float, default=None,
                        help="CGAL cell size when remeshing from images.")

    return parser.parse_known_args()[0]


def _parse_subdomain_map(map_arg: str) -> Dict[str, int]:
    candidate = Path(map_arg)
    try:
        payload = candidate.read_text(encoding="utf-8") if candidate.exists() else map_arg
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse subdomain map JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Subdomain map must be a JSON object mapping names to integers")

    result: Dict[str, int] = {}
    for key, value in data.items():
        try:
            result[str(key)] = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid label value for subdomain '{key}': {value}") from exc
    return result


def build_warp_mesh_config(args) -> Optional[WarpMeshConfig]:
    criteria: Dict[str, float] = {}
    for arg_name, criteria_key in _CRITERIA_ARG_MAP.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            criteria[criteria_key] = float(value)

    subdomain_map: Dict[str, int] = {}
    if args.subdomain_map:
        subdomain_map = _parse_subdomain_map(args.subdomain_map)

    config = WarpMeshConfig(
        mesh_path=args.warp_mesh_path,
        multi_label=args.warp_mesh_multi,
        remesh_image=args.remesh_image,
        remesh_output=args.remesh_output,
        cgal_root=args.warp_cgal_root,
        criteria=criteria,
        subdomain_map=subdomain_map,
    )
    return config if config.is_active() else None


def run_simulation(args):
    """Run the main simulation loop."""

    print("Python version:", sys.version)

    try:
        warp_mesh_config = build_warp_mesh_config(args)
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        sys.exit(2)

    # Initialize haptic controller
    haptic_controller = HapticController(scale=1.0)

    # Initialize simulation
    sim = WarpSim(
        stage_path=args.stage_path,
        num_frames=args.num_frames,
        use_opengl=not args.usd,
        warp_mesh_config=warp_mesh_config,
    )

    if args.usd:
        # Offline rendering mode
        for _ in range(args.num_frames):
            # Update haptic position and rotation
            haptic_pos = haptic_controller.get_scaled_position()
            haptic_rot = haptic_controller.get_rotation()
            sim.update_haptic_position(haptic_pos)
            sim.update_haptic_rotation(haptic_rot)

            sim.step()
            sim.render()
    else:
        # Real-time interactive mode
        while sim.is_running():
            # Get current haptic device position and rotation
            haptic_pos = haptic_controller.get_scaled_position()
            haptic_rot = haptic_controller.get_rotation()
            sim.update_haptic_position(haptic_pos)
            sim.update_haptic_rotation(haptic_rot)

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
