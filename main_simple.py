import argparse
from pathlib import Path
import warp as wp

#from haptic_device import HapticController
from warp_simulation_simple import WarpSim

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
    
    # Mesh generation arguments
    parser.add_argument("--generate-meshes", action="store_true", 
                       help="Generate tetrahedral meshes from source images automatically")
    parser.add_argument("--mesh-source", type=str, 
                       default="G:/warp/warp-surgical/liver-sliced.inr",
                       help="Source image path for mesh generation")
    parser.add_argument("--mesh-quality", choices=["fast", "low", "balanced", "high", "very_high", "ultra"], 
                       default="balanced", help="Mesh quality preset")
    parser.add_argument("--cell-size", type=float, 
                       help="Override cell size parameter for mesh generation")
    parser.add_argument("--force-regenerate", action="store_true", 
                       help="Force mesh regeneration even if mesh files exist")
    parser.add_argument("--multi-label", action="store_true", 
                       help="Generate separate meshes for each segmentation class")
    parser.add_argument("--detect-labels", action="store_true", 
                       help="Detect and display available labels in the source image")
    
    parser.add_argument("--warp-mesh-path", type=lambda x: None if x == "None" else str(x),
                       default=None, help="Optional path to a Warp mesh folder or base name")

    return parser.parse_known_args()[0]

def detect_and_display_labels(args):
    """Detect and display available labels in the source image."""
    print("=" * 60)
    print("Detecting labels in source image...")
    
    try:
        from mesh_generation_simple import SimpleMeshGenerator
        
        generator = SimpleMeshGenerator()
        if not generator.warp_cgal_available:
            print("- warp-cgal not available for label detection")
            return False
        
        source_path = getattr(args, 'mesh_source', 'G:/warp/warp-surgical/liver-sliced.inr')
        labels = generator.detect_labels(source_path)
        
        if labels:
            print(f"+ Found {len(labels)} labels: {labels}")
            print("+ You can use --multi-label to generate separate meshes for each label")
            return True
        else:
            print("- No labels detected in the image")
            return False
            
    except Exception as e:
        print(f"- Error during label detection: {e}")
        return False

def run_simulation(args):
    """Run the main simulation loop."""
    mesh_path = getattr(args, "warp_mesh_path", None)

    if args.generate_meshes:
        from mesh_generation_simple import ensure_meshes_ready

        print("Checking/generating tetrahedral meshes...")
        generated_mesh_path = ensure_meshes_ready(args)
        if generated_mesh_path:
            mesh_path = generated_mesh_path
            print("Mesh generation/validation complete!")
        else:
            print("Failed to ensure meshes are ready. Attempting to continue with existing meshes...")

    if mesh_path:
        resolved_mesh_path = Path(mesh_path)
        if resolved_mesh_path.exists():
            print(f"Using Warp mesh assets from: {resolved_mesh_path}")
            mesh_path = str(resolved_mesh_path)
        else:
            print(f"Warning: Warp mesh path not found: {resolved_mesh_path}. Falling back to bundled meshes.")
            mesh_path = None

    # Initialize haptic controller
    #haptic_controller = HapticController(scale=1.0)

    # Initialize simulation
    sim = WarpSim(
        stage_path=args.stage_path,
        num_frames=args.num_frames,
        use_opengl=not args.usd,
        mesh_path=mesh_path,
    )

    if args.usd:
        # Offline rendering mode
        for _ in range(args.num_frames):
            # Update haptic position
            #haptic_pos = haptic_controller.get_scaled_position()
            #sim.update_haptic_position(haptic_pos)

            sim.step()
            sim.render()
    else:
        # Real-time interactive mode
        while sim.is_running():
            # Get current haptic device position
            #haptic_pos = haptic_controller.get_scaled_position()
            #sim.update_haptic_position(haptic_pos)

            # Advance simulation
            sim.step()
            sim.render()

    # Save results
    #sim.save()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Handle label detection mode
    if args.detect_labels:
        detect_and_display_labels(args)
        return
    
    with wp.ScopedDevice(args.device):
        run_simulation(args)

if __name__ == "__main__":
    main()
