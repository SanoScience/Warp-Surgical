#!/usr/bin/env python3
"""
CGAL Mesh Generation Demo for WarpSim

This script demonstrates tetrahedral mesh generation from segmented images using CGAL,
following the mesh_3D_weighted_image.cpp example. It generates meshes and optionally
visualizes them in the WarpSim physics simulation environment.

Usage:
    python cgal_mesh_demo.py --input segmented.inr --output liver_cgal --visualize
    python cgal_mesh_demo.py --create-test-volume --output test_mesh --visualize
    python cgal_mesh_demo.py --simple-mesh --output simple --visualize
"""

import os
import argparse
import sys
from typing import Optional

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgal_mesh_generator import CgalMeshGenerator, create_sample_segmented_image, CGAL_AVAILABLE, CGAL_SOURCE
from cgal_to_warp import CgalToWarpConverter, convert_numpy_volume_to_warp_mesh, create_simple_test_mesh
from simple_volume_mesher import create_mesh_from_inr_file


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CGAL Tetrahedral Mesh Generation Demo for WarpSim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", 
        type=str,
        help="Path to input segmented image (INR, DICOM, etc.)"
    )
    input_group.add_argument(
        "--create-test-volume",
        action="store_true",
        help="Create a test segmented volume for demonstration"
    )
    input_group.add_argument(
        "--simple-mesh",
        action="store_true",
        help="Create a simple tetrahedral mesh for testing"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="cgal_generated",
        help="Output directory/prefix for generated mesh files"
    )
    
    # Mesh generation parameters
    parser.add_argument(
        "--facet-angle",
        type=float,
        default=30.0,
        help="Minimum facet angle in degrees"
    )
    parser.add_argument(
        "--facet-size",
        type=float,
        default=6.0,
        help="Maximum facet size"
    )
    parser.add_argument(
        "--facet-distance",
        type=float,
        default=0.5,
        help="Maximum distance between facet and domain boundary"
    )
    parser.add_argument(
        "--cell-radius-edge-ratio",
        type=float,
        default=3.0,
        help="Maximum radius-edge ratio for cells"
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=8.0,
        help="Maximum cell size"
    )
    
    # Visualization options
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Launch WarpSim visualization after mesh generation"
    )
    parser.add_argument(
        "--no-haptic",
        action="store_true",
        help="Disable haptic device for visualization"
    )
    
    # Advanced options
    parser.add_argument(
        "--sigma",
        type=float,
        help="Smoothing parameter for label weights (auto-computed if not specified)"
    )
    parser.add_argument(
        "--relative-error-bound",
        type=float,
        default=1e-6,
        help="Relative error bound for domain creation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def print_system_info():
    """Print information about available CGAL bindings."""
    print("=== CGAL Mesh Generation Demo ===")
    print(f"CGAL Available: {CGAL_AVAILABLE}")
    if CGAL_AVAILABLE:
        print(f"CGAL Source: {CGAL_SOURCE}")
    else:
        print("Warning: No CGAL bindings found. Install with: pip install cgal")
    print()


def generate_mesh_from_image(args) -> bool:
    """Generate mesh from segmented image using CGAL or fallback methods."""
    try:
        # Check if CGAL is available
        if not CGAL_AVAILABLE:
            print("CGAL bindings not available. Trying simple volume mesher fallback...")
            return generate_mesh_from_image_fallback(args)
        
        print(f"Loading segmented image: {args.input}")
        
        # Create mesh generator
        generator = CgalMeshGenerator()
        
        # Load image
        if not generator.load_segmented_image(args.input):
            print(f"Error: Failed to load image {args.input}")
            return False
        
        print("Image loaded successfully")
        
        # Generate label weights
        print("Generating label weights...")
        if not generator.generate_label_weights(args.sigma):
            print("Warning: Failed to generate label weights, proceeding without them")
        
        # Create mesh domain
        print("Creating mesh domain...")
        if not generator.create_mesh_domain(args.relative_error_bound):
            print("Warning: CGAL mesh domain creation failed, falling back to simple volume mesher")
            return generate_mesh_from_image_fallback(args)
        
        # Set mesh criteria
        criteria = {
            'facet_angle': args.facet_angle,
            'facet_size': args.facet_size,
            'facet_distance': args.facet_distance,
            'cell_radius_edge_ratio': args.cell_radius_edge_ratio,
            'cell_size': args.cell_size
        }
        generator.set_mesh_criteria(criteria)
        
        if args.verbose:
            print(f"Mesh criteria: {criteria}")
        
        # Generate mesh
        print("Generating tetrahedral mesh... (this may take a while)")
        if not generator.generate_mesh():
            print("Warning: CGAL mesh generation failed, falling back to simple volume mesher")
            return generate_mesh_from_image_fallback(args)
        
        # Get mesh statistics
        stats = generator.get_mesh_statistics()
        print(f"Mesh generated successfully!")
        print(f"Statistics: {stats}")
        
        # Convert to WarpSim format
        print("Converting to WarpSim format...")
        converter = CgalToWarpConverter()
        output_dir = f"meshes/{args.output}"
        
        if not converter.convert_cgal_mesh(generator.generated_mesh, output_dir, "model"):
            print("Error: Failed to convert mesh to WarpSim format")
            return False
        
        # Print conversion info
        mesh_info = converter.get_mesh_info()
        print(f"Conversion complete: {mesh_info}")
        
        # Export original CGAL formats
        print("Exporting CGAL formats...")
        generator.export_mesh(os.path.join(output_dir, "cgal_mesh"))
        
        print(f"Mesh saved to {output_dir}")
        return True
        
    except Exception as e:
        print(f"Error generating mesh from image: {e}")
        return False


def generate_mesh_from_image_fallback(args) -> bool:
    """Generate mesh using simple volume mesher fallback."""
    try:
        print("Using simple volume mesher (marching cubes + Delaunay triangulation)")
        
        output_dir = f"meshes/{args.output}"
        
        # Use simple volume mesher for INR files
        if args.input.endswith('.inr') or args.input.endswith('.inr.gz'):
            # Extract mesh for label 1 (you can modify this)
            label = 1
            print(f"Extracting mesh for label {label}")
            
            success = create_mesh_from_inr_file(args.input, output_dir, label=label)
            if success:
                print(f"Mesh generated successfully using simple volume mesher")
                return True
            else:
                print("Simple volume mesher failed")
                return False
        else:
            print("Simple volume mesher currently only supports INR files")
            return False
            
    except Exception as e:
        print(f"Error in fallback mesh generation: {e}")
        return False


def generate_test_volume_mesh(args) -> bool:
    """Generate mesh from a test volume."""
    try:
        print("Creating test segmented volume...")
        
        # Create test volume
        test_volume_path = f"test_volume_{args.output}.npy"
        if not create_sample_segmented_image(test_volume_path, size=32):
            print("Error: Failed to create test volume")
            return False
        
        print(f"Test volume created: {test_volume_path}")
        
        if CGAL_AVAILABLE:
            # Use CGAL to generate mesh from the test volume
            import numpy as np
            volume_data = np.load(test_volume_path)
            
            output_dir = f"meshes/{args.output}"
            criteria = {
                'facet_angle': args.facet_angle,
                'facet_size': args.facet_size,
                'facet_distance': args.facet_distance,
                'cell_radius_edge_ratio': args.cell_radius_edge_ratio,
                'cell_size': args.cell_size
            }
            
            print("Generating mesh from test volume using CGAL...")
            if not convert_numpy_volume_to_warp_mesh(volume_data, output_dir, "model", criteria):
                print("Error: Failed to generate mesh from test volume")
                return False
            
            print(f"Test volume mesh generated successfully in {output_dir}")
        else:
            print("Warning: CGAL not available, will create simple test mesh instead")
            return generate_simple_mesh(args)
        
        return True
        
    except Exception as e:
        print(f"Error generating test volume mesh: {e}")
        return False


def generate_simple_mesh(args) -> bool:
    """Generate a simple tetrahedral mesh for testing."""
    try:
        print("Creating simple tetrahedral mesh...")
        
        output_dir = f"meshes/{args.output}"
        if not create_simple_test_mesh(output_dir, "model"):
            print("Error: Failed to create simple test mesh")
            return False
        
        print(f"Simple test mesh created in {output_dir}")
        return True
        
    except Exception as e:
        print(f"Error generating simple mesh: {e}")
        return False


def visualize_mesh(args, mesh_directory: str) -> bool:
    """Launch WarpSim to visualize the generated mesh."""
    try:
        print(f"Launching WarpSim visualization for mesh in {mesh_directory}")
        
        # Check if the mesh files exist
        model_files = ["model.vertices", "model.tetras", "model.tris", "model.edges", "model.uvs"]
        for file_name in model_files:
            file_path = os.path.join(mesh_directory, file_name)
            if not os.path.exists(file_path):
                print(f"Warning: Missing mesh file {file_path}")
        
        # Create a custom version of WarpSim that uses our generated mesh
        print("Creating custom WarpSim configuration...")
        
        # Create a temporary modified mesh_loader that includes our mesh
        create_custom_warp_simulation(mesh_directory, args.output)
        
        if not args.no_haptic:
            print("Starting WarpSim with haptic support...")
            print("Note: Make sure haptic device is connected")
        else:
            print("Starting WarpSim without haptic support...")
        
        # Launch WarpSim
        import subprocess
        cmd = ["python", "main.py"]
        if args.no_haptic:
            os.environ["DISABLE_HAPTIC"] = "1"
        
        subprocess.run(cmd, cwd=os.getcwd())
        return True
        
    except Exception as e:
        print(f"Error launching visualization: {e}")
        return False


def create_custom_warp_simulation(mesh_directory: str, mesh_name: str) -> None:
    """Create a custom simulation script that loads our generated mesh."""
    try:
        # Create a demo script that uses our mesh
        demo_script = f"""#!/usr/bin/env python3
'''
Custom WarpSim demo for CGAL-generated mesh: {mesh_name}
'''

import warp as wp
import sys
import os

# Add the mesh directory to be found by mesh_loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from warp_simulation import WarpSim
from haptic_device import HapticController

def run_cgal_mesh_demo():
    '''Run WarpSim with CGAL-generated mesh.'''
    
    print("=== CGAL Mesh WarpSim Demo ===")
    print(f"Mesh: {mesh_name}")
    print(f"Location: {mesh_directory}")
    print()
    
    # Initialize haptic controller (if available)
    try:
        haptic_controller = HapticController(scale=1.0)
        print("Haptic device initialized")
    except Exception as e:
        print(f"Warning: Could not initialize haptic device: {{e}}")
        haptic_controller = None
    
    # Initialize simulation
    sim = WarpSim(use_opengl=True)
    
    print("Starting interactive simulation...")
    print("Controls:")
    print("  C - Toggle cutting")
    print("  V - Toggle heating")
    print("  B - Toggle grasping")
    print("  G - Toggle clipping")
    print("  Y - Toggle volume constraints")
    print()
    
    # Main simulation loop
    while sim.is_running():
        if haptic_controller:
            # Get current haptic device position
            haptic_pos = haptic_controller.get_scaled_position()
            sim.update_haptic_position(haptic_pos)
        
        # Advance simulation
        sim.step()
        sim.render()
    
    # Save results
    sim.save()
    print("Demo completed")

if __name__ == "__main__":
    run_cgal_mesh_demo()
"""
        
        # Save the demo script
        demo_file = f"cgal_demo_{mesh_name}.py"
        with open(demo_file, 'w') as f:
            f.write(demo_script)
        
        print(f"Custom demo script created: {demo_file}")
        print(f"You can run it with: python {demo_file}")
        
    except Exception as e:
        print(f"Error creating custom simulation: {e}")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    print_system_info()
    
    # Generate mesh based on selected option
    success = False
    output_dir = None
    
    if args.input:
        success = generate_mesh_from_image(args)
        output_dir = f"meshes/{args.output}"
    elif args.create_test_volume:
        success = generate_test_volume_mesh(args)
        output_dir = f"meshes/{args.output}"
    elif args.simple_mesh:
        success = generate_simple_mesh(args)
        output_dir = f"meshes/{args.output}"
    
    if not success:
        print("Mesh generation failed")
        sys.exit(1)
    
    print("\n=== Mesh Generation Complete ===")
    
    # Optional visualization
    if args.visualize and output_dir:
        print("\nLaunching visualization...")
        visualize_mesh(args, output_dir)
    else:
        print(f"\nMesh files are available in: {output_dir}")
        print("To visualize, run with --visualize flag")
        
        # Create the custom demo script anyway
        if output_dir:
            create_custom_warp_simulation(output_dir, args.output)
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()