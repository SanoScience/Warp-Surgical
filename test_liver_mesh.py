"""
Test application for generating tetrahedral mesh from liver-sliced.inr using warp-cgal.

This script loads the liver-sliced.inr file and generates a tetrahedral mesh
using the warp-cgal native library.
"""

import os
import sys
import time

# Add warp-cgal to Python path
warp_cgal_path = r'G:\warp\warp-cgal'
sys.path.insert(0, warp_cgal_path)

def test_liver_mesh_generation():
    """Test mesh generation with liver-sliced.inr file."""
    
    # File paths
    liver_inr_path = "liver-sliced.inr"
    output_mesh_path = "liver_mesh_output.mesh"
    
    print("=== Warp-CGAL Liver Mesh Generation Test ===")
    print(f"Input file: {liver_inr_path}")
    print(f"Output file: {output_mesh_path}")
    print()
    
    # Check input file exists
    if not os.path.exists(liver_inr_path):
        print(f"❌ Error: Input file not found: {liver_inr_path}")
        return False
    
    print(f"[OK] Found input file: {liver_inr_path}")
    file_size = os.path.getsize(liver_inr_path) / (1024 * 1024)  # MB
    print(f"   File size: {file_size:.2f} MB")
    print()
    
    try:
        # Import the library
        print("[INFO] Importing warp_cgal_python...")
        from warp_cgal_python import WarpCGALMeshGenerator, generate_mesh_from_image
        print("[OK] Successfully imported warp_cgal_python")
        print()
        
        # Test 1: Basic library functionality
        print("🔧 Testing library initialization...")
        generator = WarpCGALMeshGenerator()
        print("✅ Successfully created WarpCGALMeshGenerator")
        print()
        
        # Test 2: Load image
        print("📂 Loading liver image...")
        start_time = time.time()
        
        if not generator.load_image(liver_inr_path):
            print("❌ Failed to load image")
            return False
            
        load_time = time.time() - start_time
        print(f"✅ Successfully loaded image in {load_time:.2f} seconds")
        print()
        
        # Test 3: Set mesh criteria (optimized for medical data)
        print("⚙️  Setting mesh generation criteria...")
        generator.set_criteria(
            facet_angle=25.0,        # Good surface quality
            facet_size=8.0,          # Moderate surface mesh density
            facet_distance=0.5,      # Close surface approximation
            cell_radius_edge_ratio=3.0,  # Good cell quality
            cell_size=10.0           # Reasonable volume mesh density
        )
        print("✅ Mesh criteria set")
        print("   - Facet angle: 25°")
        print("   - Facet size: 8.0")
        print("   - Facet distance: 0.5")
        print("   - Cell radius-edge ratio: 3.0")
        print("   - Cell size: 10.0")
        print()
        
        # Test 4: Generate mesh
        print("🔄 Generating tetrahedral mesh...")
        print("   (This may take several minutes for complex geometries)")
        start_time = time.time()
        
        if not generator.generate_mesh():
            print("❌ Failed to generate mesh")
            return False
            
        mesh_time = time.time() - start_time
        print(f"✅ Successfully generated mesh in {mesh_time:.2f} seconds")
        print()
        
        # Test 5: Get mesh statistics
        print("📊 Mesh statistics:")
        stats = generator.get_mesh_statistics()
        print(f"   - Vertices: {stats['vertices']:,}")
        print(f"   - Cells (tetrahedra): {stats['cells']:,}")
        print(f"   - Facets: {stats['facets']:,}")
        print()
        
        # Test 6: Export mesh
        print("💾 Exporting mesh to file...")
        start_time = time.time()
        
        if not generator.export_mesh(output_mesh_path):
            print("❌ Failed to export mesh")
            return False
            
        export_time = time.time() - start_time
        print(f"✅ Successfully exported mesh in {export_time:.2f} seconds")
        
        # Check output file
        if os.path.exists(output_mesh_path):
            output_size = os.path.getsize(output_mesh_path) / (1024 * 1024)  # MB
            print(f"   Output file size: {output_size:.2f} MB")
        print()
        
        # Summary
        total_time = load_time + mesh_time + export_time
        print("🎉 SUCCESS! Liver mesh generation completed")
        print(f"   Total processing time: {total_time:.2f} seconds")
        print(f"   Output saved to: {output_mesh_path}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print()
        print("🔧 Troubleshooting:")
        print("1. Make sure the warp-cgal project was built successfully")
        print("2. Check that warp-cgal.pyd exists in x64/Release/ directory")
        print("3. Verify all dependencies (DLLs) are available")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_convenience_function():
    """Test the convenience function for mesh generation."""
    print("\n=== Testing Convenience Function ===")
    
    try:
        from warp_cgal_python import generate_mesh_from_image
        
        print("🚀 Using convenience function...")
        start_time = time.time()
        
        stats = generate_mesh_from_image(
            "liver-sliced.inr",
            "liver_convenience_output.mesh",
            facet_angle=30.0,
            facet_size=6.0,
            cell_size=12.0
        )
        
        total_time = time.time() - start_time
        print(f"✅ Convenience function completed in {total_time:.2f} seconds")
        print(f"   Statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Convenience function failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting warp-cgal liver mesh generation test...")
    print("=" * 60)
    
    # Test the detailed approach
    success = test_liver_mesh_generation()
    
    if success:
        # Test the convenience function
        test_convenience_function()
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed successfully!")
        print()
        print("Generated files:")
        print("- liver_mesh_output.mesh (detailed approach)")
        print("- liver_convenience_output.mesh (convenience function)")
        print()
        print("These MEDIT format files can be visualized with:")
        print("- Paraview (with MEDIT reader plugin)")
        print("- CGAL's mesh viewer")
        print("- Custom visualization tools")
        
    else:
        print("\n" + "=" * 60)
        print("❌ Test failed. Check the error messages above.")
        print()
        print("Common issues:")
        print("1. warp-cgal.pyd not found - ensure project was built")
        print("2. Missing dependencies - check DLL files in output directory")
        print("3. Invalid INR file - verify liver-sliced.inr format")