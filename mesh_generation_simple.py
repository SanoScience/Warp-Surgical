"""
Simple Mesh Generation Module for warp-surgical

This module provides automatic tetrahedral mesh generation using warp-cgal,
designed to work with the simplified warp-surgical workflow. It ensures that
proper tetrahedral meshes are available before simulation starts.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import time

# Quality presets for different use cases
QUALITY_PRESETS = {
    'preview': {
        'cell_size': 25.0,
        'facet_angle': 15.0,
        'facet_size': 20.0,
        'facet_distance': 1.0,
        'cell_radius_edge_ratio': 4.0,
    },
    'coarse': {
        'cell_size': 18.0,
        'facet_angle': 20.0,
        'facet_size': 12.0,
        'facet_distance': 0.8,
        'cell_radius_edge_ratio': 3.5,
    },
    'fast': {
        'cell_size': 15.0,
        'facet_angle': 20.0,
        'facet_size': 10.0,
        'facet_distance': 0.5,
        'cell_radius_edge_ratio': 3.0
    },
    'low': {
        'cell_size': 13.0,
        'facet_angle': 22.0,
        'facet_size': 9.0,
        'facet_distance': 0.6,
        'cell_radius_edge_ratio': 3.2,
    },
    'balanced': {
        'cell_size': 10.0,
        'facet_angle': 25.0,
        'facet_size': 8.0,
        'facet_distance': 0.5,
        'cell_radius_edge_ratio': 3.0
    },
    'high': {
        'cell_size': 6.0,
        'facet_angle': 30.0,
        'facet_size': 6.0,
        'facet_distance': 0.5,
        'cell_radius_edge_ratio': 3.0
    },
    'very_high': {
        'cell_size': 4.5,
        'facet_angle': 32.0,
        'facet_size': 4.5,
        'facet_distance': 0.4,
        'cell_radius_edge_ratio': 2.8,
    },
    'ultra': {
        'cell_size': 3.0,
        'facet_angle': 35.0,
        'facet_size': 3.0,
        'facet_distance': 0.3,
        'cell_radius_edge_ratio': 2.5,
    },
}


class SimpleMeshGenerator:
    """Simple mesh generator using warp-cgal for warp-surgical integration."""
    
    def __init__(self):
        """Initialize the mesh generator."""
        self.warp_cgal_available = False
        self._try_import_warp_cgal()
    
    def _try_import_warp_cgal(self):
        """Try to import warp-cgal Python wrapper."""
        try:
            # Add warp-cgal directory to path if needed
            warp_cgal_path = Path("G:/warp/warp-cgal").resolve()
            if str(warp_cgal_path) not in sys.path:
                sys.path.insert(0, str(warp_cgal_path))
            
            from warp_cgal_python import (generate_warp_mesh_from_image, 
                                        generate_multi_label_warp_meshes_from_image,
                                        WarpCGALMeshGenerator)
            self.generate_warp_mesh_from_image = generate_warp_mesh_from_image
            self.generate_multi_label_warp_meshes_from_image = generate_multi_label_warp_meshes_from_image
            self.WarpCGALMeshGenerator = WarpCGALMeshGenerator
            self.warp_cgal_available = True
            print("+ warp-cgal Python wrapper loaded successfully (with multi-label support)")
            
        except ImportError as e:
            print(f"- Failed to import warp-cgal: {e}")
            print("  Make sure warp-cgal.pyd is built and available")
            self.warp_cgal_available = False
    
    def is_mesh_current(self, source_path: str, target_dir: str) -> bool:
        """Check if existing mesh is newer than source image."""
        if not os.path.exists(source_path):
            print(f"- Source image not found: {source_path}")
            return False
        
        target_path = Path(target_dir)
        if not target_path.exists():
            print(f"- Target mesh directory does not exist: {target_dir}")
            return False
        
        # Check if all required mesh files exist
        required_files = ['model.vertices', 'model.tetras', 'model.tris', 'model.edges', 'model.uvs']
        for filename in required_files:
            mesh_file = target_path / filename
            if not mesh_file.exists():
                print(f"- Missing mesh file: {mesh_file}")
                return False
        
        # Check timestamps
        source_time = os.path.getmtime(source_path)
        
        # Get the oldest mesh file time (most conservative check)
        mesh_times = []
        for filename in required_files:
            mesh_file = target_path / filename
            mesh_times.append(os.path.getmtime(str(mesh_file)))
        
        oldest_mesh_time = min(mesh_times)
        
        if source_time > oldest_mesh_time:
            print(f"- Source image is newer than mesh files")
            return False
        
        print(f"+ Mesh files are up-to-date")
        return True
    
    def get_quality_parameters(self, quality_preset: str, **overrides) -> Dict[str, float]:
        """Get mesh quality parameters with optional overrides."""
        if quality_preset not in QUALITY_PRESETS:
            print(f"- Unknown quality preset '{quality_preset}', using 'balanced'")
            quality_preset = 'balanced'
        
        params = QUALITY_PRESETS[quality_preset].copy()
        params.update(overrides)
        
        print(f"+ Using quality preset '{quality_preset}' with parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        return params
    
    def detect_labels(self, source_path: str) -> Optional[list]:
        """Detect available labels in the source image."""
        if not self.warp_cgal_available:
            return None
        
        try:
            generator = self.WarpCGALMeshGenerator()
            if generator.load_image(source_path):
                labels = generator.get_image_labels()
                print(f"+ Detected {len(labels)} labels in image: {labels}")
                return labels
            else:
                print(f"- Failed to load image: {source_path}")
                return None
        except Exception as e:
            print(f"- Error detecting labels: {e}")
            return None
    
    def generate_multi_label_meshes(self, source_path: str, target_base_dir: str, quality_preset: str = 'balanced', **overrides) -> bool:
        """Generate separate tetrahedral meshes for each label using warp-cgal."""
        if not self.warp_cgal_available:
            print("- warp-cgal not available, cannot generate multi-label meshes")
            return False
        
        if not os.path.exists(source_path):
            print(f"- Source image not found: {source_path}")
            return False
        
        # Detect available labels first
        labels = self.detect_labels(source_path)
        if not labels:
            print("- No labels detected, falling back to single mesh generation")
            return self.generate_mesh(source_path, target_base_dir, quality_preset, **overrides)
        
        # Ensure target directory exists
        target_path = Path(target_base_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Get quality parameters
        params = self.get_quality_parameters(quality_preset, **overrides)
        
        print(f"+ Generating multi-label tetrahedral meshes:")
        print(f"  Source: {source_path}")
        print(f"  Target base: {target_base_dir}")
        print(f"  Quality: {quality_preset}")
        print(f"  Labels: {labels}")
        
        try:
            start_time = time.time()
            
            # Generate multi-label meshes using warp-cgal
            stats = self.generate_multi_label_warp_meshes_from_image(
                source_path,
                target_base_dir,
                **params
            )
            
            generation_time = time.time() - start_time
            
            print(f"+ Multi-label mesh generation completed in {generation_time:.1f} seconds")
            print(f"  Statistics: {stats}")
            
            # Fix Windows backslash path separator issue in generated filenames
            self._fix_backslash_filenames(target_base_dir)
            
            return True
            
        except Exception as e:
            print(f"- Multi-label mesh generation failed: {e}")
            return False
    
    def generate_mesh(self, source_path: str, target_dir: str, quality_preset: str = 'balanced', **overrides) -> bool:
        """Generate tetrahedral mesh using warp-cgal."""
        if not self.warp_cgal_available:
            print("- warp-cgal not available, cannot generate mesh")
            return False
        
        if not os.path.exists(source_path):
            print(f"- Source image not found: {source_path}")
            return False
        
        # Ensure target directory exists
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Get quality parameters
        params = self.get_quality_parameters(quality_preset, **overrides)
        
        print(f"+ Generating tetrahedral mesh:")
        print(f"  Source: {source_path}")
        print(f"  Target: {target_dir}")
        print(f"  Quality: {quality_preset}")
        
        try:
            start_time = time.time()
            
            # Generate mesh using warp-cgal
            stats = self.generate_warp_mesh_from_image(
                source_path,
                target_dir,
                **params
            )
            
            generation_time = time.time() - start_time
            
            print(f"+ Mesh generation completed in {generation_time:.1f} seconds")
            print(f"  Statistics: {stats}")
            
            return True
            
        except Exception as e:
            print(f"- Mesh generation failed: {e}")
            return False

    def _fix_backslash_filenames(self, base_dir: str):
        """Fix Windows backslash path separator issue in generated mesh filenames."""
        import shutil
        
        # Look for files with backslashes in their names in the meshes directory
        meshes_dir = Path("meshes")
        
        # Find all files with backslashes in their names
        backslash_files = []
        for file_path in meshes_dir.rglob("*"):
            if file_path.is_file() and "\\" in file_path.name:
                backslash_files.append(file_path)
        
        if not backslash_files:
            return
            
        print(f"+ Fixing {len(backslash_files)} files with backslash names...")
        
        for file_path in backslash_files:
            # Parse the intended directory and filename
            filename_with_backslash = file_path.name
            
            # Split on the last backslash to get directory and filename
            if "\\" in filename_with_backslash:
                intended_dir, intended_filename = filename_with_backslash.rsplit("\\", 1)
                
                # Create the intended directory
                target_dir = file_path.parent / intended_dir
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # Move the file to the correct location
                target_file = target_dir / intended_filename
                
                print(f"  Moving: {filename_with_backslash} -> {intended_dir}/{intended_filename}")
                shutil.move(str(file_path), str(target_file))


def ensure_meshes_ready(args) -> bool:
    """Ensure meshes are generated and ready for simulation."""
    print("=" * 60)
    print("Checking tetrahedral mesh availability...")
    
    generator = SimpleMeshGenerator()
    
    # Default paths
    source_path = getattr(args, 'mesh_source', 'G:/warp/warp-surgical/liver-sliced.inr')
    target_dir = 'meshes/liver_cgal'
    quality_preset = getattr(args, 'mesh_quality', 'balanced')
    force_regenerate = getattr(args, 'force_regenerate', False)
    multi_label = getattr(args, 'multi_label', False)
    
    print(f"+ Source image: {source_path}")
    print(f"+ Target directory: {target_dir}")
    print(f"+ Quality preset: {quality_preset}")
    print(f"+ Multi-label mode: {multi_label}")
    
    # Prepare parameter overrides
    overrides = {}
    if hasattr(args, 'cell_size') and args.cell_size is not None:
        overrides['cell_size'] = args.cell_size
        print(f"+ Cell size override: {args.cell_size}")
    
    # Check if regeneration is needed
    need_generation = force_regenerate or not generator.is_mesh_current(source_path, target_dir)
    
    if force_regenerate:
        print("+ Force regeneration requested")
    
    if need_generation:
        print("+ Mesh generation required")
        
        if not generator.warp_cgal_available:
            print("- Cannot generate mesh: warp-cgal not available")
            print("  Please build warp-cgal.pyd or use pre-generated meshes")
            return False
        
        # Choose between single mesh or multi-label mesh generation
        if multi_label:
            print("+ Using multi-label mesh generation")
            success = generator.generate_multi_label_meshes(source_path, target_dir, quality_preset, **overrides)
        else:
            print("+ Using single mesh generation")
            success = generator.generate_mesh(source_path, target_dir, quality_preset, **overrides)
            
        if not success:
            print("- Mesh generation failed")
            return False
    else:
        print("+ Using existing mesh files")
    
    print("+ Tetrahedral meshes are ready!")
    print("=" * 60)
    return True


def main():
    """Test the mesh generation functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test mesh generation")
    parser.add_argument("--mesh-source", type=str, 
                       default="G:/warp/warp-surgical/liver-sliced.inr",
                       help="Source image path")
    parser.add_argument("--mesh-quality", choices=["fast", "balanced", "high"], 
                       default="balanced", help="Mesh quality preset")
    parser.add_argument("--cell-size", type=float, help="Override cell size parameter")
    parser.add_argument("--force-regenerate", action="store_true", help="Force mesh regeneration")
    
    args = parser.parse_args()
    
    success = ensure_meshes_ready(args)
    if success:
        print("Mesh generation test completed successfully!")
    else:
        print("Mesh generation test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()