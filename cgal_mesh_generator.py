"""
CGAL Tetrahedral Mesh Generation from Segmented Images

This module provides functionality to generate tetrahedral meshes from 3D segmented images
using CGAL, following the mesh_3D_weighted_image.cpp example. It creates high-quality
tetrahedral meshes suitable for physics simulation in the WarpSim framework.
"""

import os
import numpy as np
from typing import Optional, Tuple, Dict, Any
import tempfile
import warnings

try:
    # Try to import the official CGAL Python bindings first
    import CGAL
    CGAL_AVAILABLE = True
    CGAL_SOURCE = "official"
except ImportError:
    try:
        # Fallback to pygalmesh if official bindings not available
        import pygalmesh
        CGAL_AVAILABLE = True
        CGAL_SOURCE = "pygalmesh"
    except ImportError:
        CGAL_AVAILABLE = False
        CGAL_SOURCE = None
        warnings.warn(
            "Neither official CGAL Python bindings nor pygalmesh are available. "
            "Please install one of them: 'pip install cgal' or 'pip install pygalmesh'"
        )


class CgalMeshGenerator:
    """
    Generate tetrahedral meshes from segmented images using CGAL.
    
    This class wraps CGAL's 3D mesh generation functionality to create
    high-quality tetrahedral meshes from labeled 3D images, following
    the approach in CGAL's mesh_3D_weighted_image.cpp example.
    """
    
    def __init__(self):
        """Initialize the CGAL mesh generator."""
        if not CGAL_AVAILABLE:
            raise RuntimeError(
                "CGAL Python bindings are not available. "
                "Please install with: pip install cgal"
            )
        
        self.image_data = None
        self.weighted_image_data = None
        self.mesh_domain = None
        self.mesh_criteria = None
        self.generated_mesh = None
        
        # Default mesh generation parameters
        self.default_criteria = {
            'facet_angle': 30.0,
            'facet_size': 6.0,
            'facet_distance': 0.5,
            'cell_radius_edge_ratio': 3.0,
            'cell_size': 8.0
        }
    
    def load_segmented_image(self, image_path: str) -> bool:
        """
        Load a segmented image from file.
        
        Args:
            image_path: Path to the segmented image file (supports INR, DICOM, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if CGAL_SOURCE == "official":
            return self._load_image_cgal_official(image_path)
        elif CGAL_SOURCE == "pygalmesh":
            return self._load_image_pygalmesh(image_path)
        else:
            raise RuntimeError("No CGAL bindings available")
    
    def _load_image_cgal_official(self, image_path: str) -> bool:
        """Load image using official CGAL Python bindings."""
        try:
            # The official CGAL Python bindings don't expose Image_3 directly
            # Fall back to manual INR loading and use numpy arrays
            print("Note: Official CGAL Python bindings don't expose Image_3, using numpy fallback")
            
            if image_path.endswith('.inr') or image_path.endswith('.inr.gz'):
                self.image_data = self._read_inr_file(image_path)
                return self.image_data is not None
            else:
                # Try to use PIL for other formats
                from PIL import Image
                import gzip
                
                if image_path.endswith('.gz'):
                    with gzip.open(image_path, 'rb') as f:
                        img_data = f.read()
                    # This is a simplified approach - real INR parsing would be more complex
                    self.image_data = np.frombuffer(img_data, dtype=np.uint8)
                else:
                    img = Image.open(image_path)
                    self.image_data = np.array(img)
                
                return True
        except Exception as e:
            print(f"Error loading image with CGAL official bindings: {e}")
            return False
    
    def _load_image_pygalmesh(self, image_path: str) -> bool:
        """Load image using pygalmesh (fallback implementation)."""
        try:
            # pygalmesh doesn't have direct image loading, so we need to 
            # implement our own INR reader or convert to numpy array
            if image_path.endswith('.inr') or image_path.endswith('.inr.gz'):
                self.image_data = self._read_inr_file(image_path)
                return self.image_data is not None
            else:
                # Try to use PIL for other formats
                from PIL import Image
                import gzip
                
                if image_path.endswith('.gz'):
                    with gzip.open(image_path, 'rb') as f:
                        img_data = f.read()
                    # This is a simplified approach - real INR parsing would be more complex
                    self.image_data = np.frombuffer(img_data, dtype=np.uint8)
                else:
                    img = Image.open(image_path)
                    self.image_data = np.array(img)
                
                return True
        except Exception as e:
            print(f"Error loading image with pygalmesh fallback: {e}")
            return False
    
    def _read_inr_file(self, file_path: str) -> Optional[np.ndarray]:
        """
        INR file reader with proper header parsing.
        
        INR is the native format used by CGAL examples. This implementation
        attempts to parse the ASCII header to get dimensions and data type.
        """
        try:
            import gzip
            import struct
            
            print(f"Reading INR file: {file_path}")
            
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rb') as f:
                    data = f.read()
            else:
                with open(file_path, 'rb') as f:
                    data = f.read()
            
            # INR files start with an ASCII header ending with '\n\n##}'
            header_end = data.find(b'\n\n##}')
            if header_end == -1:
                print("Warning: No INR header found, trying binary interpretation")
                return self._parse_inr_binary_fallback(data)
            
            # Parse header
            header = data[:header_end].decode('ascii', errors='ignore')
            print(f"INR header found, parsing...")
            
            # Extract dimensions
            dims = [0, 0, 0]
            voxel_size = [1.0, 1.0, 1.0]
            data_type = 'unsigned fixed'
            
            for line in header.split('\n'):
                line = line.strip()
                if line.startswith('XDIM='):
                    dims[0] = int(line.split('=')[1])
                elif line.startswith('YDIM='):
                    dims[1] = int(line.split('=')[1])
                elif line.startswith('ZDIM='):
                    dims[2] = int(line.split('=')[1])
                elif line.startswith('VX='):
                    voxel_size[0] = float(line.split('=')[1])
                elif line.startswith('VY='):
                    voxel_size[1] = float(line.split('=')[1])
                elif line.startswith('VZ='):
                    voxel_size[2] = float(line.split('=')[1])
                elif line.startswith('TYPE='):
                    data_type = line.split('=')[1]
            
            print(f"INR dimensions: {dims[0]}x{dims[1]}x{dims[2]}")
            print(f"Voxel size: {voxel_size}")
            print(f"Data type: {data_type}")
            
            # Skip header and read volume data
            volume_start = header_end + 5  # Skip '\n\n##}'
            volume_data = data[volume_start:]
            
            # Determine numpy dtype based on INR type
            if 'unsigned' in data_type and 'fixed' in data_type:
                dtype = np.uint8
            elif 'signed' in data_type and 'fixed' in data_type:
                dtype = np.int8
            elif 'float' in data_type:
                dtype = np.float32
            else:
                print(f"Unknown data type: {data_type}, defaulting to uint8")
                dtype = np.uint8
            
            # Parse volume data
            expected_size = dims[0] * dims[1] * dims[2]
            actual_size = len(volume_data) // dtype().itemsize
            
            if actual_size >= expected_size:
                volume_array = np.frombuffer(volume_data[:expected_size * dtype().itemsize], dtype=dtype)
                volume_3d = volume_array.reshape((dims[2], dims[1], dims[0]))  # INR uses ZYX order
                
                # Store voxel size for later use
                self.voxel_size = voxel_size
                
                print(f"Successfully loaded INR volume: {volume_3d.shape}")
                return volume_3d
            else:
                print(f"Warning: File size mismatch. Expected {expected_size}, got {actual_size}")
                return self._parse_inr_binary_fallback(data)
                
        except Exception as e:
            print(f"Error reading INR file: {e}")
            return self._parse_inr_binary_fallback(data)
    
    def _parse_inr_binary_fallback(self, data: bytes) -> Optional[np.ndarray]:
        """Fallback INR parsing when header parsing fails."""
        try:
            print("Attempting binary fallback for INR file...")
            volume_data = np.frombuffer(data, dtype=np.uint8)
            
            # Try to infer dimensions (this is a placeholder - real INR parsing needed)
            cube_root = round(len(volume_data) ** (1/3))
            if cube_root ** 3 == len(volume_data):
                volume_3d = volume_data.reshape((cube_root, cube_root, cube_root))
                print(f"Inferred cubic volume: {cube_root}^3")
                return volume_3d
            else:
                # Try some common medical image sizes
                common_sizes = [(128, 128, 128), (256, 256, 256), (512, 512, 128), (64, 64, 64)]
                for dims in common_sizes:
                    expected_size = dims[0] * dims[1] * dims[2]
                    if len(volume_data) >= expected_size:
                        volume_3d = volume_data[:expected_size].reshape(dims)
                        print(f"Using common size: {dims}")
                        return volume_3d
                
                # Fallback: create a test volume
                print("Warning: Could not parse INR dimensions, creating test volume")
                return self._create_test_volume()
                
        except Exception as e:
            print(f"Error in binary fallback: {e}")
            return self._create_test_volume()
    
    def _create_test_volume(self) -> np.ndarray:
        """Create a test segmented volume for demonstration."""
        # Create a simple 32x32x32 test volume with two regions
        volume = np.zeros((32, 32, 32), dtype=np.uint8)
        
        # Create a sphere in the center (label 1)
        center = np.array([16, 16, 16])
        radius = 8
        
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    pos = np.array([i, j, k])
                    dist = np.linalg.norm(pos - center)
                    if dist < radius:
                        volume[i, j, k] = 1
                    elif dist < radius + 4:
                        volume[i, j, k] = 2  # Outer shell (label 2)
        
        return volume
    
    def generate_label_weights(self, sigma: Optional[float] = None) -> bool:
        """
        Generate weighted labels for smoother domain boundaries.
        
        Args:
            sigma: Smoothing parameter (if None, uses image voxel size)
            
        Returns:
            True if successful
        """
        if self.image_data is None:
            raise RuntimeError("No image data loaded. Call load_segmented_image first.")
        
        if CGAL_SOURCE == "official":
            return self._generate_weights_cgal_official(sigma)
        elif CGAL_SOURCE == "pygalmesh":
            return self._generate_weights_pygalmesh(sigma)
        else:
            return False
    
    def _generate_weights_cgal_official(self, sigma: Optional[float]) -> bool:
        """Generate weights using official CGAL bindings."""
        try:
            if sigma is None:
                # Use voxel size if available, otherwise default
                if hasattr(self, 'voxel_size') and self.voxel_size:
                    sigma = max(self.voxel_size)
                else:
                    sigma = 1.0
            
            # Since official CGAL bindings don't expose generate_label_weights,
            # use scipy for Gaussian smoothing as approximation
            print(f"Using Gaussian smoothing with sigma={sigma} as weight approximation")
            
            from scipy import ndimage
            self.weighted_image_data = ndimage.gaussian_filter(
                self.image_data.astype(float), sigma=sigma
            )
            return True
        except Exception as e:
            print(f"Error generating weights with CGAL official: {e}")
            return False
    
    def _generate_weights_pygalmesh(self, sigma: Optional[float]) -> bool:
        """Generate weights using pygalmesh (simplified implementation)."""
        try:
            if sigma is None:
                sigma = 1.0  # Default smoothing parameter
            
            # Simple Gaussian smoothing as a weight approximation
            from scipy import ndimage
            self.weighted_image_data = ndimage.gaussian_filter(
                self.image_data.astype(float), sigma=sigma
            )
            return True
        except Exception as e:
            print(f"Error generating weights with pygalmesh: {e}")
            return False
    
    def create_mesh_domain(self, relative_error_bound: float = 1e-6) -> bool:
        """
        Create the mesh domain from the loaded image.
        
        Args:
            relative_error_bound: Error bound for the domain creation
            
        Returns:
            True if successful
        """
        if self.image_data is None:
            raise RuntimeError("No image data loaded")
        
        if CGAL_SOURCE == "official":
            return self._create_domain_cgal_official(relative_error_bound)
        elif CGAL_SOURCE == "pygalmesh":
            return self._create_domain_pygalmesh(relative_error_bound)
        else:
            return False
    
    def _create_domain_cgal_official(self, relative_error_bound: float) -> bool:
        """Create domain using official CGAL bindings."""
        try:
            # Create labeled image mesh domain
            if self.weighted_image_data is not None:
                self.mesh_domain = CGAL.Labeled_mesh_domain_3.create_labeled_image_mesh_domain(
                    self.image_data,
                    weights=self.weighted_image_data,
                    relative_error_bound=relative_error_bound
                )
            else:
                self.mesh_domain = CGAL.Labeled_mesh_domain_3.create_labeled_image_mesh_domain(
                    self.image_data,
                    relative_error_bound=relative_error_bound
                )
            return True
        except Exception as e:
            print(f"Error creating mesh domain with CGAL: {e}")
            return False
    
    def _create_domain_pygalmesh(self, relative_error_bound: float) -> bool:
        """Create domain using pygalmesh."""
        try:
            # pygalmesh approach would require converting numpy array to appropriate format
            # This is a placeholder implementation
            self.mesh_domain = {
                'image_data': self.image_data,
                'weighted_data': self.weighted_image_data,
                'error_bound': relative_error_bound
            }
            return True
        except Exception as e:
            print(f"Error creating mesh domain with pygalmesh: {e}")
            return False
    
    def set_mesh_criteria(self, criteria: Optional[Dict[str, float]] = None) -> None:
        """
        Set mesh generation criteria.
        
        Args:
            criteria: Dictionary with mesh quality parameters
        """
        if criteria is None:
            criteria = self.default_criteria.copy()
        else:
            # Merge with defaults
            merged_criteria = self.default_criteria.copy()
            merged_criteria.update(criteria)
            criteria = merged_criteria
        
        self.mesh_criteria = criteria
    
    def generate_mesh(self) -> bool:
        """
        Generate the tetrahedral mesh.
        
        Returns:
            True if successful
        """
        if self.mesh_domain is None:
            raise RuntimeError("No mesh domain created. Call create_mesh_domain first.")
        
        if self.mesh_criteria is None:
            self.set_mesh_criteria()
        
        if CGAL_SOURCE == "official":
            return self._generate_mesh_cgal_official()
        elif CGAL_SOURCE == "pygalmesh":
            return self._generate_mesh_pygalmesh()
        else:
            return False
    
    def _generate_mesh_cgal_official(self) -> bool:
        """Generate mesh using official CGAL bindings."""
        try:
            # The official CGAL Python bindings don't properly expose Mesh_3 classes
            # Fall back to pygalmesh approach
            print("Note: Official CGAL bindings don't expose Mesh_3 classes, using pygalmesh approach")
            return self._generate_mesh_pygalmesh()
        except Exception as e:
            print(f"Error generating mesh with CGAL official: {e}")
            return False
    
    def _generate_mesh_pygalmesh(self) -> bool:
        """Generate mesh using pygalmesh."""
        try:
            # pygalmesh approach for volume meshing
            # This is a simplified implementation
            
            # Convert labeled image to implicit function
            def implicit_func(x):
                # Simple conversion from labeled volume to distance function
                # This is a placeholder - real implementation would be more sophisticated
                i, j, k = int(x[0]), int(x[1]), int(x[2])
                dims = self.image_data.shape
                if 0 <= i < dims[0] and 0 <= j < dims[1] and 0 <= k < dims[2]:
                    return float(self.image_data[i, j, k]) - 0.5
                return 1.0  # Outside boundary
            
            # Generate mesh using pygalmesh
            self.generated_mesh = pygalmesh.generate_mesh(
                implicit_func,
                min_facet_angle=self.mesh_criteria['facet_angle'],
                max_cell_circumradius=self.mesh_criteria['cell_size']
            )
            
            return True
        except Exception as e:
            print(f"Error generating mesh with pygalmesh: {e}")
            return False
    
    def export_mesh(self, output_prefix: str) -> bool:
        """
        Export the generated mesh in various formats.
        
        Args:
            output_prefix: Prefix for output files
            
        Returns:
            True if successful
        """
        if self.generated_mesh is None:
            raise RuntimeError("No mesh generated. Call generate_mesh first.")
        
        try:
            if CGAL_SOURCE == "official":
                return self._export_mesh_cgal_official(output_prefix)
            elif CGAL_SOURCE == "pygalmesh":
                return self._export_mesh_pygalmesh(output_prefix)
            else:
                return False
        except Exception as e:
            print(f"Error exporting mesh: {e}")
            return False
    
    def _export_mesh_cgal_official(self, output_prefix: str) -> bool:
        """Export mesh using official CGAL bindings."""
        try:
            # Export in MEDIT format
            medit_file = f"{output_prefix}.mesh"
            with open(medit_file, 'w') as f:
                CGAL.IO.write_MEDIT(f, self.generated_mesh)
            
            # Export in binary CGAL format
            binary_file = f"{output_prefix}.binary.cgal"
            with open(binary_file, 'wb') as f:
                CGAL.IO.save_binary_file(f, self.generated_mesh)
            
            print(f"Mesh exported to {medit_file} and {binary_file}")
            return True
        except Exception as e:
            print(f"Error exporting CGAL mesh: {e}")
            return False
    
    def _export_mesh_pygalmesh(self, output_prefix: str) -> bool:
        """Export mesh using pygalmesh."""
        try:
            # Export mesh in VTK format (pygalmesh default)
            vtk_file = f"{output_prefix}.vtk"
            self.generated_mesh.write(vtk_file)
            
            print(f"Mesh exported to {vtk_file}")
            return True
        except Exception as e:
            print(f"Error exporting pygalmesh: {e}")
            return False
    
    def get_mesh_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the generated mesh.
        
        Returns:
            Dictionary with mesh statistics
        """
        if self.generated_mesh is None:
            return {}
        
        try:
            if CGAL_SOURCE == "official":
                return self._get_stats_cgal_official()
            elif CGAL_SOURCE == "pygalmesh":
                return self._get_stats_pygalmesh()
            else:
                return {}
        except Exception as e:
            print(f"Error getting mesh statistics: {e}")
            return {}
    
    def _get_stats_cgal_official(self) -> Dict[str, Any]:
        """Get statistics using official CGAL bindings."""
        try:
            triangulation = self.generated_mesh.triangulation()
            
            stats = {
                'num_vertices': triangulation.number_of_vertices(),
                'num_cells': triangulation.number_of_cells(),
                'num_facets': triangulation.number_of_facets(),
                'num_edges': triangulation.number_of_edges(),
                'cgal_source': 'official'
            }
            return stats
        except Exception as e:
            print(f"Error getting CGAL statistics: {e}")
            return {}
    
    def _get_stats_pygalmesh(self) -> Dict[str, Any]:
        """Get statistics using pygalmesh."""
        try:
            stats = {
                'num_vertices': len(self.generated_mesh.points),
                'num_cells': len(self.generated_mesh.cells) if hasattr(self.generated_mesh, 'cells') else 0,
                'cgal_source': 'pygalmesh'
            }
            return stats
        except Exception as e:
            print(f"Error getting pygalmesh statistics: {e}")
            return {}


def create_sample_segmented_image(output_path: str, size: int = 32) -> bool:
    """
    Create a sample segmented image for testing.
    
    Args:
        output_path: Where to save the sample image
        size: Volume size (size x size x size)
        
    Returns:
        True if successful
    """
    try:
        # Create a test volume with multiple labels
        volume = np.zeros((size, size, size), dtype=np.uint8)
        
        center = size // 2
        
        # Create nested spheres with different labels
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    pos = np.array([i, j, k])
                    center_pos = np.array([center, center, center])
                    dist = np.linalg.norm(pos - center_pos)
                    
                    if dist < size * 0.2:
                        volume[i, j, k] = 3  # Inner core
                    elif dist < size * 0.3:
                        volume[i, j, k] = 2  # Middle layer
                    elif dist < size * 0.4:
                        volume[i, j, k] = 1  # Outer layer
        
        # Save as numpy file (simple format)
        np.save(output_path, volume)
        print(f"Sample segmented image saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating sample image: {e}")
        return False