"""
Simple Volume Mesher

A simplified mesh generator that creates tetrahedral meshes from segmented volumes
using marching cubes and Delaunay triangulation, without requiring CGAL dependencies.
This provides a fallback solution when CGAL is not available.
"""

import numpy as np
from typing import Tuple, List, Optional
import warnings

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not available. Install with: pip install scikit-image")

try:
    from scipy.spatial import Delaunay
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Install with: pip install scipy")


class SimpleVolumeMesher:
    """
    Simple volume mesher using marching cubes and Delaunay triangulation.
    
    This provides a fallback mesh generation method when CGAL is not available,
    creating tetrahedral meshes from segmented 3D volumes.
    """
    
    def __init__(self):
        """Initialize the simple volume mesher."""
        self.volume_data = None
        self.surface_vertices = None
        self.surface_triangles = None
        self.volume_vertices = None
        self.tetrahedra = None
        self.smoothed_volume = None
        
    def load_volume(self, volume_data: np.ndarray) -> bool:
        """
        Load volume data for mesh generation.
        
        Args:
            volume_data: 3D numpy array with segmented labels
            
        Returns:
            True if successful
        """
        try:
            self.volume_data = volume_data.astype(np.uint8)
            print(f"Loaded volume: {self.volume_data.shape}")
            print(f"Unique labels: {np.unique(self.volume_data)}")
            return True
        except Exception as e:
            print(f"Error loading volume: {e}")
            return False
    
    def smooth_volume(self, sigma: float = 1.0) -> bool:
        """
        Apply Gaussian smoothing to the volume for better mesh quality.
        
        Args:
            sigma: Smoothing parameter
            
        Returns:
            True if successful
        """
        if not SCIPY_AVAILABLE:
            print("Warning: scipy not available, skipping smoothing")
            self.smoothed_volume = self.volume_data.astype(float)
            return True
        
        try:
            print(f"Smoothing volume with sigma={sigma}")
            self.smoothed_volume = ndimage.gaussian_filter(
                self.volume_data.astype(float), sigma=sigma
            )
            return True
        except Exception as e:
            print(f"Error smoothing volume: {e}")
            return False
    
    def extract_surface(self, label: int = 1, level: float = 0.5) -> bool:
        """
        Extract surface mesh using marching cubes.
        
        Args:
            label: Label value to extract surface for
            level: Isosurface level
            
        Returns:
            True if successful
        """
        if not SKIMAGE_AVAILABLE:
            print("Error: scikit-image not available for marching cubes")
            return False
        
        try:
            # Create binary volume for the specified label
            if self.smoothed_volume is not None:
                binary_volume = (self.smoothed_volume >= label - 0.5) & (self.smoothed_volume <= label + 0.5)
            else:
                binary_volume = (self.volume_data == label)
            
            print(f"Extracting surface for label {label}")
            print(f"Binary volume shape: {binary_volume.shape}, voxels: {np.sum(binary_volume)}")
            
            if np.sum(binary_volume) == 0:
                print("Warning: No voxels found for the specified label")
                return False
            
            # Use marching cubes to extract surface
            vertices, faces, normals, values = measure.marching_cubes(
                binary_volume.astype(float), level=level, spacing=(1.0, 1.0, 1.0)
            )
            
            self.surface_vertices = vertices
            self.surface_triangles = faces
            
            print(f"Extracted surface: {len(vertices)} vertices, {len(faces)} triangles")
            return True
            
        except Exception as e:
            print(f"Error extracting surface: {e}")
            return False
    
    def create_volume_mesh(self, internal_points: Optional[int] = None) -> bool:
        """
        Create volume mesh using Delaunay triangulation.
        
        Args:
            internal_points: Number of internal points to add (None for auto)
            
        Returns:
            True if successful
        """
        if not SCIPY_AVAILABLE:
            print("Error: scipy not available for Delaunay triangulation")
            return False
        
        if self.surface_vertices is None:
            print("Error: No surface vertices available. Run extract_surface first.")
            return False
        
        try:
            print("Creating volume mesh with Delaunay triangulation...")
            
            # Start with surface vertices
            all_vertices = list(self.surface_vertices)
            
            # Add internal points for better tetrahedral mesh
            if internal_points is None:
                # Auto-determine number of internal points based on volume size
                volume_size = np.prod(self.volume_data.shape)
                internal_points = min(int(volume_size ** (1/3)), 100)
            
            if internal_points > 0:
                print(f"Adding {internal_points} internal points")
                
                # Find bounding box of surface
                min_coords = np.min(self.surface_vertices, axis=0)
                max_coords = np.max(self.surface_vertices, axis=0)
                
                # Generate random internal points
                for _ in range(internal_points):
                    # Random point within bounding box
                    point = np.random.uniform(min_coords, max_coords)
                    
                    # Check if point is inside the volume (simple check)
                    i, j, k = [int(np.clip(coord, 0, dim-1)) for coord, dim in zip(point, self.volume_data.shape)]
                    if self.volume_data[i, j, k] > 0:  # Inside some label
                        all_vertices.append(point)
            
            # Convert to numpy array
            self.volume_vertices = np.array(all_vertices)
            
            # Perform Delaunay triangulation
            print(f"Performing Delaunay triangulation on {len(all_vertices)} vertices...")
            delaunay = Delaunay(self.volume_vertices)
            
            # Get tetrahedra
            self.tetrahedra = delaunay.simplices
            
            print(f"Generated {len(self.tetrahedra)} tetrahedra")
            return True
            
        except Exception as e:
            print(f"Error creating volume mesh: {e}")
            return False
    
    def get_mesh_data(self) -> dict:
        """
        Get the generated mesh data.
        
        Returns:
            Dictionary with mesh components
        """
        if self.volume_vertices is None or self.tetrahedra is None:
            return {}
        
        # Generate edges from tetrahedra
        edges = set()
        for tet in self.tetrahedra:
            # Each tetrahedron has 6 edges
            tet_edges = [
                (tet[0], tet[1]), (tet[0], tet[2]), (tet[0], tet[3]),
                (tet[1], tet[2]), (tet[1], tet[3]), (tet[2], tet[3])
            ]
            for edge in tet_edges:
                edges.add(tuple(sorted(edge)))
        
        edges_list = []
        for edge in sorted(edges):
            edges_list.extend([edge[0], edge[1]])
        
        # Use surface triangles if available, otherwise extract from tetrahedra
        if self.surface_triangles is not None:
            triangles = self.surface_triangles
        else:
            # Extract boundary faces from tetrahedra
            face_count = {}
            for tet in self.tetrahedra:
                faces = [
                    tuple(sorted([tet[0], tet[1], tet[2]])),
                    tuple(sorted([tet[0], tet[1], tet[3]])),
                    tuple(sorted([tet[0], tet[2], tet[3]])),
                    tuple(sorted([tet[1], tet[2], tet[3]]))
                ]
                for face in faces:
                    face_count[face] = face_count.get(face, 0) + 1
            
            # Boundary faces appear only once
            triangles = []
            for face, count in face_count.items():
                if count == 1:
                    triangles.append(list(face))
            triangles = np.array(triangles)
        
        # Generate simple UV coordinates
        uvs = []
        for vertex in self.volume_vertices:
            # Simple cylindrical projection
            x, y, z = vertex
            u = (x / self.volume_data.shape[0]) % 1.0
            v = (y / self.volume_data.shape[1]) % 1.0
            uvs.append([u, v])
        
        return {
            'vertices': self.volume_vertices,
            'tetrahedra': self.tetrahedra,
            'triangles': triangles,
            'edges': edges_list,
            'uvs': np.array(uvs)
        }
    
    def save_mesh(self, output_dir: str, mesh_name: str = "model") -> bool:
        """
        Save mesh in WarpSim format.
        
        Args:
            output_dir: Output directory
            mesh_name: Base mesh filename
            
        Returns:
            True if successful
        """
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            mesh_data = self.get_mesh_data()
            if not mesh_data:
                print("Error: No mesh data to save")
                return False
            
            base_path = os.path.join(output_dir, mesh_name)
            
            # Save vertices
            with open(f"{base_path}.vertices", 'w') as f:
                for vertex in mesh_data['vertices']:
                    f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
            
            # Save tetrahedra
            with open(f"{base_path}.tetras", 'w') as f:
                for tet in mesh_data['tetrahedra']:
                    f.write(f"{tet[0]} {tet[1]} {tet[2]} {tet[3]}\n")
            
            # Save triangles
            with open(f"{base_path}.tris", 'w') as f:
                for tri in mesh_data['triangles']:
                    f.write(f"{tri[0]} {tri[1]} {tri[2]}\n")
            
            # Save edges
            edges = mesh_data['edges']
            with open(f"{base_path}.edges", 'w') as f:
                for i in range(0, len(edges), 2):
                    f.write(f"{edges[i]} {edges[i+1]}\n")
            
            # Save UVs
            with open(f"{base_path}.uvs", 'w') as f:
                for uv in mesh_data['uvs']:
                    f.write(f"{uv[0]} {uv[1]}\n")
            
            print(f"Mesh saved to {output_dir}")
            print(f"  Vertices: {len(mesh_data['vertices'])}")
            print(f"  Tetrahedra: {len(mesh_data['tetrahedra'])}")
            print(f"  Triangles: {len(mesh_data['triangles'])}")
            print(f"  Edges: {len(mesh_data['edges']) // 2}")
            
            return True
            
        except Exception as e:
            print(f"Error saving mesh: {e}")
            return False


def read_inr_file_simple(file_path: str) -> Optional[np.ndarray]:
    """
    Simple INR file reader independent of CGAL.
    
    Args:
        file_path: Path to INR file
        
    Returns:
        3D numpy array or None if failed
    """
    try:
        import gzip
        
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
            # Simple fallback - try to infer cubic dimensions
            volume_data = np.frombuffer(data, dtype=np.uint8)
            cube_root = round(len(volume_data) ** (1/3))
            if cube_root ** 3 == len(volume_data):
                return volume_data.reshape((cube_root, cube_root, cube_root))
            return None
        
        # Parse header
        header = data[:header_end].decode('ascii', errors='ignore')
        print(f"INR header found, parsing...")
        
        # Extract dimensions
        dims = [64, 64, 64]  # Default
        
        for line in header.split('\n'):
            line = line.strip()
            if line.startswith('XDIM='):
                dims[0] = int(line.split('=')[1])
            elif line.startswith('YDIM='):
                dims[1] = int(line.split('=')[1])
            elif line.startswith('ZDIM='):
                dims[2] = int(line.split('=')[1])
        
        print(f"INR dimensions: {dims[0]}x{dims[1]}x{dims[2]}")
        
        # Skip header and read volume data
        volume_start = header_end + 5  # Skip '\n\n##}'
        volume_data = data[volume_start:]
        
        # Parse volume data (assume uint8 for now)
        expected_size = dims[0] * dims[1] * dims[2]
        if len(volume_data) >= expected_size:
            volume_array = np.frombuffer(volume_data[:expected_size], dtype=np.uint8)
            volume_3d = volume_array.reshape((dims[2], dims[1], dims[0]))  # INR uses ZYX order
            
            print(f"Successfully loaded INR volume: {volume_3d.shape}")
            return volume_3d
        else:
            print(f"Warning: File size mismatch. Expected {expected_size}, got {len(volume_data)}")
            return None
            
    except Exception as e:
        print(f"Error reading INR file: {e}")
        return None


def create_mesh_from_inr_file(inr_path: str, output_dir: str, label: int = 1) -> bool:
    """
    Create mesh directly from INR file using simple volume mesher.
    
    Args:
        inr_path: Path to INR file
        output_dir: Output directory for mesh files
        label: Label to extract mesh for
        
    Returns:
        True if successful
    """
    try:        
        # Read INR file independently
        volume_data = read_inr_file_simple(inr_path)
        
        if volume_data is None:
            print("Error: Could not load INR file")
            return False
        
        # Create simple volume mesher
        mesher = SimpleVolumeMesher()
        
        # Load volume
        if not mesher.load_volume(volume_data):
            return False
        
        # Smooth volume
        if not mesher.smooth_volume(sigma=1.0):
            return False
        
        # Extract surface for specified label
        if not mesher.extract_surface(label=label, level=0.5):
            return False
        
        # Create volume mesh
        if not mesher.create_volume_mesh(internal_points=50):
            return False
        
        # Save mesh
        return mesher.save_mesh(output_dir, "model")
        
    except Exception as e:
        print(f"Error creating mesh from INR: {e}")
        return False