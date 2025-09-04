"""
CGAL to WarpSim Mesh Format Converter

This module converts CGAL-generated tetrahedral meshes to the format
expected by WarpSim, including vertices, tetrahedra, edges, triangles,
and UV coordinates compatible with mesh_loader.py.
"""

import os
import numpy as np
from typing import Tuple, List, Dict, Optional, Set
import warnings

try:
    import CGAL
    CGAL_OFFICIAL_AVAILABLE = True
except ImportError:
    CGAL_OFFICIAL_AVAILABLE = False

try:
    import pygalmesh
    PYGALMESH_AVAILABLE = True
except ImportError:
    PYGALMESH_AVAILABLE = False


class CgalToWarpConverter:
    """
    Convert CGAL mesh complex to WarpSim-compatible format.
    
    This class extracts vertices, tetrahedra, surface triangles, edges,
    and generates UV coordinates from CGAL mesh data, saving them in
    the format expected by mesh_loader.py.
    """
    
    def __init__(self):
        """Initialize the converter."""
        self.vertices = []
        self.tetrahedra = []
        self.surface_triangles = []
        self.edges = []
        self.uvs = []
        
    def convert_cgal_mesh(self, cgal_mesh, output_directory: str, mesh_name: str = "model") -> bool:
        """
        Convert CGAL mesh to WarpSim format and save to files.
        
        Args:
            cgal_mesh: The CGAL mesh complex (C3t3)
            output_directory: Directory to save the mesh files
            mesh_name: Base name for the mesh files
            
        Returns:
            True if successful
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_directory, exist_ok=True)
            
            if CGAL_OFFICIAL_AVAILABLE:
                success = self._convert_cgal_official_mesh(cgal_mesh)
            elif PYGALMESH_AVAILABLE:
                success = self._convert_pygalmesh_mesh(cgal_mesh)
            else:
                raise RuntimeError("No CGAL bindings available")
            
            if not success:
                return False
            
            # Save all components to files
            return self._save_mesh_files(output_directory, mesh_name)
            
        except Exception as e:
            print(f"Error converting CGAL mesh: {e}")
            return False
    
    def _convert_cgal_official_mesh(self, cgal_mesh) -> bool:
        """Convert using official CGAL Python bindings."""
        try:
            triangulation = cgal_mesh.triangulation()
            
            # Extract vertices
            self.vertices = []
            vertex_map = {}
            vertex_index = 0
            
            for vertex in triangulation.finite_vertices():
                point = vertex.point()
                vertex_coords = [float(point.x()), float(point.y()), float(point.z())]
                self.vertices.append(vertex_coords)
                vertex_map[vertex] = vertex_index
                vertex_index += 1
            
            print(f"Extracted {len(self.vertices)} vertices")
            
            # Extract tetrahedra (cells in the complex)
            self.tetrahedra = []
            for cell in cgal_mesh.cells():
                if cgal_mesh.is_in_complex(cell):
                    vertices = []
                    for i in range(4):
                        vertex = cell.vertex(i)
                        vertices.append(vertex_map[vertex])
                    self.tetrahedra.append(vertices)
            
            print(f"Extracted {len(self.tetrahedra)} tetrahedra")
            
            # Extract surface triangles (facets in the complex)
            self.surface_triangles = []
            for facet in cgal_mesh.facets():
                if cgal_mesh.is_in_complex(facet):
                    cell, face_index = facet
                    vertices = []
                    for i in range(4):
                        if i != face_index:
                            vertex = cell.vertex(i)
                            vertices.append(vertex_map[vertex])
                    
                    if len(vertices) == 3:
                        self.surface_triangles.append(vertices)
            
            print(f"Extracted {len(self.surface_triangles)} surface triangles")
            
            # Generate edges from tetrahedra
            self._generate_edges_from_tetrahedra()
            
            # Generate UV coordinates
            self._generate_uv_coordinates()
            
            return True
            
        except Exception as e:
            print(f"Error converting CGAL official mesh: {e}")
            return False
    
    def _convert_pygalmesh_mesh(self, pygalmesh_mesh) -> bool:
        """Convert using pygalmesh format."""
        try:
            # Extract vertices
            if hasattr(pygalmesh_mesh, 'points'):
                self.vertices = pygalmesh_mesh.points.tolist()
            else:
                raise RuntimeError("Pygalmesh mesh has no points attribute")
            
            print(f"Extracted {len(self.vertices)} vertices")
            
            # Extract cells (tetrahedra)
            if hasattr(pygalmesh_mesh, 'cells') and pygalmesh_mesh.cells:
                # pygalmesh stores cells by type
                self.tetrahedra = []
                for cell_block in pygalmesh_mesh.cells:
                    if cell_block.type == "tetra":
                        for tetra in cell_block.data:
                            self.tetrahedra.append(tetra.tolist())
            
            print(f"Extracted {len(self.tetrahedra)} tetrahedra")
            
            # Extract surface triangles if available
            if hasattr(pygalmesh_mesh, 'cells') and pygalmesh_mesh.cells:
                self.surface_triangles = []
                for cell_block in pygalmesh_mesh.cells:
                    if cell_block.type == "triangle":
                        for triangle in cell_block.data:
                            self.surface_triangles.append(triangle.tolist())
            
            # If no surface triangles found, extract from tetrahedra boundary
            if not self.surface_triangles:
                self._extract_surface_triangles_from_tetrahedra()
            
            print(f"Extracted {len(self.surface_triangles)} surface triangles")
            
            # Generate edges from tetrahedra
            self._generate_edges_from_tetrahedra()
            
            # Generate UV coordinates
            self._generate_uv_coordinates()
            
            return True
            
        except Exception as e:
            print(f"Error converting pygalmesh mesh: {e}")
            return False
    
    def _generate_edges_from_tetrahedra(self) -> None:
        """Generate edge list from tetrahedra connectivity."""
        edge_set = set()
        
        # Each tetrahedron has 6 edges
        for tetra in self.tetrahedra:
            # Tetrahedron edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
            edges = [
                (tetra[0], tetra[1]),
                (tetra[0], tetra[2]),
                (tetra[0], tetra[3]),
                (tetra[1], tetra[2]),
                (tetra[1], tetra[3]),
                (tetra[2], tetra[3])
            ]
            
            for edge in edges:
                # Store edges in canonical order (smaller index first)
                canonical_edge = tuple(sorted(edge))
                edge_set.add(canonical_edge)
        
        # Convert to flat list format expected by WarpSim
        self.edges = []
        for edge in sorted(edge_set):
            self.edges.extend([edge[0], edge[1]])
        
        print(f"Generated {len(edge_set)} edges")
    
    def _extract_surface_triangles_from_tetrahedra(self) -> None:
        """Extract boundary surface triangles from tetrahedra."""
        face_count = {}
        
        # Each tetrahedron has 4 faces
        for tetra in self.tetrahedra:
            faces = [
                tuple(sorted([tetra[0], tetra[1], tetra[2]])),
                tuple(sorted([tetra[0], tetra[1], tetra[3]])),
                tuple(sorted([tetra[0], tetra[2], tetra[3]])),
                tuple(sorted([tetra[1], tetra[2], tetra[3]]))
            ]
            
            for face in faces:
                face_count[face] = face_count.get(face, 0) + 1
        
        # Boundary faces appear only once
        self.surface_triangles = []
        for face, count in face_count.items():
            if count == 1:
                self.surface_triangles.append(list(face))
        
        print(f"Extracted {len(self.surface_triangles)} boundary triangles")
    
    def _generate_uv_coordinates(self) -> None:
        """Generate UV coordinates for surface triangles."""
        # Simple UV generation based on vertex positions
        # This is a placeholder - more sophisticated UV unwrapping could be implemented
        
        self.uvs = []
        for vertex in self.vertices:
            # Simple cylindrical projection as placeholder
            x, y, z = vertex
            u = (x + 1.0) * 0.5  # Map [-1,1] to [0,1]
            v = (y + 1.0) * 0.5  # Map [-1,1] to [0,1]
            
            # Clamp to [0,1] range
            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0, v))
            
            self.uvs.append([u, v])
        
        print(f"Generated {len(self.uvs)} UV coordinates")
    
    def _save_mesh_files(self, output_directory: str, mesh_name: str) -> bool:
        """Save mesh components to WarpSim-compatible files."""
        try:
            base_path = os.path.join(output_directory, mesh_name)
            
            # Save vertices
            vertices_file = f"{base_path}.vertices"
            with open(vertices_file, 'w') as f:
                for vertex in self.vertices:
                    f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
            print(f"Saved vertices to {vertices_file}")
            
            # Save tetrahedra
            tetras_file = f"{base_path}.tetras"
            with open(tetras_file, 'w') as f:
                for tetra in self.tetrahedra:
                    f.write(f"{tetra[0]} {tetra[1]} {tetra[2]} {tetra[3]}\n")
            print(f"Saved tetrahedra to {tetras_file}")
            
            # Save surface triangles
            tris_file = f"{base_path}.tris"
            with open(tris_file, 'w') as f:
                for triangle in self.surface_triangles:
                    f.write(f"{triangle[0]} {triangle[1]} {triangle[2]}\n")
            print(f"Saved surface triangles to {tris_file}")
            
            # Save edges
            edges_file = f"{base_path}.edges"
            with open(edges_file, 'w') as f:
                for i in range(0, len(self.edges), 2):
                    f.write(f"{self.edges[i]} {self.edges[i+1]}\n")
            print(f"Saved edges to {edges_file}")
            
            # Save UV coordinates
            uvs_file = f"{base_path}.uvs"
            with open(uvs_file, 'w') as f:
                for uv in self.uvs:
                    f.write(f"{uv[0]} {uv[1]}\n")
            print(f"Saved UV coordinates to {uvs_file}")
            
            return True
            
        except Exception as e:
            print(f"Error saving mesh files: {e}")
            return False
    
    def get_mesh_info(self) -> Dict[str, int]:
        """Get information about the converted mesh."""
        return {
            'num_vertices': len(self.vertices),
            'num_tetrahedra': len(self.tetrahedra),
            'num_surface_triangles': len(self.surface_triangles),
            'num_edges': len(self.edges) // 2,
            'num_uvs': len(self.uvs)
        }


def convert_numpy_volume_to_warp_mesh(volume_data: np.ndarray, 
                                      output_directory: str,
                                      mesh_name: str = "model",
                                      mesh_criteria: Optional[Dict] = None) -> bool:
    """
    Convert a numpy volume directly to WarpSim mesh format.
    
    This is a simplified approach for testing when CGAL bindings are not available.
    
    Args:
        volume_data: 3D numpy array with segmented data
        output_directory: Directory to save mesh files
        mesh_name: Base name for mesh files
        mesh_criteria: Mesh quality parameters
        
    Returns:
        True if successful
    """
    try:
        from cgal_mesh_generator import CgalMeshGenerator
        
        # Create CGAL mesh generator
        generator = CgalMeshGenerator()
        
        # Set the volume data directly (bypass file loading)
        generator.image_data = volume_data
        
        # Generate weights
        if not generator.generate_label_weights():
            print("Warning: Could not generate label weights")
        
        # Create domain
        if not generator.create_mesh_domain():
            print("Error: Could not create mesh domain")
            return False
        
        # Set criteria
        if mesh_criteria:
            generator.set_mesh_criteria(mesh_criteria)
        
        # Generate mesh
        if not generator.generate_mesh():
            print("Error: Could not generate mesh")
            return False
        
        # Convert to WarpSim format
        converter = CgalToWarpConverter()
        return converter.convert_cgal_mesh(generator.generated_mesh, output_directory, mesh_name)
        
    except Exception as e:
        print(f"Error converting numpy volume to mesh: {e}")
        return False


def create_simple_test_mesh(output_directory: str, mesh_name: str = "simple_test") -> bool:
    """
    Create a simple test mesh in WarpSim format for verification.
    
    This creates a basic tetrahedral mesh manually to test the pipeline
    without requiring CGAL.
    
    Args:
        output_directory: Directory to save mesh files
        mesh_name: Base name for mesh files
        
    Returns:
        True if successful
    """
    try:
        os.makedirs(output_directory, exist_ok=True)
        
        # Create a simple tetrahedron
        vertices = [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.5, 1.0, 0.0],  # 2
            [0.5, 0.5, 1.0]   # 3
        ]
        
        # One tetrahedron
        tetrahedra = [
            [0, 1, 2, 3]
        ]
        
        # Surface triangles (all faces of the tetrahedron)
        triangles = [
            [0, 1, 2],  # Bottom face
            [0, 1, 3],  # Front face
            [0, 2, 3],  # Left face
            [1, 2, 3]   # Right face
        ]
        
        # Edges
        edges = [
            0, 1,  # Edge 0-1
            0, 2,  # Edge 0-2
            0, 3,  # Edge 0-3
            1, 2,  # Edge 1-2
            1, 3,  # Edge 1-3
            2, 3   # Edge 2-3
        ]
        
        # UV coordinates (simple)
        uvs = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.5]
        ]
        
        # Save files
        base_path = os.path.join(output_directory, mesh_name)
        
        # Vertices
        with open(f"{base_path}.vertices", 'w') as f:
            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
        
        # Tetrahedra
        with open(f"{base_path}.tetras", 'w') as f:
            for t in tetrahedra:
                f.write(f"{t[0]} {t[1]} {t[2]} {t[3]}\n")
        
        # Triangles
        with open(f"{base_path}.tris", 'w') as f:
            for tri in triangles:
                f.write(f"{tri[0]} {tri[1]} {tri[2]}\n")
        
        # Edges
        with open(f"{base_path}.edges", 'w') as f:
            for i in range(0, len(edges), 2):
                f.write(f"{edges[i]} {edges[i+1]}\n")
        
        # UVs
        with open(f"{base_path}.uvs", 'w') as f:
            for uv in uvs:
                f.write(f"{uv[0]} {uv[1]}\n")
        
        print(f"Simple test mesh saved to {output_directory}")
        print(f"Vertices: {len(vertices)}, Tetrahedra: {len(tetrahedra)}, Triangles: {len(triangles)}")
        
        return True
        
    except Exception as e:
        print(f"Error creating simple test mesh: {e}")
        return False