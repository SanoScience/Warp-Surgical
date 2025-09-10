"""
Python wrapper for the warp-cgal native library.

This module provides a Python interface to the CGAL mesh generation functionality
implemented in the native C++ DLL.
"""

import ctypes
import os
import sys
from typing import Optional

class WarpCGALMeshGenerator:
    """Python wrapper for the native CGAL mesh generator."""
    
    def __init__(self):
        """Initialize the mesh generator by loading the native library."""
        # Determine the library extension based on platform
        if sys.platform.startswith('win'):
            lib_ext = ".pyd"
            lib_name = "warp-cgal"
        else:
            lib_ext = ".so" 
            lib_name = "warp_cgal"
        
        # Try to find the library in common locations
        dll_paths = [
            f"{lib_name}{lib_ext}",  # Same directory
            f"x64/Debug/{lib_name}.pyd",  # Windows paths
            f"x64/Release/{lib_name}.pyd",
            f"build/{lib_name}{lib_ext}",  # Linux build directory
            os.path.join(os.path.dirname(__file__), f"{lib_name}{lib_ext}"),
            os.path.join(os.path.dirname(__file__), "x64", "Debug", f"{lib_name}.pyd"),
            os.path.join(os.path.dirname(__file__), "x64", "Release", f"{lib_name}.pyd"),
            os.path.join(os.path.dirname(__file__), "build", f"{lib_name}{lib_ext}"),
        ]
        
        self._dll = None
        for path in dll_paths:
            if os.path.exists(path):
                try:
                    self._dll = ctypes.CDLL(path)
                    break
                except OSError:
                    continue
        
        if self._dll is None:
            raise RuntimeError(f"Could not find or load {lib_name}{lib_ext}")
        
        # Set up function signatures
        self._setup_function_signatures()
        
        # Create the native mesh generator
        self._generator = self._dll.create_mesh_generator()
        if not self._generator:
            raise RuntimeError("Failed to create native mesh generator")
    
    def _setup_function_signatures(self):
        """Set up ctypes function signatures for the DLL functions."""
        # create_mesh_generator() -> void*
        self._dll.create_mesh_generator.restype = ctypes.c_void_p
        self._dll.create_mesh_generator.argtypes = []
        
        # destroy_mesh_generator(void*)
        self._dll.destroy_mesh_generator.restype = None
        self._dll.destroy_mesh_generator.argtypes = [ctypes.c_void_p]
        
        # load_image(void*, const char*) -> int
        self._dll.load_image.restype = ctypes.c_int
        self._dll.load_image.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        
        # set_criteria(void*, double, double, double, double, double)
        self._dll.set_criteria.restype = None
        self._dll.set_criteria.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, 
                                         ctypes.c_double, ctypes.c_double, ctypes.c_double]
        
        # generate_mesh(void*) -> int
        self._dll.generate_mesh.restype = ctypes.c_int
        self._dll.generate_mesh.argtypes = [ctypes.c_void_p]
        
        # export_mesh(void*, const char*) -> int
        self._dll.export_mesh.restype = ctypes.c_int
        self._dll.export_mesh.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        
        # export_mesh_warp_format(void*, const char*) -> int
        self._dll.export_mesh_warp_format.restype = ctypes.c_int
        self._dll.export_mesh_warp_format.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        
        # get_number_of_vertices(void*) -> int
        self._dll.get_number_of_vertices.restype = ctypes.c_int
        self._dll.get_number_of_vertices.argtypes = [ctypes.c_void_p]
        
        # get_number_of_cells(void*) -> int
        self._dll.get_number_of_cells.restype = ctypes.c_int
        self._dll.get_number_of_cells.argtypes = [ctypes.c_void_p]
        
        # get_number_of_facets(void*) -> int
        self._dll.get_number_of_facets.restype = ctypes.c_int
        self._dll.get_number_of_facets.argtypes = [ctypes.c_void_p]
        
        # Multi-label mesh generation functions
        # generate_multi_label_meshes(void*) -> int
        self._dll.generate_multi_label_meshes.restype = ctypes.c_int
        self._dll.generate_multi_label_meshes.argtypes = [ctypes.c_void_p]
        
        # export_multi_label_meshes(void*, const char*) -> int
        self._dll.export_multi_label_meshes.restype = ctypes.c_int
        self._dll.export_multi_label_meshes.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        
        # export_multi_label_meshes_warp_format(void*, const char*) -> int
        self._dll.export_multi_label_meshes_warp_format.restype = ctypes.c_int
        self._dll.export_multi_label_meshes_warp_format.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        
        # get_image_labels_count(void*) -> int
        self._dll.get_image_labels_count.restype = ctypes.c_int
        self._dll.get_image_labels_count.argtypes = [ctypes.c_void_p]
        
        # get_image_labels(void*, int*)
        self._dll.get_image_labels.restype = None
        self._dll.get_image_labels.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        
        # get_number_of_vertices_for_label(void*, int) -> int
        self._dll.get_number_of_vertices_for_label.restype = ctypes.c_int
        self._dll.get_number_of_vertices_for_label.argtypes = [ctypes.c_void_p, ctypes.c_int]
        
        # get_number_of_cells_for_label(void*, int) -> int
        self._dll.get_number_of_cells_for_label.restype = ctypes.c_int
        self._dll.get_number_of_cells_for_label.argtypes = [ctypes.c_void_p, ctypes.c_int]
        
        # get_number_of_facets_for_label(void*, int) -> int
        self._dll.get_number_of_facets_for_label.restype = ctypes.c_int
        self._dll.get_number_of_facets_for_label.argtypes = [ctypes.c_void_p, ctypes.c_int]
    
    def __del__(self):
        """Clean up the native mesh generator."""
        if hasattr(self, '_generator') and self._generator and hasattr(self, '_dll'):
            self._dll.destroy_mesh_generator(self._generator)
    
    def load_image(self, filename: str) -> bool:
        """
        Load a segmented image from file.
        
        Args:
            filename: Path to the image file (INR format)
            
        Returns:
            True if successful, False otherwise
        """
        result = self._dll.load_image(self._generator, filename.encode('utf-8'))
        return result != 0
    
    def set_criteria(self, facet_angle: float = 30.0, facet_size: float = 6.0, 
                    facet_distance: float = 0.5, cell_radius_edge_ratio: float = 3.0, 
                    cell_size: float = 8.0) -> None:
        """
        Set mesh generation criteria.
        
        Args:
            facet_angle: Minimum angle for surface facets
            facet_size: Maximum size for surface facets
            facet_distance: Maximum distance from surface
            cell_radius_edge_ratio: Maximum radius-edge ratio for cells
            cell_size: Maximum size for cells
        """
        self._dll.set_criteria(self._generator, facet_angle, facet_size, 
                              facet_distance, cell_radius_edge_ratio, cell_size)
    
    def generate_mesh(self) -> bool:
        """
        Generate the tetrahedral mesh.
        
        Returns:
            True if successful, False otherwise
        """
        result = self._dll.generate_mesh(self._generator)
        return result != 0
    
    def export_mesh(self, filename: str) -> bool:
        """
        Export the generated mesh to file in MEDIT format.
        
        Args:
            filename: Output filename (MEDIT format)
            
        Returns:
            True if successful, False otherwise
        """
        result = self._dll.export_mesh(self._generator, filename.encode('utf-8'))
        return result != 0
    
    def export_mesh_warp_format(self, folder_name: str) -> bool:
        """
        Export the generated mesh to WarpSim format.
        
        This creates a folder with multiple model files:
        - {folder_name}/model.vertices - vertex coordinates
        - {folder_name}/model.tetras - tetrahedron indices
        - {folder_name}/model.tris - surface triangle indices
        - {folder_name}/model.edges - edge indices
        - {folder_name}/model.uvs - UV coordinates
        
        Args:
            folder_name: Name of folder to create (e.g. 'liver_mesh')
            
        Returns:
            True if successful, False otherwise
        """
        result = self._dll.export_mesh_warp_format(self._generator, folder_name.encode('utf-8'))
        return result != 0
    
    def get_number_of_vertices(self) -> int:
        """Get the number of vertices in the generated mesh."""
        return self._dll.get_number_of_vertices(self._generator)
    
    def get_number_of_cells(self) -> int:
        """Get the number of cells in the generated mesh."""
        return self._dll.get_number_of_cells(self._generator)
    
    def get_number_of_facets(self) -> int:
        """Get the number of facets in the generated mesh."""
        return self._dll.get_number_of_facets(self._generator)
    
    def get_mesh_statistics(self) -> dict:
        """
        Get statistics about the generated mesh.
        
        Returns:
            Dictionary with mesh statistics
        """
        return {
            'vertices': self.get_number_of_vertices(),
            'cells': self.get_number_of_cells(),
            'facets': self.get_number_of_facets()
        }
    
    def generate_multi_label_meshes(self) -> bool:
        """
        Generate separate tetrahedral meshes for each label in the image.
        
        Returns:
            True if successful, False otherwise
        """
        result = self._dll.generate_multi_label_meshes(self._generator)
        return result != 0
    
    def export_multi_label_meshes(self, base_filename: str) -> bool:
        """
        Export the generated multi-label meshes to separate MEDIT format files.
        
        Args:
            base_filename: Base filename for output (e.g., "mesh" -> "mesh_label_1.mesh", etc.)
            
        Returns:
            True if successful, False otherwise
        """
        result = self._dll.export_multi_label_meshes(self._generator, base_filename.encode('utf-8'))
        return result != 0
    
    def export_multi_label_meshes_warp_format(self, base_folder_name: str) -> bool:
        """
        Export the generated multi-label meshes to WarpSim format.
        
        This creates separate folders for each label:
        - {base_folder_name}_label_1/model.vertices, model.tetras, etc.
        - {base_folder_name}_label_2/model.vertices, model.tetras, etc.
        
        Args:
            base_folder_name: Base name for output folders
            
        Returns:
            True if successful, False otherwise
        """
        result = self._dll.export_multi_label_meshes_warp_format(self._generator, base_folder_name.encode('utf-8'))
        return result != 0
    
    def get_image_labels(self) -> list:
        """
        Get the list of labels found in the loaded image.
        
        Returns:
            List of integer labels
        """
        count = self._dll.get_image_labels_count(self._generator)
        if count == 0:
            return []
        
        labels_array = (ctypes.c_int * count)()
        self._dll.get_image_labels(self._generator, labels_array)
        return [labels_array[i] for i in range(count)]
    
    def get_mesh_statistics_for_label(self, label: int) -> dict:
        """
        Get statistics about the generated mesh for a specific label.
        
        Args:
            label: The label to get statistics for
            
        Returns:
            Dictionary with mesh statistics for the label
        """
        return {
            'label': label,
            'vertices': self._dll.get_number_of_vertices_for_label(self._generator, label),
            'cells': self._dll.get_number_of_cells_for_label(self._generator, label),
            'facets': self._dll.get_number_of_facets_for_label(self._generator, label)
        }
    
    def get_multi_label_mesh_statistics(self) -> dict:
        """
        Get statistics for all labels in the multi-label meshes.
        
        Returns:
            Dictionary with statistics for each label
        """
        labels = self.get_image_labels()
        stats = {}
        total_vertices = 0
        total_cells = 0
        total_facets = 0
        
        for label in labels:
            label_stats = self.get_mesh_statistics_for_label(label)
            stats[f'label_{label}'] = label_stats
            total_vertices += label_stats['vertices']
            total_cells += label_stats['cells']
            total_facets += label_stats['facets']
        
        stats['total'] = {
            'vertices': total_vertices,
            'cells': total_cells,
            'facets': total_facets,
            'labels_count': len(labels)
        }
        
        return stats


def generate_mesh_from_image(image_path: str, output_path: str, **criteria) -> dict:
    """
    Convenience function to generate a mesh from an image file.
    
    Args:
        image_path: Path to the input image file
        output_path: Path for the output mesh file
        **criteria: Mesh generation criteria (see set_criteria for parameters)
        
    Returns:
        Dictionary with mesh statistics
    """
    generator = WarpCGALMeshGenerator()
    
    if not generator.load_image(image_path):
        raise RuntimeError(f"Failed to load image: {image_path}")
    
    generator.set_criteria(**criteria)
    
    if not generator.generate_mesh():
        raise RuntimeError("Failed to generate mesh")
    
    if not generator.export_mesh(output_path):
        raise RuntimeError(f"Failed to export mesh: {output_path}")
    
    return generator.get_mesh_statistics()


def generate_warp_mesh_from_image(image_path: str, output_folder: str, **criteria) -> dict:
    """
    Convenience function to generate a mesh from an image file in WarpSim format.
    
    This creates a folder with multiple model files:
    - {output_folder}/model.vertices - vertex coordinates
    - {output_folder}/model.tetras - tetrahedron indices  
    - {output_folder}/model.tris - surface triangle indices
    - {output_folder}/model.edges - edge indices
    - {output_folder}/model.uvs - UV coordinates
    
    Args:
        image_path: Path to the input image file
        output_folder: Name of folder to create for output files
        **criteria: Mesh generation criteria (see set_criteria for parameters)
        
    Returns:
        Dictionary with mesh statistics
    """
    generator = WarpCGALMeshGenerator()
    
    if not generator.load_image(image_path):
        raise RuntimeError(f"Failed to load image: {image_path}")
    
    generator.set_criteria(**criteria)
    
    if not generator.generate_mesh():
        raise RuntimeError("Failed to generate mesh")
    
    if not generator.export_mesh_warp_format(output_folder):
        raise RuntimeError(f"Failed to export WarpSim format mesh: {output_folder}")
    
    return generator.get_mesh_statistics()


def generate_multi_label_meshes_from_image(image_path: str, output_base: str, **criteria) -> dict:
    """
    Convenience function to generate separate meshes for each label in an image file.
    
    Args:
        image_path: Path to the input image file
        output_base: Base name for output files (e.g., "mesh" -> "mesh_label_1.mesh", etc.)
        **criteria: Mesh generation criteria (see set_criteria for parameters)
        
    Returns:
        Dictionary with statistics for all labels
    """
    generator = WarpCGALMeshGenerator()
    
    if not generator.load_image(image_path):
        raise RuntimeError(f"Failed to load image: {image_path}")
    
    labels = generator.get_image_labels()
    if not labels:
        raise RuntimeError(f"No labels found in image: {image_path}")
    
    print(f"Found {len(labels)} labels: {labels}")
    
    generator.set_criteria(**criteria)
    
    if not generator.generate_multi_label_meshes():
        raise RuntimeError("Failed to generate multi-label meshes")
    
    if not generator.export_multi_label_meshes(output_base):
        raise RuntimeError(f"Failed to export multi-label meshes: {output_base}")
    
    return generator.get_multi_label_mesh_statistics()


def generate_multi_label_warp_meshes_from_image(image_path: str, output_base: str, **criteria) -> dict:
    """
    Convenience function to generate separate WarpSim format meshes for each label in an image file.
    
    This creates separate folders for each label:
    - {output_base}_label_1/model.vertices, model.tetras, etc.
    - {output_base}_label_2/model.vertices, model.tetras, etc.
    
    Args:
        image_path: Path to the input image file
        output_base: Base name for output folders
        **criteria: Mesh generation criteria (see set_criteria for parameters)
        
    Returns:
        Dictionary with statistics for all labels
    """
    generator = WarpCGALMeshGenerator()
    
    if not generator.load_image(image_path):
        raise RuntimeError(f"Failed to load image: {image_path}")
    
    labels = generator.get_image_labels()
    if not labels:
        raise RuntimeError(f"No labels found in image: {image_path}")
    
    print(f"Found {len(labels)} labels: {labels}")
    
    generator.set_criteria(**criteria)
    
    if not generator.generate_multi_label_meshes():
        raise RuntimeError("Failed to generate multi-label meshes")
    
    if not generator.export_multi_label_meshes_warp_format(output_base):
        raise RuntimeError(f"Failed to export multi-label WarpSim format meshes: {output_base}")
    
    return generator.get_multi_label_mesh_statistics()


if __name__ == "__main__":
    # Test the library
    try:
        generator = WarpCGALMeshGenerator()
        print("WarpCGAL library loaded successfully!")
        print(f"Generator created: {generator._generator}")
    except Exception as e:
        print(f"Error: {e}")
