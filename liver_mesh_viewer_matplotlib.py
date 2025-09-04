"""
Matplotlib-based 3D mesh viewer for MEDIT format files.

This viewer provides interactive 3D visualization of tetrahedral meshes
using matplotlib's 3D plotting capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches
from matplotlib.widgets import Button, CheckButtons, Slider

class MEDITMeshLoader:
    """Loads MEDIT format mesh files."""
    
    def __init__(self):
        self.vertices = []
        self.triangles = []
        self.tetrahedra = []
        
    def load_mesh(self, filename):
        """Load a MEDIT format mesh file."""
        print(f"[INFO] Loading mesh from {filename}")
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line == "Vertices":
                i += 1
                num_vertices = int(lines[i].strip())
                print(f"[INFO] Loading {num_vertices} vertices...")
                
                for j in range(num_vertices):
                    i += 1
                    parts = lines[i].strip().split()
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    self.vertices.append([x, y, z])
            
            elif line == "Triangles":
                i += 1
                num_triangles = int(lines[i].strip())
                print(f"[INFO] Loading {num_triangles} triangles...")
                
                for j in range(num_triangles):
                    i += 1
                    parts = lines[i].strip().split()
                    # MEDIT uses 1-based indexing, convert to 0-based
                    v1, v2, v3 = int(parts[0])-1, int(parts[1])-1, int(parts[2])-1
                    self.triangles.append([v1, v2, v3])
                    
            elif line == "Tetrahedra":
                i += 1
                num_tetrahedra = int(lines[i].strip())
                print(f"[INFO] Loading {num_tetrahedra} tetrahedra...")
                
                for j in range(num_tetrahedra):
                    i += 1
                    parts = lines[i].strip().split()
                    # MEDIT uses 1-based indexing, convert to 0-based
                    v1 = int(parts[0])-1
                    v2 = int(parts[1])-1 
                    v3 = int(parts[2])-1
                    v4 = int(parts[3])-1
                    self.tetrahedra.append([v1, v2, v3, v4])
            
            i += 1
        
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.triangles = np.array(self.triangles, dtype=np.int32)
        self.tetrahedra = np.array(self.tetrahedra, dtype=np.int32)
        
        print(f"[OK] Loaded mesh: {len(self.vertices)} vertices, {len(self.triangles)} triangles, {len(self.tetrahedra)} tetrahedra")
        
        # Center and normalize the mesh
        self._center_mesh()
        
        # Extract surface triangles from tetrahedra
        if len(self.tetrahedra) > 0:
            self.surface_triangles = self._extract_surface_triangles()
        else:
            self.surface_triangles = self.triangles
        
        return True
        
    def _center_mesh(self):
        """Center the mesh and normalize its size."""
        if len(self.vertices) == 0:
            return
            
        # Find bounding box
        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)
        
        # Center the mesh
        center = (min_coords + max_coords) / 2
        self.vertices -= center
        
        # Scale to reasonable size
        size = np.max(max_coords - min_coords)
        if size > 0:
            self.vertices = self.vertices * (2.0 / size)  # Scale to fit in 2x2x2 box
            
        print(f"[INFO] Mesh centered and scaled. Original size: {size:.2f}")
    
    def _extract_surface_triangles(self):
        """Extract surface triangles from tetrahedra."""
        if len(self.tetrahedra) == 0:
            return self.triangles
            
        # For each face of each tetrahedron, count how many tets share it
        face_count = {}
        
        for tet in self.tetrahedra:
            # Four faces of a tetrahedron
            faces = [
                tuple(sorted([tet[0], tet[1], tet[2]])),
                tuple(sorted([tet[0], tet[1], tet[3]])),
                tuple(sorted([tet[0], tet[2], tet[3]])),
                tuple(sorted([tet[1], tet[2], tet[3]]))
            ]
            
            for face in faces:
                face_count[face] = face_count.get(face, 0) + 1
        
        # Surface faces appear only once
        surface_triangles = []
        for face, count in face_count.items():
            if count == 1:
                surface_triangles.append(list(face))
        
        print(f"[INFO] Extracted {len(surface_triangles)} surface triangles from tetrahedra")
        return np.array(surface_triangles, dtype=np.int32)


class InteractiveMeshViewer:
    """Interactive 3D mesh viewer using matplotlib."""
    
    def __init__(self, mesh_file):
        # Load mesh
        self.loader = MEDITMeshLoader()
        if not self.loader.load_mesh(mesh_file):
            raise Exception("Failed to load mesh")
        
        # View parameters
        self.show_wireframe = False
        self.show_tetrahedra = False
        self.alpha = 0.7
        self.triangle_limit = 5000  # Limit for performance
        
        # Setup the plot
        self.setup_plot()
        self.create_controls()
        self.update_plot()
        
    def setup_plot(self):
        """Setup the matplotlib 3D plot."""
        self.fig = plt.figure(figsize=(15, 10))
        
        # Main 3D plot
        self.ax = self.fig.add_subplot(121, projection='3d')
        self.ax.set_title(f'Liver Mesh - {len(self.loader.vertices):,} vertices')
        
        # Info panel
        self.info_ax = self.fig.add_subplot(122)
        self.info_ax.axis('off')
        
        # Set equal aspect ratio for 3D plot
        self.ax.set_box_aspect([1,1,1])
        
        # Adjust layout
        plt.tight_layout()
        
    def create_controls(self):
        """Create interactive controls."""
        # Make space for controls
        plt.subplots_adjust(bottom=0.2)
        
        # Wireframe checkbox
        wireframe_ax = plt.axes([0.1, 0.05, 0.15, 0.04])
        self.wireframe_check = CheckButtons(wireframe_ax, ['Wireframe'], [False])
        self.wireframe_check.on_clicked(self.toggle_wireframe)
        
        # Tetrahedra checkbox
        tetrahedra_ax = plt.axes([0.3, 0.05, 0.15, 0.04])
        self.tetrahedra_check = CheckButtons(tetrahedra_ax, ['Show Tets'], [False])
        self.tetrahedra_check.on_clicked(self.toggle_tetrahedra)
        
        # Alpha slider
        alpha_ax = plt.axes([0.1, 0.12, 0.35, 0.02])
        self.alpha_slider = Slider(alpha_ax, 'Transparency', 0.1, 1.0, valinit=0.7)
        self.alpha_slider.on_changed(self.update_alpha)
        
        # Triangle limit slider
        limit_ax = plt.axes([0.1, 0.15, 0.35, 0.02])
        self.limit_slider = Slider(limit_ax, 'Triangle Limit', 1000, 20000, valinit=5000, valfmt='%d')
        self.limit_slider.on_changed(self.update_limit)
        
        # Update button
        update_ax = plt.axes([0.55, 0.05, 0.1, 0.04])
        self.update_button = Button(update_ax, 'Update')
        self.update_button.on_clicked(lambda x: self.update_plot())
        
        # Reset view button
        reset_ax = plt.axes([0.70, 0.05, 0.1, 0.04])
        self.reset_button = Button(reset_ax, 'Reset View')
        self.reset_button.on_clicked(self.reset_view)
        
    def toggle_wireframe(self, label):
        """Toggle wireframe mode."""
        self.show_wireframe = not self.show_wireframe
        print(f"[INFO] Wireframe mode: {'ON' if self.show_wireframe else 'OFF'}")
        self.update_plot()
        
    def toggle_tetrahedra(self, label):
        """Toggle tetrahedra view."""
        self.show_tetrahedra = not self.show_tetrahedra
        mode = "tetrahedra" if self.show_tetrahedra else "surface"
        print(f"[INFO] Display mode: {mode}")
        self.update_plot()
        
    def update_alpha(self, val):
        """Update transparency."""
        self.alpha = val
        self.update_plot()
        
    def update_limit(self, val):
        """Update triangle limit."""
        self.triangle_limit = int(val)
        self.update_plot()
        
    def reset_view(self, event):
        """Reset the 3D view."""
        self.ax.view_init(elev=20, azim=45)
        self.ax.dist = 10
        plt.draw()
        
    def update_plot(self):
        """Update the 3D plot."""
        self.ax.clear()
        
        if len(self.loader.vertices) == 0:
            self.ax.text(0, 0, 0, 'No mesh data loaded', fontsize=12)
            plt.draw()
            return
            
        # Choose triangles to display
        if self.show_tetrahedra and len(self.loader.tetrahedra) > 0:
            triangles = self._get_tetrahedra_faces()
        else:
            triangles = self.loader.surface_triangles
        
        # Limit triangles for performance
        if len(triangles) > self.triangle_limit:
            step = len(triangles) // self.triangle_limit
            triangles = triangles[::step]
        
        # Create triangle collection
        triangle_verts = []
        colors = []
        
        for tri in triangles:
            if (tri[0] < len(self.loader.vertices) and 
                tri[1] < len(self.loader.vertices) and 
                tri[2] < len(self.loader.vertices)):
                
                v1 = self.loader.vertices[tri[0]]
                v2 = self.loader.vertices[tri[1]]
                v3 = self.loader.vertices[tri[2]]
                
                triangle_verts.append([v1, v2, v3])
                
                # Color based on Z coordinate
                z_avg = (v1[2] + v2[2] + v3[2]) / 3
                colors.append(plt.cm.viridis((z_avg + 1) / 2))  # Normalize to [0,1]
        
        if triangle_verts:
            # Create the 3D polygon collection
            if self.show_wireframe:
                collection = Poly3DCollection(triangle_verts, alpha=self.alpha, 
                                            facecolors='none', edgecolors='black', linewidths=0.5)
            else:
                collection = Poly3DCollection(triangle_verts, alpha=self.alpha, 
                                            facecolors=colors, edgecolors='black', linewidths=0.1)
            
            self.ax.add_collection3d(collection)
        
        # Set plot properties
        vertices = self.loader.vertices
        if len(vertices) > 0:
            self.ax.set_xlim(np.min(vertices[:, 0]), np.max(vertices[:, 0]))
            self.ax.set_ylim(np.min(vertices[:, 1]), np.max(vertices[:, 1]))
            self.ax.set_zlim(np.min(vertices[:, 2]), np.max(vertices[:, 2]))
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'Liver Mesh - {len(triangles):,} triangles displayed')
        
        # Update info panel
        self.update_info_panel()
        
        plt.draw()
        
    def _get_tetrahedra_faces(self):
        """Get faces from tetrahedra for wireframe display."""
        faces = []
        
        # Sample tetrahedra for performance
        step = max(1, len(self.loader.tetrahedra) // (self.triangle_limit // 4))
        
        for tet in self.loader.tetrahedra[::step]:
            # Four faces of a tetrahedron
            faces.extend([
                [tet[0], tet[1], tet[2]],
                [tet[0], tet[1], tet[3]],
                [tet[0], tet[2], tet[3]],
                [tet[1], tet[2], tet[3]]
            ])
        
        return np.array(faces, dtype=np.int32)
        
    def update_info_panel(self):
        """Update the information panel."""
        self.info_ax.clear()
        self.info_ax.axis('off')
        
        # Create info text
        info_text = f"""LIVER MESH STATISTICS

Geometry:
• Vertices: {len(self.loader.vertices):,}
• Surface Triangles: {len(self.loader.surface_triangles):,}
• Tetrahedra: {len(self.loader.tetrahedra):,}

Display Settings:
• Mode: {'Tetrahedra' if self.show_tetrahedra else 'Surface'}
• Style: {'Wireframe' if self.show_wireframe else 'Solid'}
• Transparency: {self.alpha:.1f}
• Triangles Shown: {min(self.triangle_limit, len(self.loader.surface_triangles)):,}

Mesh Quality:
"""
        
        # Add quality assessment
        vertex_count = len(self.loader.vertices)
        if vertex_count < 5000:
            info_text += "• Resolution: COARSE\n• Simulation: Real-time capable\n• Use case: Training, haptics"
        elif vertex_count < 20000:
            info_text += "• Resolution: MEDIUM\n• Simulation: Good performance\n• Use case: Standard surgery sim"
        elif vertex_count < 50000:
            info_text += "• Resolution: FINE\n• Simulation: GPU recommended\n• Use case: High-fidelity research"
        else:
            info_text += "• Resolution: VERY FINE\n• Simulation: HPC required\n• Use case: Detailed analysis"
        
        if len(self.loader.surface_triangles) > 0:
            # Calculate mesh bounds
            vertices = self.loader.vertices
            bounds = np.max(vertices, axis=0) - np.min(vertices, axis=0)
            info_text += f"""

Mesh Bounds (normalized):
• X: {bounds[0]:.3f}
• Y: {bounds[1]:.3f}  
• Z: {bounds[2]:.3f}

Medical Simulation Notes:
• Mesh appears suitable for liver
  biomechanical simulation
• Consider material properties:
  - Young's modulus: 5-50 kPa
  - Poisson's ratio: 0.45-0.49
  - Density: ~1000 kg/m³
"""

        # Display the text
        self.info_ax.text(0.05, 0.95, info_text, transform=self.info_ax.transAxes,
                         verticalalignment='top', fontfamily='monospace', fontsize=9)
        
    def show(self):
        """Display the viewer."""
        print("\n=== Interactive Liver Mesh Viewer ===")
        print("Controls:")
        print("  - Mouse: Rotate, zoom, pan")
        print("  - Checkboxes: Toggle wireframe/tetrahedra")
        print("  - Sliders: Adjust transparency and triangle limit")  
        print("  - Buttons: Update view, reset camera")
        print()
        
        plt.show()


def main():
    """Main entry point."""
    import sys
    import os
    
    # Default to the generated mesh file
    mesh_file = "liver_mesh_output.mesh"
    
    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    
    if not os.path.exists(mesh_file):
        print(f"[ERROR] Mesh file not found: {mesh_file}")
        print("Usage: python liver_mesh_viewer_matplotlib.py [mesh_file.mesh]")
        print("Make sure to run test_liver_mesh_simple.py first to generate the mesh.")
        return
    
    try:
        # Create and show viewer
        viewer = InteractiveMeshViewer(mesh_file)
        viewer.show()
        
    except Exception as e:
        print(f"[ERROR] Failed to run viewer: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()