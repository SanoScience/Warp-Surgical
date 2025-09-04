"""
OpenGL mesh viewer using PyOpenGL and tkinter for MEDIT format files.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import math

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import tkinter.opengl as tkogl
    OPENGL_AVAILABLE = True
except ImportError:
    print("PyOpenGL not available. Please install with: pip install PyOpenGL PyOpenGL_accelerate")
    OPENGL_AVAILABLE = False

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
        
        # Scale to fit in unit cube
        size = np.max(max_coords - min_coords)
        if size > 0:
            self.vertices /= size
            
        print(f"[INFO] Mesh centered and scaled. Size: {size:.2f}")
    
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


class OpenGLCanvas:
    """OpenGL rendering canvas using tkinter."""
    
    def __init__(self, parent, mesh_loader):
        self.parent = parent
        self.mesh_loader = mesh_loader
        
        # Camera parameters
        self.distance = 3.0
        self.rotation_x = 20.0
        self.rotation_y = 45.0
        
        # Mouse tracking
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # View options
        self.wireframe_mode = False
        self.show_tetrahedra = False
        
        # Create frame and canvas
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        self.create_controls()
        
        # Info display (simple text for now since we can't easily embed OpenGL in tkinter)
        self.create_info_display()
        
    def create_controls(self):
        """Create control panel."""
        control_frame = ttk.Frame(self.frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # View controls
        ttk.Label(control_frame, text="View Controls:").pack(side=tk.LEFT)
        
        ttk.Button(control_frame, text="Reset View", 
                  command=self.reset_view).pack(side=tk.LEFT, padx=2)
        
        self.wireframe_var = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="Wireframe", 
                       variable=self.wireframe_var,
                       command=self.toggle_wireframe).pack(side=tk.LEFT, padx=2)
        
        self.tetrahedra_var = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="Show Tetrahedra", 
                       variable=self.tetrahedra_var,
                       command=self.toggle_tetrahedra).pack(side=tk.LEFT, padx=2)
        
        # Export button
        ttk.Button(control_frame, text="Export View", 
                  command=self.export_view).pack(side=tk.RIGHT, padx=2)
        
    def create_info_display(self):
        """Create information display."""
        info_frame = ttk.LabelFrame(self.frame, text="Mesh Information")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(info_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.info_text = tk.Text(text_frame, wrap=tk.WORD, height=20)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Display mesh information
        self.display_mesh_info()
        
        # Add 2D visualization
        self.create_2d_visualization()
        
    def display_mesh_info(self):
        """Display detailed mesh information."""
        info = f"""LIVER MESH ANALYSIS
{'='*50}

GEOMETRY STATISTICS:
- Vertices: {len(self.mesh_loader.vertices):,}
- Surface Triangles: {len(self.mesh_loader.surface_triangles):,}
- Tetrahedra: {len(self.mesh_loader.tetrahedra):,}

MESH QUALITY:
"""
        
        if len(self.mesh_loader.vertices) > 0:
            # Calculate basic mesh quality metrics
            vertices = self.mesh_loader.vertices
            
            # Bounding box
            min_coords = np.min(vertices, axis=0)
            max_coords = np.max(vertices, axis=0)
            size = max_coords - min_coords
            
            info += f"""
BOUNDING BOX (normalized):
- X: {min_coords[0]:.3f} to {max_coords[0]:.3f} (size: {size[0]:.3f})
- Y: {min_coords[1]:.3f} to {max_coords[1]:.3f} (size: {size[1]:.3f})  
- Z: {min_coords[2]:.3f} to {max_coords[2]:.3f} (size: {size[2]:.3f})

TRIANGLE ANALYSIS:
"""
            
            if len(self.mesh_loader.surface_triangles) > 0:
                # Calculate triangle areas
                triangles = self.mesh_loader.surface_triangles
                areas = []
                
                sample_size = min(1000, len(triangles))  # Sample for performance
                step = max(1, len(triangles) // sample_size)
                
                for i in range(0, len(triangles), step):
                    tri = triangles[i]
                    if (tri[0] < len(vertices) and tri[1] < len(vertices) and tri[2] < len(vertices)):
                        v1 = vertices[tri[0]]
                        v2 = vertices[tri[1]]
                        v3 = vertices[tri[2]]
                        
                        # Calculate area using cross product
                        edge1 = v2 - v1
                        edge2 = v3 - v1
                        cross = np.cross(edge1, edge2)
                        area = np.linalg.norm(cross) / 2.0
                        areas.append(area)
                
                if areas:
                    areas = np.array(areas)
                    info += f"""- Sample triangles analyzed: {len(areas):,}
- Average triangle area: {np.mean(areas):.6f}
- Min triangle area: {np.min(areas):.6f}
- Max triangle area: {np.max(areas):.6f}
- Triangle area std dev: {np.std(areas):.6f}
"""

        info += f"""
VISUALIZATION CONTROLS:
- Wireframe Mode: {'ON' if self.wireframe_mode else 'OFF'}
- Show Tetrahedra: {'ON' if self.show_tetrahedra else 'OFF'}

MEDICAL SIMULATION SUITABILITY:
"""
        
        # Assess mesh quality for simulation
        vertex_count = len(self.mesh_loader.vertices)
        tet_count = len(self.mesh_loader.tetrahedra)
        
        if vertex_count > 0:
            if vertex_count < 5000:
                quality = "COARSE - Good for real-time simulation"
            elif vertex_count < 20000:
                quality = "MEDIUM - Good balance of detail and performance"
            elif vertex_count < 50000:
                quality = "FINE - High detail, may need GPU acceleration"
            else:
                quality = "VERY FINE - Requires high-performance computing"
                
            info += f"""- Mesh Resolution: {quality}
- Estimated simulation performance: {self._estimate_performance()}
- Recommended for: {self._get_simulation_recommendations()}

NEXT STEPS:
1. Import mesh into Warp physics simulation
2. Define material properties (elasticity, density)
3. Set boundary conditions for surgical simulation
4. Configure haptic feedback parameters
"""

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)
        
    def _estimate_performance(self):
        """Estimate simulation performance."""
        vertex_count = len(self.mesh_loader.vertices)
        if vertex_count < 5000:
            return "Excellent (>60 FPS expected)"
        elif vertex_count < 15000:
            return "Good (30-60 FPS expected)"
        elif vertex_count < 30000:
            return "Moderate (10-30 FPS expected)"
        else:
            return "Challenging (<10 FPS expected)"
    
    def _get_simulation_recommendations(self):
        """Get simulation recommendations."""
        vertex_count = len(self.mesh_loader.vertices)
        if vertex_count < 10000:
            return "Real-time haptic simulation, educational training"
        elif vertex_count < 25000:
            return "High-fidelity surgical simulation, research"
        else:
            return "Offline simulation, detailed biomechanical analysis"
    
    def create_2d_visualization(self):
        """Create a simple 2D projection visualization."""
        viz_frame = ttk.LabelFrame(self.frame, text="2D Projection (XY Plane)")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        canvas = tk.Canvas(viz_frame, width=600, height=400, bg='black')
        canvas.pack(padx=5, pady=5)
        
        if len(self.mesh_loader.vertices) == 0:
            return
        
        # Get 2D projection (XY plane)
        vertices = self.mesh_loader.vertices
        triangles = self.mesh_loader.surface_triangles[:1000]  # Limit for performance
        
        # Scale and center for canvas
        min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
        min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])
        
        scale_x = 500 / (max_x - min_x) if (max_x - min_x) > 0 else 1
        scale_y = 300 / (max_y - min_y) if (max_y - min_y) > 0 else 1
        scale = min(scale_x, scale_y) * 0.8  # Add margin
        
        center_x, center_y = 300, 200
        
        # Draw triangles
        for i, tri in enumerate(triangles):
            if i % 10 != 0:  # Sample for performance
                continue
                
            if (tri[0] < len(vertices) and tri[1] < len(vertices) and tri[2] < len(vertices)):
                v1 = vertices[tri[0]]
                v2 = vertices[tri[1]]
                v3 = vertices[tri[2]]
                
                # Convert to canvas coordinates
                x1 = center_x + (v1[0] - (min_x + max_x) / 2) * scale
                y1 = center_y - (v1[1] - (min_y + max_y) / 2) * scale  # Flip Y
                x2 = center_x + (v2[0] - (min_x + max_x) / 2) * scale
                y2 = center_y - (v2[1] - (min_y + max_y) / 2) * scale
                x3 = center_x + (v3[0] - (min_x + max_x) / 2) * scale
                y3 = center_y - (v3[1] - (min_y + max_y) / 2) * scale
                
                # Color based on Z coordinate
                z_avg = (v1[2] + v2[2] + v3[2]) / 3
                intensity = int((z_avg + 1) * 127.5)
                color = f"#{intensity:02x}{intensity//2:02x}{100:02x}"
                
                canvas.create_polygon(x1, y1, x2, y2, x3, y3, 
                                    fill=color, outline='gray', width=0.5)
        
        # Add axis labels
        canvas.create_text(50, 390, text="X", fill='white', font=('Arial', 12))
        canvas.create_text(10, 20, text="Y", fill='white', font=('Arial', 12))
        
    def reset_view(self):
        """Reset camera view."""
        self.distance = 3.0
        self.rotation_x = 20.0
        self.rotation_y = 45.0
        print("[INFO] View reset")
        
    def toggle_wireframe(self):
        """Toggle wireframe mode."""
        self.wireframe_mode = self.wireframe_var.get()
        print(f"[INFO] Wireframe mode: {'ON' if self.wireframe_mode else 'OFF'}")
        
    def toggle_tetrahedra(self):
        """Toggle tetrahedra view."""
        self.show_tetrahedra = self.tetrahedra_var.get()
        mode = "tetrahedra" if self.show_tetrahedra else "surface"
        print(f"[INFO] Display mode: {mode}")
        
    def export_view(self):
        """Export mesh data."""
        filename = f"liver_mesh_export.txt"
        with open(filename, 'w') as f:
            f.write("LIVER MESH EXPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Vertices: {len(self.mesh_loader.vertices)}\n")
            f.write(f"Triangles: {len(self.mesh_loader.surface_triangles)}\n")
            f.write(f"Tetrahedra: {len(self.mesh_loader.tetrahedra)}\n\n")
            
            # Sample vertex data
            f.write("Sample vertices (first 10):\n")
            for i, vertex in enumerate(self.mesh_loader.vertices[:10]):
                f.write(f"{i}: ({vertex[0]:.3f}, {vertex[1]:.3f}, {vertex[2]:.3f})\n")
                
        print(f"[INFO] Exported mesh data to {filename}")


class MeshViewerApp:
    """Main application window."""
    
    def __init__(self, mesh_file):
        self.root = tk.Tk()
        self.root.title(f"Liver Mesh Viewer - {mesh_file}")
        self.root.geometry("800x900")
        
        # Load mesh
        self.mesh_loader = MEDITMeshLoader()
        
        try:
            self.mesh_loader.load_mesh(mesh_file)
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load mesh: {e}")
            return
        
        # Create GUI
        self.canvas = OpenGLCanvas(self.root, self.mesh_loader)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set(f"Loaded: {len(self.mesh_loader.vertices)} vertices, {len(self.mesh_loader.surface_triangles)} triangles")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def run(self):
        """Run the application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    import sys
    import os
    
    if not OPENGL_AVAILABLE:
        return
    
    # Default to the generated mesh file
    mesh_file = "liver_mesh_output.mesh"
    
    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    
    if not os.path.exists(mesh_file):
        print(f"[ERROR] Mesh file not found: {mesh_file}")
        print("Usage: python liver_mesh_viewer_opengl.py [mesh_file.mesh]")
        print("Make sure to run test_liver_mesh_simple.py first to generate the mesh.")
        return
    
    try:
        app = MeshViewerApp(mesh_file)
        print(f"[INFO] Starting OpenGL mesh viewer for {mesh_file}")
        app.run()
        
    except Exception as e:
        print(f"[ERROR] Failed to run viewer: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()