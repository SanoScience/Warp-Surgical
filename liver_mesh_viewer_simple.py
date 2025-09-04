"""
Simple Pyglet-based 3D mesh viewer for MEDIT format files.

This is a simplified version that avoids complex OpenGL calls that might not be available.
"""

import pyglet
import math
import numpy as np

class SimpleMeshLoader:
    """Simple MEDIT format mesh loader."""
    
    def __init__(self):
        self.vertices = []
        self.triangles = []
        
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
            
            i += 1
        
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.triangles = np.array(self.triangles, dtype=np.int32)
        
        print(f"[OK] Loaded mesh: {len(self.vertices)} vertices, {len(self.triangles)} triangles")
        
        # Center and normalize the mesh
        self._center_mesh()
        
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
        
        # Scale to fit in reasonable size
        size = np.max(max_coords - min_coords)
        if size > 0:
            self.vertices = self.vertices * (2.0 / size)  # Scale to fit in 2x2x2 box
            
        print(f"[INFO] Mesh centered and scaled. Original size: {size:.2f}")


class SimpleMeshViewer(pyglet.window.Window):
    """Simple mesh viewer using basic pyglet graphics."""
    
    def __init__(self, mesh_file):
        super().__init__(1200, 800, "Simple Liver Mesh Viewer", resizable=True)
        
        # Load mesh
        self.loader = SimpleMeshLoader()
        self.loader.load_mesh(mesh_file)
        
        # Camera parameters
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.zoom = 1.0
        
        # Mouse tracking
        self.mouse_pressed = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # View mode
        self.wireframe = False
        
        # Convert mesh to drawable format
        self._create_batch()
        
        print("\n=== Simple Liver Mesh Viewer ===")
        print("Controls:")
        print("  Mouse: Rotate view")
        print("  Mouse wheel: Zoom")
        print("  W: Toggle wireframe mode")
        print("  ESC: Exit")
        print()
        
    def _create_batch(self):
        """Create pyglet batch for efficient rendering."""
        self.batch = pyglet.graphics.Batch()
        
        if len(self.loader.triangles) == 0:
            print("[WARNING] No triangles to display")
            return
        
        # Limit triangles for performance (sample every nth triangle)
        max_triangles = 5000
        step = max(1, len(self.loader.triangles) // max_triangles)
        
        vertices_data = []
        colors_data = []
        
        triangle_count = 0
        for i in range(0, len(self.loader.triangles), step):
            tri = self.loader.triangles[i]
            
            # Check if triangle indices are valid
            if (tri[0] >= len(self.loader.vertices) or tri[1] >= len(self.loader.vertices) or 
                tri[2] >= len(self.loader.vertices)):
                continue
            
            v1 = self.loader.vertices[tri[0]]
            v2 = self.loader.vertices[tri[1]]
            v3 = self.loader.vertices[tri[2]]
            
            # Add triangle vertices
            vertices_data.extend([
                v1[0], v1[1], v2[0], v2[1], v3[0], v3[1]  # 2D projection for simplicity
            ])
            
            # Color based on Z coordinate
            color1 = int((v1[2] + 1.0) * 127.5)
            color2 = int((v2[2] + 1.0) * 127.5)
            color3 = int((v3[2] + 1.0) * 127.5)
            
            colors_data.extend([
                color1, 100, 150,  # v1 color
                color2, 100, 150,  # v2 color  
                color3, 100, 150   # v3 color
            ])
            
            triangle_count += 1
        
        print(f"[INFO] Created batch with {triangle_count} triangles")
        
        if vertices_data:
            self.mesh_list = self.batch.add_indexed(
                triangle_count * 3, 
                pyglet.gl.GL_TRIANGLES,
                None,
                list(range(triangle_count * 3)),
                ('v2f', vertices_data),
                ('c3B', colors_data)
            )
        else:
            self.mesh_list = None
    
    def on_draw(self):
        """Render the scene."""
        self.clear()
        
        # Simple 2D projection for now
        pyglet.gl.glPushMatrix()
        
        # Apply transformations
        pyglet.gl.glTranslatef(self.width/2, self.height/2, 0)
        pyglet.gl.glScalef(self.zoom * 100, self.zoom * 100, 1.0)
        pyglet.gl.glRotatef(self.rotation_x, 1, 0, 0)
        pyglet.gl.glRotatef(self.rotation_y, 0, 0, 1)
        
        # Set rendering mode
        if self.wireframe:
            pyglet.gl.glPolygonMode(pyglet.gl.GL_FRONT_AND_BACK, pyglet.gl.GL_LINE)
        else:
            pyglet.gl.glPolygonMode(pyglet.gl.GL_FRONT_AND_BACK, pyglet.gl.GL_FILL)
        
        # Draw mesh
        if self.batch:
            self.batch.draw()
        
        pyglet.gl.glPopMatrix()
    
    def on_mouse_press(self, x, y, button, modifiers):
        """Handle mouse press."""
        if button == pyglet.window.mouse.LEFT:
            self.mouse_pressed = True
            self.last_mouse_x = x
            self.last_mouse_y = y
    
    def on_mouse_release(self, x, y, button, modifiers):
        """Handle mouse release."""
        if button == pyglet.window.mouse.LEFT:
            self.mouse_pressed = False
            
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Handle mouse drag."""
        if buttons & pyglet.window.mouse.LEFT:
            self.rotation_y += dx * 0.5
            self.rotation_x += dy * 0.5
            
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        """Handle mouse scroll."""
        self.zoom *= (1.1 if scroll_y > 0 else 0.9)
        self.zoom = max(0.1, min(5.0, self.zoom))
    
    def on_key_press(self, symbol, modifiers):
        """Handle key press."""
        if symbol == pyglet.window.key.ESCAPE:
            self.close()
        elif symbol == pyglet.window.key.W:
            self.wireframe = not self.wireframe
            print(f"[INFO] Wireframe mode: {'ON' if self.wireframe else 'OFF'}")


def main():
    """Main entry point."""
    import sys
    
    # Default to the generated mesh file
    mesh_file = "liver_mesh_output.mesh"
    
    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    
    import os
    if not os.path.exists(mesh_file):
        print(f"[ERROR] Mesh file not found: {mesh_file}")
        print("Usage: python liver_mesh_viewer_simple.py [mesh_file.mesh]")
        print("Make sure to run test_liver_mesh_simple.py first to generate the mesh.")
        return
    
    try:
        # Create and run viewer
        viewer = SimpleMeshViewer(mesh_file)
        
        print(f"[INFO] Starting simple viewer for {mesh_file}")
        pyglet.app.run()
        
    except Exception as e:
        print(f"[ERROR] Failed to run viewer: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()