#!/usr/bin/env python3

"""
Position Based Fluids (PBF) Example with OpenGL Rendering

This example demonstrates PBF fluid simulation using NVIDIA Warp with real-time
OpenGL particle visualization. Based on the paper "Position based fluids" by
Macklin & Müller (2013).

Key differences from SPH:
- Uses position constraints instead of pressure forces
- Iterative constraint solver for incompressibility  
- Better stability and visual quality
- Spatial hash grid for efficient neighbor finding

Controls:
- Mouse: Rotate camera
- WASD: Move camera
- Space: Pause/unpause simulation
- R: Reset simulation
"""

import math
import numpy as np
import warp as wp

try:
    import warp.render
    from warp.render.imgui_manager import ImGuiManager
    IMGUI_AVAILABLE = True
except ImportError:
    print("Warp render module not available. Using basic OpenGL fallback.")
    warp.render = None
    IMGUI_AVAILABLE = False

try:
    from OpenGL.GL import *
    from OpenGL.GLUT import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    print("PyOpenGL not available. Install with: pip install PyOpenGL PyOpenGL_accelerate")
    OPENGL_AVAILABLE = False

import threading
import time
from pbf_kernels import (
    pbf_predict_positions,
    pbf_compute_density,
    pbf_compute_lambda,
    pbf_compute_delta_positions,
    pbf_apply_delta_positions,
    pbf_apply_boundaries,
    pbf_update_velocities_positions,
)


# Simulation parameters (stable defaults)
SIM_PARAMS = {
    'particle_radius': 0.015,
    'smoothing_radius': 0.04,
    'rest_density': 1000.0,
    'constraint_iterations': 6,
    'damping': 0.96,
    'gravity': -5.0,
    'dt': 1.0/120.0,
    'max_velocity': 8.0,
    'constraint_epsilon': 200.0,
    'grid_dim': 64,
    'domain_min': np.array([-1.2, -0.8, -1.2]),
    'domain_max': np.array([1.2, 2.0, 1.2])
}

class PBFImGuiManager(ImGuiManager):
    """ImGui manager for PBF simulation parameter control."""

    def __init__(self, renderer, simulation, window_pos=(10, 10), window_size=(320, 480)):
        super().__init__(renderer)
        if not self.is_available:
            return

        self.sim = simulation
        self.window_pos = window_pos
        self.window_size = window_size
        
        # Store original parameters for reset
        self.original_params = self.sim.params.copy()
        
        # UI state
        self.show_debug_info = True
        self.show_performance = True
        self.auto_reset_on_change = False

    def draw_ui(self):
        if not self.is_available:
            return
            
        # Set window position and size
        self.imgui.set_next_window_size(self.window_size[0], self.window_size[1], self.imgui.ONCE)
        self.imgui.set_next_window_position(self.window_pos[0], self.window_pos[1], self.imgui.ONCE)

        self.imgui.begin("PBF Simulation Controls")
        
        # Simulation control
        self.imgui.text("Simulation Control:")
        
        if self.imgui.button("Pause" if not self.sim.paused else "Resume"):
            self.sim.paused = not self.sim.paused
        
        self.imgui.same_line()
        if self.imgui.button("Reset"):
            self.sim.reset_particles()
        
        self.imgui.same_line()
        if self.imgui.button("Reset Params"):
            self.sim.params = self.original_params.copy()
            
        self.imgui.separator()
        
        # Core fluid parameters
        self.imgui.text("Fluid Parameters:")
        
        changed, new_dt = self.imgui.slider_float(
            "Timestep", self.sim.params['dt'], 1.0/240.0, 1.0/30.0, "%.4f"
        )
        if changed:
            self.sim.params['dt'] = new_dt
        
        changed, new_iterations = self.imgui.slider_int(
            "Constraint Iterations", self.sim.params['constraint_iterations'], 4, 12
        )
        if changed:
            self.sim.params['constraint_iterations'] = new_iterations
            
        changed, new_damping = self.imgui.slider_float(
            "Damping", self.sim.params['damping'], 0.90, 0.99, "%.3f"
        )
        if changed:
            self.sim.params['damping'] = new_damping
            
        changed, new_gravity = self.imgui.slider_float(
            "Gravity", self.sim.params['gravity'], -12.0, -2.0, "%.1f"
        )
        if changed:
            self.sim.params['gravity'] = new_gravity
        
        self.imgui.separator()
        
        # Particle parameters
        self.imgui.text("Particle Parameters:")
        
        changed, new_radius = self.imgui.slider_float(
            "Particle Radius", self.sim.params['particle_radius'], 0.010, 0.030, "%.3f"
        )
        if changed:
            self.sim.params['particle_radius'] = new_radius
            
        changed, new_smoothing = self.imgui.slider_float(
            "Smoothing Radius", self.sim.params['smoothing_radius'], 0.025, 0.080, "%.3f"
        )
        if changed:
            self.sim.params['smoothing_radius'] = new_smoothing
            
        changed, new_density = self.imgui.slider_float(
            "Rest Density", self.sim.params['rest_density'], 500.0, 2000.0, "%.0f"
        )
        if changed:
            self.sim.params['rest_density'] = new_density

        changed, new_maxvel = self.imgui.slider_float(
            "Max Velocity", self.sim.params['max_velocity'], 4.0, 12.0, "%.1f"
        )
        if changed:
            self.sim.params['max_velocity'] = new_maxvel

        changed, new_eps = self.imgui.slider_float(
            "Constraint Epsilon", self.sim.params['constraint_epsilon'], 50.0, 400.0, "%.1f"
        )
        if changed:
            self.sim.params['constraint_epsilon'] = new_eps
        
        self.imgui.separator()
        
        # Domain bounds
        self.imgui.text("Domain Bounds:")
        
        domain_min = list(self.sim.params['domain_min'])
        domain_max = list(self.sim.params['domain_max'])
        
        changed_min, domain_min = self.drag_vec3("Min Bounds", wp.vec3(*domain_min), speed=0.1)
        if changed_min:
            self.sim.params['domain_min'] = np.array([domain_min[0], domain_min[1], domain_min[2]])
            
        changed_max, domain_max = self.drag_vec3("Max Bounds", wp.vec3(*domain_max), speed=0.1)
        if changed_max:
            self.sim.params['domain_max'] = np.array([domain_max[0], domain_max[1], domain_max[2]])
        
        self.imgui.separator()
        
        # Quick presets
        self.imgui.text("Quick Presets:")
        
        if self.imgui.button("Conservative"):
            self.sim.params.update({
                'dt': 1.0/150.0,
                'constraint_iterations': 8,
                'damping': 0.96,
                'gravity': -5.0,
                'constraint_epsilon': 220.0,
                'max_velocity': 8.0
            })
            
        self.imgui.same_line()
        if self.imgui.button("Fast"):
            self.sim.params.update({
                'dt': 1.0/90.0,
                'constraint_iterations': 6,
                'damping': 0.95,
                'gravity': -7.0,
                'constraint_epsilon': 180.0,
                'max_velocity': 9.0
            })
            
        self.imgui.same_line()
        if self.imgui.button("High Quality"):
            self.sim.params.update({
                'dt': 1.0/180.0,
                'constraint_iterations': 10,
                'damping': 0.97,
                'gravity': -5.0,
                'constraint_epsilon': 250.0,
                'max_velocity': 8.0
            })
        
        self.imgui.separator()
        
        # Debug info
        changed, self.show_debug_info = self.imgui.checkbox("Show Debug Info", self.show_debug_info)
        
        if self.show_debug_info:
            self.imgui.text(f"Particles: {self.sim.num_particles}")
            self.imgui.text(f"Status: {'PAUSED' if self.sim.paused else 'RUNNING'}")
            
            # Get some debug stats
            try:
                positions = self.sim.positions.numpy()
                velocities = self.sim.velocities.numpy()
                
                vel_magnitudes = np.linalg.norm(velocities, axis=1)
                avg_vel = np.mean(vel_magnitudes)
                max_vel = np.max(vel_magnitudes)
                
                bbox_min = np.min(positions, axis=0)
                bbox_max = np.max(positions, axis=0)
                spread = np.max(bbox_max - bbox_min)
                
                self.imgui.text(f"Avg Velocity: {avg_vel:.2f}")
                self.imgui.text(f"Max Velocity: {max_vel:.2f}")
                self.imgui.text(f"Particle Spread: {spread:.2f}")
                
            except Exception:
                self.imgui.text("Debug data unavailable")
        
        self.imgui.separator()
        
        # Performance info
        changed, self.show_performance = self.imgui.checkbox("Show Performance", self.show_performance)
        
        if self.show_performance:
            fps = self.imgui.get_io().framerate
            frame_time = 1000.0 / fps if fps > 0 else 0
            self.imgui.text(f"FPS: {fps:.1f}")
            self.imgui.text(f"Frame Time: {frame_time:.2f} ms")
        
        # Tips
        self.imgui.separator()
        self.imgui.text("Tips:")
        self.imgui.text_wrapped("• Lower timestep for stability")
        self.imgui.text_wrapped("• More iterations for accuracy")
        self.imgui.text_wrapped("• Higher damping for stability")
        self.imgui.text_wrapped("• Adjust smoothing radius carefully")

        self.imgui.end()

class PBFSimulation:
    def __init__(self, num_particles=2048):
        self.num_particles = num_particles
        self.params = SIM_PARAMS.copy()
        self.paused = False
        
        # Initialize Warp
        wp.init()
        self.device = wp.get_device()
        
        # Create particle arrays
        self.positions = wp.zeros(num_particles, dtype=wp.vec3, device=self.device)
        self.velocities = wp.zeros(num_particles, dtype=wp.vec3, device=self.device)
        self.predicted_positions = wp.zeros(num_particles, dtype=wp.vec3, device=self.device)
        self.densities = wp.zeros(num_particles, dtype=wp.float32, device=self.device)
        self.lambdas = wp.zeros(num_particles, dtype=wp.float32, device=self.device)
        self.delta_positions = wp.zeros(num_particles, dtype=wp.vec3, device=self.device)
        
        # Create spatial hash grid
        self.hash_grid = wp.HashGrid(
            dim_x=self.params['grid_dim'],
            dim_y=self.params['grid_dim'],
            dim_z=self.params['grid_dim'],
            device=self.device
        )
        
        # Initialize particle positions
        self.reset_particles()
        
        print(f"PBF Simulation initialized with {num_particles} particles")
        
    def reset_particles(self):
        """Initialize particles in a dam-break setup."""
        positions = []
        
        # Create a dam of particles
        spacing = self.params['particle_radius'] * 2.0
        layers_x = int(1.0 / spacing)
        layers_y = int(1.5 / spacing) 
        layers_z = int(1.0 / spacing)
        
        count = 0
        for i in range(layers_x):
            for j in range(layers_y):
                for k in range(layers_z):
                    if count >= self.num_particles:
                        break
                        
                    x = -1.0 + i * spacing + (np.random.random() - 0.5) * spacing * 0.1
                    y = -0.8 + j * spacing + (np.random.random() - 0.5) * spacing * 0.1
                    z = -0.5 + k * spacing + (np.random.random() - 0.5) * spacing * 0.1
                    
                    positions.append([x, y, z])
                    count += 1
                    
                if count >= self.num_particles:
                    break
            if count >= self.num_particles:
                break
        
        # Fill remaining particles if needed
        while count < self.num_particles:
            x = np.random.uniform(-1.0, 0.0)
            y = np.random.uniform(-0.8, 0.5)
            z = np.random.uniform(-0.5, 0.5)
            positions.append([x, y, z])
            count += 1
        
        # Copy to GPU
        positions_array = np.array(positions, dtype=np.float32)
        wp.copy(self.positions, wp.array(positions_array, dtype=wp.vec3, device=self.device))
        
        # Reset velocities
        wp.copy(self.velocities, wp.zeros(self.num_particles, dtype=wp.vec3, device=self.device))
        
    def step(self):
        """Run one simulation step."""
        if self.paused:
            return
            
        # Step 1: Predict positions
        gravity = wp.vec3(0.0, self.params['gravity'], 0.0)
        wp.launch(
            pbf_predict_positions,
            dim=self.num_particles,
            inputs=[
                self.positions, self.velocities, self.predicted_positions,
                gravity, self.params['dt'], self.params['max_velocity']
            ],
            device=self.device
        )
        
        # Step 2: Build spatial hash
        self.hash_grid.build(self.predicted_positions, self.params['smoothing_radius'])
        
        # Step 3: Constraint projection iterations
        for iteration in range(self.params['constraint_iterations']):
            # Compute density
            wp.launch(
                pbf_compute_density,
                dim=self.num_particles,
                inputs=[self.predicted_positions, self.densities, self.hash_grid.id, self.params['smoothing_radius']],
                device=self.device
            )
            
            # Compute lambda
            wp.launch(
                pbf_compute_lambda,
                dim=self.num_particles,
                inputs=[
                    self.predicted_positions, self.densities, self.lambdas, self.hash_grid.id,
                    self.params['smoothing_radius'], self.params['rest_density'], self.params['constraint_epsilon']
                ],
                device=self.device
            )
            
            # Compute position corrections
            wp.launch(
                pbf_compute_delta_positions,
                dim=self.num_particles,
                inputs=[
                    self.predicted_positions, self.lambdas, self.delta_positions, self.hash_grid.id,
                    self.params['smoothing_radius'], self.params['rest_density']
                ],
                device=self.device
            )
            
            # Apply corrections
            wp.launch(
                pbf_apply_delta_positions,
                dim=self.num_particles,
                inputs=[self.predicted_positions, self.delta_positions],
                device=self.device
            )
        
        # Step 4: Apply boundary conditions
        domain_min = wp.vec3(*self.params['domain_min'])
        domain_max = wp.vec3(*self.params['domain_max'])
        wp.launch(
            pbf_apply_boundaries,
            dim=self.num_particles,
            inputs=[self.predicted_positions, self.velocities, domain_min, domain_max, 0.2],
            device=self.device
        )
        
        # Step 5: Update velocities and positions
        wp.launch(
            pbf_update_velocities_positions,
            dim=self.num_particles,
            inputs=[
                self.positions, self.velocities, self.predicted_positions,
                self.params['damping'], self.params['dt'], self.params['max_velocity']
            ],
            device=self.device
        )

# Enhanced OpenGL Renderer with ImGui integration
class OpenGLRenderer:
    def __init__(self, simulation, use_imgui=True):
        self.sim = simulation
        self.camera_distance = 6.0
        self.camera_theta = 45.0
        self.camera_phi = 30.0
        self.mouse_last_x = 0
        self.mouse_last_y = 0
        self.mouse_dragging = False
        
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Initialize Warp renderer and ImGui if available
        self.use_imgui = use_imgui and IMGUI_AVAILABLE
        if self.use_imgui:
            try:
                self.warp_renderer = warp.render.OpenGLRenderer(vsync=False)
                self.imgui_manager = PBFImGuiManager(self.warp_renderer, simulation)
                if self.imgui_manager.is_available:
                    self.warp_renderer.render_2d_callbacks.append(self.imgui_manager.render_frame)
                else:
                    self.use_imgui = False
                    print("ImGui not available, falling back to basic OpenGL")
            except Exception as e:
                print(f"Failed to initialize ImGui: {e}")
                self.use_imgui = False
        
    def init_gl(self):
        """Initialize OpenGL settings."""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glClearColor(0.1, 0.1, 0.2, 1.0)
        glPointSize(8.0)
        
    def setup_camera(self):
        """Setup camera projection and view."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, 1.0, 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Camera position
        cam_x = self.camera_distance * math.sin(math.radians(self.camera_theta)) * math.cos(math.radians(self.camera_phi))
        cam_y = self.camera_distance * math.sin(math.radians(self.camera_phi))
        cam_z = self.camera_distance * math.cos(math.radians(self.camera_theta)) * math.cos(math.radians(self.camera_phi))
        
        gluLookAt(cam_x, cam_y, cam_z,  # eye
                  0.0, 0.0, 0.0,        # center
                  0.0, 1.0, 0.0)        # up
        
    def render_particles(self):
        """Render particles as points."""
        # Get particle positions from GPU
        positions_gpu = self.sim.positions.numpy()
        
        glBegin(GL_POINTS)
        for i in range(self.sim.num_particles):
            pos = positions_gpu[i]
            
            # Color based on height (blue to white gradient)
            height_factor = max(0.0, min(1.0, (pos[1] + 1.0) / 4.0))
            
            glColor4f(0.3 + height_factor * 0.5,  # Red
                     0.5 + height_factor * 0.3,  # Green  
                     0.8 + height_factor * 0.2,  # Blue
                     0.8)                         # Alpha
            
            glVertex3f(pos[0], pos[1], pos[2])
            
        glEnd()
        
    def render_boundaries(self):
        """Render simulation boundaries."""
        glColor4f(0.5, 0.5, 0.5, 0.3)
        glBegin(GL_LINES)
        
        # Draw wireframe box
        min_pt = self.sim.params['domain_min']
        max_pt = self.sim.params['domain_max']
        
        # Bottom face
        glVertex3f(min_pt[0], min_pt[1], min_pt[2])
        glVertex3f(max_pt[0], min_pt[1], min_pt[2])
        
        glVertex3f(max_pt[0], min_pt[1], min_pt[2])
        glVertex3f(max_pt[0], min_pt[1], max_pt[2])
        
        glVertex3f(max_pt[0], min_pt[1], max_pt[2])
        glVertex3f(min_pt[0], min_pt[1], max_pt[2])
        
        glVertex3f(min_pt[0], min_pt[1], max_pt[2])
        glVertex3f(min_pt[0], min_pt[1], min_pt[2])
        
        # Top face  
        glVertex3f(min_pt[0], max_pt[1], min_pt[2])
        glVertex3f(max_pt[0], max_pt[1], min_pt[2])
        
        glVertex3f(max_pt[0], max_pt[1], min_pt[2])
        glVertex3f(max_pt[0], max_pt[1], max_pt[2])
        
        glVertex3f(max_pt[0], max_pt[1], max_pt[2])
        glVertex3f(min_pt[0], max_pt[1], max_pt[2])
        
        glVertex3f(min_pt[0], max_pt[1], max_pt[2])
        glVertex3f(min_pt[0], max_pt[1], min_pt[2])
        
        # Vertical edges
        glVertex3f(min_pt[0], min_pt[1], min_pt[2])
        glVertex3f(min_pt[0], max_pt[1], min_pt[2])
        
        glVertex3f(max_pt[0], min_pt[1], min_pt[2])
        glVertex3f(max_pt[0], max_pt[1], min_pt[2])
        
        glVertex3f(max_pt[0], min_pt[1], max_pt[2])
        glVertex3f(max_pt[0], max_pt[1], max_pt[2])
        
        glVertex3f(min_pt[0], min_pt[1], max_pt[2])
        glVertex3f(min_pt[0], max_pt[1], max_pt[2])
        
        glEnd()
        
    def render_text(self, x, y, text):
        """Render text on screen."""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, 800, 0, 600, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glColor3f(1.0, 1.0, 1.0)
        glRasterPos2f(x, y)
        
        for char in text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
    def render(self):
        """Main render function with ImGui support."""
        # Clear and setup 3D rendering
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.setup_camera()
        
        # Render 3D scene
        self.render_boundaries()
        self.render_particles()
        
        if self.use_imgui:
            # Render ImGui interface on top of 3D scene
            try:
                # Switch to 2D rendering for ImGui
                glMatrixMode(GL_PROJECTION)
                glPushMatrix()
                glLoadIdentity()
                glOrtho(0, glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT), 0, -1, 1)
                
                glMatrixMode(GL_MODELVIEW)
                glPushMatrix()
                glLoadIdentity()
                
                # Disable depth testing for ImGui
                glDisable(GL_DEPTH_TEST)
                
                # Call ImGui render directly
                self.imgui_manager.draw_ui()
                
                # Re-enable depth testing
                glEnable(GL_DEPTH_TEST)
                
                # Restore matrices
                glPopMatrix()
                glMatrixMode(GL_PROJECTION)
                glPopMatrix()
                glMatrixMode(GL_MODELVIEW)
                
            except Exception as e:
                print(f"ImGui render error: {e}")
                self.use_imgui = False  # Disable on error
        else:
            # Fallback to basic OpenGL rendering
            self.render_text(10, 580, f"PBF Simulation - {self.sim.num_particles} particles")
            self.render_text(10, 560, f"{'PAUSED' if self.sim.paused else 'RUNNING'}")
            self.render_text(10, 540, "Controls: Space=Pause, R=Reset, Mouse=Camera")
            
            if not IMGUI_AVAILABLE:
                self.render_text(10, 520, "ImGui not available - install warp.render")
            
            # Calculate FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time > 1.0:
                fps = self.frame_count / (current_time - self.last_fps_time)
                self.render_text(10, 500, f"FPS: {fps:.1f}")
                self.frame_count = 0
                self.last_fps_time = current_time
        
        glutSwapBuffers()
    
    def cleanup(self):
        """Cleanup resources."""
        if self.use_imgui:
            try:
                self.imgui_manager.shutdown()
                self.warp_renderer.clear()
            except Exception:
                pass

# Global variables for GLUT callbacks
sim = None
renderer = None

def display():
    """GLUT display callback."""
    global sim, renderer
    
    # Run simulation step
    sim.step()
    
    # Render
    renderer.render()

def reshape(width, height):
    """GLUT reshape callback."""
    glViewport(0, 0, width, height)

def keyboard(key, x, y):
    """GLUT keyboard callback."""
    global sim
    
    if key == b' ':  # Space - pause/unpause
        sim.paused = not sim.paused
        print("Simulation", "paused" if sim.paused else "resumed")
    elif key == b'r' or key == b'R':  # R - reset
        sim.reset_particles()
        print("Simulation reset")
    elif key == b'\x1b':  # Escape - quit
        print("Exiting...")
        import sys
        sys.exit(0)

def mouse(button, state, x, y):
    """GLUT mouse callback."""
    global renderer
    
    if button == GLUT_LEFT_BUTTON:
        if state == GLUT_DOWN:
            renderer.mouse_dragging = True
            renderer.mouse_last_x = x
            renderer.mouse_last_y = y
        else:
            renderer.mouse_dragging = False

def motion(x, y):
    """GLUT mouse motion callback."""
    global renderer
    
    if renderer.mouse_dragging:
        dx = x - renderer.mouse_last_x
        dy = y - renderer.mouse_last_y
        
        renderer.camera_theta += dx * 0.5
        renderer.camera_phi += dy * 0.5
        
        # Clamp phi
        renderer.camera_phi = max(-89.0, min(89.0, renderer.camera_phi))
        
        renderer.mouse_last_x = x
        renderer.mouse_last_y = y

def idle():
    """GLUT idle callback."""
    glutPostRedisplay()

def cleanup():
    """Cleanup function."""
    global sim, renderer
    if renderer:
        renderer.cleanup()
    print("Cleanup completed")

def main():
    """Main function with ImGui support."""
    global sim, renderer
    
    if not OPENGL_AVAILABLE:
        print("OpenGL not available. Cannot run visualization.")
        return
    
    print("Position Based Fluids (PBF) Simulation with ImGui Controls")
    print("=========================================================")
    
    # Check ImGui availability
    if IMGUI_AVAILABLE:
        print("ImGui available - advanced parameter control enabled")
    else:
        print("ImGui not available - using basic controls only")
        print("To enable ImGui: ensure warp.render module is available")
    
    # Create simulation with reasonable particle count for GUI testing
    sim = PBFSimulation(num_particles=2048)
    renderer = OpenGLRenderer(sim, use_imgui=True)
    
    # Initialize GLUT
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(1280, 720)
    glutCreateWindow(b"PBF Simulation with ImGui")
    
    # Set callbacks
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutIdleFunc(idle)
    
    # Initialize OpenGL
    renderer.init_gl()
    
    print("\nSimulation started!")
    if IMGUI_AVAILABLE:
        print("ImGui Controls:")
        print("  • Real-time parameter sliders")
        print("  • Quick preset buttons")
        print("  • Debug information display")
        print("  • Performance monitoring")
    else:
        print("Basic Controls:")
        print("  Space: Pause/unpause simulation")
        print("  R: Reset simulation")
        print("  Mouse: Rotate camera")
        print("  Escape: Exit")
    
    try:
        # Start main loop
        glutMainLoop()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cleanup()

if __name__ == "__main__":
    main()
