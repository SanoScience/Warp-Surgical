#!/usr/bin/env python3

"""
Debug Version of Position Based Fluids (PBF) Simulation

This version includes:
- Stability checks and numerical limits
- Adjustable parameters with runtime controls
- Debug output and particle tracking
- Fallback mechanisms for instability
- Visual debugging aids

Controls:
- Mouse: Rotate camera
- WASD: Move camera  
- Space: Pause/unpause simulation
- R: Reset simulation
- 1-9: Adjust timestep
- Q/E: Adjust constraint iterations
- Z/C: Adjust damping
- T: Toggle debug info
- F: Toggle freeze unstable particles
"""

import math
import numpy as np
import warp as wp

try:
    import warp.render
except ImportError:
    print("Warp render module not available. Using basic OpenGL fallback.")
    warp.render = None

try:
    from OpenGL.GL import *
    from OpenGL.GLUT import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    print("PyOpenGL not available. Install with: pip install PyOpenGL PyOpenGL_accelerate")
    OPENGL_AVAILABLE = False

import time
from pbf_kernels import (
    pbf_predict_positions_debug,
    pbf_compute_density_debug,
    pbf_compute_lambda_debug,
    pbf_compute_delta_positions_debug,
    pbf_apply_delta_positions_debug,
    pbf_update_velocities_positions_debug,
    reset_stability_flags,
    pbf_apply_boundaries,
)


# Debug simulation parameters with safe defaults
DEBUG_PARAMS = {
    'particle_radius': 0.02,
    'smoothing_radius': 0.04,  # Smaller for better stability
    'rest_density': 1000.0,
    'constraint_iterations': 6,  # More iterations for stability
    'damping': 0.95,  # More damping
    'gravity': -4.0,  # Reduced gravity
    'dt': 1.0/120.0,  # Smaller timestep
    'grid_dim': 64,
    'domain_min': np.array([-1.5, -1.0, -1.5]),
    'domain_max': np.array([1.5, 2.0, 1.5]),
    'max_velocity': 5.0,  # Velocity clamping
    'max_density': 5000.0,  # Density clamping
    'min_density': 100.0,   # Minimum density
    'constraint_epsilon': 100.0,  # Regularization parameter
    'freeze_unstable': True,  # Freeze particles that go unstable
    'debug_output': False,
    'stability_check': True
}

class PBFDebugSimulation:
    def __init__(self, num_particles=1024):
        self.num_particles = num_particles
        self.params = DEBUG_PARAMS.copy()
        self.paused = False
        self.step_count = 0
        
        # Debug state
        self.debug_mode = True
        self.last_debug_time = time.time()
        
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
        
        # Debug arrays
        self.unstable_flags = wp.zeros(num_particles, dtype=wp.int32, device=self.device)
        
        # Create spatial hash grid
        self.hash_grid = wp.HashGrid(
            dim_x=self.params['grid_dim'],
            dim_y=self.params['grid_dim'],
            dim_z=self.params['grid_dim'],
            device=self.device
        )
        
        # Initialize particle positions
        self.reset_particles()
        
        print(f"PBF Debug Simulation initialized with {num_particles} particles")
        print(f"Timestep: {self.params['dt']:.4f}s, Iterations: {self.params['constraint_iterations']}")
        
    def reset_particles(self):
        """Initialize particles in a more stable configuration."""
        positions = []
        
        # Create a smaller, more stable dam
        spacing = self.params['particle_radius'] * 2.1  # Slightly more spacing
        layers_x = int(0.8 / spacing)  # Smaller dam
        layers_y = int(1.0 / spacing) 
        layers_z = int(0.8 / spacing)
        
        count = 0
        for i in range(layers_x):
            for j in range(layers_y):
                for k in range(layers_z):
                    if count >= self.num_particles:
                        break
                        
                    x = -0.7 + i * spacing + (np.random.random() - 0.5) * spacing * 0.05  # Less randomness
                    y = -0.6 + j * spacing + (np.random.random() - 0.5) * spacing * 0.05
                    z = -0.4 + k * spacing + (np.random.random() - 0.5) * spacing * 0.05
                    
                    positions.append([x, y, z])
                    count += 1
                    
                if count >= self.num_particles:
                    break
            if count >= self.num_particles:
                break
        
        # Fill remaining particles if needed
        while count < self.num_particles:
            x = np.random.uniform(-0.7, 0.0)
            y = np.random.uniform(-0.6, 0.2)
            z = np.random.uniform(-0.4, 0.4)
            positions.append([x, y, z])
            count += 1
        
        # Copy to GPU
        positions_array = np.array(positions, dtype=np.float32)
        wp.copy(self.positions, wp.array(positions_array, dtype=wp.vec3, device=self.device))
        
        # Reset velocities and flags
        wp.copy(self.velocities, wp.zeros(self.num_particles, dtype=wp.vec3, device=self.device))
        wp.copy(self.unstable_flags, wp.zeros(self.num_particles, dtype=wp.int32, device=self.device))
        
        self.step_count = 0
        
    def adjust_timestep(self, factor):
        """Adjust timestep for stability."""
        self.params['dt'] *= factor
        self.params['dt'] = max(1.0/240.0, min(1.0/30.0, self.params['dt']))  # Clamp to reasonable range
        print(f"Timestep: {self.params['dt']:.4f}s")
        
    def adjust_iterations(self, delta):
        """Adjust constraint iterations."""
        self.params['constraint_iterations'] += delta
        self.params['constraint_iterations'] = max(1, min(20, self.params['constraint_iterations']))
        print(f"Constraint iterations: {self.params['constraint_iterations']}")
        
    def adjust_damping(self, delta):
        """Adjust damping factor."""
        self.params['damping'] += delta
        self.params['damping'] = max(0.5, min(0.99, self.params['damping']))
        print(f"Damping: {self.params['damping']:.3f}")
        
    def get_debug_stats(self):
        """Get debug statistics."""
        if not self.debug_mode:
            return {}
            
        # Get data from GPU
        positions_cpu = self.positions.numpy()
        velocities_cpu = self.velocities.numpy()
        densities_cpu = self.densities.numpy()
        unstable_flags_cpu = self.unstable_flags.numpy()
        
        # Compute stats
        vel_magnitudes = np.linalg.norm(velocities_cpu, axis=1)
        
        stats = {
            'unstable_count': np.sum(unstable_flags_cpu),
            'avg_velocity': np.mean(vel_magnitudes),
            'max_velocity': np.max(vel_magnitudes),
            'avg_density': np.mean(densities_cpu),
            'max_density': np.max(densities_cpu),
            'min_density': np.min(densities_cpu),
            'step_count': self.step_count
        }
        
        return stats
        
    def step(self):
        """Run one simulation step with debug checks."""
        if self.paused:
            return
            
        self.step_count += 1
        
        # Reset ALL stability flags after initial steps to allow particles to stabilize
        if self.step_count == 10:  # Reset after 10 steps
            wp.copy(self.unstable_flags, wp.zeros(self.num_particles, dtype=wp.int32, device=self.device))
            print("Reset all stability flags after initial steps")
        
        # Reset some stability flags periodically (less aggressive)
        if self.step_count % 120 == 0 and self.step_count > 10:  # Every 2 seconds after initial reset
            domain_min = wp.vec3(*self.params['domain_min'])
            domain_max = wp.vec3(*self.params['domain_max'])
            wp.launch(
                reset_stability_flags,
                dim=self.num_particles,
                inputs=[self.unstable_flags, self.positions, self.velocities, domain_min, domain_max],
                device=self.device
            )
        
        # Step 1: Predict positions
        gravity = wp.vec3(0.0, self.params['gravity'], 0.0)
        wp.launch(
            pbf_predict_positions_debug,
            dim=self.num_particles,
            inputs=[
                self.positions, self.velocities, self.predicted_positions, self.unstable_flags,
                gravity, self.params['dt'], self.params['max_velocity']
            ],
            device=self.device
        )
        
        # Step 2: Build spatial hash
        self.hash_grid.build(self.predicted_positions, self.params['smoothing_radius'])
        
        # Step 3: Constraint projection iterations
        max_delta = self.params['smoothing_radius'] * 0.5  # Limit position corrections
        
        for iteration in range(self.params['constraint_iterations']):
            # Compute density
            wp.launch(
                pbf_compute_density_debug,
                dim=self.num_particles,
                inputs=[
                    self.predicted_positions, self.densities, self.unstable_flags, self.hash_grid.id,
                    self.params['smoothing_radius'], self.params['max_density'], self.params['min_density']
                ],
                device=self.device
            )
            
            # Compute lambda
            wp.launch(
                pbf_compute_lambda_debug,
                dim=self.num_particles,
                inputs=[
                    self.predicted_positions, self.densities, self.lambdas, self.unstable_flags,
                    self.hash_grid.id, self.params['smoothing_radius'], self.params['rest_density'],
                    self.params['constraint_epsilon']
                ],
                device=self.device
            )
            
            # Compute position corrections
            wp.launch(
                pbf_compute_delta_positions_debug,
                dim=self.num_particles,
                inputs=[
                    self.predicted_positions, self.lambdas, self.delta_positions, self.unstable_flags,
                    self.hash_grid.id, self.params['smoothing_radius'], self.params['rest_density'], max_delta
                ],
                device=self.device
            )
            
            # Apply corrections
            wp.launch(
                pbf_apply_delta_positions_debug,
                dim=self.num_particles,
                inputs=[self.predicted_positions, self.delta_positions, self.unstable_flags],
                device=self.device
            )
        
        # Step 4: Apply boundary conditions (same as original)
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
            pbf_update_velocities_positions_debug,
            dim=self.num_particles,
            inputs=[
                self.positions, self.velocities, self.predicted_positions, self.unstable_flags,
                self.params['damping'], self.params['dt'], self.params['max_velocity']
            ],
            device=self.device
        )



# OpenGL Renderer with Debug Features
class DebugOpenGLRenderer:
    def __init__(self, simulation):
        self.sim = simulation
        self.camera_distance = 4.0
        self.camera_theta = 45.0
        self.camera_phi = 20.0
        self.mouse_last_x = 0
        self.mouse_last_y = 0
        self.mouse_dragging = False
        
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.show_debug = True
        
    def init_gl(self):
        """Initialize OpenGL settings."""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glClearColor(0.05, 0.05, 0.1, 1.0)
        glPointSize(6.0)
        
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
        """Render particles with stability color coding."""
        # Get particle data from GPU
        positions_gpu = self.sim.positions.numpy()
        unstable_flags_gpu = self.sim.unstable_flags.numpy()
        velocities_gpu = self.sim.velocities.numpy()
        
        glBegin(GL_POINTS)
        for i in range(self.sim.num_particles):
            pos = positions_gpu[i]
            is_unstable = unstable_flags_gpu[i]
            vel_magnitude = np.linalg.norm(velocities_gpu[i])
            
            # Color coding: unstable=red, high velocity=yellow, normal=blue-white
            if is_unstable:
                glColor4f(1.0, 0.2, 0.2, 0.9)  # Red for unstable
            elif vel_magnitude > 2.0:
                glColor4f(1.0, 1.0, 0.2, 0.8)  # Yellow for high velocity
            else:
                # Normal particles: height-based gradient
                height_factor = max(0.0, min(1.0, (pos[1] + 1.0) / 3.0))
                glColor4f(0.2 + height_factor * 0.5,  # Red
                         0.4 + height_factor * 0.4,  # Green
                         0.8 + height_factor * 0.2,  # Blue
                         0.8)                         # Alpha
            
            glVertex3f(pos[0], pos[1], pos[2])
        glEnd()
        
    def render_boundaries(self):
        """Render simulation boundaries."""
        glColor4f(0.3, 0.3, 0.3, 0.5)
        glBegin(GL_LINES)
        
        min_pt = self.sim.params['domain_min']
        max_pt = self.sim.params['domain_max']
        
        # Draw wireframe box (same as original)
        vertices = [
            [min_pt[0], min_pt[1], min_pt[2]], [max_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], min_pt[2]], [max_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], min_pt[1], max_pt[2]], [min_pt[0], min_pt[1], max_pt[2]],
            [min_pt[0], min_pt[1], max_pt[2]], [min_pt[0], min_pt[1], min_pt[2]],
            
            [min_pt[0], max_pt[1], min_pt[2]], [max_pt[0], max_pt[1], min_pt[2]],
            [max_pt[0], max_pt[1], min_pt[2]], [max_pt[0], max_pt[1], max_pt[2]],
            [max_pt[0], max_pt[1], max_pt[2]], [min_pt[0], max_pt[1], max_pt[2]],
            [min_pt[0], max_pt[1], max_pt[2]], [min_pt[0], max_pt[1], min_pt[2]],
            
            [min_pt[0], min_pt[1], min_pt[2]], [min_pt[0], max_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], min_pt[2]], [max_pt[0], max_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], max_pt[2]], [max_pt[0], max_pt[1], max_pt[2]],
            [min_pt[0], min_pt[1], max_pt[2]], [min_pt[0], max_pt[1], max_pt[2]]
        ]
        
        for i in range(0, len(vertices), 2):
            glVertex3f(*vertices[i])
            glVertex3f(*vertices[i+1])
        
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
        """Main render function with debug info."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self.setup_camera()
        
        # Render scene
        self.render_boundaries()
        self.render_particles()
        
        # Render UI
        line_height = 20
        y_pos = 580
        
        self.render_text(10, y_pos, f"PBF Debug - {self.sim.num_particles} particles")
        y_pos -= line_height
        
        self.render_text(10, y_pos, f"{'PAUSED' if self.sim.paused else 'RUNNING'}")
        y_pos -= line_height
        
        # Debug stats
        if self.show_debug:
            stats = self.sim.get_debug_stats()
            self.render_text(10, y_pos, f"Unstable: {stats.get('unstable_count', 0)}")
            y_pos -= line_height
            
            self.render_text(10, y_pos, f"Avg Vel: {stats.get('avg_velocity', 0):.2f}")
            y_pos -= line_height
            
            self.render_text(10, y_pos, f"Max Vel: {stats.get('max_velocity', 0):.2f}")
            y_pos -= line_height
            
            self.render_text(10, y_pos, f"Density: {stats.get('avg_density', 0):.0f}")
            y_pos -= line_height
        
        # Parameters
        self.render_text(10, y_pos, f"dt: {self.sim.params['dt']:.4f}")
        y_pos -= line_height
        
        self.render_text(10, y_pos, f"Iter: {self.sim.params['constraint_iterations']}")
        y_pos -= line_height
        
        self.render_text(10, y_pos, f"Damp: {self.sim.params['damping']:.2f}")
        y_pos -= line_height
        
        # Controls
        self.render_text(10, 60, "Controls: Space=Pause R=Reset T=Debug")
        self.render_text(10, 40, "1-9=Timestep Q/E=Iterations Z/C=Damping")
        self.render_text(10, 20, "Red=Unstable Yellow=Fast Blue=Normal")
        
        # Calculate FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time > 1.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            self.render_text(10, 100, f"FPS: {fps:.1f}")
            self.frame_count = 0
            self.last_fps_time = current_time
        
        glutSwapBuffers()

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
    """GLUT keyboard callback with debug controls."""
    global sim, renderer
    
    if key == b' ':  # Space - pause/unpause
        sim.paused = not sim.paused
        print("Simulation", "paused" if sim.paused else "resumed")
    elif key == b'r' or key == b'R':  # R - reset
        sim.reset_particles()
        print("Simulation reset")
    elif key == b't' or key == b'T':  # T - toggle debug
        renderer.show_debug = not renderer.show_debug
        print("Debug info", "on" if renderer.show_debug else "off")
    elif key in b'123456789':  # Adjust timestep
        factor = 0.5 + (int(chr(key[0])) - 1) * 0.1  # 0.5 to 1.3
        sim.params['dt'] = factor / 60.0
        print(f"Timestep: {sim.params['dt']:.4f}s")
    elif key == b'q' or key == b'Q':  # Decrease iterations
        sim.adjust_iterations(-1)
    elif key == b'e' or key == b'E':  # Increase iterations
        sim.adjust_iterations(1)
    elif key == b'z' or key == b'Z':  # Decrease damping
        sim.adjust_damping(-0.02)
    elif key == b'c' or key == b'C':  # Increase damping
        sim.adjust_damping(0.02)
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

def main():
    """Main function."""
    global sim, renderer
    
    if not OPENGL_AVAILABLE:
        print("OpenGL not available. Cannot run visualization.")
        return
    
    print("PBF Debug Simulation")
    print("====================")
    
    # Create debug simulation
    sim = PBFDebugSimulation(num_particles=512)  # Smaller count for debugging
    renderer = DebugOpenGLRenderer(sim)
    
    # Initialize GLUT
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"PBF Debug Simulation")
    
    # Set callbacks
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutIdleFunc(idle)
    
    # Initialize OpenGL
    renderer.init_gl()
    
    print("\nDebug simulation started!")
    print("Color coding:")
    print("  Red: Unstable particles")
    print("  Yellow: High velocity particles")
    print("  Blue-white: Normal particles")
    print("\nControls:")
    print("  Space: Pause/unpause")
    print("  R: Reset simulation")
    print("  T: Toggle debug info")
    print("  1-9: Adjust timestep")
    print("  Q/E: Adjust constraint iterations")
    print("  Z/C: Adjust damping")
    print("  Mouse: Rotate camera")
    print("  Escape: Exit")
    
    # Start main loop
    glutMainLoop()

if __name__ == "__main__":
    main()
