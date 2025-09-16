#!/usr/bin/env python3

"""
Stable Position Based Fluids (PBF) Simulation

This is a simplified, more stable version of the PBF simulation that fixes
the main causes of particle explosions:

1. Conservative timestep (1/120s instead of 1/60s)
2. More constraint iterations (6 instead of 4)
3. Better velocity and density clamping
4. Smaller initial particle spacing
5. Higher damping factor

Controls:
- Mouse: Rotate camera
- Space: Pause/unpause simulation
- R: Reset simulation
- 1-5: Adjust timestep (1=slowest, 5=fastest)
- Q/E: Adjust constraint iterations
- Escape: Exit
"""

import math
import numpy as np
import warp as wp

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
    pbf_predict_positions_stable,
    pbf_compute_density_stable,
    pbf_compute_lambda_stable,
    pbf_compute_delta_positions_stable,
    pbf_apply_delta_positions,
    pbf_apply_boundaries_stable,
    pbf_update_velocities_positions_stable,
)


# Conservative simulation parameters for stability
STABLE_PARAMS = {
    'particle_radius': 0.015,    # Smaller particles
    'smoothing_radius': 0.035,   # Smaller smoothing radius
    'rest_density': 1000.0,
    'constraint_iterations': 6,   # More iterations
    'damping': 0.96,             # More damping
    'gravity': -5.0,             # Reduced gravity
    'dt': 1.0/120.0,            # Smaller timestep
    'grid_dim': 64,
    'domain_min': np.array([-1.2, -0.8, -1.2]),
    'domain_max': np.array([1.2, 2.0, 1.2]),
    'max_velocity': 8.0,         # Velocity limit
    'constraint_epsilon': 200.0   # More regularization
}

class StablePBFSimulation:
    def __init__(self, num_particles=1024):
        self.num_particles = num_particles
        self.params = STABLE_PARAMS.copy()
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
        
        print(f"Stable PBF Simulation initialized with {num_particles} particles")
        print(f"Conservative settings: dt={self.params['dt']:.4f}s, iterations={self.params['constraint_iterations']}")
        
    def reset_particles(self):
        """Initialize particles in a stable, compact configuration."""
        positions = []
        
        # Create a stable dam with tighter spacing
        spacing = self.params['particle_radius'] * 2.05  # Tight but not overlapping
        layers_x = int(0.6 / spacing)  # Smaller dam
        layers_y = int(0.8 / spacing) 
        layers_z = int(0.6 / spacing)
        
        count = 0
        for i in range(layers_x):
            for j in range(layers_y):
                for k in range(layers_z):
                    if count >= self.num_particles:
                        break
                        
                    x = -0.5 + i * spacing
                    y = -0.6 + j * spacing
                    z = -0.3 + k * spacing
                    
                    # Add very small random perturbation to break symmetry
                    x += (np.random.random() - 0.5) * spacing * 0.02
                    y += (np.random.random() - 0.5) * spacing * 0.02
                    z += (np.random.random() - 0.5) * spacing * 0.02
                    
                    positions.append([x, y, z])
                    count += 1
                    
                if count >= self.num_particles:
                    break
            if count >= self.num_particles:
                break
        
        # Fill remaining particles in a grid pattern
        while count < self.num_particles:
            layer = (count - (layers_x * layers_y * layers_z)) // (layers_x * layers_z)
            remainder = (count - (layers_x * layers_y * layers_z)) % (layers_x * layers_z)
            i = remainder % layers_x
            k = remainder // layers_x
            
            x = -0.5 + i * spacing
            y = -0.6 + (layers_y + layer) * spacing
            z = -0.3 + k * spacing
            
            positions.append([x, y, z])
            count += 1
        
        # Copy to GPU
        positions_array = np.array(positions, dtype=np.float32)
        wp.copy(self.positions, wp.array(positions_array, dtype=wp.vec3, device=self.device))
        
        # Reset velocities
        wp.copy(self.velocities, wp.zeros(self.num_particles, dtype=wp.vec3, device=self.device))
        
    def adjust_timestep(self, level):
        """Adjust timestep (1=slowest, 5=fastest)."""
        timesteps = [1.0/240.0, 1.0/180.0, 1.0/120.0, 1.0/90.0, 1.0/60.0]
        self.params['dt'] = timesteps[max(0, min(4, level-1))]
        print(f"Timestep level {level}: {self.params['dt']:.4f}s")
        
    def adjust_iterations(self, delta):
        """Adjust constraint iterations."""
        self.params['constraint_iterations'] += delta
        self.params['constraint_iterations'] = max(1, min(12, self.params['constraint_iterations']))
        print(f"Constraint iterations: {self.params['constraint_iterations']}")
        
    def step(self):
        """Run one stable simulation step."""
        if self.paused:
            return
            
        # Step 1: Predict positions
        gravity = wp.vec3(0.0, self.params['gravity'], 0.0)
        wp.launch(
            pbf_predict_positions_stable,
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
                pbf_compute_density_stable,
                dim=self.num_particles,
                inputs=[
                    self.predicted_positions, self.densities, self.hash_grid.id, self.params['smoothing_radius']
                ],
                device=self.device
            )
            
            # Compute lambda
            wp.launch(
                pbf_compute_lambda_stable,
                dim=self.num_particles,
                inputs=[
                    self.predicted_positions, self.densities, self.lambdas, self.hash_grid.id,
                    self.params['smoothing_radius'], self.params['rest_density'], self.params['constraint_epsilon']
                ],
                device=self.device
            )
            
            # Compute position corrections
            wp.launch(
                pbf_compute_delta_positions_stable,
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
            pbf_apply_boundaries_stable,
            dim=self.num_particles,
            inputs=[self.predicted_positions, self.velocities, domain_min, domain_max, 0.3],
            device=self.device
        )
        
        # Step 5: Update velocities and positions
        wp.launch(
            pbf_update_velocities_positions_stable,
            dim=self.num_particles,
            inputs=[
                self.positions, self.velocities, self.predicted_positions,
                self.params['damping'], self.params['dt'], self.params['max_velocity']
            ],
            device=self.device
        )

# Simple OpenGL Renderer
class StableRenderer:
    def __init__(self, simulation):
        self.sim = simulation
        self.camera_distance = 3.0
        self.camera_theta = 45.0
        self.camera_phi = 20.0
        self.mouse_last_x = 0
        self.mouse_last_y = 0
        self.mouse_dragging = False
        
        self.frame_count = 0
        self.last_fps_time = time.time()
        
    def init_gl(self):
        """Initialize OpenGL settings."""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glClearColor(0.02, 0.02, 0.05, 1.0)
        glPointSize(5.0)
        
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
        """Render particles as colored points."""
        positions_gpu = self.sim.positions.numpy()
        
        glBegin(GL_POINTS)
        for i in range(self.sim.num_particles):
            pos = positions_gpu[i]
            
            # Color based on height (blue to cyan gradient)
            height_factor = max(0.0, min(1.0, (pos[1] + 0.8) / 2.8))
            
            glColor4f(0.1 + height_factor * 0.4,  # Red
                     0.3 + height_factor * 0.5,  # Green  
                     0.8 + height_factor * 0.2,  # Blue
                     0.9)                         # Alpha
            
            glVertex3f(pos[0], pos[1], pos[2])
            
        glEnd()
        
    def render_boundaries(self):
        """Render simulation boundaries."""
        glColor4f(0.3, 0.3, 0.3, 0.4)
        glBegin(GL_LINES)
        
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
        
        glColor3f(0.9, 0.9, 1.0)
        glRasterPos2f(x, y)
        
        for char in text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
    def render(self):
        """Main render function."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self.setup_camera()
        
        # Render scene
        self.render_boundaries()
        self.render_particles()
        
        # Render UI
        self.render_text(10, 580, f"Stable PBF - {self.sim.num_particles} particles")
        self.render_text(10, 560, f"{'PAUSED' if self.sim.paused else 'RUNNING'}")
        self.render_text(10, 540, f"dt: {self.sim.params['dt']:.4f}s, iter: {self.sim.params['constraint_iterations']}")
        self.render_text(10, 40, "Space=Pause R=Reset 1-5=Speed Q/E=Iterations")
        
        # Calculate FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time > 1.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            self.render_text(10, 520, f"FPS: {fps:.1f}")
            self.frame_count = 0
            self.last_fps_time = current_time
        
        glutSwapBuffers()

# Global variables for GLUT callbacks
sim = None
renderer = None

def display():
    """GLUT display callback."""
    global sim, renderer
    sim.step()
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
    elif key in b'12345':  # Adjust timestep
        level = int(chr(key[0]))
        sim.adjust_timestep(level)
    elif key == b'q' or key == b'Q':  # Decrease iterations
        sim.adjust_iterations(-1)
    elif key == b'e' or key == b'E':  # Increase iterations
        sim.adjust_iterations(1)
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
    
    print("Stable PBF Simulation")
    print("=====================")
    
    # Create stable simulation
    sim = StablePBFSimulation(num_particles=768)  # Moderate particle count
    renderer = StableRenderer(sim)
    
    # Initialize GLUT
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"Stable PBF Simulation")
    
    # Set callbacks
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutIdleFunc(idle)
    
    # Initialize OpenGL
    renderer.init_gl()
    
    print("\nStable simulation started!")
    print("This version should NOT explode!")
    print("\nControls:")
    print("  Space: Pause/unpause")
    print("  R: Reset simulation")
    print("  1-5: Adjust timestep (1=slowest, 5=fastest)")
    print("  Q/E: Adjust constraint iterations")
    print("  Mouse: Rotate camera")
    print("  Escape: Exit")
    
    # Start main loop
    glutMainLoop()

if __name__ == "__main__":
    main()
