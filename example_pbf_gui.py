#!/usr/bin/env python3

"""
PBF Simulation with Simple GUI Controls

This version uses basic OpenGL-based GUI elements that work reliably with GLUT,
providing interactive parameter control without complex ImGui dependencies.

Controls:
- Mouse: Rotate camera (right side of window)
- Left side: Interactive parameter panel
- Click sliders to adjust parameters
- Buttons for presets and controls
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

# Stable simulation parameters
SIM_PARAMS = {
    'particle_radius': 0.015,
    'smoothing_radius': 0.04,
    'rest_density': 1000.0,
    'constraint_iterations': 6,
    'damping': 0.96,
    'gravity': -5.0,
    'dt': 1.0/120.0,
    'substeps': 6,  # Number of substeps per frame (PBD-style)
    'max_velocity': 8.0,
    'constraint_epsilon': 200.0,
    'grid_dim': 64,
    'domain_min': np.array([-1.2, -0.8, -1.2]),
    'domain_max': np.array([1.2, 2.0, 1.2])
}

# Import PBF kernels from the stable version
from example_pbf_stable import (
    pbf_predict_positions_stable as pbf_predict_positions,
    pbf_compute_density_stable as pbf_compute_density,
    pbf_compute_lambda_stable as pbf_compute_lambda,
    pbf_compute_delta_positions_stable as pbf_compute_delta_positions,
    pbf_apply_delta_positions,
    pbf_apply_boundaries_stable as pbf_apply_boundaries,
    pbf_update_velocities_positions_stable as pbf_update_velocities_positions
)

# from example_pbf_opengl import (
#     pbf_predict_positions as pbf_predict_positions,
#     pbf_compute_density as pbf_compute_density,
#     pbf_compute_lambda as pbf_compute_lambda,
#     pbf_compute_delta_positions as pbf_compute_delta_positions,
#     pbf_apply_delta_positions,
#     pbf_apply_boundaries as pbf_apply_boundaries,
#     pbf_update_velocities_positions as pbf_update_velocities_positions
# )

class SimpleGUIElement:
    """Base class for simple GUI elements."""
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.hovered = False
        self.clicked = False
    
    def contains_point(self, px, py):
        return (self.x <= px <= self.x + self.width and 
                self.y <= py <= self.y + self.height)
    
    def render(self):
        pass
    
    def handle_click(self, x, y):
        return False

class SimpleSlider(SimpleGUIElement):
    """Simple slider control."""
    def __init__(self, x, y, width, height, min_val, max_val, current_val, label):
        super().__init__(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.current_val = current_val
        self.label = label
        self.dragging = False
        
    def get_normalized_value(self):
        return (self.current_val - self.min_val) / (self.max_val - self.min_val)
    
    def set_normalized_value(self, norm_val):
        norm_val = max(0, min(1, norm_val))
        self.current_val = self.min_val + norm_val * (self.max_val - self.min_val)
        
    def render(self):
        # Draw background
        glColor3f(0.2, 0.2, 0.25)
        glBegin(GL_QUADS)
        glVertex2f(self.x, self.y)
        glVertex2f(self.x + self.width, self.y)
        glVertex2f(self.x + self.width, self.y + self.height)
        glVertex2f(self.x, self.y + self.height)
        glEnd()
        
        # Draw filled portion
        fill_width = self.width * self.get_normalized_value()
        if self.hovered or self.dragging:
            glColor3f(0.4, 0.6, 0.8)
        else:
            glColor3f(0.3, 0.5, 0.7)
        glBegin(GL_QUADS)
        glVertex2f(self.x, self.y)
        glVertex2f(self.x + fill_width, self.y)
        glVertex2f(self.x + fill_width, self.y + self.height)
        glVertex2f(self.x, self.y + self.height)
        glEnd()
        
        # Draw border
        glColor3f(0.5, 0.5, 0.5)
        glBegin(GL_LINE_LOOP)
        glVertex2f(self.x, self.y)
        glVertex2f(self.x + self.width, self.y)
        glVertex2f(self.x + self.width, self.y + self.height)
        glVertex2f(self.x, self.y + self.height)
        glEnd()
        
    def handle_click(self, x, y):
        if self.contains_point(x, y):
            # Calculate normalized position within slider
            norm_x = (x - self.x) / self.width
            self.set_normalized_value(norm_x)
            return True
        return False
    
    def handle_drag(self, x, y):
        if self.dragging and self.contains_point(x, y):
            norm_x = (x - self.x) / self.width
            self.set_normalized_value(norm_x)

class SimpleButton(SimpleGUIElement):
    """Simple button control."""
    def __init__(self, x, y, width, height, label, callback=None):
        super().__init__(x, y, width, height)
        self.label = label
        self.callback = callback
        self.pressed = False
        
    def render(self):
        # Choose colors based on state
        if self.pressed:
            glColor3f(0.2, 0.4, 0.6)
        elif self.hovered:
            glColor3f(0.3, 0.5, 0.7)
        else:
            glColor3f(0.25, 0.45, 0.65)
            
        # Draw button background
        glBegin(GL_QUADS)
        glVertex2f(self.x, self.y)
        glVertex2f(self.x + self.width, self.y)
        glVertex2f(self.x + self.width, self.y + self.height)
        glVertex2f(self.x, self.y + self.height)
        glEnd()
        
        # Draw border
        glColor3f(0.6, 0.6, 0.6)
        glBegin(GL_LINE_LOOP)
        glVertex2f(self.x, self.y)
        glVertex2f(self.x + self.width, self.y)
        glVertex2f(self.x + self.width, self.y + self.height)
        glVertex2f(self.x, self.y + self.height)
        glEnd()
        
    def handle_click(self, x, y):
        if self.contains_point(x, y):
            self.pressed = True
            if self.callback:
                self.callback()
            return True
        return False

class PBFSimulation:
    def __init__(self, num_particles=1024):
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
        
        print(f"PBF Simulation with GUI initialized - {num_particles} particles")
        
    def reset_particles(self):
        """Initialize particles in a stable configuration."""
        positions = []
        
        # Create a stable dam
        spacing = self.params['particle_radius'] * 3
        layers_x = int(0.6 / spacing)
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
                    
                    # Small random perturbation
                    x += (np.random.random() - 0.5) * spacing * 0.05
                    y += (np.random.random() - 0.5) * spacing * 0.05
                    z += (np.random.random() - 0.5) * spacing * 0.05
                    
                    positions.append([x, y, z])
                    count += 1
                    
                if count >= self.num_particles:
                    break
            if count >= self.num_particles:
                break
        
        # Fill remaining particles
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
        wp.copy(self.velocities, wp.zeros(self.num_particles, dtype=wp.vec3, device=self.device))
        
    def step(self):
        """Run one simulation step with substepping (PBD-style)."""
        if self.paused:
            return
            
        # PBD-style substepping: divide timestep into smaller substeps
        substeps = self.params['substeps']
        sub_dt = self.params['dt'] / substeps
        
        for substep in range(substeps):
            self.substep(sub_dt)
    
    def substep(self, dt):
        """Run one simulation substep."""
        # Step 1: Predict positions
        gravity = wp.vec3(0.0, self.params['gravity'], 0.0)
        wp.launch(
            pbf_predict_positions,
            dim=self.num_particles,
            inputs=[
                self.positions, self.velocities, self.predicted_positions,
                gravity, dt, self.params['max_velocity']
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
                inputs=[
                    self.predicted_positions, self.densities, self.hash_grid.id, self.params['smoothing_radius']
                ],
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
            inputs=[self.predicted_positions, self.velocities, domain_min, domain_max, 0.3],
            device=self.device
        )
        
        # Step 5: Update velocities and positions
        wp.launch(
            pbf_update_velocities_positions,
            dim=self.num_particles,
            inputs=[
                self.positions, self.velocities, self.predicted_positions,
                self.params['damping'], dt, self.params['max_velocity']
            ],
            device=self.device
        )

class GUIRenderer:
    """Renderer with simple GUI controls."""
    def __init__(self, simulation):
        self.sim = simulation
        self.camera_distance = 3.0
        self.camera_theta = 45.0
        self.camera_phi = 20.0
        self.mouse_last_x = 0
        self.mouse_last_y = 0
        self.mouse_dragging = False
        
        self.window_width = 1200
        self.window_height = 800
        self.gui_width = 300
        
        # GUI elements
        self.gui_elements = []
        self.setup_gui()
        
        # FPS tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        
    def setup_gui(self):
        """Setup GUI elements."""
        y_pos = 50
        spacing = 35
        
        # Timestep slider
        self.dt_slider = SimpleSlider(20, y_pos, 250, 20, 1.0/240.0, 1.0/60.0, self.sim.params['dt'], "Timestep")
        self.gui_elements.append(self.dt_slider)
        y_pos += spacing
        
        # Iterations slider
        self.iter_slider = SimpleSlider(20, y_pos, 250, 20, 4, 12, self.sim.params['constraint_iterations'], "Iterations")
        self.gui_elements.append(self.iter_slider)
        y_pos += spacing
        
        # Substeps slider
        self.substeps_slider = SimpleSlider(20, y_pos, 250, 20, 4, 12, self.sim.params['substeps'], "Substeps")
        self.gui_elements.append(self.substeps_slider)
        y_pos += spacing
        
        # Damping slider
        self.damping_slider = SimpleSlider(20, y_pos, 250, 20, 0.90, 0.99, self.sim.params['damping'], "Damping")
        self.gui_elements.append(self.damping_slider)
        y_pos += spacing
        
        # Gravity slider
        self.gravity_slider = SimpleSlider(20, y_pos, 250, 20, -12.0, -2.0, self.sim.params['gravity'], "Gravity")
        self.gui_elements.append(self.gravity_slider)
        y_pos += spacing
        
        # Particle radius slider
        self.particle_radius_slider = SimpleSlider(20, y_pos, 250, 20, 0.010, 0.030, self.sim.params['particle_radius'], "Particle Radius")
        self.gui_elements.append(self.particle_radius_slider)
        y_pos += spacing
        
        # Smoothing radius slider
        self.smoothing_radius_slider = SimpleSlider(20, y_pos, 250, 20, 0.025, 0.080, self.sim.params['smoothing_radius'], "Smoothing Radius")
        self.gui_elements.append(self.smoothing_radius_slider)
        y_pos += spacing
        
        # Rest density slider
        self.rest_density_slider = SimpleSlider(20, y_pos, 250, 20, 100.0, 5000.0, self.sim.params['rest_density'], "Rest Density")
        self.gui_elements.append(self.rest_density_slider)
        y_pos += spacing
        
        
        # Max velocity slider
        self.max_velocity_slider = SimpleSlider(20, y_pos, 250, 20, 4.0, 12.0, self.sim.params['max_velocity'], "Max Velocity")
        self.gui_elements.append(self.max_velocity_slider)
        y_pos += spacing
        
        # Constraint epsilon slider
        self.constraint_epsilon_slider = SimpleSlider(20, y_pos, 250, 20, 50.0, 400.0, self.sim.params['constraint_epsilon'], "Constraint Epsilon")
        self.gui_elements.append(self.constraint_epsilon_slider)
        y_pos += spacing * 2
        
        # Control buttons
        self.pause_btn = SimpleButton(20, y_pos, 80, 25, "Pause", self.toggle_pause)
        self.gui_elements.append(self.pause_btn)
        
        self.reset_btn = SimpleButton(110, y_pos, 80, 25, "Reset", self.reset_simulation)
        self.gui_elements.append(self.reset_btn)
        y_pos += spacing
        
        # Preset buttons
        self.conservative_btn = SimpleButton(20, y_pos, 80, 25, "Safe", self.set_conservative)
        self.gui_elements.append(self.conservative_btn)
        
        self.fast_btn = SimpleButton(110, y_pos, 80, 25, "Fast", self.set_fast)
        self.gui_elements.append(self.fast_btn)
        
        self.quality_btn = SimpleButton(200, y_pos, 70, 25, "Quality", self.set_quality)
        self.gui_elements.append(self.quality_btn)
        
    def toggle_pause(self):
        self.sim.paused = not self.sim.paused
        self.pause_btn.label = "Resume" if self.sim.paused else "Pause"
        
    def reset_simulation(self):
        self.sim.reset_particles()
        
    def set_conservative(self):
        self.sim.params.update({
            'dt': 1.0/150.0, 'constraint_iterations': 8, 'substeps': 8,
            'damping': 0.96, 'gravity': -5.0, 'constraint_epsilon': 220.0, 'max_velocity': 8.0
        })
        self.update_sliders()
        
    def set_fast(self):
        self.sim.params.update({
            'dt': 1.0/90.0, 'constraint_iterations': 6, 'substeps': 4,
            'damping': 0.95, 'gravity': -7.0, 'constraint_epsilon': 180.0, 'max_velocity': 9.0
        })
        self.update_sliders()
        
    def set_quality(self):
        self.sim.params.update({
            'dt': 1.0/180.0, 'constraint_iterations': 10, 'substeps': 10,
            'damping': 0.97, 'gravity': -5.0, 'constraint_epsilon': 250.0, 'max_velocity': 8.0
        })
        self.update_sliders()
        
    def update_sliders(self):
        """Update slider values from simulation parameters."""
        self.dt_slider.current_val = self.sim.params['dt']
        self.iter_slider.current_val = self.sim.params['constraint_iterations']
        self.substeps_slider.current_val = self.sim.params['substeps']
        self.damping_slider.current_val = self.sim.params['damping']
        self.gravity_slider.current_val = self.sim.params['gravity']
        self.particle_radius_slider.current_val = self.sim.params['particle_radius']
        self.smoothing_radius_slider.current_val = self.sim.params['smoothing_radius']
        self.rest_density_slider.current_val = self.sim.params['rest_density']
        self.max_velocity_slider.current_val = self.sim.params['max_velocity']
        self.constraint_epsilon_slider.current_val = self.sim.params['constraint_epsilon']
        
    def update_simulation_from_sliders(self):
        """Update simulation parameters from slider values."""
        self.sim.params['dt'] = self.dt_slider.current_val
        self.sim.params['constraint_iterations'] = int(self.iter_slider.current_val)
        self.sim.params['substeps'] = int(self.substeps_slider.current_val)
        self.sim.params['damping'] = self.damping_slider.current_val
        self.sim.params['gravity'] = self.gravity_slider.current_val
        self.sim.params['particle_radius'] = self.particle_radius_slider.current_val
        self.sim.params['smoothing_radius'] = self.smoothing_radius_slider.current_val
        self.sim.params['rest_density'] = self.rest_density_slider.current_val
        self.sim.params['max_velocity'] = self.max_velocity_slider.current_val
        self.sim.params['constraint_epsilon'] = self.constraint_epsilon_slider.current_val
        
    def render_text(self, x, y, text):
        """Render text using GLUT."""
        glColor3f(1.0, 1.0, 1.0)
        glRasterPos2f(x, y)
        for char in text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))
            
    def setup_3d_camera(self):
        """Setup 3D camera for particle rendering."""
        # Set viewport for 3D rendering (right side)
        glViewport(self.gui_width, 0, self.window_width - self.gui_width, self.window_height)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = (self.window_width - self.gui_width) / self.window_height
        gluPerspective(60.0, aspect, 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Camera position
        cam_x = self.camera_distance * math.sin(math.radians(self.camera_theta)) * math.cos(math.radians(self.camera_phi))
        cam_y = self.camera_distance * math.sin(math.radians(self.camera_phi))
        cam_z = self.camera_distance * math.cos(math.radians(self.camera_theta)) * math.cos(math.radians(self.camera_phi))
        
        gluLookAt(cam_x, cam_y, cam_z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        
    def setup_2d_gui(self):
        """Setup 2D rendering for GUI."""
        # Set viewport for GUI (left side)
        glViewport(0, 0, self.gui_width, self.window_height)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.gui_width, self.window_height, 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
    def render_particles(self):
        """Render particles in 3D with pressure-based coloring."""
        positions_gpu = self.sim.positions.numpy()
        densities_gpu = self.sim.densities.numpy()
        
        # Calculate pressure from density
        rest_density = self.sim.params['rest_density']
        pressures = (densities_gpu / rest_density) - 1.0
        
        # Normalize pressure for coloring (clamp extreme values)
        min_pressure = np.percentile(pressures, 5)
        max_pressure = np.percentile(pressures, 95)
        pressure_range = max(max_pressure - min_pressure, 0.1)
        
        glPointSize(5.0)
        glBegin(GL_POINTS)
        for i in range(self.sim.num_particles):
            pos = positions_gpu[i]
            pressure = pressures[i]
            
            # Normalize pressure to [0, 1]
            pressure_factor = max(0.0, min(1.0, (pressure - min_pressure) / pressure_range))
            
            # Color mapping: Blue (low pressure) -> Green -> Yellow -> Red (high pressure)
            if pressure_factor < 0.5:
                # Blue to Green
                t = pressure_factor * 2.0
                r = 0.0
                g = 0.2 + t * 0.6
                b = 0.9 - t * 0.4
            else:
                # Green to Red
                t = (pressure_factor - 0.5) * 2.0
                r = t * 0.9
                g = 0.8 - t * 0.4
                b = 0.1
            
            glColor4f(r, g, b, 0.85)
            glVertex3f(pos[0], pos[1], pos[2])
            
        glEnd()
        
    def render_boundaries(self):
        """Render simulation boundaries."""
        glColor4f(0.3, 0.3, 0.3, 0.4)
        glBegin(GL_LINES)
        
        min_pt = self.sim.params['domain_min']
        max_pt = self.sim.params['domain_max']
        
        # Draw wireframe box edges
        vertices = [
            # Bottom face
            [min_pt[0], min_pt[1], min_pt[2]], [max_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], min_pt[2]], [max_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], min_pt[1], max_pt[2]], [min_pt[0], min_pt[1], max_pt[2]],
            [min_pt[0], min_pt[1], max_pt[2]], [min_pt[0], min_pt[1], min_pt[2]],
            
            # Vertical edges
            [min_pt[0], min_pt[1], min_pt[2]], [min_pt[0], max_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], min_pt[2]], [max_pt[0], max_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], max_pt[2]], [max_pt[0], max_pt[1], max_pt[2]],
            [min_pt[0], min_pt[1], max_pt[2]], [min_pt[0], max_pt[1], max_pt[2]]
        ]
        
        for i in range(0, len(vertices), 2):
            glVertex3f(*vertices[i])
            glVertex3f(*vertices[i+1])
        
        glEnd()
        
    def render_gui(self):
        """Render GUI elements."""
        # Draw GUI background
        glColor3f(0.15, 0.15, 0.2)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(self.gui_width, 0)
        glVertex2f(self.gui_width, self.window_height)
        glVertex2f(0, self.window_height)
        glEnd()
        
        # Draw separator line
        glColor3f(0.4, 0.4, 0.4)
        glBegin(GL_LINES)
        glVertex2f(self.gui_width, 0)
        glVertex2f(self.gui_width, self.window_height)
        glEnd()
        
        # Render GUI elements
        for element in self.gui_elements:
            element.render()
        
        # Render text labels and info
        y_pos = 30
        self.render_text(20, y_pos, "PBF Simulation Controls")
        
        y_pos = 70
        self.render_text(20, y_pos, f"dt: {self.sim.params['dt']:.4f}")
        y_pos += 35
        self.render_text(20, y_pos, f"iter: {int(self.sim.params['constraint_iterations'])}")
        y_pos += 35
        self.render_text(20, y_pos, f"substeps: {int(self.sim.params['substeps'])}")
        y_pos += 35
        self.render_text(20, y_pos, f"damp: {self.sim.params['damping']:.3f}")
        y_pos += 35
        self.render_text(20, y_pos, f"grav: {self.sim.params['gravity']:.1f}")
        y_pos += 35
        self.render_text(20, y_pos, f"radius: {self.sim.params['particle_radius']:.3f}")
        y_pos += 35
        self.render_text(20, y_pos, f"smooth: {self.sim.params['smoothing_radius']:.3f}")
        y_pos += 35
        self.render_text(20, y_pos, f"density: {self.sim.params['rest_density']:.0f}")
        y_pos += 35
        self.render_text(20, y_pos, f"max_vel: {self.sim.params['max_velocity']:.1f}")
        y_pos += 35
        self.render_text(20, y_pos, f"epsilon: {self.sim.params['constraint_epsilon']:.1f}")
        
        # Status info
        y_pos += 80
        self.render_text(20, y_pos, f"Particles: {self.sim.num_particles}")
        y_pos += 20
        self.render_text(20, y_pos, f"Status: {'PAUSED' if self.sim.paused else 'RUNNING'}")
        y_pos += 20
        self.render_text(20, y_pos, f"FPS: {self.fps:.1f}")
        
        # Debug info
        try:
            positions = self.sim.positions.numpy()
            velocities = self.sim.velocities.numpy()
            densities = self.sim.densities.numpy()
            
            # Velocity stats
            vel_magnitudes = np.linalg.norm(velocities, axis=1)
            avg_vel = np.mean(vel_magnitudes)
            max_vel = np.max(vel_magnitudes)
            
            # Pressure stats
            rest_density = self.sim.params['rest_density']
            pressures = (densities / rest_density) - 1.0
            avg_pressure = np.mean(pressures)
            max_pressure = np.max(pressures)
            min_pressure = np.min(pressures)
            
            y_pos += 30
            self.render_text(20, y_pos, f"Avg Vel: {avg_vel:.2f}")
            y_pos += 20
            self.render_text(20, y_pos, f"Max Vel: {max_vel:.2f}")
            y_pos += 20
            self.render_text(20, y_pos, f"Avg P: {avg_pressure:.3f}")
            y_pos += 20
            self.render_text(20, y_pos, f"P Range: [{min_pressure:.2f}, {max_pressure:.2f}]")
        except:
            pass
        
        # Color legend for pressure
        y_pos += 30
        self.render_text(20, y_pos, "Pressure Colors:")
        y_pos += 20
        glColor3f(0.0, 0.2, 0.9)  # Blue
        self.render_text(20, y_pos, "• Blue: Low pressure")
        y_pos += 20
        glColor3f(0.0, 0.8, 0.5)  # Green
        self.render_text(20, y_pos, "• Green: Medium pressure")
        y_pos += 20
        glColor3f(0.9, 0.4, 0.1)  # Red
        self.render_text(20, y_pos, "• Red: High pressure")
        
        # Instructions
        y_pos = self.window_height - 120
        glColor3f(1.0, 1.0, 1.0)  # Reset to white
        self.render_text(20, y_pos, "Instructions:")
        y_pos += 20
        self.render_text(20, y_pos, "• Click sliders to adjust")
        y_pos += 20
        self.render_text(20, y_pos, "• Use preset buttons")
        y_pos += 20
        self.render_text(20, y_pos, "• Drag in 3D area to rotate")
        y_pos += 20
        self.render_text(20, y_pos, "• ESC to exit")
        
    def render(self):
        """Main render function."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Update simulation parameters from sliders
        self.update_simulation_from_sliders()
        
        # Render 3D scene
        self.setup_3d_camera()
        glEnable(GL_DEPTH_TEST)
        self.render_boundaries()
        self.render_particles()
        
        # Render 2D GUI
        self.setup_2d_gui()
        glDisable(GL_DEPTH_TEST)
        self.render_gui()
        
        # Calculate FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time > 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        
        glutSwapBuffers()
        
    def handle_mouse(self, button, state, x, y):
        """Handle mouse input."""
        if x < self.gui_width:  # GUI area
            if state == GLUT_DOWN:
                for element in self.gui_elements:
                    if element.handle_click(x, y):
                        break
        else:  # 3D area - camera control
            if button == GLUT_LEFT_BUTTON:
                if state == GLUT_DOWN:
                    self.mouse_dragging = True
                    self.mouse_last_x = x
                    self.mouse_last_y = y
                else:
                    self.mouse_dragging = False
                    
    def handle_mouse_motion(self, x, y):
        """Handle mouse motion."""
        if x < self.gui_width:  # GUI area
            # Update hover states
            for element in self.gui_elements:
                element.hovered = element.contains_point(x, y)
        else:  # 3D area
            if self.mouse_dragging:
                dx = x - self.mouse_last_x
                dy = y - self.mouse_last_y
                
                self.camera_theta += dx * 0.5
                self.camera_phi += dy * 0.5
                
                # Clamp phi
                self.camera_phi = max(-89.0, min(89.0, self.camera_phi))
                
                self.mouse_last_x = x
                self.mouse_last_y = y

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
    global renderer
    renderer.window_width = width
    renderer.window_height = height
    glViewport(0, 0, width, height)

def mouse(button, state, x, y):
    """GLUT mouse callback."""
    global renderer
    renderer.handle_mouse(button, state, x, y)

def motion(x, y):
    """GLUT mouse motion callback."""
    global renderer
    renderer.handle_mouse_motion(x, y)

def passive_motion(x, y):
    """GLUT passive mouse motion callback."""
    global renderer
    renderer.handle_mouse_motion(x, y)

def keyboard(key, x, y):
    """GLUT keyboard callback."""
    global sim
    
    if key == b' ':  # Space - pause/unpause
        sim.paused = not sim.paused
        renderer.pause_btn.label = "Resume" if sim.paused else "Pause"
    elif key == b'r' or key == b'R':  # R - reset
        sim.reset_particles()
    elif key == b'\x1b':  # Escape - quit
        print("Exiting...")
        import sys
        sys.exit(0)

def idle():
    """GLUT idle callback."""
    glutPostRedisplay()

def main():
    """Main function."""
    global sim, renderer
    
    if not OPENGL_AVAILABLE:
        print("OpenGL not available. Cannot run visualization.")
        return
    
    print("PBF Simulation with Simple GUI")
    print("===============================")
    
    # Create simulation and renderer
    sim = PBFSimulation(num_particles=32384)
    renderer = GUIRenderer(sim)
    
    # Initialize GLUT
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(renderer.window_width, renderer.window_height)
    glutCreateWindow(b"PBF Simulation - GUI Controls")
    
    # Set callbacks
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutPassiveMotionFunc(passive_motion)
    glutIdleFunc(idle)
    
    # Initialize OpenGL
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_POINT_SMOOTH)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.02, 0.02, 0.05, 1.0)
    
    print("\nSimulation started!")
    print("GUI Controls:")
    print("  • Left panel: Interactive sliders and buttons")
    print("  • Right panel: 3D fluid visualization")
    print("  • Click sliders to adjust parameters")
    print("  • Use preset buttons for quick settings")
    print("  • Drag in 3D area to rotate camera")
    print("  • Space: Pause/Resume")
    print("  • R: Reset particles")
    print("  • Escape: Exit")
    
    # Start main loop
    glutMainLoop()

if __name__ == "__main__":
    main()
