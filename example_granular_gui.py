#!/usr/bin/env python3

"""
Granular Particle Simulation with Spatial Hash Grid Collisions

This creates a granular (non-PBF) particle simulation using Warp's spatial hash grid
for efficient particle-particle collision detection. Unlike PBF which uses density
constraints, this uses simple distance-based collision response.

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

# PBD Granular simulation parameters
GRANULAR_PARAMS = {
    'particle_radius': 0.02,
    'collision_radius': 0.04,  # Slightly larger than visual radius
    'restitution': 0.3,
    'friction': 0.8,
    'gravity': -9.81,
    'dt': 0.001,
    'substeps': 4,
    'constraint_iterations': 4,  # PBD constraint iterations
    'max_velocity': 10.0,
    'air_damping': 0.99,
    'grid_dim': 64,
    'domain_min': np.array([-2.0, -1.0, -2.0]),
    'domain_max': np.array([2.0, 3.0, 2.0])
}

# Warp kernels for granular simulation
@wp.kernel
def granular_predict_positions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    predicted_positions: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: wp.float32,
    max_velocity: wp.float32
):
    tid = wp.tid()

    # Apply gravity
    vel = velocities[tid]
    vel = vel + gravity * dt

    # Clamp velocity
    vel_mag = wp.length(vel)
    if vel_mag > max_velocity:
        vel = vel * (max_velocity / vel_mag)
        velocities[tid] = vel

    # Predict position
    predicted_positions[tid] = positions[tid] + vel * dt

@wp.kernel
def granular_collision_constraints(
    predicted_positions: wp.array(dtype=wp.vec3),
    position_deltas: wp.array(dtype=wp.vec3),
    constraint_counts: wp.array(dtype=wp.int32),
    hash_grid: wp.uint64,
    collision_radius: wp.float32
):
    tid = wp.tid()

    pos_i = predicted_positions[tid]
    delta_sum = wp.vec3(0.0, 0.0, 0.0)
    constraint_count = int(0)  # Declare as dynamic variable

    # Query neighbors using spatial hash
    for j in wp.hash_grid_query(hash_grid, pos_i, collision_radius):
        if j == tid:
            continue

        pos_j = predicted_positions[j]

        # Distance between particles
        delta = pos_i - pos_j
        dist = wp.length(delta)

        # Check collision (particles overlap)
        if dist > 0.001 and dist < collision_radius:  # Avoid division by zero
            # Collision constraint: maintain minimum distance
            constraint_violation = collision_radius - dist
            if constraint_violation > 0.0:
                # Collision normal (from j to i)
                normal = delta / dist

                # PBD position correction (half the correction for each particle)
                correction = normal * constraint_violation * 0.5
                delta_sum = delta_sum + correction
                constraint_count = constraint_count + 1

    position_deltas[tid] = delta_sum
    constraint_counts[tid] = constraint_count

@wp.kernel
def granular_apply_position_deltas(
    predicted_positions: wp.array(dtype=wp.vec3),
    position_deltas: wp.array(dtype=wp.vec3),
    constraint_counts: wp.array(dtype=wp.int32),
    stiffness: wp.float32
):
    tid = wp.tid()

    # Apply accumulated position corrections
    delta = position_deltas[tid]
    count = constraint_counts[tid]

    if count > 0:
        # Average the corrections and apply with stiffness
        avg_correction = delta * (stiffness / wp.float32(count))
        predicted_positions[tid] = predicted_positions[tid] + avg_correction

@wp.kernel
def granular_apply_boundaries(
    predicted_positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    domain_min: wp.vec3,
    domain_max: wp.vec3,
    restitution: wp.float32,
    particle_radius: wp.float32
):
    tid = wp.tid()

    pos = predicted_positions[tid]
    vel = velocities[tid]

    # X boundaries
    if pos[0] - particle_radius < domain_min[0]:
        pos = wp.vec3(domain_min[0] + particle_radius, pos[1], pos[2])
        vel = wp.vec3(-vel[0] * restitution, vel[1], vel[2])
    elif pos[0] + particle_radius > domain_max[0]:
        pos = wp.vec3(domain_max[0] - particle_radius, pos[1], pos[2])
        vel = wp.vec3(-vel[0] * restitution, vel[1], vel[2])

    # Y boundaries (floor and ceiling)
    if pos[1] - particle_radius < domain_min[1]:
        pos = wp.vec3(pos[0], domain_min[1] + particle_radius, pos[2])
        vel = wp.vec3(vel[0], -vel[1] * restitution, vel[2])
    elif pos[1] + particle_radius > domain_max[1]:
        pos = wp.vec3(pos[0], domain_max[1] - particle_radius, pos[2])
        vel = wp.vec3(vel[0], -vel[1] * restitution, vel[2])

    # Z boundaries
    if pos[2] - particle_radius < domain_min[2]:
        pos = wp.vec3(pos[0], pos[1], domain_min[2] + particle_radius)
        vel = wp.vec3(vel[0], vel[1], -vel[2] * restitution)
    elif pos[2] + particle_radius > domain_max[2]:
        pos = wp.vec3(pos[0], pos[1], domain_max[2] - particle_radius)
        vel = wp.vec3(vel[0], vel[1], -vel[2] * restitution)

    predicted_positions[tid] = pos
    velocities[tid] = vel

@wp.kernel
def granular_update_velocities_positions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    predicted_positions: wp.array(dtype=wp.vec3),
    air_damping: wp.float32,
    dt: wp.float32,
    max_velocity: wp.float32
):
    tid = wp.tid()

    # Update velocity based on position change (PBD style)
    old_pos = positions[tid]
    new_pos = predicted_positions[tid]
    vel = (new_pos - old_pos) / dt

    # Apply air damping
    vel = vel * air_damping

    # Clamp velocity
    vel_mag = wp.length(vel)
    if vel_mag > max_velocity:
        vel = vel * (max_velocity / vel_mag)

    velocities[tid] = vel
    positions[tid] = new_pos

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

class GranularSimulation:
    def __init__(self, num_particles=2048):
        self.num_particles = num_particles
        self.params = GRANULAR_PARAMS.copy()
        self.paused = False

        # Initialize Warp
        wp.init()
        self.device = wp.get_device()

        # Create particle arrays for PBD
        self.positions = wp.zeros(num_particles, dtype=wp.vec3, device=self.device)
        self.velocities = wp.zeros(num_particles, dtype=wp.vec3, device=self.device)
        self.predicted_positions = wp.zeros(num_particles, dtype=wp.vec3, device=self.device)
        self.position_deltas = wp.zeros(num_particles, dtype=wp.vec3, device=self.device)
        self.constraint_counts = wp.zeros(num_particles, dtype=wp.int32, device=self.device)

        # Create spatial hash grid
        self.hash_grid = wp.HashGrid(
            dim_x=self.params['grid_dim'],
            dim_y=self.params['grid_dim'],
            dim_z=self.params['grid_dim'],
            device=self.device
        )

        # Initialize particle positions
        self.reset_particles()

        print(f"Granular Simulation initialized - {num_particles} particles")

    def reset_particles(self):
        """Initialize particles in a stable configuration."""
        positions = []

        # Create a dense granular pile - use closer spacing for more collisions
        spacing = self.params['collision_radius'] * 1.8  # Slightly overlapping for collisions
        layers_x = int(1.2 / spacing)
        layers_y = int(1.5 / spacing)
        layers_z = int(1.2 / spacing)

        count = 0
        for i in range(layers_x):
            for j in range(layers_y):
                for k in range(layers_z):
                    if count >= self.num_particles:
                        break

                    x = -0.6 + i * spacing
                    y = 0.5 + j * spacing  # Start above ground for visible fall
                    z = -0.6 + k * spacing

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

        # Fill remaining particles in a second layer for more interactions
        while count < self.num_particles:
            layer = (count - (layers_x * layers_y * layers_z)) // (layers_x * layers_z)
            remainder = (count - (layers_x * layers_y * layers_z)) % (layers_x * layers_z)
            i = remainder % layers_x
            k = remainder // layers_x

            x = -0.6 + i * spacing + (np.random.random() - 0.5) * spacing * 0.1
            y = 0.5 + (layers_y + layer) * spacing + (np.random.random() - 0.5) * spacing * 0.1
            z = -0.6 + k * spacing + (np.random.random() - 0.5) * spacing * 0.1

            positions.append([x, y, z])
            count += 1

        # Copy to GPU
        positions_array = np.array(positions, dtype=np.float32)
        wp.copy(self.positions, wp.array(positions_array, dtype=wp.vec3, device=self.device))
        wp.copy(self.velocities, wp.zeros(self.num_particles, dtype=wp.vec3, device=self.device))

    def step(self):
        """Run one simulation step with substepping."""
        if self.paused:
            return

        # Substepping for stability
        substeps = self.params['substeps']
        sub_dt = self.params['dt'] / substeps

        for substep in range(substeps):
            self.substep(sub_dt)

    def substep(self, dt):
        """Run one simulation substep using PBD."""
        # Step 1: Predict positions
        gravity = wp.vec3(0.0, self.params['gravity'], 0.0)
        wp.launch(
            granular_predict_positions,
            dim=self.num_particles,
            inputs=[
                self.positions, self.velocities, self.predicted_positions,
                gravity, dt, self.params['max_velocity']
            ],
            device=self.device
        )

        # Step 2: Build spatial hash for collision detection
        self.hash_grid.build(self.predicted_positions, self.params['collision_radius'])

        # Step 3: PBD constraint iterations
        stiffness = 1.0  # Full constraint stiffness
        for iteration in range(self.params['constraint_iterations']):
            # Reset constraint accumulators
            self.position_deltas.zero_()
            self.constraint_counts.zero_()

            # Compute collision constraints
            wp.launch(
                granular_collision_constraints,
                dim=self.num_particles,
                inputs=[
                    self.predicted_positions, self.position_deltas, self.constraint_counts,
                    self.hash_grid.id, self.params['collision_radius']
                ],
                device=self.device
            )

            # Apply position corrections
            wp.launch(
                granular_apply_position_deltas,
                dim=self.num_particles,
                inputs=[
                    self.predicted_positions, self.position_deltas, self.constraint_counts,
                    stiffness
                ],
                device=self.device
            )

        # Step 4: Apply boundary conditions
        domain_min = wp.vec3(*self.params['domain_min'])
        domain_max = wp.vec3(*self.params['domain_max'])
        wp.launch(
            granular_apply_boundaries,
            dim=self.num_particles,
            inputs=[
                self.predicted_positions, self.velocities, domain_min, domain_max,
                self.params['restitution'], self.params['particle_radius']
            ],
            device=self.device
        )

        # Step 5: Update velocities and positions (PBD style)
        wp.launch(
            granular_update_velocities_positions,
            dim=self.num_particles,
            inputs=[
                self.positions, self.velocities, self.predicted_positions,
                self.params['air_damping'], dt, self.params['max_velocity']
            ],
            device=self.device
        )

class GUIRenderer:
    """Renderer with simple GUI controls."""
    def __init__(self, simulation):
        self.sim = simulation
        self.camera_distance = 4.0
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
        self.dt_slider = SimpleSlider(20, y_pos, 250, 20, 1.0/240.0, 1.0/30.0, self.sim.params['dt'], "Timestep")
        self.gui_elements.append(self.dt_slider)
        y_pos += spacing

        # Substeps slider
        self.substeps_slider = SimpleSlider(20, y_pos, 250, 20, 1, 8, self.sim.params['substeps'], "Substeps")
        self.gui_elements.append(self.substeps_slider)
        y_pos += spacing

        # Constraint iterations slider
        self.iterations_slider = SimpleSlider(20, y_pos, 250, 20, 1, 10, self.sim.params['constraint_iterations'], "PBD Iterations")
        self.gui_elements.append(self.iterations_slider)
        y_pos += spacing

        # Restitution slider
        self.restitution_slider = SimpleSlider(20, y_pos, 250, 20, 0.0, 1.0, self.sim.params['restitution'], "Restitution")
        self.gui_elements.append(self.restitution_slider)
        y_pos += spacing

        # Friction slider
        self.friction_slider = SimpleSlider(20, y_pos, 250, 20, 0.0, 1.0, self.sim.params['friction'], "Friction")
        self.gui_elements.append(self.friction_slider)
        y_pos += spacing

        # Gravity slider
        self.gravity_slider = SimpleSlider(20, y_pos, 250, 20, -20.0, -1.0, self.sim.params['gravity'], "Gravity")
        self.gui_elements.append(self.gravity_slider)
        y_pos += spacing

        # Particle radius slider
        self.particle_radius_slider = SimpleSlider(20, y_pos, 250, 20, 0.005, 0.05, self.sim.params['particle_radius'], "Particle Radius")
        self.gui_elements.append(self.particle_radius_slider)
        y_pos += spacing

        # Collision radius slider
        self.collision_radius_slider = SimpleSlider(20, y_pos, 250, 20, 0.01, 0.1, self.sim.params['collision_radius'], "Collision Radius")
        self.gui_elements.append(self.collision_radius_slider)
        y_pos += spacing

        # Air damping slider
        self.air_damping_slider = SimpleSlider(20, y_pos, 250, 20, 0.9, 1.0, self.sim.params['air_damping'], "Air Damping")
        self.gui_elements.append(self.air_damping_slider)
        y_pos += spacing

        # Max velocity slider
        self.max_velocity_slider = SimpleSlider(20, y_pos, 250, 20, 2.0, 20.0, self.sim.params['max_velocity'], "Max Velocity")
        self.gui_elements.append(self.max_velocity_slider)
        y_pos += spacing * 2

        # Control buttons
        self.pause_btn = SimpleButton(20, y_pos, 80, 25, "Pause", self.toggle_pause)
        self.gui_elements.append(self.pause_btn)

        self.reset_btn = SimpleButton(110, y_pos, 80, 25, "Reset", self.reset_simulation)
        self.gui_elements.append(self.reset_btn)
        y_pos += spacing

        # Preset buttons
        self.bouncy_btn = SimpleButton(20, y_pos, 80, 25, "Bouncy", self.set_bouncy)
        self.gui_elements.append(self.bouncy_btn)

        self.sand_btn = SimpleButton(110, y_pos, 80, 25, "Sand", self.set_sand)
        self.gui_elements.append(self.sand_btn)

        self.water_btn = SimpleButton(200, y_pos, 70, 25, "Water", self.set_water)
        self.gui_elements.append(self.water_btn)

    def toggle_pause(self):
        self.sim.paused = not self.sim.paused
        self.pause_btn.label = "Resume" if self.sim.paused else "Pause"

    def reset_simulation(self):
        self.sim.reset_particles()

    def set_bouncy(self):
        self.sim.params.update({
            'restitution': 0.8, 'friction': 0.2, 'air_damping': 0.999,
            'gravity': -9.81, 'substeps': 2, 'constraint_iterations': 2
        })
        self.update_sliders()

    def set_sand(self):
        self.sim.params.update({
            'restitution': 0.1, 'friction': 0.9, 'air_damping': 0.95,
            'gravity': -9.81, 'substeps': 4, 'constraint_iterations': 6
        })
        self.update_sliders()

    def set_water(self):
        self.sim.params.update({
            'restitution': 0.3, 'friction': 0.3, 'air_damping': 0.98,
            'gravity': -9.81, 'substeps': 6, 'constraint_iterations': 4
        })
        self.update_sliders()

    def update_sliders(self):
        """Update slider values from simulation parameters."""
        self.dt_slider.current_val = self.sim.params['dt']
        self.substeps_slider.current_val = self.sim.params['substeps']
        self.iterations_slider.current_val = self.sim.params['constraint_iterations']
        self.restitution_slider.current_val = self.sim.params['restitution']
        self.friction_slider.current_val = self.sim.params['friction']
        self.gravity_slider.current_val = self.sim.params['gravity']
        self.particle_radius_slider.current_val = self.sim.params['particle_radius']
        self.collision_radius_slider.current_val = self.sim.params['collision_radius']
        self.air_damping_slider.current_val = self.sim.params['air_damping']
        self.max_velocity_slider.current_val = self.sim.params['max_velocity']

    def update_simulation_from_sliders(self):
        """Update simulation parameters from slider values."""
        self.sim.params['dt'] = self.dt_slider.current_val
        self.sim.params['substeps'] = int(self.substeps_slider.current_val)
        self.sim.params['constraint_iterations'] = int(self.iterations_slider.current_val)
        self.sim.params['restitution'] = self.restitution_slider.current_val
        self.sim.params['friction'] = self.friction_slider.current_val
        self.sim.params['gravity'] = self.gravity_slider.current_val
        self.sim.params['particle_radius'] = self.particle_radius_slider.current_val
        self.sim.params['collision_radius'] = self.collision_radius_slider.current_val
        self.sim.params['air_damping'] = self.air_damping_slider.current_val
        self.sim.params['max_velocity'] = self.max_velocity_slider.current_val

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
        """Render particles in 3D with velocity-based coloring."""
        positions_gpu = self.sim.positions.numpy()
        velocities_gpu = self.sim.velocities.numpy()

        # Calculate speed for coloring
        speeds = np.linalg.norm(velocities_gpu, axis=1)
        max_speed = max(np.max(speeds), 0.1)

        glPointSize(6.0)
        glBegin(GL_POINTS)
        for i in range(self.sim.num_particles):
            pos = positions_gpu[i]
            speed = speeds[i]

            # Normalize speed to [0, 1]
            speed_factor = min(speed / max_speed, 1.0)

            # Color mapping: Blue (slow) -> Green -> Yellow -> Red (fast)
            if speed_factor < 0.33:
                # Blue to Green
                t = speed_factor * 3.0
                r = 0.1
                g = 0.3 + t * 0.4
                b = 0.8 - t * 0.3
            elif speed_factor < 0.66:
                # Green to Yellow
                t = (speed_factor - 0.33) * 3.0
                r = t * 0.6
                g = 0.7 + t * 0.2
                b = 0.5 - t * 0.4
            else:
                # Yellow to Red
                t = (speed_factor - 0.66) * 3.0
                r = 0.6 + t * 0.3
                g = 0.9 - t * 0.6
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
        self.render_text(20, y_pos, "Granular Simulation Controls")

        y_pos = 70
        self.render_text(20, y_pos, f"dt: {self.sim.params['dt']:.4f}")
        y_pos += 35
        self.render_text(20, y_pos, f"substeps: {int(self.sim.params['substeps'])}")
        y_pos += 35
        self.render_text(20, y_pos, f"iterations: {int(self.sim.params['constraint_iterations'])}")
        y_pos += 35
        self.render_text(20, y_pos, f"restitution: {self.sim.params['restitution']:.2f}")
        y_pos += 35
        self.render_text(20, y_pos, f"friction: {self.sim.params['friction']:.2f}")
        y_pos += 35
        self.render_text(20, y_pos, f"gravity: {self.sim.params['gravity']:.1f}")
        y_pos += 35
        self.render_text(20, y_pos, f"radius: {self.sim.params['particle_radius']:.3f}")
        y_pos += 35
        self.render_text(20, y_pos, f"collision: {self.sim.params['collision_radius']:.3f}")
        y_pos += 35
        self.render_text(20, y_pos, f"damping: {self.sim.params['air_damping']:.3f}")
        y_pos += 35
        self.render_text(20, y_pos, f"max_vel: {self.sim.params['max_velocity']:.1f}")

        # Status info
        y_pos += 70
        self.render_text(20, y_pos, f"Particles: {self.sim.num_particles}")
        y_pos += 20
        self.render_text(20, y_pos, f"Status: {'PAUSED' if self.sim.paused else 'RUNNING'}")
        y_pos += 20
        self.render_text(20, y_pos, f"FPS: {self.fps:.1f}")

        # Velocity stats
        try:
            velocities = self.sim.velocities.numpy()
            vel_magnitudes = np.linalg.norm(velocities, axis=1)
            avg_vel = np.mean(vel_magnitudes)
            max_vel = np.max(vel_magnitudes)

            y_pos += 30
            self.render_text(20, y_pos, f"Avg Speed: {avg_vel:.2f}")
            y_pos += 20
            self.render_text(20, y_pos, f"Max Speed: {max_vel:.2f}")
        except:
            pass

        # Color legend for velocity
        y_pos += 30
        self.render_text(20, y_pos, "Speed Colors:")
        y_pos += 20
        glColor3f(0.1, 0.3, 0.8)  # Blue
        self.render_text(20, y_pos, "• Blue: Slow")
        y_pos += 20
        glColor3f(0.0, 0.7, 0.5)  # Green
        self.render_text(20, y_pos, "• Green: Medium")
        y_pos += 20
        glColor3f(0.9, 0.3, 0.1)  # Red
        self.render_text(20, y_pos, "• Red: Fast")

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
    global sim, renderer

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

    print("PBD Granular Particle Simulation with Spatial Hash Collisions")
    print("=============================================================")

    # Create simulation and renderer
    sim = GranularSimulation(num_particles=4096)
    renderer = GUIRenderer(sim)

    # Initialize GLUT
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(renderer.window_width, renderer.window_height)
    glutCreateWindow(b"PBD Granular Simulation - Position Based Dynamics")

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

    print("\nPBD Granular simulation started!")
    print("GUI Controls:")
    print("  • Left panel: Interactive sliders and buttons")
    print("  • Right panel: 3D particle visualization")
    print("  • Click sliders to adjust parameters")
    print("  • PBD Iterations: Controls constraint solver accuracy")
    print("  • Use preset buttons for different materials")
    print("  • Drag in 3D area to rotate camera")
    print("  • Space: Pause/Resume")
    print("  • R: Reset particles")
    print("  • Escape: Exit")

    # Start main loop
    glutMainLoop()

if __name__ == "__main__":
    main()