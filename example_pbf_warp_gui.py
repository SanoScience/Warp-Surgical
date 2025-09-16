#!/usr/bin/env python3

"""
Position Based Fluids (PBF) with Warp Renderer and ImGui

This version uses Warp's OpenGLRenderer with ImGui for parameter control,
completely removing PyOpenGL dependencies. Based on Warp's rendering examples.

Controls:
- ImGui sliders for real-time parameter adjustment
- Mouse/keyboard controls through Warp renderer
- Professional parameter control interface
"""

import numpy as np
import warp as wp
import warp.render
from warp.render.imgui_manager import ImGuiManager
from pbf_kernels import (
    pbf_predict_positions,
    pbf_compute_density,
    pbf_compute_lambda,
    pbf_compute_delta_positions,
    pbf_apply_delta_positions,
    pbf_apply_boundaries,
    pbf_update_velocities_positions,
)


# Stable PBF parameters
PBF_PARAMS = {
    'particle_radius': 0.02,
    'smoothing_radius': 0.04,
    'rest_density': 1000.0,
    'constraint_iterations': 6,
    'damping': 0.95,
    'gravity': -5.0,
    'dt': 1.0/120.0,
    'max_velocity': 8.0,
    'constraint_epsilon': 200.0,
    'grid_dim': 64,
    'domain_min': np.array([-1.2, -0.8, -1.2]),
    'domain_max': np.array([1.2, 2.0, 1.2])
}

# PBF kernels (stable versions)

class PBFImGuiManager(ImGuiManager):
    """ImGui manager for PBF simulation parameter control."""

    def __init__(self, renderer, simulation, window_pos=(10, 10), window_size=(350, 550)):
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
        if self.imgui.button("Reset Particles"):
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
            "Constraint Iterations", self.sim.params['constraint_iterations'], 1, 12
        )
        if changed:
            self.sim.params['constraint_iterations'] = new_iterations
            
        changed, new_damping = self.imgui.slider_float(
            "Damping", self.sim.params['damping'], 0.5, 0.99, "%.3f"
        )
        if changed:
            self.sim.params['damping'] = new_damping
            
        changed, new_gravity = self.imgui.slider_float(
            "Gravity", self.sim.params['gravity'], -20.0, -1.0, "%.1f"
        )
        if changed:
            self.sim.params['gravity'] = new_gravity
        
        self.imgui.separator()
        
        # Particle parameters
        self.imgui.text("Particle Parameters:")
        
        changed, new_radius = self.imgui.slider_float(
            "Particle Radius", self.sim.params['particle_radius'], 0.005, 0.05, "%.3f"
        )
        if changed:
            self.sim.params['particle_radius'] = new_radius
            
        changed, new_smoothing = self.imgui.slider_float(
            "Smoothing Radius", self.sim.params['smoothing_radius'], 0.01, 0.1, "%.3f"
        )
        if changed:
            self.sim.params['smoothing_radius'] = new_smoothing
            
        changed, new_density = self.imgui.slider_float(
            "Rest Density", self.sim.params['rest_density'], 500.0, 2000.0, "%.0f"
        )
        if changed:
            self.sim.params['rest_density'] = new_density
            
        changed, new_max_vel = self.imgui.slider_float(
            "Max Velocity", self.sim.params['max_velocity'], 2.0, 20.0, "%.1f"
        )
        if changed:
            self.sim.params['max_velocity'] = new_max_vel
            
        changed, new_epsilon = self.imgui.slider_float(
            "Constraint Epsilon", self.sim.params['constraint_epsilon'], 10.0, 1000.0, "%.1f"
        )
        if changed:
            self.sim.params['constraint_epsilon'] = new_epsilon
            
        changed, new_grid_dim = self.imgui.slider_int(
            "Grid Dimension", self.sim.params['grid_dim'], 32, 128
        )
        if changed:
            self.sim.params['grid_dim'] = new_grid_dim
            # Note: Grid dimension change requires recreation of hash grid
            # This will take effect on next reset
        
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
                'damping': 0.95,
                'gravity': -5.0,
                'max_velocity': 6.0,
                'constraint_epsilon': 300.0
            })
            
        self.imgui.same_line()
        if self.imgui.button("Fast"):
            self.sim.params.update({
                'dt': 1.0/90.0,
                'constraint_iterations': 4,
                'damping': 0.98,
                'gravity': -9.81,
                'max_velocity': 10.0,
                'constraint_epsilon': 150.0
            })
            
        self.imgui.same_line()
        if self.imgui.button("High Quality"):
            self.sim.params.update({
                'dt': 1.0/180.0,
                'constraint_iterations': 10,
                'damping': 0.96,
                'gravity': -7.0,
                'max_velocity': 8.0,
                'constraint_epsilon': 250.0
            })
        
        self.imgui.separator()
        
        # Debug info
        changed, self.show_debug_info = self.imgui.checkbox("Show Debug Info", self.show_debug_info)
        
        if self.show_debug_info:
            self.imgui.text(f"Particles: {self.sim.num_particles}")
            self.imgui.text(f"Status: {'PAUSED' if self.sim.paused else 'RUNNING'}")
            
            # Get debug stats if available
            if hasattr(self.sim, 'debug_stats') and self.sim.debug_stats:
                stats = self.sim.debug_stats
                self.imgui.text(f"Avg Velocity: {stats.get('avg_velocity', 0):.2f}")
                self.imgui.text(f"Max Velocity: {stats.get('max_velocity', 0):.2f}")
                self.imgui.text(f"Particle Spread: {stats.get('spread', 0):.2f}")
        
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
        self.imgui.text_wrapped("• Adjust max velocity to prevent explosions")

        self.imgui.end()

class PBFSimulation:
    """Position Based Fluids simulation using Warp."""
    
    def __init__(self, num_particles=1024):
        self.num_particles = num_particles
        self.params = PBF_PARAMS.copy()
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
        
        # Debug stats
        self.debug_stats = {}
        
        # Initialize particle positions
        self.reset_particles()
        
        print(f"PBF Simulation initialized with {num_particles} particles")
        
    def reset_particles(self):
        """Initialize particles in a stable, compact configuration."""
        positions = []
        
        # Create a stable dam with tighter spacing
        spacing = self.params['particle_radius'] * 2.05
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
                    
                    # Add very small random perturbation
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
        
    def update_debug_stats(self):
        """Update debug statistics."""
        try:
            positions = self.positions.numpy()
            velocities = self.velocities.numpy()
            
            vel_magnitudes = np.linalg.norm(velocities, axis=1)
            bbox_min = np.min(positions, axis=0)
            bbox_max = np.max(positions, axis=0)
            
            self.debug_stats = {
                'avg_velocity': np.mean(vel_magnitudes),
                'max_velocity': np.max(vel_magnitudes),
                'spread': np.max(bbox_max - bbox_min)
            }
        except Exception:
            self.debug_stats = {}
        
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
                self.params['damping'], self.params['dt'], self.params['max_velocity']
            ],
            device=self.device
        )
        
        # Update debug stats periodically
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 60 == 0:  # Update every 60 frames
            self.update_debug_stats()

class PBFExample:
    """Main PBF example class using Warp renderer."""
    
    def __init__(self, num_particles=1024, use_imgui=True):
        self.sim = PBFSimulation(num_particles)
        
        # Create Warp renderer
        self.renderer = wp.render.OpenGLRenderer(vsync=False)
        
        # Setup ImGui if available
        self.use_imgui = use_imgui
        if self.use_imgui:
            self.imgui_manager = PBFImGuiManager(self.renderer, self.sim)
            if self.imgui_manager.is_available:
                self.renderer.render_2d_callbacks.append(self.imgui_manager.render_frame)
            else:
                self.use_imgui = False
                print("ImGui not available, using basic rendering")
        
        # Setup camera
        self.setup_camera()
        
        print("PBF Example with Warp Renderer initialized")
        
    def setup_camera(self):
        """Setup camera position and orientation."""
        # Position camera to view the fluid simulation
        eye = (2.0, 1.0, 2.0)
        target = (0.0, 0.0, 0.0)
        up = (0.0, 1.0, 0.0)
        
        self.renderer.camera_fov = 60.0
        self.renderer.camera_near_plane = 0.1
        self.renderer.camera_far_plane = 100.0
        
        # Set camera view
        import math
        # Convert to camera transform matrix if needed
        
    def render_particles(self):
        """Render fluid particles using Warp renderer."""
        # Get particle positions
        positions = self.sim.positions.numpy()
        
        # Create colors based on height (blue to cyan gradient)
        colors = []
        for pos in positions:
            # Height-based coloring
            height_factor = max(0.0, min(1.0, (pos[1] + 0.8) / 2.8))
            r = 0.1 + height_factor * 0.4
            g = 0.3 + height_factor * 0.5  
            b = 0.8 + height_factor * 0.2
            colors.append((r, g, b))
        
        colors = np.array(colors, dtype=np.float32)
        
        # Render particles as spheres
        self.renderer.render_points(
            points=positions,
            radius=self.sim.params['particle_radius'],
            colors=colors,
            name="fluid_particles"
        )
        
    def render_boundaries(self):
        """Render simulation boundaries."""
        min_pt = self.sim.params['domain_min']
        max_pt = self.sim.params['domain_max']
        
        # Create wireframe box
        lines = []
        # Bottom face
        lines.extend([
            [min_pt[0], min_pt[1], min_pt[2]], [max_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], min_pt[2]], [max_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], min_pt[1], max_pt[2]], [min_pt[0], min_pt[1], max_pt[2]],
            [min_pt[0], min_pt[1], max_pt[2]], [min_pt[0], min_pt[1], min_pt[2]]
        ])
        # Top face
        lines.extend([
            [min_pt[0], max_pt[1], min_pt[2]], [max_pt[0], max_pt[1], min_pt[2]],
            [max_pt[0], max_pt[1], min_pt[2]], [max_pt[0], max_pt[1], max_pt[2]],
            [max_pt[0], max_pt[1], max_pt[2]], [min_pt[0], max_pt[1], max_pt[2]],
            [min_pt[0], max_pt[1], max_pt[2]], [min_pt[0], max_pt[1], min_pt[2]]
        ])
        # Vertical edges
        lines.extend([
            [min_pt[0], min_pt[1], min_pt[2]], [min_pt[0], max_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], min_pt[2]], [max_pt[0], max_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], max_pt[2]], [max_pt[0], max_pt[1], max_pt[2]],
            [min_pt[0], min_pt[1], max_pt[2]], [min_pt[0], max_pt[1], max_pt[2]]
        ])
        
        # Convert to numpy array and render as lines
        lines = np.array(lines, dtype=np.float32)
        
        # Render boundary as line strips
        if len(lines) > 0:
            self.renderer.render_line_strip(
                vertices=lines,
                color=(0.5, 0.5, 0.5),
                name="boundary_lines"
            )
        
    def render(self):
        """Main render function."""
        # Run simulation step
        self.sim.step()
        
        # Begin frame
        time = self.renderer.clock_time
        self.renderer.begin_frame(time)
        
        # Render 3D scene
        self.render_boundaries()
        self.render_particles()
        
        # End frame (renders ImGui if available)
        self.renderer.end_frame()
        
    def run(self):
        """Run the simulation loop."""
        print("\nStarting PBF simulation...")
        
        if self.use_imgui:
            print("Controls:")
            print("  • ImGui panel: Real-time parameter sliders")
            print("  • Mouse: Rotate camera")
            print("  • WASD: Move camera")
            print("  • ESC: Exit")
        else:
            print("Controls:")
            print("  • Mouse: Rotate camera") 
            print("  • WASD: Move camera")
            print("  • ESC: Exit")
            
        # Main loop
        while self.renderer.is_running():
            self.render()
            
        # Cleanup
        self.cleanup()
        
    def cleanup(self):
        """Cleanup resources."""
        if self.use_imgui:
            self.imgui_manager.shutdown()
        self.renderer.clear()

def main():
    """Main function."""
    print("Position Based Fluids (PBF) with Warp Renderer and ImGui")
    print("=========================================================")
    
    # Create and run example
    example = PBFExample(num_particles=1024, use_imgui=True)
    example.run()

if __name__ == "__main__":
    with wp.ScopedDevice("cuda"):
        main()
