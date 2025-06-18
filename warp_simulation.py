import warp as wp
import newton

from newton.utils.render import SimRendererOpenGL

import numpy as np

from render_surgsim_opengl import SurgSimRendererOpenGL

from mesh_loader import load_mesh_and_build_model
from render_opengl import CustomOpenGLRenderer
from simulation_kernels import (
    clear_jacobian_accumulator,
    apply_jacobian_deltas,
    apply_tri_points_constraints_jacobian,
    set_body_position,
    paint_vertices_near_haptic
)

class WarpSim:
    def __init__(self, stage_path="output.usd", num_frames=300, use_opengl=True):
        self.sim_substeps = 32
        self.num_frames = num_frames
        self.fps = 120

        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.sim_constraint_iterations = 5

        # Initialize model
        self._build_model()
        self.vertex_colors = wp.zeros(self.model.particle_count, dtype=wp.vec3f, device=wp.get_device())
        
        # Initialize simulation components
        self._setup_simulation()
        
        # Initialize rendering
        self._setup_renderer(stage_path, use_opengl)

        # Init component metadata
        #self.mesh_ranges = {
        #    'liver': {'vertex_start': 0, 'vertex_count': 0, 'index_start': 0, 'index_count': 0},
        #    'fat': {'vertex_start': 0, 'vertex_count': 0, 'index_start': 0, 'index_count': 0},
        #    'gallbladder': {'vertex_start': 0, 'vertex_count': 0, 'index_start': 0, 'index_count': 0}
        #}

        self.paint_color_buffer = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=wp.get_device())  # Default red
        self.paint_strength_buffer = wp.array([0.1], dtype=wp.float32, device=wp.get_device())  # Default strength
        self.set_paint_strength(0.0)
    

        # Load textures
        if self.use_opengl and self.renderer:
            self.liver_texture = self.renderer.load_texture("textures/liver-base.png")
            self.fat_texture = self.renderer.load_texture("textures/fat-base.png") 
            self.gallbladder_texture = self.renderer.load_texture("textures/gallbladder-base.png")

            # self.renderer.set_input_callbacks(
            #     on_key_press=self._on_key_press,
            #     on_key_release=self._on_key_release
            # )

        # Setup CUDA graph if available
        self._setup_cuda_graph()

    def _on_key_press(self, symbol, modifiers):
        """Handle key press events."""
        from pyglet.window import key
        
        if symbol == key.R:
            self.set_paint_color([1.0, 0.0, 0.0])
            self.set_paint_strength(1.0)
        elif symbol == key.F:
            self.set_paint_color([0.0, 1.0, 0.0])
            self.set_paint_strength(1.0)

    def _on_key_release(self, symbol, modifiers):
        """Handle key release events."""
        from pyglet.window import key
        
        # Stop painting when any paint key is released
        if symbol in [key.R, key.F]:
            self.set_paint_strength(0.0)

    def _build_model(self):
        """Build the simulation model with mesh and haptic device."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        
        # Import the mesh
        self.tri_points_connectors, self.surface_tris, uvs, self.mesh_ranges = load_mesh_and_build_model(builder, vertical_offset=-3.0)
        self.surface_tris_wp = wp.array(self.surface_tris, dtype=wp.int32, device=wp.get_device())
        self.uvs_wp = wp.array(uvs, dtype=wp.vec2f, device=wp.get_device())

        # Add haptic device collision body
        self.haptic_body_id = builder.add_body(
            xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
            mass=1.0,  # Zero mass makes it kinematic
            armature=0.0
        )

        builder.add_shape_sphere(
            body=self.haptic_body_id,
            xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
            radius=0.2,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=1000
            )
        )
        


        self.model = builder.finalize()

    def _paint_vertices_near_haptic_proxy(self, paint_radius=0.25, falloff_power=2.0):
        """Paint vertex colors near the haptic proxy position."""
        wp.launch(
            paint_vertices_near_haptic,
            dim=self.model.particle_count,
            inputs=[
                self.state_0.particle_q,  # vertex positions
                self.state_0.body_q,      # haptic position
                self.vertex_colors,       # vertex colors to modify
                paint_radius,             # paint radius
                self.paint_color_buffer,  # paint color (from array)
                self.paint_strength_buffer,  # paint strength (from array)
                falloff_power             # falloff power for smooth edges
            ],
            device=wp.get_device()
        )

    def set_paint_color(self, color):
        """Set the paint color from CPU."""
        self.paint_color_buffer.assign([wp.vec3(color[0], color[1], color[2])])

    def set_paint_strength(self, strength):
        """Set the paint strength from CPU."""
        self.paint_strength_buffer.assign([strength])

    def _setup_simulation(self):
        """Initialize simulation states and integrator."""
        self.integrator = newton.solvers.XPBDSolver(self.model, iterations=5)
        
        self.dev_pos_buffer = wp.array([0.0, 0.0, 0.0], dtype=wp.vec3, device=wp.get_device())
        
        self.rest = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.collide(self.state_0)
        
        # Create Jacobian accumulators
        self.delta_accumulator = wp.zeros(self.model.particle_count, dtype=wp.vec3f, device=wp.get_device())
        self.count_accumulator = wp.zeros(self.model.particle_count, dtype=wp.int32, device=wp.get_device())

    def _setup_renderer(self, stage_path, use_opengl):
        """Initialize the appropriate renderer."""
        self.use_opengl = use_opengl
        
        if self.use_opengl:
            self.renderer = SurgSimRendererOpenGL(self.model, "Warp Surgical Simulation", scaling=1.0)
        elif stage_path:
            self.renderer = newton.render.SimRenderer(self.model, stage_path, scaling=20.0)
        else:
            self.renderer = None

    def _setup_cuda_graph(self):
        """Setup CUDA graph for performance optimization."""
        self.use_cuda_graph = False #wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        """Run one simulation step with all substeps."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # Update haptic device position
            wp.launch(
                set_body_position,
                dim=1,
                inputs=[self.state_0.body_q, self.state_0.body_qd, self.haptic_body_id, self.dev_pos_buffer],
                device=self.state_0.body_q.device,
            )

            for _ in range(self.sim_constraint_iterations):
                # Clear Jacobian accumulators
                wp.launch(
                    clear_jacobian_accumulator,
                    dim=self.model.particle_count,
                    inputs=[self.delta_accumulator, self.count_accumulator],
                    device=self.state_0.particle_q.device,
                )

                # Apply constraints
                wp.launch(
                    apply_tri_points_constraints_jacobian,
                    dim=len(self.tri_points_connectors),
                    inputs=[
                        self.state_0.particle_q,
                        self.tri_points_connectors,
                        self.delta_accumulator,
                        self.count_accumulator
                    ],
                    device=self.state_0.particle_q.device,
                )

                # Apply accumulated deltas
                wp.launch(
                    apply_jacobian_deltas,
                    dim=self.model.particle_count,
                    inputs=[self.state_0.particle_q, self.delta_accumulator, self.count_accumulator],
                    device=self.state_0.particle_q.device,
                )

           # self._paint_vertices_near_haptic_proxy()

            # Run collision detection and integration
            self.contacts = self.model.collide(self.state_0)
            #if self.contacts:
            #    print(f"Contacts detected: {self.contacts.soft_contact_normal}")

            self.integrator.step(self.model, self.state_0, self.state_1, None, self.contacts, self.sim_dt)
            
            # Swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        """Advance simulation by one frame."""
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        """Render the current simulation state."""
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            if self.use_opengl:
                self.renderer.begin_frame()
                
                self.renderer.render_contacts(self.state_0, self.contacts)
                
                
                self.renderer.render_sphere(
                    "sphere",
                    [self.haptic_pos_right[0] * 0.01, self.haptic_pos_right[1] * 0.01, self.haptic_pos_right[2] * 0.01],
                    [0.0, 0.0, 0.0, 1.0],
                    0.1,
                )

                self.renderer.render_sphere(
                    "locking_sphere",
                    [0.5, 1.5, -5.0],
                    [0.0, 0.0, 0.0, 1.0],
                    1,
                )
                # Render liver mesh
                if self.mesh_ranges['liver']['index_count'] > 0:
                    self.renderer.render_mesh_warp_range(
                        name="liver_mesh",
                        points=self.state_0.particle_q,
                        indices=self.surface_tris_wp,
                        texture_coords=self.uvs_wp,
                        texture=self.liver_texture,
                        colors=self.vertex_colors,
                        index_start=self.mesh_ranges['liver']['index_start'],
                        index_count=self.mesh_ranges['liver']['index_count'],
                        update_topology=False
                    )
                
                # Render fat mesh
                if self.mesh_ranges['fat']['index_count'] > 0:
                    self.renderer.render_mesh_warp_range(
                        name="fat_mesh",
                        points=self.state_0.particle_q,
                        indices=self.surface_tris_wp,
                        texture_coords=self.uvs_wp,
                        texture=self.fat_texture,
                        colors=self.vertex_colors,
                        index_start=self.mesh_ranges['fat']['index_start'],
                        index_count=self.mesh_ranges['fat']['index_count'],
                        update_topology=False
                    )
                
                # Render gallbladder mesh
                if self.mesh_ranges['gallbladder']['index_count'] > 0:
                    self.renderer.render_mesh_warp_range(
                        name="gallbladder_mesh",
                        points=self.state_0.particle_q,
                        indices=self.surface_tris_wp,
                        texture_coords=self.uvs_wp,
                        texture=self.gallbladder_texture,
                        colors=self.vertex_colors,
                        index_start=self.mesh_ranges['gallbladder']['index_start'],
                        index_count=self.mesh_ranges['gallbladder']['index_count'],
                        update_topology=False
                    )
                
                self.renderer.end_frame()
            else:
                self.renderer.begin_frame(self.sim_time)
                self.renderer.render(self.state_0)

                self.renderer.end_frame()

    def update_haptic_position(self, position):
        """Update the haptic device position in the simulation."""
        
        haptic_pos = wp.vec3(position[0], position[1], position[2] - 500.0)  # Offset to avoid collision with ground;
        self.haptic_pos_right = [haptic_pos[0], haptic_pos[1], haptic_pos[2]];
        wp.copy(self.dev_pos_buffer, wp.array([haptic_pos], dtype=wp.vec3, device=wp.get_device()))

    def is_running(self):
        """Check if the simulation should continue running."""
        return self.renderer.is_running() if self.use_opengl else True
    
    def save(self):
        """Save the simulation results."""
        if self.renderer:
            self.renderer.save()
