import warp as wp
import warp.sim
import warp.sim.render
from warp.sim.render import SimRendererOpenGL

from mesh_loader import load_mesh_and_build_model
from simulation_kernels import (
    clear_jacobian_accumulator,
    apply_jacobian_deltas,
    apply_tri_points_constraints_jacobian,
    set_body_position
)

class WarpSim:
    def __init__(self, stage_path="output.usd", num_frames=300, use_opengl=True):
        self.sim_substeps = 64
        self.num_frames = num_frames
        fps = 60

        self.frame_dt = 1.0 / fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # Initialize model
        self._build_model()
        
        # Initialize simulation components
        self._setup_simulation()
        
        # Initialize rendering
        self._setup_renderer(stage_path, use_opengl)
        
        # Setup CUDA graph if available
        self._setup_cuda_graph()

    def _build_model(self):
        """Build the simulation model with mesh and haptic device."""
        builder = wp.sim.ModelBuilder()
        
        # Import the mesh
        self.tri_points_connectors = load_mesh_and_build_model(builder, vertical_offset=-3.0)
        
        # Add haptic device collision body
        self.haptic_body_id = builder.add_body(
            origin=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
            m=0.0,  # Zero mass makes it kinematic
            armature=0.0
        )

        builder.add_shape_sphere(
            self.haptic_body_id,
            has_ground_collision=False,
            has_shape_collision=True,
            radius=0.2,
            pos=wp.vec3(0.0, 0.0, 0.0),
            density=100
        )

        self.model = builder.finalize()
        self.model.ground = True

    def _setup_simulation(self):
        """Initialize simulation states and integrator."""
        self.integrator = wp.sim.XPBDIntegrator(iterations=5)
        
        self.dev_pos_buffer = wp.array([0.0, 0.0, 0.0], dtype=wp.vec3, device=wp.get_device())
        
        self.rest = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        
        # Create Jacobian accumulators
        self.delta_accumulator = wp.zeros(self.model.particle_count, dtype=wp.vec3f, device=wp.get_device())
        self.count_accumulator = wp.zeros(self.model.particle_count, dtype=wp.int32, device=wp.get_device())

    def _setup_renderer(self, stage_path, use_opengl):
        """Initialize the appropriate renderer."""
        self.use_opengl = use_opengl
        
        if self.use_opengl:
            self.renderer = SimRendererOpenGL(self.model, "Warp Surgical Simulation", scaling=1.0)
        elif stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=20.0)
        else:
            self.renderer = None

    def _setup_cuda_graph(self):
        """Setup CUDA graph for performance optimization."""
        self.use_cuda_graph = wp.get_device().is_cuda
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

            # Run collision detection and integration
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            
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
                self.renderer.render(self.state_0)
                self.renderer.end_frame()
            else:
                self.renderer.begin_frame(self.sim_time)
                self.renderer.render(self.state_0)
                self.renderer.end_frame()

    def update_haptic_position(self, position):
        """Update the haptic device position in the simulation."""
        haptic_pos = wp.vec3(position[0], position[1], position[2])
        wp.copy(self.dev_pos_buffer, wp.array([haptic_pos], dtype=wp.vec3, device=wp.get_device()))

    def is_running(self):
        """Check if the simulation should continue running."""
        return self.renderer.is_running() if self.use_opengl else True
    
    def save(self):
        """Save the simulation results."""
        if self.renderer:
            self.renderer.save()
