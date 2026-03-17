import warp as wp

from simulation_system import SimulationSystem
from collision_kernels import collide_triangles_vs_sphere
from simulation_kernels import apply_deltas_and_zero_accumulators

from omnisurg.scene_builder import HapticProxyState


class HapticSphereCollisionSystem(SimulationSystem):
    """Single-sphere collision for the haptic proxy.

    Owns fixed-size device buffers allocated once in initialize().
    Uses collide_triangles_vs_sphere with the proxy's persistent GPU buffer
    so no per-frame host-to-device copies are needed.
    """

    def __init__(self, proxy: HapticProxyState, priority: int = 80):
        super().__init__(priority=priority)
        self.proxy = proxy
        self._accumulator: wp.array | None = None
        self._count: wp.array | None = None

    def initialize(self, model):
        self._accumulator = wp.zeros(
            model.particle_count, dtype=wp.vec3f, device=model.device,
        )
        self._count = wp.zeros(
            model.particle_count, dtype=wp.int32, device=model.device,
        )

    def solve_constraints(
        self, model, state_in, state_out,
        particle_q, particle_qd, particle_deltas,
        body_q, body_qd, body_deltas,
        dt, iteration,
    ):
        if model.tri_count == 0:
            return

        self._accumulator.zero_()
        self._count.zero_()

        wp.launch(
            kernel=collide_triangles_vs_sphere,
            dim=model.tri_count,
            inputs=[
                particle_q,
                particle_qd,
                model.particle_inv_mass,
                model.tri_indices,
                self.proxy.center_current,
                self.proxy.radius,
                0.0,
                dt,
            ],
            outputs=[self._accumulator, self._count],
            device=model.device,
        )

        wp.launch(
            kernel=apply_deltas_and_zero_accumulators,
            dim=model.particle_count,
            inputs=[self._accumulator, self._count],
            outputs=[particle_deltas],
            device=model.device,
        )
