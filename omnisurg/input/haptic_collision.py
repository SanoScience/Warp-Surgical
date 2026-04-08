import warp as wp

from omnisurg.input.haptic_proxy import HapticProxyState
from omnisurg.physics.base import SimulationSystem, SolverStage
from omnisurg.physics.collision import collide_triangles_vs_sphere
from omnisurg.physics.kernels import apply_deltas_and_zero_accumulators


class HapticSphereCollisionSystem(SimulationSystem):
    """Triangle-vs-sphere collision system for the Phase 1 haptic proxy."""

    def __init__(self, proxy: HapticProxyState, priority: int = 80):
        super().__init__(priority=priority, stage=SolverStage.PROJECTION)
        self.proxy = proxy
        self._cull_radius = proxy.radius + proxy.max_tri_extent
        self._accumulator: wp.array | None = None
        self._count: wp.array | None = None

    def get_accumulators(self):
        if self._accumulator is None:
            return None
        return (self._accumulator, self._count)

    def initialize(self, model):
        self._accumulator = wp.zeros(model.particle_count, dtype=wp.vec3f, device=model.device)
        self._count = wp.zeros(model.particle_count, dtype=wp.int32, device=model.device)

    def solve_constraints(
        self,
        model,
        state_in,
        state_out,
        particle_q,
        particle_qd,
        particle_deltas,
        body_q,
        body_qd,
        body_deltas,
        dt,
        iteration,
    ):
        if model.tri_count == 0:
            return

        wp.launch(
            kernel=collide_triangles_vs_sphere,
            dim=model.tri_count,
            inputs=[
                particle_q,
                particle_qd,
                model.particle_inv_mass,
                model.tri_indices,
                self.proxy.center_scaled,
                self.proxy.radius,
                1.0,
                0.0,
                dt,
                self._cull_radius,
            ],
            outputs=[self._accumulator, self._count],
            device=model.device,
        )

        if not self.deferred_apply:
            wp.launch(
                kernel=apply_deltas_and_zero_accumulators,
                dim=model.particle_count,
                inputs=[self._accumulator, self._count],
                outputs=[particle_deltas],
                device=model.device,
            )

