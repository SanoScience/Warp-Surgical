import warp as wp
from newton._src.geometry import ParticleFlags

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


@wp.kernel
def compute_proxy_vertex_truncation_factors(
    particle_flags: wp.array(dtype=wp.int32),
    base_positions: wp.array(dtype=wp.vec3f),
    surface_vertex_ids: wp.array(dtype=wp.int32),
    displacement_in: wp.array(dtype=wp.vec3f),
    sphere_center_prev: wp.array(dtype=wp.vec3f),
    sphere_center: wp.array(dtype=wp.vec3f),
    sphere_radius: wp.float32,
    motion_samples: int,
    radius_margin: wp.float32,
    safety: wp.float32,
    truncation_t_out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid >= surface_vertex_ids.shape[0] * motion_samples or motion_samples <= 0:
        return

    vertex_idx = tid // motion_samples
    sample_idx = tid % motion_samples
    vertex_id = surface_vertex_ids[vertex_idx]
    if (particle_flags[vertex_id] & ParticleFlags.ACTIVE) == 0:
        return

    displacement = displacement_in[vertex_id]
    if wp.length_sq(displacement) <= 1.0e-12:
        return

    radius = sphere_radius + radius_margin
    if radius <= 0.0:
        return

    sample_factor = wp.float32(sample_idx + 1) / wp.float32(motion_samples)
    center = wp.lerp(sphere_center_prev[0], sphere_center[0], sample_factor)

    x0 = base_positions[vertex_id]
    x1 = x0 + displacement

    normal = x1 - center
    if wp.length_sq(normal) <= 1.0e-12:
        normal = x0 - center
    if wp.length_sq(normal) <= 1.0e-12:
        normal = -displacement
    if wp.length_sq(normal) <= 1.0e-12:
        return

    n = wp.normalize(normal)
    plane_point = center + n * radius

    s0 = wp.dot(n, x0 - plane_point)
    s1 = wp.dot(n, x1 - plane_point)
    if s1 >= 0.0:
        return

    t = wp.float32(1.0)
    if s0 > 0.0:
        denom = s0 - s1
        if denom <= 1.0e-8:
            return
        crossing_t = s0 / denom
        t = wp.clamp(wp.min(crossing_t * safety, crossing_t - 1.0e-3), 0.0, 1.0)
    elif s1 < s0:
        t = 0.0
    else:
        return

    wp.atomic_min(truncation_t_out, vertex_id, t)


@wp.kernel
def apply_surface_vertex_truncation(
    surface_vertex_ids: wp.array(dtype=wp.int32),
    truncation_t: wp.array(dtype=wp.float32),
    displacements: wp.array(dtype=wp.vec3f),
):
    tid = wp.tid()
    if tid >= surface_vertex_ids.shape[0]:
        return

    vertex_id = surface_vertex_ids[tid]
    displacements[vertex_id] = displacements[vertex_id] * truncation_t[vertex_id]


class HapticSphereTruncationSystem(SimulationSystem):
    """Truncate tissue motion against the haptic proxy sphere before commit."""

    def __init__(
        self,
        proxy: HapticProxyState,
        *,
        surface_vertex_ids: wp.array,
        motion_samples: int = 4,
        contact_margin: float = 0.0,
        safety: float = 0.90,
        truncate_prediction: bool = True,
        priority: int = 80,
    ):
        super().__init__(priority=priority, stage=SolverStage.TRUNCATION)
        self.proxy = proxy
        self.surface_vertex_ids = surface_vertex_ids
        self.motion_samples = motion_samples
        self.contact_margin = contact_margin
        self.safety = safety
        self.truncate_prediction_enabled = truncate_prediction
        self._truncation_t: wp.array | None = None

    def initialize(self, model):
        self._truncation_t = wp.zeros(model.particle_count, dtype=wp.float32, device=model.device)

    def truncate_prediction(
        self,
        model,
        state_in,
        state_out,
        base_positions,
        displacements,
        dt,
    ):
        if not self.truncate_prediction_enabled or model.tri_count == 0 or self._truncation_t is None:
            return

        self._apply_truncation(
            model,
            base_positions=base_positions,
            displacements=displacements,
            sphere_center_prev=self.proxy.center_scaled_prev,
            sphere_center=self.proxy.center_scaled,
            motion_samples=max(1, int(self.motion_samples)),
        )

    def truncate_deltas(
        self,
        model,
        state_in,
        state_out,
        particle_q,
        particle_qd,
        particle_deltas,
        dt,
        iteration,
    ):
        if model.tri_count == 0 or self._truncation_t is None:
            return

        self._apply_truncation(
            model,
            base_positions=particle_q,
            displacements=particle_deltas,
            sphere_center_prev=self.proxy.center_scaled,
            sphere_center=self.proxy.center_scaled,
            motion_samples=1,
        )

    def _apply_truncation(
        self,
        model,
        *,
        base_positions: wp.array,
        displacements: wp.array,
        sphere_center_prev: wp.array,
        sphere_center: wp.array,
        motion_samples: int,
    ):
        if self._truncation_t is None or len(self.surface_vertex_ids) == 0 or motion_samples <= 0:
            return

        self._truncation_t.fill_(1.0)
        wp.launch(
            kernel=compute_proxy_vertex_truncation_factors,
            dim=len(self.surface_vertex_ids) * motion_samples,
            inputs=[
                model.particle_flags,
                base_positions,
                self.surface_vertex_ids,
                displacements,
                sphere_center_prev,
                sphere_center,
                self.proxy.radius,
                motion_samples,
                self.contact_margin,
                self.safety,
            ],
            outputs=[self._truncation_t],
            device=model.device,
        )
        wp.launch(
            kernel=apply_surface_vertex_truncation,
            dim=len(self.surface_vertex_ids),
            inputs=[self.surface_vertex_ids, self._truncation_t],
            outputs=[displacements],
            device=model.device,
        )

