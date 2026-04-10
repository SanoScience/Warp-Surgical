import numpy as np
import warp as wp
from newton._src.geometry import ParticleFlags
from newton._src.geometry.kernels import triangle_closest_point

from omnisurg.input.haptic_proxy import HapticProxyState
from omnisurg.physics.base import SimulationSystem, SolverStage
from omnisurg.physics.kernels import apply_deltas_and_zero_accumulators


@wp.func
def _double_sided_reaction_dir(
    v0: wp.vec3f,
    v1: wp.vec3f,
    v2: wp.vec3f,
    contact_point: wp.vec3f,
    reference_dir: wp.vec3f,
) -> wp.vec3f:
    if wp.length_sq(reference_dir) > 1.0e-12:
        return wp.normalize(reference_dir)

    centroid = (v0 + v1 + v2) / 3.0
    centroid_dir = centroid - contact_point
    if wp.length_sq(centroid_dir) > 1.0e-12:
        return wp.normalize(centroid_dir)

    d0 = v0 - contact_point
    d1 = v1 - contact_point
    d2 = v2 - contact_point

    best = d0
    if wp.length_sq(d1) > wp.length_sq(best):
        best = d1
    if wp.length_sq(d2) > wp.length_sq(best):
        best = d2

    if wp.length_sq(best) > 1.0e-12:
        return wp.normalize(best)

    return wp.vec3f(1.0, 0.0, 0.0)


@wp.kernel
def collide_triangles_vs_haptic_sphere_with_reaction(
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    inv_masses: wp.array(dtype=wp.float32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    sphere_center: wp.array(dtype=wp.vec3f),
    sphere_radius: wp.float32,
    sphere_center_scale: wp.float32,
    restitution: wp.float32,
    dt: wp.float32,
    cull_radius: wp.float32,
    delta_accumulator: wp.array(dtype=wp.vec3f),
    delta_counter: wp.array(dtype=wp.int32),
    reaction_accumulator: wp.array(dtype=wp.vec3f),
    reaction_counter: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= tri_indices.shape[0]:
        return

    t1 = tri_indices[tid, 0]
    t2 = tri_indices[tid, 1]
    t3 = tri_indices[tid, 2]

    p1 = positions[t1]
    p2 = positions[t2]
    p3 = positions[t3]

    sphere_pos = sphere_center[0] * sphere_center_scale

    if cull_radius > 0.0:
        centroid = (p1 + p2 + p3) / 3.0
        if wp.length(centroid - sphere_pos) > cull_radius:
            return

    w1 = inv_masses[t1]
    w2 = inv_masses[t2]
    w3 = inv_masses[t3]
    weight = w1 + w2 + w3
    if weight <= 0.0:
        return

    closest_p, bary, feature_type = triangle_closest_point(p1, p2, p3, sphere_pos)
    to_sphere = closest_p - sphere_pos
    dist = wp.length(to_sphere)
    if dist >= sphere_radius:
        return

    penetration = sphere_radius - dist
    if dist > 1.0e-8:
        correction_dir = to_sphere / dist
    else:
        correction_dir = _double_sided_reaction_dir(
            p1,
            p2,
            p3,
            sphere_pos,
            -((velocities[t1] + velocities[t2] + velocities[t3]) / 3.0),
        )

    total_correction = correction_dir * penetration
    d1 = total_correction * (w1 / weight)
    d2 = total_correction * (w2 / weight)
    d3 = total_correction * (w3 / weight)

    wp.atomic_add(delta_accumulator, t1, d1)
    wp.atomic_add(delta_accumulator, t2, d2)
    wp.atomic_add(delta_accumulator, t3, d3)
    wp.atomic_add(delta_counter, t1, 1)
    wp.atomic_add(delta_counter, t2, 1)
    wp.atomic_add(delta_counter, t3, 1)

    # Equal-and-opposite contact correction that can be mapped into a haptic force.
    wp.atomic_add(reaction_accumulator, 0, -total_correction)
    wp.atomic_add(reaction_counter, 0, 1)


class HapticSphereCollisionSystem(SimulationSystem):
    """Triangle-vs-sphere collision system for the Phase 1 haptic proxy."""

    def __init__(self, proxy: HapticProxyState, priority: int = 80):
        super().__init__(priority=priority, stage=SolverStage.PROJECTION)
        self.proxy = proxy
        self._cull_radius = proxy.radius + proxy.max_tri_extent
        self._accumulator: wp.array | None = None
        self._count: wp.array | None = None
        self._reaction_accumulator: wp.array | None = None
        self._reaction_count: wp.array | None = None

    def get_accumulators(self):
        if self._accumulator is None:
            return None
        return (self._accumulator, self._count)

    def initialize(self, model):
        self._accumulator = wp.zeros(model.particle_count, dtype=wp.vec3f, device=model.device)
        self._count = wp.zeros(model.particle_count, dtype=wp.int32, device=model.device)
        self._reaction_accumulator = wp.zeros(1, dtype=wp.vec3f, device=model.device)
        self._reaction_count = wp.zeros(1, dtype=wp.int32, device=model.device)

    def clear_reaction(self):
        if self._reaction_accumulator is None or self._reaction_count is None:
            return
        self._reaction_accumulator.zero_()
        self._reaction_count.zero_()

    def get_reaction_average(self) -> tuple[np.ndarray, int]:
        if self._reaction_accumulator is None or self._reaction_count is None:
            return np.zeros(3, dtype=np.float32), 0

        reaction = np.asarray(self._reaction_accumulator.numpy()[0], dtype=np.float32)
        count = int(self._reaction_count.numpy()[0])
        if count <= 0:
            return np.zeros(3, dtype=np.float32), 0

        return reaction / float(count), count

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
        if model.tri_count == 0 or self._reaction_accumulator is None or self._reaction_count is None:
            return

        wp.launch(
            kernel=collide_triangles_vs_haptic_sphere_with_reaction,
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
            outputs=[
                self._accumulator,
                self._count,
                self._reaction_accumulator,
                self._reaction_count,
            ],
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

