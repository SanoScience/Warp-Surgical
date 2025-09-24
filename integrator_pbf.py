import math
import weakref
from dataclasses import dataclass

import warp as wp

from warp.sim.integrator import Integrator
from warp.sim.model import PARTICLE_FLAG_ACTIVE, Model, State


POLY6_BASE = 315.0 / (64.0 * math.pi)
SPIKY_BASE = -45.0 / math.pi
MIN_REST_DENSITY = 1.0e-6
SCORR_DELTA_SCALE = 0.1
SCORR_POWER = 4.0
SURFACE_TENSION_NORMAL_EPS = 1.0e-5
DEFAULT_TENSILE_INSTABILITY = 0.0


@dataclass
class _TempBuffers:
    count: int
    device: str
    x_pred: wp.array
    v_tmp: wp.array
    density: wp.array
    lambdas: wp.array
    delta: wp.array
    curl: wp.array
    curl_mag: wp.array


@wp.func
def _poly6_kernel(r: wp.vec3, h: float, coeff: float):
    h2 = h * h
    r2 = wp.dot(r, r)
    if r2 >= h2:
        return 0.0
    diff = h2 - r2
    return coeff * diff * diff * diff


@wp.func
def _poly6_laplacian(r: wp.vec3, h: float, coeff: float):
    h2 = h * h
    r2 = wp.dot(r, r)
    if r2 >= h2:
        return 0.0
    diff = h2 - r2
    return -6.0 * coeff * diff * (3.0 * h2 - 7.0 * r2)


@wp.func
def _spiky_gradient(r: wp.vec3, h: float, coeff: float):
    rl = wp.length(r)
    if rl == 0.0 or rl > h:
        return wp.vec3(0.0)
    s = coeff * (h - rl) * (h - rl) / rl
    return r * s


@wp.kernel
def _pbf_predict_positions(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    flags: wp.array(dtype=wp.uint32),
    gravity: wp.vec3,
    dt: float,
    v_max: float,
    x_pred: wp.array(dtype=wp.vec3),
    v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    x0 = x[tid]
    vel = v[tid]

    if (flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        x_pred[tid] = x0
        v_out[tid] = wp.vec3(0.0)
        return

    w = inv_mass[tid]
    if w > 0.0:
        accel = f[tid] * w + gravity
        vel = vel + accel * dt
    else:
        vel = wp.vec3(0.0)

    speed = wp.length(vel)
    if speed > v_max:
        vel *= v_max / speed

    x_pred[tid] = x0 + vel * dt
    v_out[tid] = vel


@wp.kernel
def _pbf_compute_density(
    grid_id: wp.uint64,
    x_pred: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
    flags: wp.array(dtype=wp.uint32),
    h: float,
    coeff_poly6: float,
    rest_density: float,
    density_out: wp.array(dtype=float),
):
    tid = wp.tid()

    if (flags[tid] & PARTICLE_FLAG_ACTIVE) == 0 or masses[tid] == 0.0:
        density_out[tid] = rest_density
        return

    xi = x_pred[tid]
    rho = masses[tid] * _poly6_kernel(wp.vec3(0.0), h, coeff_poly6)

    neighbors = wp.hash_grid_query(grid_id, xi, h)
    for j in neighbors:
        if j == tid:
            continue
        if (flags[j] & PARTICLE_FLAG_ACTIVE) == 0 or masses[j] == 0.0:
            continue
        rho += masses[j] * _poly6_kernel(xi - x_pred[j], h, coeff_poly6)

    density_out[tid] = wp.max(rho, MIN_REST_DENSITY)


@wp.kernel
def _pbf_compute_lambdas(
    grid_id: wp.uint64,
    x_pred: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
    flags: wp.array(dtype=wp.uint32),
    density: wp.array(dtype=float),
    h: float,
    coeff_spiky: float,
    rest_density: float,
    relaxation: float,
    lambdas_out: wp.array(dtype=float),
):
    tid = wp.tid()

    if (flags[tid] & PARTICLE_FLAG_ACTIVE) == 0 or masses[tid] == 0.0:
        lambdas_out[tid] = 0.0
        return

    xi = x_pred[tid]
    inv_rest_density = 1.0 / rest_density
    ci = density[tid] * inv_rest_density - 1.0
    ci = wp.max(ci, 0.0)

    grad_i = wp.vec3(0.0)
    sum_grad_sq = float(0.0)

    neighbors = wp.hash_grid_query(grid_id, xi, h)
    for j in neighbors:
        if j == tid:
            continue
        if (flags[j] & PARTICLE_FLAG_ACTIVE) == 0 or masses[j] == 0.0:
            continue
        grad = _spiky_gradient(xi - x_pred[j], h, coeff_spiky)
        grad *= masses[j] * inv_rest_density
        sum_grad_sq += wp.dot(grad, grad)
        grad_i += grad

    sum_grad_sq += wp.dot(grad_i, grad_i)

    denom = sum_grad_sq + relaxation
    if denom > 0.0:
        lambdas_out[tid] = -ci / denom
    else:
        lambdas_out[tid] = 0.0


@wp.kernel
def _pbf_compute_position_deltas(
    grid_id: wp.uint64,
    x_pred: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
    flags: wp.array(dtype=wp.uint32),
    lambdas: wp.array(dtype=float),
    density: wp.array(dtype=float),
    h: float,
    coeff_spiky: float,
    coeff_poly6: float,
    surface_tension: float,
    tensile_instability: float,
    scorr_w0: float,
    scorr_power: float,
    rest_density: float,
    delta_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    if (flags[tid] & PARTICLE_FLAG_ACTIVE) == 0 or masses[tid] == 0.0:
        delta_out[tid] = wp.vec3(0.0)
        return

    xi = x_pred[tid]
    lambda_i = lambdas[tid]
    delta = wp.vec3(0.0)
    color_grad = wp.vec3(0.0)
    laplacian_sum = float(0.0)
    mass_i = masses[tid]

    neighbors = wp.hash_grid_query(grid_id, xi, h)
    for j in neighbors:
        if j == tid:
            continue
        if (flags[j] & PARTICLE_FLAG_ACTIVE) == 0 or masses[j] == 0.0:
            continue
        rij = xi - x_pred[j]
        grad = _spiky_gradient(rij, h, coeff_spiky)
        scorr = 0.0
        if tensile_instability > 0.0 and scorr_w0 > 0.0:
            w = _poly6_kernel(rij, h, coeff_poly6)
            if w > 0.0:
                ratio = w / scorr_w0
                scorr = -tensile_instability * wp.pow(ratio, scorr_power)
        delta += (lambda_i + lambdas[j] + scorr) * grad

        if surface_tension > 0.0:
            density_j = wp.max(density[j], MIN_REST_DENSITY)
            mass_term = masses[j] / density_j
            color_grad += grad * mass_term
            laplacian_sum += mass_term * _poly6_laplacian(rij, h, coeff_poly6)

    if surface_tension > 0.0 and mass_i > 0.0:
        grad_len = wp.length(color_grad)
        if grad_len > SURFACE_TENSION_NORMAL_EPS:
            curvature = -laplacian_sum
            curvature = wp.max(curvature, 0.0)
            if curvature > 0.0:
                normal = color_grad / grad_len
                delta -= surface_tension * curvature * normal

    inv_rest_density = 1.0 / rest_density
    delta *= mass_i * inv_rest_density

    delta_out[tid] = delta



@wp.kernel
def _pbf_accumulate_delta(
    x_pred_in: wp.array(dtype=wp.vec3),
    delta: wp.array(dtype=wp.vec3),
    x_pred_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    x_pred_out[tid] = x_pred_in[tid] + delta[tid]


@wp.kernel
def _pbf_update_velocities(
    x_orig: wp.array(dtype=wp.vec3),
    x_pred: wp.array(dtype=wp.vec3),
    flags: wp.array(dtype=wp.uint32),
    inv_mass: wp.array(dtype=float),
    dt: float,
    v_max: float,
    v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    if (flags[tid] & PARTICLE_FLAG_ACTIVE) == 0 or inv_mass[tid] == 0.0:
        v_out[tid] = wp.vec3(0.0)
        return

    vel = (x_pred[tid] - x_orig[tid]) / dt
    speed = wp.length(vel)
    if speed > v_max:
        vel *= v_max / speed

    v_out[tid] = vel


@wp.kernel
def _pbf_apply_viscosity(
    grid_id: wp.uint64,
    x_pred: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    flags: wp.array(dtype=wp.uint32),
    h: float,
    coeff_poly6: float,
    rest_density: float,
    viscosity: float,
):
    tid = wp.tid()

    if (flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return

    xi = x_pred[tid]
    vi = v[tid]
    delta_v = wp.vec3(0.0)

    neighbors = wp.hash_grid_query(grid_id, xi, h)
    for j in neighbors:
        if j == tid:
            continue
        if (flags[j] & PARTICLE_FLAG_ACTIVE) == 0:
            continue
        delta_v -= (vi - v[j]) * _poly6_kernel(xi - x_pred[j], h, coeff_poly6)

    delta_v *= viscosity / rest_density
    v[tid] = vi + delta_v


@wp.kernel
def _pbf_compute_curl(
    grid_id: wp.uint64,
    x_pred: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
    flags: wp.array(dtype=wp.uint32),
    h: float,
    coeff_spiky: float,
    rest_density: float,
    curl_out: wp.array(dtype=wp.vec3),
    curl_mag_out: wp.array(dtype=float),
):
    tid = wp.tid()

    if (flags[tid] & PARTICLE_FLAG_ACTIVE) == 0 or masses[tid] == 0.0:
        curl_out[tid] = wp.vec3(0.0)
        curl_mag_out[tid] = 0.0
        return

    xi = x_pred[tid]
    vi = v[tid]
    omega = wp.vec3(0.0)
    inv_rest_density = 1.0 / rest_density

    neighbors = wp.hash_grid_query(grid_id, xi, h)
    for j in neighbors:
        if j == tid:
            continue
        if (flags[j] & PARTICLE_FLAG_ACTIVE) == 0 or masses[j] == 0.0:
            continue
        grad = _spiky_gradient(xi - x_pred[j], h, coeff_spiky)
        mass_term = masses[j] * inv_rest_density
        omega += wp.cross(v[j] - vi, grad) * mass_term

    curl_out[tid] = omega
    curl_mag_out[tid] = wp.length(omega)


@wp.kernel
def _pbf_apply_vorticity_correction(
    grid_id: wp.uint64,
    x_pred: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
    inv_mass: wp.array(dtype=float),
    flags: wp.array(dtype=wp.uint32),
    curl: wp.array(dtype=wp.vec3),
    curl_mag: wp.array(dtype=float),
    h: float,
    coeff_spiky: float,
    rest_density: float,
    strength: float,
    dt: float,
):
    tid = wp.tid()

    if (flags[tid] & PARTICLE_FLAG_ACTIVE) == 0 or masses[tid] == 0.0 or inv_mass[tid] == 0.0:
        return

    omega_i = curl[tid]
    curl_mag_i = curl_mag[tid]
    eta = wp.vec3(0.0)
    xi = x_pred[tid]
    inv_rest_density = 1.0 / rest_density

    neighbors = wp.hash_grid_query(grid_id, xi, h)
    for j in neighbors:
        if j == tid:
            continue
        if (flags[j] & PARTICLE_FLAG_ACTIVE) == 0 or masses[j] == 0.0:
            continue
        grad = _spiky_gradient(xi - x_pred[j], h, coeff_spiky)
        mass_term = masses[j] * inv_rest_density
        eta += (curl_mag[j] - curl_mag_i) * grad * mass_term

    eta_len = wp.length(eta)
    if eta_len == 0.0:
        return

    eta /= eta_len
    force = wp.cross(eta, omega_i) * strength
    v[tid] = v[tid] + force * dt

@wp.kernel
def _pbf_handle_boundaries(
    x_pred_in: wp.array(dtype=wp.vec3),
    v_in: wp.array(dtype=wp.vec3),
    flags: wp.array(dtype=wp.uint32),
    bounds_min: wp.vec3,
    bounds_max: wp.vec3,
    padding: float,
    restitution: float,
    friction: float,
    x_pred_out: wp.array(dtype=wp.vec3),
    v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    pos = x_pred_in[tid]
    vel = v_in[tid]

    if (flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        x_pred_out[tid] = pos
        v_out[tid] = vel
        return

    min_x = bounds_min[0] + padding
    max_x = bounds_max[0] - padding
    min_y = bounds_min[1] + padding
    max_y = bounds_max[1] - padding
    min_z = bounds_min[2] + padding
    max_z = bounds_max[2] - padding

    if pos[0] < min_x:
        pos[0] = min_x
        if vel[0] < 0.0:
            vel[0] *= -restitution
            vel[1] *= friction
            vel[2] *= friction
    elif pos[0] > max_x:
        pos[0] = max_x
        if vel[0] > 0.0:
            vel[0] *= -restitution
            vel[1] *= friction
            vel[2] *= friction

    if pos[1] < min_y:
        pos[1] = min_y
        if vel[1] < 0.0:
            vel[1] *= -restitution
            vel[0] *= friction
            vel[2] *= friction
    elif pos[1] > max_y:
        pos[1] = max_y
        if vel[1] > 0.0:
            vel[1] *= -restitution
            vel[0] *= friction
            vel[2] *= friction

    if pos[2] < min_z:
        pos[2] = min_z
        if vel[2] < 0.0:
            vel[2] *= -restitution
            vel[0] *= friction
            vel[1] *= friction
    elif pos[2] > max_z:
        pos[2] = max_z
        if vel[2] > 0.0:
            vel[2] *= -restitution
            vel[0] *= friction
            vel[1] *= friction

    x_pred_out[tid] = pos
    v_out[tid] = vel


class PBFIntegrator(Integrator):
    def __init__(
        self,
        smoothing_radius: float,
        rest_density: float,
        relaxation: float = 1.0e-6,
        iterations: int = 4,
        viscosity: float = 0.0,
        vorticity: float = 0.0,
        surface_tension: float = 0.0,
        tensile_instability: float = DEFAULT_TENSILE_INSTABILITY,
        boundary_min: wp.vec3 | None = None,
        boundary_max: wp.vec3 | None = None,
        boundary_padding: float = 0.1,
        restitution: float = 1.0,
        friction: float = 1.0,
    ) -> None:
        super().__init__()
        self.smoothing_radius = smoothing_radius
        self.rest_density = max(rest_density, MIN_REST_DENSITY)
        self.relaxation = relaxation
        self.num_iterations = max(1, iterations)
        self.viscosity = max(viscosity, 0.0)
        self.vorticity = max(vorticity, 0.0)
        self.surface_tension = max(surface_tension, 0.0)
        self.tensile_instability = max(tensile_instability, 0.0)
        self.boundary_min = boundary_min
        self.boundary_max = boundary_max
        self.boundary_padding = boundary_padding
        self.restitution = restitution
        self.friction = friction
        self._poly6_coeff = 0.0
        self._spiky_coeff = 0.0
        self._scorr_w0 = 0.0
        self._scorr_delta_q = 0.0
        self._temps = weakref.WeakKeyDictionary()
        self._update_kernel_norms()

    def _update_kernel_norms(self) -> None:
        h = max(self.smoothing_radius, 1.0e-4)
        self._poly6_coeff = POLY6_BASE / (h ** 9)
        self._spiky_coeff = SPIKY_BASE / (h ** 6)
        self._scorr_delta_q = SCORR_DELTA_SCALE * h
        if self._scorr_delta_q >= h:
            self._scorr_delta_q = h * 0.99
        diff = h * h - self._scorr_delta_q * self._scorr_delta_q
        if diff > 0.0:
            self._scorr_w0 = self._poly6_coeff * (diff ** 3)
        else:
            self._scorr_w0 = 0.0

    def set_iterations(self, iterations: int) -> None:
        self.num_iterations = max(1, int(iterations))

    def set_viscosity(self, viscosity: float) -> None:
        self.viscosity = max(0.0, float(viscosity))

    def set_vorticity(self, vorticity: float) -> None:
        self.vorticity = max(0.0, float(vorticity))

    def set_surface_tension(self, surface_tension: float) -> None:
        self.surface_tension = max(0.0, float(surface_tension))

    def set_smoothing_radius(self, smoothing_radius: float) -> None:
        self.smoothing_radius = max(smoothing_radius, 1.0e-4)
        self._update_kernel_norms()

    def set_rest_density(self, rest_density: float) -> None:
        self.rest_density = max(float(rest_density), MIN_REST_DENSITY)

    def set_relaxation(self, relaxation: float) -> None:
        self.relaxation = max(float(relaxation), 0.0)

    def set_boundary_padding(self, boundary_padding: float) -> None:
        self.boundary_padding = max(float(boundary_padding), 0.0)

    def set_restitution(self, restitution: float) -> None:
        self.restitution = max(float(restitution), 0.0)

    def set_friction(self, friction: float) -> None:
        self.friction = max(float(friction), 0.0)

    def _get_temps(self, model: Model) -> _TempBuffers:
        temp = self._temps.get(model)
        count = model.particle_count
        if temp is None or temp.count != count or temp.device != model.device:
            temp = _TempBuffers(
                count=count,
                device=model.device,
                x_pred=wp.empty(count, dtype=wp.vec3, device=model.device),
                v_tmp=wp.empty(count, dtype=wp.vec3, device=model.device),
                density=wp.empty(count, dtype=float, device=model.device),
                lambdas=wp.empty(count, dtype=float, device=model.device),
                delta=wp.empty(count, dtype=wp.vec3, device=model.device),
                curl=wp.empty(count, dtype=wp.vec3, device=model.device),
                curl_mag=wp.empty(count, dtype=float, device=model.device),
            )
            self._temps[model] = temp
        return temp

    def simulate(self, model: Model, state_in: State, state_out: State, dt: float, control=None):
        if model.body_count:
            self.integrate_bodies(model, state_in, state_out, dt)

        if model.particle_count == 0:
            if state_out.particle_q and state_in.particle_q:
                state_out.particle_q.assign(state_in.particle_q)
            if state_out.particle_qd and state_in.particle_qd:
                state_out.particle_qd.assign(state_in.particle_qd)
            return

        temp = self._get_temps(model)

        wp.launch(
            kernel=_pbf_predict_positions,
            dim=model.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_f,
                model.particle_inv_mass,
                model.particle_flags,
                model.gravity,
                dt,
                model.particle_max_velocity,
            ],
            outputs=[temp.x_pred, temp.v_tmp],
            device=model.device,
        )

        model.particle_grid.build(temp.x_pred, self.smoothing_radius)

        for _ in range(self.num_iterations):
            wp.launch(
                kernel=_pbf_compute_density,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    temp.x_pred,
                    model.particle_mass,
                    model.particle_flags,
                    self.smoothing_radius,
                    self._poly6_coeff,
                    self.rest_density,
                ],
                outputs=[temp.density],
                device=model.device,
            )

            wp.launch(
                kernel=_pbf_compute_lambdas,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    temp.x_pred,
                    model.particle_mass,
                    model.particle_flags,
                    temp.density,
                    self.smoothing_radius,
                    self._spiky_coeff,
                    self.rest_density,
                    self.relaxation,
                ],
                outputs=[temp.lambdas],
                device=model.device,
            )

            wp.launch(
                kernel=_pbf_compute_position_deltas,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    temp.x_pred,
                    model.particle_mass,
                    model.particle_flags,
                    temp.lambdas,
                    temp.density,
                    self.smoothing_radius,
                    self._spiky_coeff,
                    self._poly6_coeff,
                    self.surface_tension,
                    self.tensile_instability,
                    self._scorr_w0,
                    SCORR_POWER,
                    self.rest_density,
                ],
                outputs=[temp.delta],
                device=model.device,
            )

            wp.launch(
                kernel=_pbf_accumulate_delta,
                dim=model.particle_count,
                inputs=[temp.x_pred, temp.delta],
                outputs=[temp.x_pred],
                device=model.device,
            )

            if self.boundary_min is not None and self.boundary_max is not None:
                wp.launch(
                    kernel=_pbf_handle_boundaries,
                    dim=model.particle_count,
                    inputs=[
                        temp.x_pred,
                        temp.v_tmp,
                        model.particle_flags,
                        self.boundary_min,
                        self.boundary_max,
                        self.boundary_padding,
                        self.restitution,
                        self.friction,
                    ],
                    outputs=[temp.x_pred, temp.v_tmp],
                    device=model.device,
                )

        wp.launch(
            kernel=_pbf_update_velocities,
            dim=model.particle_count,
            inputs=[
                state_in.particle_q,
                temp.x_pred,
                model.particle_flags,
                model.particle_inv_mass,
                dt,
                model.particle_max_velocity,
            ],
            outputs=[temp.v_tmp],
            device=model.device,
        )

        if self.vorticity > 0.0:
            model.particle_grid.build(temp.x_pred, self.smoothing_radius)
            wp.launch(
                kernel=_pbf_compute_curl,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    temp.x_pred,
                    temp.v_tmp,
                    model.particle_mass,
                    model.particle_flags,
                    self.smoothing_radius,
                    self._spiky_coeff,
                    self.rest_density,
                ],
                outputs=[temp.curl, temp.curl_mag],
                device=model.device,
            )

            wp.launch(
                kernel=_pbf_apply_vorticity_correction,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    temp.x_pred,
                    temp.v_tmp,
                    model.particle_mass,
                    model.particle_inv_mass,
                    model.particle_flags,
                    temp.curl,
                    temp.curl_mag,
                    self.smoothing_radius,
                    self._spiky_coeff,
                    self.rest_density,
                    self.vorticity,
                    dt,
                ],
                device=model.device,
            )
        if self.viscosity > 0.0:
            model.particle_grid.build(temp.x_pred, self.smoothing_radius)
            wp.launch(
                kernel=_pbf_apply_viscosity,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    temp.x_pred,
                    temp.v_tmp,
                    model.particle_flags,
                    self.smoothing_radius,
                    self._poly6_coeff,
                    self.rest_density,
                    self.viscosity,
                ],
                device=model.device,
            )

        if self.boundary_min is not None and self.boundary_max is not None:
            wp.launch(
                kernel=_pbf_handle_boundaries,
                dim=model.particle_count,
                inputs=[
                    temp.x_pred,
                    temp.v_tmp,
                    model.particle_flags,
                    self.boundary_min,
                    self.boundary_max,
                    self.boundary_padding,
                    self.restitution,
                    self.friction,
                ],
                outputs=[temp.x_pred, temp.v_tmp],
                device=model.device,
            )

        state_out.particle_q.assign(temp.x_pred)
        state_out.particle_qd.assign(temp.v_tmp)

        if state_out.particle_f and state_in.particle_f:
            state_out.particle_f.assign(state_in.particle_f)
