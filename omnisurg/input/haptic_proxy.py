from dataclasses import dataclass

import warp as wp


@dataclass
class HapticProxyState:
    """GPU-resident state that drives the kinematic haptic sphere."""

    center_prev: wp.array
    center_target: wp.array
    center_current: wp.array
    center_scaled_prev: wp.array
    center_scaled: wp.array
    body_id: int
    radius: float
    max_tri_extent: float = 0.0


def _as_vec3f(value) -> wp.vec3f:
    return wp.vec3f(float(value[0]), float(value[1]), float(value[2]))


def create_haptic_proxy_state(
    *,
    body_id: int,
    radius: float,
    device,
    initial_position=None,
    initial_scaled_position=None,
    max_tri_extent: float = 0.0,
) -> HapticProxyState:
    raw = _as_vec3f(initial_position) if initial_position is not None else wp.vec3f(0.0, 0.0, 0.0)
    scaled = _as_vec3f(initial_scaled_position) if initial_scaled_position is not None else wp.vec3f(0.0, 0.0, 0.0)
    return HapticProxyState(
        center_prev=wp.array([raw], dtype=wp.vec3f, device=device),
        center_target=wp.array([raw], dtype=wp.vec3f, device=device),
        center_current=wp.array([raw], dtype=wp.vec3f, device=device),
        center_scaled_prev=wp.array([scaled], dtype=wp.vec3f, device=device),
        center_scaled=wp.array([scaled], dtype=wp.vec3f, device=device),
        body_id=body_id,
        radius=radius,
        max_tri_extent=max_tri_extent,
    )


def create_vec3_staging_buffer():
    staging = wp.zeros(1, dtype=wp.vec3f, device="cpu")
    return staging, staging.numpy()


@wp.kernel
def update_haptic_proxy(
    center_prev: wp.array(dtype=wp.vec3f),
    center_target: wp.array(dtype=wp.vec3f),
    center_current: wp.array(dtype=wp.vec3f),
    center_scaled_prev: wp.array(dtype=wp.vec3f),
    center_scaled: wp.array(dtype=wp.vec3f),
    body_q: wp.array(dtype=wp.transformf),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    body_id: int,
    factor: float,
    position_scale: float,
    dt: float,
):
    if wp.tid() != 0:
        return

    current = wp.lerp(center_prev[0], center_target[0], factor)
    center_current[0] = current

    center_scaled_prev[0] = center_scaled[0]
    scaled = current * position_scale
    center_scaled[0] = scaled

    xform = body_q[body_id]
    prev = wp.transform_get_translation(xform)
    body_q[body_id] = wp.transform(scaled, wp.transform_get_rotation(xform))

    vel = (scaled - prev) / dt
    body_qd[body_id] = wp.spatial_vector(vel, wp.vec3f(0.0, 0.0, 0.0))


@wp.kernel
def scale_position(
    src: wp.array(dtype=wp.vec3f),
    dst: wp.array(dtype=wp.vec3f),
    scale: float,
):
    if wp.tid() != 0:
        return

    dst[0] = src[0] * scale
