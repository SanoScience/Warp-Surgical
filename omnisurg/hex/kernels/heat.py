# SPDX-License-Identifier: Apache-2.0
"""Electrocautery heat + damage kernels.

Ports the three loops in ``SpringGrid::updateHeat`` (springGrid.h:514-610):

* **heat injection** (lines 544-569): for each particle near an electrode
  capsule, smooth ``heat`` toward ``equipment.power`` with factor 0.1.
* **damage** (lines 582-593): particles whose ``heat`` exceeds their material
  resistance lose their ACTIVE flag and their outgoing springs are zeroed.
* **diffusion** (lines 596-603): paper Eq. 1 - each active particle blends
  neighbour heat weighted by ``(c_k + c_i)/2``.

Heat is double-buffered (``heat_in`` / ``heat_out``) to avoid an in-place read
/ write race in the diffusion sweep; the caller swaps buffers each frame.
"""

from __future__ import annotations

import warp as wp

from newton._src.geometry.flags import ParticleFlags

_ACTIVE_BIT = wp.constant(wp.int32(int(ParticleFlags.ACTIVE)))
_HEAT_RETAIN = wp.constant(0.98)
_HEAT_DIFFUSE = wp.constant(0.16)
_HEAT_INJECT_RETAIN = wp.constant(0.9)
_HEAT_INJECT_NEW = wp.constant(0.1)
_BURN_HEAT_THRESHOLD = wp.constant(15.0)
_BURN_SLOPE = wp.constant(0.04)
_BURN_BIAS = wp.constant(0.05)


@wp.func
def _point_segment_distance_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:
    """Squared distance from point ``p`` to the segment ``[a, b]`` (inclusive)."""
    ab = b - a
    ap = p - a
    denom = wp.dot(ab, ab)
    if denom < 1.0e-12:
        return wp.dot(ap, ap)
    t = wp.dot(ap, ab) / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    closest = a + ab * t
    d = p - closest
    return wp.dot(d, d)


@wp.kernel
def heat_apply_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tool_p0: wp.array(dtype=wp.vec3),
    tool_p1: wp.array(dtype=wp.vec3),
    tool_radius: wp.array(dtype=wp.float32),
    tool_role_electrode: wp.array(dtype=wp.int32),
    num_segments: int,
    tool_power: float,
    tool_active: int,
    particle_heat: wp.array(dtype=wp.float32),
):
    """Smooth ``particle_heat`` toward ``tool_power`` for particles inside any
    active electrode capsule.

    ``tool_active`` is a 0/1 flag corresponding to the pedal state in the
    reference implementation. When 0, this kernel is a no-op.
    """
    i = wp.tid()
    if tool_active == 0:
        return
    if (particle_flags[i] & _ACTIVE_BIT) == 0:
        return
    p = particle_q[i]
    for s in range(num_segments):
        if tool_role_electrode[s] == 0:
            continue
        r = tool_radius[s]
        d2 = _point_segment_distance_sq(p, tool_p0[s], tool_p1[s])
        if d2 <= r * r:
            particle_heat[i] = particle_heat[i] * _HEAT_INJECT_RETAIN + tool_power * _HEAT_INJECT_NEW
            return


@wp.kernel
def heat_diffuse_kernel(
    particle_flags: wp.array(dtype=wp.int32),
    particle_neighbors: wp.array2d(dtype=wp.int32),
    particle_material: wp.array(dtype=wp.int32),
    material_conductivity: wp.array(dtype=wp.float32),
    heat_in: wp.array(dtype=wp.float32),
    dt: float,
    heat_out: wp.array(dtype=wp.float32),
):
    """One Jacobi diffusion sweep over the 6-neighbour graph (paper Eq. 1).

    ``heat_out`` is written from ``heat_in``; swap buffers between frames to
    avoid a read/write race on the same array.
    """
    i = wp.tid()
    if (particle_flags[i] & _ACTIVE_BIT) == 0:
        heat_out[i] = heat_in[i]
        return
    h = heat_in[i]
    c_i = material_conductivity[particle_material[i]]
    accum = float(0.0)
    for d in range(6):
        nb = particle_neighbors[i, d]
        if nb < 0:
            continue
        if (particle_flags[nb] & _ACTIVE_BIT) == 0:
            continue
        c_n = material_conductivity[particle_material[nb]]
        accum += (heat_in[nb] - h) * 0.5 * (c_n + c_i)
    heat_out[i] = h * _HEAT_RETAIN + accum * _HEAT_DIFFUSE


@wp.kernel
def apply_damage_kernel(
    particle_material: wp.array(dtype=wp.int32),
    material_resistance: wp.array(dtype=wp.float32),
    particle_heat: wp.array(dtype=wp.float32),
    particle_burnt: wp.array(dtype=wp.float32),
    fulguration: float,
    particle_flags: wp.array(dtype=wp.int32),
):
    """Clear ACTIVE on particles whose heat exceeded their resistance threshold.

    Also updates the ``particle_burnt`` visualisation factor per the reference
    (springGrid.h:582-585). Runs after diffusion so ``particle_heat`` is the
    post-sweep value.
    """
    i = wp.tid()
    if (particle_flags[i] & _ACTIVE_BIT) == 0:
        return
    h = particle_heat[i]
    # Burn accumulation (cosmetic only; feeds vertex colour later).
    if h > _BURN_HEAT_THRESHOLD:
        b = particle_burnt[i] + (_BURN_SLOPE * (h - _BURN_HEAT_THRESHOLD) + _BURN_BIAS) * fulguration
        if b > 1.0:
            b = 1.0
        particle_burnt[i] = b
    r = material_resistance[particle_material[i]]
    if r > 0.0 and h > r:
        particle_flags[i] = particle_flags[i] & (~_ACTIVE_BIT)


@wp.kernel
def disable_cut_springs_kernel(
    spring_indices: wp.array(dtype=wp.int32),
    particle_flags: wp.array(dtype=wp.int32),
    particle_material: wp.array(dtype=wp.int32),
    material_stiffness_scale: wp.array(dtype=wp.float32),
    spring_stiffness_base: wp.array(dtype=wp.float32),
    spring_enabled: wp.array(dtype=wp.int32),
    spring_stiffness: wp.array(dtype=wp.float32),
):
    """Zero the stiffness of cut springs; scale the rest by material settings.

    Each frame:

    * If either endpoint's ACTIVE bit was cleared, ``spring_enabled`` is
      flipped to 0 permanently and ``spring_stiffness`` is zeroed so the
      XPBD ``solve_springs`` kernel skips the constraint.
    * Otherwise ``spring_stiffness = spring_stiffness_base * 0.5 * (scale_i
      + scale_j)``, the average of the two endpoint materials' live stiffness
      multipliers driven from the UI.

    Once disabled a spring stays disabled - matches the reference's
    ``SpringGrid::remove`` behaviour.
    """
    tid = wp.tid()
    if spring_enabled[tid] == 0:
        return
    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]
    if (particle_flags[i] & _ACTIVE_BIT) == 0 or (particle_flags[j] & _ACTIVE_BIT) == 0:
        spring_enabled[tid] = 0
        spring_stiffness[tid] = 0.0
        return
    scale = 0.5 * (
        material_stiffness_scale[particle_material[i]]
        + material_stiffness_scale[particle_material[j]]
    )
    spring_stiffness[tid] = spring_stiffness_base[tid] * scale
