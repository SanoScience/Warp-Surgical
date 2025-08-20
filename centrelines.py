import warp as wp

@wp.struct
class CentrelinePointInfo:
    point_start_id: wp.int32
    point_count: wp.int32

    edge_start_id: wp.int32
    edge_count: wp.int32

    stiffness: wp.float32
    rest_length_mul: wp.float32
    radius: wp.float32
    position: wp.vec4f

@wp.struct
class ClampConstraint:
    id: wp.int32
    dist: wp.float32


@wp.kernel
def compute_centreline_positions(
    centreline_points: wp.array(dtype=CentrelinePointInfo),   # [num_points]
    clamp_cnstr: wp.array(dtype=ClampConstraint),             # [total_point_cnstrs]
    particle_q: wp.array(dtype=wp.vec3f),                     # [num_particles]
    out_positions: wp.array(dtype=wp.vec3f)                   # [num_points]
):
    idx = wp.tid()
    info = centreline_points[idx]
    sum_pos = wp.vec3f(0.0, 0.0, 0.0)
    for i in range(info.point_count):
        pid = clamp_cnstr[info.point_start_id + i].id
        sum_pos += particle_q[pid]
    if info.point_count > 0:
        avg_pos = sum_pos / float(info.point_count)
    else:
        avg_pos = sum_pos
    out_positions[idx] = avg_pos

@wp.kernel
def attach_clip_to_nearest_centreline(
    centreline_points: wp.array(dtype=CentrelinePointInfo),   # [num_points]
    centreline_positions: wp.array(dtype=wp.vec3f),           # [num_points]
    instr_pos: wp.array(dtype=wp.vec3f),                      # Instrument position buffer [1]
    clip_attached: wp.array(dtype=wp.int32),                  # [num_points], 0 = not attached, 1 = attached
    clip_indices: wp.array(dtype=wp.int32),                   # [max_clips], stores centreline index for each clip
    clip_count: wp.array(dtype=wp.int32),                     # [1], running count of clips
    max_clips: int,
    clip_radius: float
):
    tid = wp.tid()
    if tid == 0:
        min_dist = float(1e10)
        min_idx = int(-1)
        for i in range(centreline_points.shape[0]):
            pos = centreline_positions[i]
            dist = wp.length(pos - instr_pos[0])
            if dist < min_dist and clip_attached[i] == 0: #and dist < clip_radius:
                min_dist = dist
                min_idx = i

        if min_idx >= 0:
            clip_attached[min_idx] = 1
            idx = wp.atomic_add(clip_count, 0, 1)
            if idx < max_clips:
                clip_indices[idx] = min_idx


@wp.kernel
def cut_centrelines_near_haptic(
    centreline_points: wp.array(dtype=CentrelinePointInfo),   # [num_points]
    centreline_positions: wp.array(dtype=wp.vec3f),           # [num_points]
    instr_pos: wp.array(dtype=wp.vec3f),                      # Instrument position buffer [1]
    cut_flags: wp.array(dtype=wp.int32),                      # [num_points], 0 = not cut, 1 = cut
    cut_radius: float
):
    tid = wp.tid()
    if tid < centreline_points.shape[0]:
        pos = centreline_positions[tid]
        dist = wp.length(pos - instr_pos[0] * 0.01)
        if dist < cut_radius:
            cut_flags[tid] = 1

@wp.kernel
def update_centreline_leaks(
    states: wp.array(dtype=wp.int32),           # [num_points], state per centreline point
    num_points: int,
    out_clipping_ready_to_cut: wp.array(dtype=wp.int32),  # [1]
    out_clipping_done: wp.array(dtype=wp.int32),          # [1]
    out_clipping_error: wp.array(dtype=wp.int32),         # [1]
    out_valid_ids_to_cut: wp.array(dtype=wp.int32),       # [num_points]
    out_valid_ids_count: wp.array(dtype=wp.int32),        # [1]
):
    # Only thread 0 does this logic
    if wp.tid() > 0:
        return

    # Constants for state
    STATE_DEFAULT = 0
    STATE_LOOSELY_CLIPPED = 1
    STATE_PARTIALLY_CLIPPED = 2
    STATE_CLIPPED = 3
    STATE_CUT = 4

    clippingStarted = bool(False)
    for i in range(num_points):
        if states[i] == STATE_CLIPPED:
            clippingStarted = True

    # Find first cut (A)
    cutIdA = int(-1)
    for i in range(num_points):
        if states[i] == STATE_CUT:
            cutIdA = i
            break

    # Find highest clip state before cutIdA
    clipIdA = int(-1)
    highestClipStateA = STATE_DEFAULT
    if cutIdA != -1:
        for i in range(cutIdA, -1, -1):
            s = states[i]
            if s == STATE_LOOSELY_CLIPPED or s == STATE_PARTIALLY_CLIPPED or s == STATE_CLIPPED:
                if s > highestClipStateA:
                    highestClipStateA = s
                    clipIdA = i

    # Find last cut (B)
    cutIdB = int(-1)
    for i in range(num_points-1, -1, -1):
        if states[i] == STATE_CUT:
            cutIdB = i
            break

    # Find highest clip state after cutIdB
    clipIdB = int(-1)
    highestClipStateB = STATE_DEFAULT
    if cutIdB != -1:
        for i in range(cutIdB, num_points):
            s = states[i]
            if s == STATE_LOOSELY_CLIPPED or s == STATE_PARTIALLY_CLIPPED or s == STATE_CLIPPED:
                if s > highestClipStateB:
                    highestClipStateB = s
                    clipIdB = i

    # Clipping error logic
    clippingError = int(0)

    clipA = int(0)
    if clipIdA != -1 and cutIdA != -1:
        if highestClipStateA == STATE_CLIPPED and clipIdA < cutIdA:
            clipA = 1
        if highestClipStateA == STATE_CLIPPED and clipIdA >= cutIdA:
            clippingError = 1
    if clipIdA == -1 and cutIdA != -1:
        clippingError = 1

    clipB = int(0)
    if clipIdB != -1 and cutIdB != -1:
        if highestClipStateB == STATE_CLIPPED and clipIdB > cutIdB:
            clipB = 1
        if highestClipStateB == STATE_CLIPPED and clipIdB <= cutIdB:
            clippingError = 1
    if clipIdB == -1 and cutIdB != -1:
        clippingError = 1

    # Count clipped
    clippedCount = int(0)
    for i in range(num_points):
        if states[i] == STATE_CLIPPED:
            clippedCount += 1

    # Ready to cut if at least 3 clipped
    clippingReadyToCut = int(0)
    validIdsCount = int(0)
    if clippedCount >= 3:
        clippingReadyToCut = int(1)
        # Find lowest and highest clipped ids
        lowestClippedId = int(-1)
        highestClippedId = int(-1)
        for i in range(num_points):
            if states[i] == STATE_CLIPPED:
                lowestClippedId = i
                break
        for i in range(num_points-1, -1, -1):
            if states[i] == STATE_CLIPPED:
                highestClippedId = i
                break
        # Fill valid ids to cut
        if lowestClippedId != -1 and highestClippedId != -1 and lowestClippedId < highestClippedId:
            for i in range(lowestClippedId, highestClippedId):
                out_valid_ids_to_cut[validIdsCount] = i
                validIdsCount += 1

    # Clipping done if both sides clipped
    clippingDone = 1 if (clipA and clipB) else 0

    # Write outputs
    out_clipping_ready_to_cut[0] = clippingReadyToCut
    out_clipping_done[0] = clippingDone
    out_clipping_error[0] = clippingError
    out_valid_ids_count[0] = validIdsCount


@wp.kernel
def emit_bleed_particles(
    centreline_positions: wp.array(dtype=wp.vec3f),   # [num_points]
    cut_flags: wp.array(dtype=wp.int32),              # [num_points]
    bleed_positions: wp.array(dtype=wp.vec3f),        # [max_bleed_particles]
    bleed_velocities: wp.array(dtype=wp.vec3f),       # [max_bleed_particles]
    bleed_lifetimes: wp.array(dtype=wp.float32),      # [max_bleed_particles]
    bleed_active: wp.array(dtype=wp.int32),           # [max_bleed_particles]
    bleed_next_id: wp.array(dtype=wp.int32),          # [1]
    max_bleed_particles: int,
    dt: float
):
    tid = wp.tid()
    if tid >= centreline_positions.shape[0]:
        return

    if cut_flags[tid] == 1:
        # Emit a particle from node if cut
        idx = wp.atomic_add(bleed_next_id, 0, 1) % max_bleed_particles
        bleed_positions[idx] = centreline_positions[tid]

        # Randomized velocity
        bleed_velocities[idx] = wp.vec3f(0.0, 0.2, 0.0) + 0.05 * wp.vec3f(float(tid % 3 - 1), 1.0, float((tid * 7) % 3 - 1))
        bleed_lifetimes[idx] = 1.5  # seconds
        bleed_active[idx] = 1


@wp.kernel
def update_bleed_particles(
    bleed_positions: wp.array(dtype=wp.vec3f),
    bleed_velocities: wp.array(dtype=wp.vec3f),
    bleed_lifetimes: wp.array(dtype=wp.float32),
    bleed_active: wp.array(dtype=wp.int32),
    max_bleed_particles: int,
    dt: float
):
    tid = wp.tid()
    if tid >= max_bleed_particles:
        return
    if bleed_active[tid] == 1:
        # Basic gravity
        bleed_velocities[tid] += wp.vec3f(0.0, -9.8, 0.0) * dt
        bleed_positions[tid] += bleed_velocities[tid] * dt
        bleed_lifetimes[tid] -= dt
        if bleed_lifetimes[tid] <= 0.0:
            bleed_active[tid] = 0