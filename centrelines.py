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
    max_clips: int
):
    tid = wp.tid()
    if tid == 0:
        min_dist = float(1e10)
        min_idx = int(-1)
        for i in range(centreline_points.shape[0]):
            pos = centreline_positions[i]
            dist = wp.length(pos - instr_pos[0])
            if dist < min_dist and clip_attached[i] == 0:
                min_dist = dist
                min_idx = i

        if min_idx >= 0:
            clip_attached[min_idx] = 1
            idx = wp.atomic_add(clip_count, 0, 1)
            if idx < max_clips:
                clip_indices[idx] = min_idx