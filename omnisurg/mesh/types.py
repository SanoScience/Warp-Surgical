import warp as wp


@wp.struct
class Tetrahedron:
    ids: wp.vec4i
    rest_volume: wp.float32


@wp.struct
class TriPointsConnector:
    particle_id: wp.int32
    rest_dist: wp.float32
    tri_ids: wp.vec3i
    tri_bar: wp.vec3f


def compute_tet_volume(p0, p1, p2, p3):
    return abs(wp.dot(wp.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0)
