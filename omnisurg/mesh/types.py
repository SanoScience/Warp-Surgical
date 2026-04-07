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


def parse_connector_file(filepath: str, particle_id_offset: int = 0, tri_id_offset: int = 0) -> list[TriPointsConnector]:
    connectors = []
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) != 8:
                raise ValueError(f"Line does not have 8 elements: {line}")

            connector = TriPointsConnector()
            connector.particle_id = int(parts[0]) + particle_id_offset
            connector.rest_dist = float(parts[1])
            connector.tri_ids = wp.vec3i(
                int(parts[2]) + tri_id_offset,
                int(parts[3]) + tri_id_offset,
                int(parts[4]) + tri_id_offset,
            )
            connector.tri_bar = wp.vec3f(float(parts[5]), float(parts[6]), float(parts[7]))
            connectors.append(connector)
    return connectors
