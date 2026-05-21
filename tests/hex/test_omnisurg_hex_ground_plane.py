from __future__ import annotations

import numpy as np

from omnisurg.hex.app_runtime import _make_ground_plane_mesh


def test_make_ground_plane_mesh_expands_xy_footprint():
    particle_q = np.asarray(
        [
            [1.0, -2.0, 0.4],
            [3.0, 2.0, 0.8],
            [2.0, 0.0, -0.1],
        ],
        dtype=np.float32,
    )

    points, indices = _make_ground_plane_mesh(particle_q, ground_height=-0.25, min_margin=0.5)

    assert points.shape == (4, 3)
    assert indices.shape == (6,)
    assert np.allclose(points[:, 2], -0.25)
    assert np.allclose(points[:, 0], [-1.0, 5.0, 5.0, -1.0])
    assert np.allclose(points[:, 1], [-4.0, -4.0, 4.0, 4.0])
    assert np.array_equal(indices, np.asarray([0, 1, 2, 0, 2, 3], dtype=np.int32))


def test_make_ground_plane_mesh_uses_min_margin_for_tiny_footprint():
    particle_q = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)

    points, _ = _make_ground_plane_mesh(particle_q, ground_height=0.0, min_margin=0.25)

    assert np.allclose(points[:, 0], [-0.25, 0.25, 0.25, -0.25])
    assert np.allclose(points[:, 1], [-0.25, -0.25, 0.25, 0.25])
