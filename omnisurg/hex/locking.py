# SPDX-License-Identifier: Apache-2.0
"""Locking and active-cell selection helpers for the hex runtime."""

from __future__ import annotations

import numpy as np
import warp as wp


def _active_cells_for_material(
    cell_material: np.ndarray,
    cell_active: np.ndarray,
    material_idx: int,
) -> np.ndarray:
    active = (cell_active != 0) & (cell_material == int(material_idx))
    return np.nonzero(active)[0].astype(np.int32, copy=False)


def _active_cells_outside_cluster_coverage(
    cell_active: np.ndarray,
    cell_to_cluster: np.ndarray,
) -> np.ndarray:
    outside = (cell_active != 0) & (cell_to_cluster < 0)
    return np.nonzero(outside)[0].astype(np.int32, copy=False)


@wp.kernel(enable_backward=False)
def _enforce_locked_nodes_kernel(
    locked_indices: wp.array(dtype=wp.int32),
    locked_positions: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle_idx = locked_indices[tid]
    if particle_idx < 0:
        return

    particle_q[particle_idx] = locked_positions[tid]
    particle_qd[particle_idx] = wp.vec3(0.0, 0.0, 0.0)


def _enforce_locked_nodes(
    state,
    locked_indices: wp.array,
    locked_positions: wp.array,
    locked_count: int,
    device,
) -> None:
    if locked_count <= 0:
        return
    wp.launch(
        kernel=_enforce_locked_nodes_kernel,
        dim=int(locked_count),
        inputs=[locked_indices, locked_positions],
        outputs=[state.particle_q, state.particle_qd],
        device=device,
    )


def _merge_locked_node_positions(
    locked_indices_host: np.ndarray,
    locked_positions_host: np.ndarray,
    locked_slot_by_node: dict[int, int],
    locked_count: int,
    node_indices: np.ndarray,
    node_positions: np.ndarray,
) -> int:
    nodes = np.asarray(node_indices, dtype=np.int32).reshape(-1)
    positions = np.asarray(node_positions, dtype=np.float32)
    if positions.shape != (nodes.size, 3):
        raise ValueError(f"locked node positions shape mismatch: expected {(nodes.size, 3)}, got {positions.shape}")

    for node, position in zip(nodes.tolist(), positions, strict=True):
        if node < 0:
            continue
        slot = locked_slot_by_node.get(node)
        if slot is None:
            if locked_count >= locked_indices_host.shape[0]:
                raise RuntimeError("locked node buffer is full")
            slot = int(locked_count)
            locked_slot_by_node[node] = slot
            locked_indices_host[slot] = node
            locked_count += 1
        locked_positions_host[slot] = position
    return int(locked_count)


__all__ = [
    "_active_cells_for_material",
    "_active_cells_outside_cluster_coverage",
    "_enforce_locked_nodes",
    "_enforce_locked_nodes_kernel",
    "_merge_locked_node_positions",
]
