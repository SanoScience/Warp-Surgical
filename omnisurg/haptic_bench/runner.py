"""Single-run bench engine: algorithm + virtual device + contact trace → metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from omnisurg.haptic_bench.algorithms import (
    HapticStepInput,
    create as create_algorithm,
)
from omnisurg.haptic_bench.metrics import BenchTrace, compute_metrics
from omnisurg.haptic_bench.virtual_device import (
    VirtualHandpiece,
    VirtualHandpieceParams,
)
from omnisurg.haptic_feedback import HapticFeedbackSettings


CONTACT_TRACE_SCHEMA = 1


def load_contact_trace(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    version = int(data["schema_version"]) if "schema_version" in data.files else 0
    if version != CONTACT_TRACE_SCHEMA:
        raise ValueError(
            f"Contact trace schema mismatch: got {version}, expected {CONTACT_TRACE_SCHEMA}"
        )
    return {key: data[key] for key in data.files}


def run_single(
    algorithm_name: str,
    settings: HapticFeedbackSettings,
    contact_trace: dict[str, np.ndarray],
    device_params: VirtualHandpieceParams | None = None,
) -> tuple[BenchTrace, dict[str, float]]:
    algorithm = create_algorithm(algorithm_name, settings)
    device = VirtualHandpiece(device_params)

    t = contact_trace["t"]
    dt = contact_trace["dt"]
    ref_pos = contact_trace["device_position"]
    ref_vel = contact_trace["device_velocity"]
    reaction = contact_trace["avg_reaction_offset"]
    contact_count = contact_trace["contact_count"]
    n = t.shape[0]

    device_pos_out = np.zeros((n, 3), dtype=np.float32)
    device_vel_out = np.zeros((n, 3), dtype=np.float32)
    cmd_force_out = np.zeros((n, 3), dtype=np.float32)
    applied_force_out = np.zeros((n, 3), dtype=np.float32)

    algorithm.reset(ref_pos[0].astype(np.float32))
    device.reset()

    prev_cmd_force = np.zeros(3, dtype=np.float32)

    for i in range(n):
        device_pos, device_vel, applied_force = device.step(
            prev_cmd_force, ref_pos[i], ref_vel[i], float(dt[i])
        )
        device_pos_out[i] = device_pos
        device_vel_out[i] = device_vel
        applied_force_out[i] = applied_force

        if int(contact_count[i]) <= 0:
            algorithm.reset(device_pos)
            prev_cmd_force = np.zeros(3, dtype=np.float32)
            cmd_force_out[i] = prev_cmd_force
            continue

        step_input = HapticStepInput(
            device_position=device_pos,
            avg_reaction_offset=reaction[i],
            contact_count=int(contact_count[i]),
            dt=float(dt[i]),
        )
        result = algorithm.compute(step_input)
        cmd_force_out[i] = result.force
        prev_cmd_force = result.force.copy()

    trace = BenchTrace(
        t=t.astype(np.float32),
        dt=dt.astype(np.float32),
        device_position=device_pos_out,
        reference_position=ref_pos.astype(np.float32),
        device_velocity=device_vel_out,
        commanded_force=cmd_force_out,
        applied_force=applied_force_out,
        contact_count=contact_count.astype(np.int32),
        reaction_offset=reaction.astype(np.float32),
    )
    metrics = compute_metrics(trace)
    return trace, metrics
