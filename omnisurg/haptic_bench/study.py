"""Optuna-backed multi-objective sweep over algorithm parameters."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from omnisurg.haptic_bench.runner import load_contact_trace, run_single
from omnisurg.haptic_bench.virtual_device import VirtualHandpieceParams
from omnisurg.haptic_feedback import HapticFeedbackSettings


DEFAULT_OBJECTIVES: tuple[str, ...] = (
    "oscillation_peak",
    "energy_injection_abs",
    "jerk_rms",
    "free_space_force_mean",
    "contact_rise_time_ms",
)


def _sample_param(trial, name: str, spec: dict) -> Any:
    kind = spec.get("type", "float")
    if kind == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
        )
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"Unknown param type '{kind}' for '{name}'")


def _metric_objective(metrics: dict[str, float], name: str) -> float:
    if name == "energy_injection_abs":
        return float(abs(metrics.get("energy_injection", 0.0)))
    return float(metrics.get(name, 0.0))


def run_study(
    *,
    algorithm_name: str,
    space: dict[str, dict],
    contact_trace_path: str | Path,
    n_trials: int,
    objectives: tuple[str, ...] = DEFAULT_OBJECTIVES,
    base_settings: HapticFeedbackSettings | None = None,
    device_params: VirtualHandpieceParams | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "optuna is required for haptic-bench study. Install with `uv pip install optuna`."
        ) from exc

    contact_trace = load_contact_trace(contact_trace_path)
    base = base_settings if base_settings is not None else HapticFeedbackSettings()
    directions = ["minimize"] * len(objectives)

    def objective(trial):
        trial_settings = replace(base)
        for name, spec in space.items():
            value = _sample_param(trial, name, spec)
            if not hasattr(trial_settings, name):
                raise AttributeError(f"HapticFeedbackSettings has no field '{name}'")
            setattr(trial_settings, name, value)
        _, metrics = run_single(
            algorithm_name, trial_settings, contact_trace, device_params
        )
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("params", {k: getattr(trial_settings, k) for k in space})
        return tuple(_metric_objective(metrics, name) for name in objectives)

    sampler = optuna.samplers.NSGAIISampler(seed=seed)
    study = optuna.create_study(directions=directions, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    pareto = []
    for trial in study.best_trials:
        pareto.append(
            {
                "trial_number": trial.number,
                "params": trial.user_attrs.get("params", {}),
                "metrics": trial.user_attrs.get("metrics", {}),
                "values": list(trial.values) if trial.values is not None else [],
            }
        )

    return {
        "algorithm": algorithm_name,
        "n_trials": n_trials,
        "objectives": list(objectives),
        "pareto_front": pareto,
        "all_trials": [
            {
                "number": trial.number,
                "params": trial.user_attrs.get("params", {}),
                "metrics": trial.user_attrs.get("metrics", {}),
                "values": list(trial.values) if trial.values is not None else [],
            }
            for trial in study.trials
        ],
    }
