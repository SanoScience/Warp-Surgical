"""Haptic research bench CLI.

Usage:
    python -m omnisurg.haptic_bench <command> [options] [--json]

Commands:
    list-algorithms               Print registered algorithm names.
    run                           Score one (algorithm, settings) pair against a contact trace.
    study                         Multi-objective Optuna sweep over algorithm params.
    compare                       Aggregate multiple `run` outputs into a table.

All commands accept --json for machine-readable output wrapped in
`{"schema":"haptic-bench/v1","data":...}`. Exit codes: 0 ok, 2 arg error,
3 runtime, 4 missing dep, 5 timeout.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from omnisurg.haptic_bench.algorithms import list_algorithms
from omnisurg.haptic_bench.runner import load_contact_trace, run_single
from omnisurg.haptic_bench.virtual_device import VirtualHandpieceParams
from omnisurg.haptic_feedback import (
    HapticFeedbackSettings,
    settings_from_dict,
    settings_to_dict,
)


SCHEMA = "haptic-bench/v1"


def _emit(data: dict, as_json: bool) -> None:
    if as_json:
        sys.stdout.write(json.dumps({"schema": SCHEMA, "data": data}, indent=2) + "\n")
    else:
        sys.stdout.write(json.dumps(data, indent=2) + "\n")


def _load_settings(path: str | None) -> HapticFeedbackSettings:
    if path is None:
        return HapticFeedbackSettings()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return settings_from_dict(payload)


def _load_device_params(path: str | None) -> VirtualHandpieceParams | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return VirtualHandpieceParams(
        mass=float(payload.get("mass", VirtualHandpieceParams.mass)),
        damping=float(payload.get("damping", VirtualHandpieceParams.damping)),
        hand_stiffness=float(
            payload.get("hand_stiffness", VirtualHandpieceParams.hand_stiffness)
        ),
        actuator_tau=float(
            payload.get("actuator_tau", VirtualHandpieceParams.actuator_tau)
        ),
        max_force=float(payload.get("max_force", VirtualHandpieceParams.max_force)),
    )


def _cmd_list(args) -> int:
    _emit({"algorithms": list_algorithms()}, as_json=args.json)
    return 0


def _cmd_run(args) -> int:
    settings = _load_settings(args.settings)
    device_params = _load_device_params(args.device_params)
    contact_trace = load_contact_trace(args.contact_trace)
    _, metrics = run_single(args.algorithm, settings, contact_trace, device_params)

    data = {
        "algorithm": args.algorithm,
        "settings": settings_to_dict(settings),
        "device_params": asdict(device_params) if device_params is not None else None,
        "contact_trace": str(args.contact_trace),
        "metrics": metrics,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"schema": SCHEMA, "data": data}, indent=2) + "\n",
            encoding="utf-8",
        )
    _emit(data, as_json=args.json)
    return 0


def _cmd_study(args) -> int:
    from omnisurg.haptic_bench.study import run_study

    space = json.loads(Path(args.space).read_text(encoding="utf-8"))
    base_settings = _load_settings(args.settings)
    device_params = _load_device_params(args.device_params)

    result = run_study(
        algorithm_name=args.algorithm,
        space=space,
        contact_trace_path=args.contact_trace,
        n_trials=args.n_trials,
        base_settings=base_settings,
        device_params=device_params,
        seed=args.seed,
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"schema": SCHEMA, "data": result}, indent=2) + "\n",
            encoding="utf-8",
        )
    summary = {
        "algorithm": result["algorithm"],
        "n_trials": result["n_trials"],
        "objectives": result["objectives"],
        "pareto_size": len(result["pareto_front"]),
        "pareto_front": result["pareto_front"],
    }
    _emit(summary, as_json=args.json)
    return 0


def _cmd_compare(args) -> int:
    rows = []
    for path in args.runs:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        data = payload.get("data", payload)
        rows.append(
            {
                "path": path,
                "algorithm": data.get("algorithm"),
                "metrics": data.get("metrics", {}),
            }
        )
    data = {"rows": rows}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"schema": SCHEMA, "data": data}, indent=2) + "\n",
            encoding="utf-8",
        )
    _emit(data, as_json=args.json)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--json", action="store_true", help="Emit newton-cli-style envelope")

    parser = argparse.ArgumentParser(prog="omnisurg.haptic_bench", parents=[parent])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-algorithms", parents=[parent], help="Print registered algorithm names")

    run_p = sub.add_parser("run", parents=[parent], help="Score one (algorithm, settings) pair")
    run_p.add_argument("--algorithm", required=True)
    run_p.add_argument("--settings", default=None, help="Preset JSON path")
    run_p.add_argument("--device-params", default=None, help="Virtual handpiece params JSON")
    run_p.add_argument("--contact-trace", required=True)
    run_p.add_argument("--out", default=None)

    study_p = sub.add_parser("study", parents=[parent], help="Optuna multi-objective sweep")
    study_p.add_argument("--algorithm", required=True)
    study_p.add_argument("--space", required=True, help="Param space JSON")
    study_p.add_argument("--settings", default=None, help="Baseline preset JSON")
    study_p.add_argument("--device-params", default=None)
    study_p.add_argument("--contact-trace", required=True)
    study_p.add_argument("--n-trials", type=int, default=100)
    study_p.add_argument("--seed", type=int, default=0)
    study_p.add_argument("--out", default=None)

    compare_p = sub.add_parser("compare", parents=[parent], help="Aggregate multiple run outputs")
    compare_p.add_argument("--runs", nargs="+", required=True)
    compare_p.add_argument("--out", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list-algorithms":
            return _cmd_list(args)
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "study":
            return _cmd_study(args)
        if args.command == "compare":
            return _cmd_compare(args)
    except FileNotFoundError as exc:
        sys.stderr.write(f"error: file not found: {exc}\n")
        return 2
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"error: malformed JSON input: {exc}\n")
        return 2
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    except KeyError as exc:
        sys.stderr.write(f"error: unknown key {exc}\n")
        return 2
    except ImportError as exc:
        sys.stderr.write(f"error: missing dependency: {exc}\n")
        return 4
    except RuntimeError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 3
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
