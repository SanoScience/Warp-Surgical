import argparse
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_vec3(values) -> np.ndarray:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected exactly 3 values")
    try:
        vector = np.array([float(value) for value in values], dtype=np.float32)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("vector values must be numeric") from exc
    return vector


def _clamp_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0.0:
        return np.zeros(3, dtype=np.float32)
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= max_norm or magnitude <= 1.0e-8:
        return vector.astype(np.float32, copy=True)
    return (vector * (max_norm / magnitude)).astype(np.float32, copy=True)


def _compute_spring_damper_force(
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    anchor: np.ndarray,
    spring_k: float,
    damping_b: float,
    max_force: float,
    deadband: float,
) -> tuple[np.ndarray, np.ndarray]:
    displacement = anchor - position
    force = spring_k * displacement - damping_b * velocity
    if float(np.linalg.norm(force)) < deadband:
        force = np.zeros(3, dtype=np.float32)
    force = _clamp_norm(force, max_force)
    return force, displacement.astype(np.float32, copy=True)


def _format_vec3(vector: np.ndarray) -> str:
    return f"({vector[0]: .3f}, {vector[1]: .3f}, {vector[2]: .3f})"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone OpenHaptics spring-damper test. "
            "The anchor starts at the current device position plus --anchor-offset. "
            "Press the device button to re-anchor."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device-name", type=str, default="Default Device", help="OpenHaptics device name")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Position scale passed to HapticController; 1.0 keeps raw device units",
    )
    parser.add_argument("--spring-k", type=float, default=0.003, help="Spring gain")
    parser.add_argument("--damping-b", type=float, default=0.0002, help="Damping gain")
    parser.add_argument("--max-force", type=float, default=0.05, help="Clamp on commanded force magnitude")
    parser.add_argument("--deadband", type=float, default=0.0005, help="Zero force below this magnitude")
    parser.add_argument("--ramp-seconds", type=float, default=0.5, help="Linear force ramp-in time")
    parser.add_argument("--update-hz", type=float, default=250.0, help="Controller update rate from Python")
    parser.add_argument("--status-hz", type=float, default=10.0, help="Console status print rate")
    parser.add_argument(
        "--velocity-alpha",
        type=float,
        default=0.25,
        help="Low-pass factor for finite-difference velocity",
    )
    parser.add_argument("--duration", type=float, default=0.0, help="Run time in seconds, 0 means until Ctrl+C")
    parser.add_argument(
        "--anchor-offset",
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="Offset added to the current position when creating or resetting the anchor",
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    anchor_offset = _parse_vec3(args.anchor_offset)
    velocity_alpha = float(np.clip(args.velocity_alpha, 0.0, 1.0))
    update_period = 1.0 / max(float(args.update_hz), 1.0)
    status_period = 1.0 / max(float(args.status_hz), 0.1)
    ramp_seconds = max(float(args.ramp_seconds), 0.0)

    from omnisurg.input.device import HapticController

    controller = None
    try:
        controller = HapticController(device_name=args.device_name, scale=float(args.scale))
        position = np.asarray(controller.get_scaled_position(), dtype=np.float32)
        velocity = np.zeros(3, dtype=np.float32)
        anchor = position + anchor_offset
        previous_position = position.copy()
        previous_time = time.perf_counter()
        start_time = previous_time
        last_status_time = previous_time
        previous_button = bool(controller.is_button_pressed())

        print("OpenHaptics spring-damper test")
        print(f"Device: {args.device_name}")
        print(
            "Defaults are intentionally conservative. "
            "Hold the stylus securely, use Ctrl+C to exit, and press the device button to re-anchor."
        )
        print(f"Initial anchor: {_format_vec3(anchor)}")

        while True:
            loop_started = time.perf_counter()
            dt = max(loop_started - previous_time, 1.0e-4)

            position = np.asarray(controller.get_scaled_position(), dtype=np.float32)
            button = bool(controller.is_button_pressed())
            if button and not previous_button:
                anchor = position + anchor_offset
                print(f"Re-anchored to {_format_vec3(anchor)}")
            previous_button = button

            raw_velocity = (position - previous_position) / dt
            velocity = velocity_alpha * raw_velocity + (1.0 - velocity_alpha) * velocity

            force, displacement = _compute_spring_damper_force(
                position=position,
                velocity=velocity,
                anchor=anchor,
                spring_k=max(float(args.spring_k), 0.0),
                damping_b=max(float(args.damping_b), 0.0),
                max_force=max(float(args.max_force), 0.0),
                deadband=max(float(args.deadband), 0.0),
            )

            if ramp_seconds > 0.0:
                ramp = min((loop_started - start_time) / ramp_seconds, 1.0)
                force = (force * ramp).astype(np.float32, copy=False)

            controller.set_force(force.tolist())

            if loop_started - last_status_time >= status_period:
                print(
                    f"pos={_format_vec3(position)} "
                    f"disp={_format_vec3(displacement)} |disp|={np.linalg.norm(displacement):.3f} "
                    f"vel={_format_vec3(velocity)} "
                    f"force={_format_vec3(force)} |F|={np.linalg.norm(force):.4f} "
                    f"button={'1' if button else '0'}"
                )
                last_status_time = loop_started

            previous_position = position.copy()
            previous_time = loop_started

            if args.duration > 0.0 and loop_started - start_time >= float(args.duration):
                break

            elapsed = time.perf_counter() - loop_started
            if elapsed < update_period:
                time.sleep(update_period - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        if controller is not None:
            try:
                controller.set_force([0.0, 0.0, 0.0])
                time.sleep(0.05)
            finally:
                controller.close()


if __name__ == "__main__":
    main()
