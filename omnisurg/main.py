import argparse
import sys

import numpy as np
import warp as wp

from omnisurg.input.factory import INPUT_BACKENDS, RoleInputConfig, normalize_input_backend, open_input_sources

LIVE_INPUT_BACKENDS = INPUT_BACKENDS
INPUT_BACKEND_METAVAR = "{" + ",".join(INPUT_BACKENDS) + "}"


def _parse_input_backend(value: str) -> str:
    try:
        return normalize_input_backend(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="OmniSurg Phase 2 - soft-body simulation with configurable live input backends",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default=None, help="Warp device override")
    parser.add_argument(
        "--viewer",
        type=str,
        default="gl",
        choices=[
            "surgsim",
            "gl",
            "rtx",
            "headless",
            "slang",
            "slang-d3d12",
            "slang-vulkan",
            "slang-vk",
        ],
        help="Viewer backend",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="chole",
        choices=["single", "chole"],
        help="Scene preset",
    )
    parser.add_argument(
        "--right-replay",
        dest="right_replay",
        type=str,
        default=None,
        help="Path to right-controller .npy haptic replay trace",
    )
    parser.add_argument("--replay", dest="right_replay", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--left-replay",
        type=str,
        default=None,
        help="Path to left-controller .npy haptic replay trace",
    )
    parser.add_argument(
        "--replay-force-feedback",
        action="store_true",
        help="During replay, also open the live haptic device(s) and dispatch "
        "sim-generated forces to them (positions still come from the trace).",
    )
    parser.add_argument(
        "--force-telemetry",
        action="store_true",
        help="Show live per-frame force / proxy-offset / contact plots in the "
        "viewer's Plots window (GL viewer only).",
    )
    parser.add_argument(
        "--force-telemetry-csv",
        type=str,
        default=None,
        help="Path to write per-frame force-feedback telemetry as CSV for "
        "offline FFT/stability analysis. Implies --force-telemetry sampling.",
    )
    parser.add_argument(
        "--contact-trace",
        type=str,
        default=None,
        help="Path to write a per-frame haptic-algorithm input trace (.npz). "
        "Feeds the offline haptic_bench for algorithm research.",
    )
    parser.add_argument(
        "--stage3-validate",
        action="store_true",
        help="Drive the real haptic device to physically follow the right-controller "
        "replay trace while still dispatching sim-computed contact forces. Unattended — "
        "do not hold the stylus. Implies --replay-force-feedback; requires --right-replay.",
    )
    parser.add_argument("--stage3-kp", type=float, default=0.02,
                        help="Position-tracking P gain in N per native position unit "
                             "(Touch traces are mm, so this is N/mm). 0.02 → 0.2 N at "
                             "10 mm error; stays well under --stage3-force-cap.")
    parser.add_argument("--stage3-kd", type=float, default=0.005,
                        help="Position-tracking D gain (N·s per native position unit).")
    parser.add_argument("--stage3-force-cap", type=float, default=1.5,
                        help="Max force magnitude at the device (N). Safety cap.")
    parser.add_argument("--stage3-abort-mm", type=float, default=10.0,
                        help="Abort if tracking error stays above this many native "
                             "position units (mm on Touch) for --stage3-abort-ms milliseconds "
                             "(catches human grabbing stylus).")
    parser.add_argument("--stage3-abort-ms", type=float, default=50.0,
                        help="Sustained-error window triggering abort (ms).")
    parser.add_argument("--stage3-ramp-ms", type=float, default=1500.0,
                        help="Ramp-in window (ms) to smoothly move device from its "
                             "current pose to the first replay sample.")
    parser.add_argument("--stage3-telemetry", type=str, default=None,
                        help="Path to write a per-frame stage-3 validation trace (.npz).")
    parser.add_argument("--stage3-abort-action", type=str, default="zero",
                        choices=["zero", "hold", "continue"],
                        help="What to do when tracking-error threshold is held: "
                             "'zero' drops forces to 0 and exits (safest, default); "
                             "'hold' stops dispatching so motor holds last command + exits; "
                             "'continue' logs once per excursion and keeps running.")
    parser.add_argument("--stage3-dry-run", action="store_true",
                        help="Scan the replay trace, print worst-case PD-force estimate "
                             "given --stage3-kp/kd/force-cap, and exit without touching "
                             "the device. Use to sanity-check gains before a live run.")
    parser.add_argument(
        "--record-right",
        type=str,
        default=None,
        help="Path to record the right-controller haptic trace (.npy). Toggle with R in the viewer.",
    )
    parser.add_argument(
        "--record-left",
        type=str,
        default=None,
        help="Path to record the left-controller haptic trace (.npy). Toggle with R in the viewer.",
    )
    parser.add_argument(
        "--right-input-backend",
        type=_parse_input_backend,
        default="openhaptics",
        metavar=INPUT_BACKEND_METAVAR,
        help="Live input backend for the right controller",
    )
    parser.add_argument(
        "--left-input-backend",
        type=_parse_input_backend,
        default="openhaptics",
        metavar=INPUT_BACKEND_METAVAR,
        help="Live input backend for the left controller",
    )
    parser.add_argument(
        "--input-backend",
        dest="legacy_input_backend",
        type=_parse_input_backend,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--right-device-name",
        type=str,
        default="Default Device",
        help="OpenHaptics device name for the right controller",
    )
    parser.add_argument(
        "--left-device-name",
        type=str,
        default="Left Device",
        help="OpenHaptics device name for the left controller",
    )
    parser.add_argument(
        "--right-device-index",
        type=int,
        default=0,
        help="Device index for backends that enumerate identical devices such as MiniMou",
    )
    parser.add_argument(
        "--left-device-index",
        type=int,
        default=1,
        help="Device index for backends that enumerate identical devices such as MiniMou",
    )
    parser.add_argument("--device-name", dest="right_device_name", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--asset",
        type=str,
        default="liver",
        help="Single-asset mesh name used when --scene=single",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=0,
        help="Max frames to run (0 = unlimited)",
    )
    parser.add_argument("--no-vsync", action="store_true", help="Disable display vsync for profiling")
    parser.add_argument("--no-pacing", action="store_true", help="Disable frame pacing sleep for profiling")
    parser.add_argument(
        "--no-internal-profiling",
        action="store_true",
        help="Disable OmniSurg's internal Warp timers when using an external profiler",
    )
    parser.add_argument(
        "--no-textures",
        action="store_true",
        help="Start with organ textures disabled",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["quality", "balanced", "performance"],
        help="Simulation preset (overrides default substeps/fps)",
    )
    args = parser.parse_args(argv)
    if args.legacy_input_backend is not None:
        args.right_input_backend = args.legacy_input_backend
    args.right_input_backend = normalize_input_backend(args.right_input_backend)
    args.left_input_backend = normalize_input_backend(args.left_input_backend)
    if args.right_device_name is None:
        args.right_device_name = "Default Device"
    args.replay = args.right_replay
    return args


def _device_name_candidates(device_name: str) -> tuple[str, ...]:
    aliases = {
        "Device Left": "Left Device",
        "Left Device": "Device Left",
    }
    candidates = [device_name]
    alias = aliases.get(device_name)
    if alias is not None and alias not in candidates:
        candidates.append(alias)
    return tuple(candidates)


def _build_openhaptics_source(controller_id: str, requested_name: str):
    from omnisurg.haptics import LiveHapticSource

    last_exc = None
    for candidate_name in _device_name_candidates(requested_name):
        try:
            source = LiveHapticSource(device_name=candidate_name)
            if candidate_name != requested_name:
                print(
                    f'{controller_id} device name fallback: using "{candidate_name}" instead of "{requested_name}"'
                )
            return source, candidate_name
        except Exception as exc:
            last_exc = exc

    assert last_exc is not None
    raise last_exc


class _SampleTransformSource:
    def __init__(self, source, *, position_transform=None, rotation_transform=None):
        self._source = source
        self._position_transform = position_transform
        self._rotation_transform = rotation_transform

    def __getattr__(self, name: str):
        return getattr(self._source, name)

    def poll(self):
        sample = self._source.poll()
        if not sample:
            return sample

        transformed = dict(sample)
        position = transformed.get("position")
        if position is not None and self._position_transform is not None:
            transformed["position"] = self._position_transform(np.asarray(position, dtype=np.float32))

        rotation = transformed.get("rotation")
        if rotation is not None and self._rotation_transform is not None:
            transformed["rotation"] = self._rotation_transform(np.asarray(rotation, dtype=np.float32))

        return transformed

    def close(self):
        self._source.close()


def _reflect_minimou_quaternion_x(rotation: np.ndarray) -> np.ndarray:
    corrected = np.asarray(rotation, dtype=np.float32).copy()
    if corrected.shape[0] >= 1:
        corrected[0] *= -1.0
    return corrected


def _build_minimou_source(device_index: int):
    from omnisurg.haptics import LiveMiniMouSource

    source = LiveMiniMouSource(device_index=device_index)
    source = _SampleTransformSource(source, rotation_transform=_reflect_minimou_quaternion_x)
    descriptor = f"MiniMou[{device_index}]"
    controller = getattr(getattr(source, "_ctrl", None), "_controller", None)
    device_id = getattr(controller, "device_id", None)
    if device_id is not None:
        descriptor = f"{descriptor}/id={device_id}"
    return source, descriptor


def _build_live_input_rig(args):
    from omnisurg.haptics import MultiSourceRig

    configs = []
    for controller_id in ("right", "left"):
        backend = normalize_input_backend(getattr(args, f"{controller_id}_input_backend"))
        if backend in {"off", "replay"}:
            continue
        configs.append(
            RoleInputConfig(
                role=controller_id,
                backend=backend,
                device_name=getattr(args, f"{controller_id}_device_name", None),
                device_index=getattr(args, f"{controller_id}_device_index", None),
                replay_path=getattr(args, f"{controller_id}_replay", None),
                force_feedback=backend == "openhaptics",
            )
        )

    result = open_input_sources(configs, require_all=False)

    if result.failures:
        print("Unavailable live devices: " + "; ".join(result.failures))

    if not result.sources:
        return None

    print(
        "Using live input devices: "
        + ", ".join(f"{controller_id}={descriptor}" for controller_id, descriptor in result.descriptors.items())
    )
    return MultiSourceRig(result.sources)


def _dispatch_haptic_force_commands(input_rig, force_commands):
    if input_rig is None:
        return

    dispatch = getattr(input_rig, "set_force_commands", None)
    if callable(dispatch):
        dispatch(force_commands)
        return

    set_force = getattr(input_rig, "set_force", None)
    if callable(set_force):
        right_force = force_commands.get("right", np.zeros(3, dtype=np.float32))
        set_force(right_force)


def _zero_haptic_force_commands(input_rig):
    zero = np.zeros(3, dtype=np.float32)
    _dispatch_haptic_force_commands(
        input_rig,
        {
            "right": zero,
            "left": zero.copy(),
        },
    )


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)

    if argv and argv[0] == "hex":
        from omnisurg.hex.cli import main as hex_main

        return hex_main(argv[1:])

    args = parse_args(argv)

    print(f"Python {sys.version}")
    print(
        f"OmniSurg Phase 2 - scene={args.scene}, asset={args.asset}, viewer={args.viewer}, "
        f"textures={'off' if args.no_textures else 'on'}"
    )

    from omnisurg.config import BoundsConfig, HapticConfig, SceneConfig, SimulationConfig, SIMULATION_PRESETS, ViewerConfig
    from omnisurg.haptics import BimanualReplayRig, RecordingRig, ReplayForceFeedbackRig
    from omnisurg.input.position_tracking import Stage3ValidationRig, analyse_trace_for_stage3
    from omnisurg.runtime import Runtime
    from omnisurg.telemetry import ContactTraceRecorder, ForceTelemetry, Stage3TelemetryRecorder

    if args.stage3_validate and not args.replay:
        print("--stage3-validate requires --right-replay (a right-controller trace to track).")
        sys.exit(2)

    if args.stage3_dry_run:
        if not args.replay:
            print("--stage3-dry-run requires --right-replay.")
            sys.exit(2)
        if args.preset:
            sim_dt = SIMULATION_PRESETS[args.preset].frame_dt
        else:
            sim_dt = SimulationConfig().frame_dt
        summary = analyse_trace_for_stage3(
            args.replay,
            sim_frame_dt=sim_dt,
            kp=args.stage3_kp,
            kd=args.stage3_kd,
            force_cap=args.stage3_force_cap,
            abort_error=args.stage3_abort_mm,
        )
        print("[stage3 dry-run] trace analysis:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        if summary.get("exceeds_force_cap"):
            print(
                "[stage3 dry-run] WARNING: estimated worst-case PD force exceeds "
                "--stage3-force-cap. Lower Kp/Kd or raise the cap before a live run."
            )
        sys.exit(0)

    if args.preset:
        sim_config = SIMULATION_PRESETS[args.preset]
    else:
        sim_config = SimulationConfig()
    scene_config = SceneConfig(scene_preset=args.scene, asset_name=args.asset)
    haptic_config = HapticConfig()
    viewer_config = ViewerConfig(
        backend=args.viewer,
        vsync=not args.no_vsync,
        textures_enabled=not args.no_textures,
    )
    bounds_config = BoundsConfig()

    input_rig = None
    if args.replay or args.left_replay:
        input_rig = BimanualReplayRig(right_path=args.replay, left_path=args.left_replay)
        replay_desc = []
        if args.replay:
            replay_desc.append(f"right={args.replay}")
        if args.left_replay:
            replay_desc.append(f"left={args.left_replay}")
        print("Using replay input: " + ", ".join(replay_desc))
        if args.stage3_validate:
            force_rig = _build_live_input_rig(args)
            if force_rig is None:
                print("--stage3-validate requires a live haptic device; none available. Aborting.")
                sys.exit(3)
            abort_frames = max(1, int(round(args.stage3_abort_ms * 1.0e-3 / sim_config.frame_dt)))
            ramp_frames = max(0, int(round(args.stage3_ramp_ms * 1.0e-3 / sim_config.frame_dt)))
            input_rig = Stage3ValidationRig(
                input_rig,
                force_rig,
                sim_frame_dt=sim_config.frame_dt,
                controller_id="right",
                kp=args.stage3_kp,
                kd=args.stage3_kd,
                force_cap=args.stage3_force_cap,
                abort_error=args.stage3_abort_mm,
                abort_error_frames=abort_frames,
                ramp_in_frames=ramp_frames,
                abort_action=args.stage3_abort_action,
            )
            print(
                f"[stage3] armed: kp={args.stage3_kp} kd={args.stage3_kd} "
                f"cap={args.stage3_force_cap} N  abort>{args.stage3_abort_mm}mm for "
                f"{abort_frames} frames ({args.stage3_abort_ms}ms)  ramp={ramp_frames} frames "
                f"({args.stage3_ramp_ms}ms). DO NOT HOLD THE STYLUS."
            )
        elif args.replay_force_feedback:
            force_rig = _build_live_input_rig(args)
            if force_rig is None:
                print(
                    "Replay force-feedback requested but no live haptic devices "
                    "were available; continuing without force output."
                )
            else:
                input_rig = ReplayForceFeedbackRig(input_rig, force_rig)
                print("Replay force-feedback armed: sim forces will be sent to live device(s)")
    elif args.viewer == "headless":
        print("Headless mode: running without live haptic input")
    else:
        input_rig = _build_live_input_rig(args)
        if input_rig is None:
            print("Haptic devices not available, running without input")

    recording_rig: RecordingRig | None = None
    if args.record_right or args.record_left:
        if input_rig is None:
            print("No live input rig available; ignoring --record-right/--record-left")
        else:
            record_paths = {}
            if args.record_right:
                record_paths["right"] = args.record_right
            if args.record_left:
                record_paths["left"] = args.record_left
            recording_rig = RecordingRig(input_rig, record_paths)
            input_rig = recording_rig
            print(
                "Recording armed (press R to start/stop): "
                + ", ".join(f"{cid}={path}" for cid, path in record_paths.items())
            )

    is_replay = bool(args.replay or args.left_replay)

    if args.viewer in {"gl", "surgsim"}:
        print("Press T in the viewer to toggle textures")
        if recording_rig is not None:
            print("Press R in the viewer to start/stop recording")
        if is_replay:
            print("Press P in the viewer to restart playback")

    with wp.ScopedDevice(args.device):
        rt = Runtime(sim_config, scene_config, haptic_config, viewer_config, bounds_config)
        if args.no_internal_profiling:
            rt.profiling_enabled = False
            rt.profiling_console_enabled = False

        if recording_rig is not None:
            def _record_key_hook(symbol, modifiers, rig=recording_rig):
                import pyglet
                if symbol == pyglet.window.key.R:
                    rig.toggle_recording()
            rt.register_key_press_hook(_record_key_hook)

        if is_replay and input_rig is not None:
            def _replay_key_hook(symbol, modifiers, rig=input_rig):
                import pyglet
                if symbol == pyglet.window.key.P:
                    rig.reset()
                    print("[replay] restarted")
            rt.register_key_press_hook(_replay_key_hook)

        telemetry: ForceTelemetry | None = None
        if args.force_telemetry or args.force_telemetry_csv:
            telemetry = ForceTelemetry(
                rt,
                viewer_plots=args.force_telemetry,
                csv_path=args.force_telemetry_csv,
            )
            if telemetry.enabled:
                desc = []
                if args.force_telemetry:
                    desc.append("viewer plots")
                if args.force_telemetry_csv:
                    desc.append(f"csv={args.force_telemetry_csv}")
                print("Force telemetry enabled: " + ", ".join(desc))
            else:
                telemetry = None
                print("Force telemetry requested but unsupported for this viewer/backend")

        contact_trace: ContactTraceRecorder | None = None
        if args.contact_trace:
            contact_trace = ContactTraceRecorder(rt, args.contact_trace)
            print(f"Contact-trace recording enabled: {args.contact_trace}")

        stage3_telemetry: Stage3TelemetryRecorder | None = None
        if args.stage3_telemetry and isinstance(input_rig, Stage3ValidationRig):
            stage3_telemetry = Stage3TelemetryRecorder(
                input_rig,
                args.stage3_telemetry,
                sim_frame_dt=sim_config.frame_dt,
            )
            print(f"Stage3 telemetry enabled: {args.stage3_telemetry}")

        frame = 0
        while rt.is_running():
            if input_rig:
                rt.poll_input(input_rig)
            rt.step()
            if input_rig:
                _dispatch_haptic_force_commands(input_rig, rt.get_haptic_force_commands())
            if telemetry is not None:
                telemetry.record()
            if contact_trace is not None:
                contact_trace.record()
            if stage3_telemetry is not None:
                stage3_telemetry.record()
            rt.render()
            if not args.no_pacing:
                rt.pace()

            frame += 1
            if args.num_frames > 0 and frame >= args.num_frames:
                break
            if isinstance(input_rig, Stage3ValidationRig) and input_rig.aborted:
                print("[stage3] aborted; exiting sim loop.")
                break

        if telemetry is not None:
            telemetry.close()
        if contact_trace is not None:
            contact_trace.close()
        if stage3_telemetry is not None:
            stage3_telemetry.close()
        if input_rig:
            _zero_haptic_force_commands(input_rig)
            input_rig.close()
        rt.close()


if __name__ == "__main__":
    main()
