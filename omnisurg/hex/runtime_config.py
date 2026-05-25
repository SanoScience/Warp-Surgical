# SPDX-License-Identifier: Apache-2.0
"""Shared argument registration and normalization for internal hex runtimes."""

from __future__ import annotations

import argparse
from typing import Any

from omnisurg.input.factory import INPUT_BACKENDS, normalize_input_backend
from omnisurg.rendering.slang_cryo import SLANG_RENDER_BACKENDS, is_slang_backend

from .setup import (
    GS_WEIGHTING_BY_NAME,
    HIERARCHICAL_MODE_BY_NAME,
    L0_SHAPE_MATCHING_MODE_BY_NAME,
    L2_HIERARCHICAL_MODE_BY_NAME,
)

INSTRUMENT_COUNT = 2
INSTRUMENT_TOOL_MODES = ("diathermy", "grasper", "scissors", "bipolar")
INSTRUMENT_TOOL_MODE_ALIASES = {"cutting": "diathermy"}
INPUT_BACKEND_METAVAR = "{" + ",".join(INPUT_BACKENDS) + "}"


def _parse_input_backend(value: str) -> str:
    try:
        return normalize_input_backend(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def normalize_instrument_tool_mode(mode: str) -> str:
    normalized = INSTRUMENT_TOOL_MODE_ALIASES.get(str(mode).strip().lower(), str(mode).strip().lower())
    if normalized not in INSTRUMENT_TOOL_MODES:
        choices = ", ".join((*INSTRUMENT_TOOL_MODES, *sorted(INSTRUMENT_TOOL_MODE_ALIASES)))
        raise ValueError(f"unknown instrument tool mode {mode!r}; expected one of: {choices}")
    return normalized


def add_hex_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Register shared runtime options used by both hex runtime drivers."""
    parser.add_argument("--atlas", type=str, default="Digimouse/atlas/atlas")
    parser.add_argument("--size", type=int, default=0, help="If > 0, use a synthetic NxNxN block instead of Digimouse.")
    parser.add_argument("--voxel", type=float, default=0.005, help="Block mode voxel size (ignored for Digimouse).")
    parser.add_argument(
        "--atlas-pad",
        type=int,
        default=1,
        help="Empty-voxel margin padded around the Digimouse atlas so MC closes caps where the mouse touches a grid edge. Default 1.",
    )
    parser.add_argument(
        "--viewer",
        choices=("gl", "headless", *sorted(SLANG_RENDER_BACKENDS)),
        default="gl",
        help=(
            "Renderer backend. 'gl' keeps ViewerGL; 'headless' creates no window; "
            "'slang' selects Vulkan on Linux and D3D12 on Windows."
        ),
    )
    parser.add_argument("--usd", type=str, default=None)
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Number of frames to run. Defaults to infinite for ViewerGL; required with --usd.",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--timer-report-secs",
        type=float,
        default=0.0,
        help="print aggregated ScopedTimer summaries every N wall-clock seconds (0 disables; default).",
    )
    parser.add_argument(
        "--timer-sync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="synchronize selected hot-path timers so their wall times include queued GPU work.",
    )
    parser.add_argument(
        "--timer-gpu-activities",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="collect CUDA kernel/memcpy/memset activity totals for selected hot-path timers.",
    )
    parser.add_argument(
        "--cut-debug-validate",
        action="store_true",
        help="run synchronizing GPU deletion-state validation after cut attempts.",
    )
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--drop-height", type=float, default=0.04)
    parser.add_argument("--ground-height", type=float, default=0.0)
    parser.add_argument(
        "--global-scale",
        type=float,
        default=1.0,
        help=(
            "Uniformly scales spatial quantities (voxel size, origin, drop height, particle radius, render extents). "
            "Stiffness, damping, mass, and friction are untouched."
        ),
    )
    parser.add_argument("--gravity-on", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture the per-frame substep loop into a CUDA graph and replay it.",
    )
    parser.add_argument(
        "--particle-particle-collisions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable particle-particle self-collision in the local solver (default OFF).",
    )
    parser.add_argument(
        "--gl-interop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable Newton's CUDA/OpenGL VBO interop path for ViewerGL (default ON; use --no-gl-interop to force CPU uploads).",
    )
    parser.add_argument(
        "--viewer-log-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="call Newton's viewer.log_state() each frame (default ON).",
    )
    parser.add_argument(
        "--right-input-backend",
        type=_parse_input_backend,
        default=None,
        metavar=INPUT_BACKEND_METAVAR,
        help="pose source for the right kinematic instrument sphere.",
    )
    parser.add_argument(
        "--left-input-backend",
        type=_parse_input_backend,
        default=None,
        metavar=INPUT_BACKEND_METAVAR,
        help="pose source for the left kinematic instrument sphere.",
    )
    parser.add_argument("--input-backend", dest="input_backend", type=_parse_input_backend, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--right-device-name", default=None, help="right OpenHaptics device name")
    parser.add_argument("--left-device-name", default=None, help="left OpenHaptics device name")
    parser.add_argument("--device-name", dest="device_name", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--right-device-index", type=int, default=0, help="right indexed-device backend index")
    parser.add_argument("--left-device-index", type=int, default=1, help="left indexed-device backend index")
    parser.add_argument("--right-replay", default=None, help="right-controller replay trace for replay input backend")
    parser.add_argument("--left-replay", default=None, help="left-controller replay trace for replay input backend")
    parser.add_argument(
        "--position-scale",
        type=float,
        default=0.001,
        help="world units per input-device millimetre; 0.001 maps mm to metres.",
    )
    parser.add_argument(
        "--device-x-offsets",
        type=float,
        nargs=2,
        default=(-0.17, 0.17),
        metavar=("FIRST_X", "SECOND_X"),
        help="world-space X offsets for first/right and second/left instrument bases.",
    )
    parser.add_argument(
        "--instrument-radius",
        "--instrument-radius-scale",
        dest="instrument_radius_scale",
        type=float,
        default=5.0,
        help="kinematic instrument sphere radius in voxel widths.",
    )
    parser.add_argument(
        "--instrument-collision-relaxation",
        type=float,
        default=0.9,
        help="sphere-vs-particle contact correction blend in [0, 1].",
    )
    parser.add_argument(
        "--instrument-contact-iterations",
        type=int,
        default=1,
        help="sphere-vs-particle contact projection passes per solver constraint iteration.",
    )
    parser.add_argument(
        "--instrument-max-correction-scale",
        type=float,
        default=1.0,
        help="maximum instrument contact position correction per pass in voxel widths; 0 disables the cap.",
    )
    parser.add_argument(
        "--show-instruments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="render the two kinematic instrument spheres.",
    )
    parser.add_argument(
        "--instrument-follow-camera",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "move the instrument spheres in the camera frame. "
            "Default: ON for fallback input, OFF for hardware input."
        ),
    )
    parser.add_argument(
        "--instrument-tool-modes",
        nargs=INSTRUMENT_COUNT,
        choices=(*INSTRUMENT_TOOL_MODES, *sorted(INSTRUMENT_TOOL_MODE_ALIASES)),
        default=None,
        metavar=("FIRST", "SECOND"),
        help=(
            "per-instrument mode for the two input devices; defaults to grasper for MiniMou "
            "and diathermy for other backends. Deprecated cutting aliases to diathermy."
        ),
    )
    parser.add_argument("--show-heat-overlay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diathermy-power", type=float, default=400.0)
    parser.add_argument("--heat-diffusion", type=float, default=0.25)
    parser.add_argument("--heat-cooling", type=float, default=0.10)
    parser.add_argument("--heat-substeps", type=int, default=1)
    parser.add_argument("--blade-length", "--blade-length-scale", dest="blade_length_scale", type=float, default=8.0)
    parser.add_argument("--blade-radius", "--blade-radius-scale", dest="blade_radius_scale", type=float, default=0.75)
    parser.add_argument("--shape-matching-stiffness", type=float, default=1.0)
    parser.add_argument("--shape-matching-relaxation", type=float, default=1.0)
    parser.add_argument("--shape-matching-passes", type=int, default=1)
    parser.add_argument(
        "--shape-matching-mode",
        choices=tuple(L0_SHAPE_MATCHING_MODE_BY_NAME),
        default=None,
        help="L0 shape-matching solve mode: scatter, gather, or gs.",
    )
    parser.add_argument(
        "--shape-matching-gs-weighting",
        choices=tuple(GS_WEIGHTING_BY_NAME),
        default="averaged",
        help="Colored-GS support weighting: averaged, sqrt, or full.",
    )
    parser.add_argument(
        "--shape-matching-gs-support-alpha",
        type=float,
        default=-1.0,
        help="Colored-GS support exponent in [0, 1]; negative uses the selected weighting preset.",
    )
    parser.add_argument(
        "--shape-matching-gather",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use the experimental particle-gather shape-matching apply path (default uses cluster scatter)",
    )
    parser.add_argument(
        "--shape-matching-computed-prolongation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compute hierarchy/sleep prolongation parents from grid coordinates instead of reading precomputed tables",
    )
    parser.add_argument(
        "--volume-preservation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable the optional L0 hexahedral volume-preservation pass",
    )
    parser.add_argument("--volume-preservation-stiffness", type=float, default=0.0)
    parser.add_argument("--volume-preservation-passes", type=int, default=1)
    parser.add_argument(
        "--sleep-l0-shape-matching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="skip L0 shape matching inside intact active outer8 hierarchy blocks and trilinearly project children",
    )
    parser.add_argument("--sleep-l0-wake-halo-blocks", type=int, default=1)
    parser.add_argument(
        "--hierarchical-shape-matching",
        choices=tuple(HIERARCHICAL_MODE_BY_NAME),
        default="outer8",
        help="L1 prepass mode: off, outer8 with prolongation, or full27 direct.",
    )
    parser.add_argument("--hierarchical-shape-matching-stiffness", type=float, default=1.0)
    parser.add_argument("--hierarchical-shape-matching-relaxation", type=float, default=1.0)
    parser.add_argument("--hierarchical-shape-matching-passes", type=int, default=1)
    parser.add_argument(
        "--hierarchical-shape-matching-gs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run the L1 hierarchy prepass as colored Gauss-Seidel instead of scatter Jacobi",
    )
    parser.add_argument(
        "--hierarchical-shape-matching-outer8-prolongation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply L1 Outer8 coarse-vertex corrections to skipped fine nodes by trilinear prolongation",
    )
    parser.add_argument(
        "--l2-hierarchical-shape-matching",
        choices=tuple(L2_HIERARCHICAL_MODE_BY_NAME),
        default="outer8",
        help="L2 prepass mode: off, outer8 with optional prolongation, or full125 direct over 4x4x4 blocks.",
    )
    parser.add_argument("--l2-hierarchical-shape-matching-stiffness", type=float, default=1.0)
    parser.add_argument("--l2-hierarchical-shape-matching-relaxation", type=float, default=1.0)
    parser.add_argument("--l2-hierarchical-shape-matching-passes", type=int, default=1)
    parser.add_argument(
        "--l2-hierarchical-shape-matching-gs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run the L2 hierarchy prepass as colored Gauss-Seidel instead of scatter Jacobi",
    )
    parser.add_argument(
        "--l2-hierarchical-shape-matching-outer8-prolongation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply L2 Outer8 coarse-vertex corrections to skipped fine nodes by trilinear prolongation",
    )
    parser.add_argument(
        "--hierarchical-shape-matching-outer8-absolute-projection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="set L1/L2 Outer8 prolonged children to absolute trilinear vertex positions instead of applying vertex deltas",
    )
    parser.add_argument("--particle-radius-scale", type=float, default=0.18)
    parser.add_argument("--node-render-radius-scale", type=float, default=0.22)
    parser.add_argument("--cell-render-radius-scale", type=float, default=0.5)
    parser.add_argument(
        "--drag-radius-scale",
        type=float,
        default=8.0,
        help="RMB drag selection radius in voxel widths.",
    )
    parser.add_argument(
        "--drag-pull-stiffness",
        type=float,
        default=1.0,
        help="RMB/instrument grab distance-constraint blend per solver iteration, in [0, 1].",
    )
    parser.add_argument(
        "--show-grab-constraints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show GL pull-point lines and particles for mouse and grasper grabs.",
    )
    parser.add_argument(
        "--plane-cut-depth-scale",
        type=float,
        default=8.0,
        help="Left Shift + MMB plane-cut ray length in voxel widths.",
    )
    parser.add_argument(
        "--ray-cut-depth-scale",
        type=float,
        default=8.0,
        help="Left Alt ray-cut depth in voxel widths.",
    )
    parser.add_argument(
        "--render-particles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show the cell-centred particle overlay (default OFF).",
    )
    parser.add_argument("--overlay-line-width", type=float, default=0.00035)
    parser.add_argument(
        "--cryo-texture",
        type=str,
        default="Digimouse/cryo_texture.npy",
        help="Packed 3D RGB volume from tools/build_cryo_texture.py. Pass empty string to disable texturing.",
    )
    parser.add_argument(
        "--cryo-renderer",
        choices=("volume", "atlas", "off"),
        default="volume",
        help="MC surface cryo texturing path: direct GL 3D volume, baked 2D atlas, or disabled.",
    )
    parser.add_argument(
        "--slang-procedural-material",
        type=str,
        default="",
        help="Material Maker 3D Slang export to hot-load for the Slang procedural surface.",
    )
    parser.add_argument(
        "--slang-procedural-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to UV3/procedural coordinates before sampling the Material Maker shader.",
    )
    parser.add_argument(
        "--slang-procedural-hot-reload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Watch and hot-reload the external Material Maker Slang shader and sibling textures.",
    )
    parser.add_argument(
        "--slang-environment-map",
        type=str,
        default="environments/photo_studio_01_1k.hdr",
        help="HDRI/equirectangular environment map sampled by the Slang surface shader. Pass empty string to disable.",
    )
    parser.add_argument(
        "--slang-environment-intensity",
        type=float,
        default=1.0,
        help="Brightness multiplier for the Slang HDRI environment map.",
    )
    parser.add_argument(
        "--slang-environment-background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the Slang HDRI environment map behind the scene.",
    )
    parser.add_argument(
        "--slang-environment-rotation-deg",
        type=float,
        default=0.0,
        help="Yaw rotation applied to Slang HDRI sampling and background, in degrees.",
    )
    parser.add_argument(
        "--slang-environment-pitch-deg",
        type=float,
        default=-90.0,
        help="Pitch rotation applied to Slang HDRI sampling and background, in degrees.",
    )
    parser.add_argument(
        "--color-by-segmentation-map",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Color the MC surface by Digimouse/material segmentation labels; takes precedence over cryo texturing.",
    )
    parser.add_argument(
        "--color-by-stress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Color the MC surface by per-cell stretch using a cold-to-warm ramp.",
    )
    parser.add_argument(
        "--stress-color-scale",
        type=float,
        default=30.0,
        help="Multiplier applied to per-cell strain before mapping stress colours.",
    )
    parser.add_argument(
        "--active-cut-fast-surface",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use cheaper surface shading while continuous/ray cutting is active.",
    )
    parser.add_argument(
        "--active-cut-smooth-normals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep smooth MC normals enabled during active continuous/ray cutting.",
    )
    parser.add_argument(
        "--active-cut-taubin-iterations",
        type=int,
        default=0,
        help="Taubin smoothing iterations while continuous/ray cutting is active.",
    )
    parser.add_argument("--atlas-tile-size", type=int, default=8)
    parser.add_argument(
        "--segmentation-panel-settings",
        type=str,
        default="segmentation_panel_settings.json",
        help="JSON path used by the Tissue Classes panel Load/Save buttons.",
    )
    parser.add_argument(
        "--exit-after-init",
        action="store_true",
        help="Print the startup phase breakdown and exit before entering the frame loop.",
    )


def _fail(parser: argparse.ArgumentParser | None, message: str) -> None:
    if parser is not None:
        parser.error(message)
    raise SystemExit(message)


def _has_arg(args: Any, name: str) -> bool:
    return hasattr(args, name)


def normalize_hex_runtime_args(args: argparse.Namespace, *, parser: argparse.ArgumentParser | None = None) -> bool:
    """Normalize and validate a parsed hex runtime namespace.

    The namespace is mutated in place. The return value records whether the
    normalized viewer is a Slang backend.
    """
    _normalize_hex_input_args(args, parser=parser)

    slang_procedural_material = getattr(args, "slang_procedural_material", "")
    if slang_procedural_material and str(getattr(args, "viewer", "")) == "gl":
        args.viewer = "slang"
    slang_viewer_requested = is_slang_backend(getattr(args, "viewer", ""))

    if getattr(args, "instrument_follow_camera", None) is None:
        args.instrument_follow_camera = _all_configured_backends(args, "fallback")

    if getattr(args, "instrument_tool_modes", None) is None:
        args.instrument_tool_modes = tuple(
            "grasper" if backend == "minimou" else "diathermy"
            for backend in (args.right_input_backend, args.left_input_backend)
        )
    try:
        args.instrument_tool_modes = tuple(
            normalize_instrument_tool_mode(mode) for mode in getattr(args, "instrument_tool_modes")
        )
    except ValueError as exc:
        _fail(parser, str(exc))

    if getattr(args, "usd", None) is not None and getattr(args, "frames", None) is None:
        _fail(parser, "--frames is required when using --usd")
    if getattr(args, "usd", None) is not None and slang_viewer_requested:
        _fail(parser, "--viewer slang cannot be combined with --usd")
    if _has_arg(args, "crop_visible_margin_voxels") and int(getattr(args, "crop_visible_margin_voxels")) < 0:
        _fail(parser, "--crop-visible-margin-voxels must be >= 0")
    if float(getattr(args, "instrument_radius_scale", 0.0)) < 0.0:
        _fail(parser, "--instrument-radius-scale must be >= 0")
    if int(getattr(args, "instrument_contact_iterations", 1)) < 1:
        _fail(parser, "--instrument-contact-iterations must be >= 1")
    if float(getattr(args, "instrument_max_correction_scale", 0.0)) < 0.0:
        _fail(parser, "--instrument-max-correction-scale must be >= 0")

    return bool(slang_viewer_requested)


def _normalize_hex_input_args(args: argparse.Namespace, *, parser: argparse.ArgumentParser | None = None) -> None:
    legacy_backend = getattr(args, "input_backend", None)
    try:
        if legacy_backend is not None:
            legacy_backend = normalize_input_backend(legacy_backend)
        right_backend = getattr(args, "right_input_backend", None)
        left_backend = getattr(args, "left_input_backend", None)
        args.right_input_backend = normalize_input_backend(
            right_backend if right_backend is not None else (legacy_backend or "fallback")
        )
        args.left_input_backend = normalize_input_backend(
            left_backend if left_backend is not None else (legacy_backend or "fallback")
        )
    except ValueError as exc:
        _fail(parser, str(exc))

    if getattr(args, "right_device_name", None) is None:
        args.right_device_name = getattr(args, "device_name", None) or "Default Device"
    if getattr(args, "left_device_name", None) is None:
        args.left_device_name = "Left Device"
    args.device_name = args.right_device_name
    if args.right_input_backend == args.left_input_backend:
        args.input_backend = args.right_input_backend
    else:
        args.input_backend = "mixed"


def _all_configured_backends(args: argparse.Namespace, backend: str) -> bool:
    return (
        str(getattr(args, "right_input_backend", "")) == backend
        and str(getattr(args, "left_input_backend", "")) == backend
    )
