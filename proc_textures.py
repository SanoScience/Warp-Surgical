"""Procedural texture generation using Nvidia Warp."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import warp as wp

wp.init()

NOISE_OCTAVES = 6


@wp.func
def _fract(value: float) -> float:
    return value - wp.floor(value)


@wp.func
def _hash_vec2(cell: wp.vec2f) -> wp.vec2f:
    dot1 = wp.dot(cell, wp.vec2f(127.1, 311.7))
    dot2 = wp.dot(cell, wp.vec2f(269.5, 183.3))
    sin1 = wp.sin(dot1) * 43758.5453123
    sin2 = wp.sin(dot2) * 43758.5453123
    return wp.vec2f(_fract(sin1), _fract(sin2))


@wp.func
def fbm_noise(uv: wp.vec2f, time: float, base_frequency: float, gain: float, lacunarity: float) -> float:
    amplitude = 0.5
    frequency = base_frequency
    total_weight = 0.0
    value = 0.0

    for octave in range(NOISE_OCTAVES):
        angle = (0.7548776662466927 * float(octave)) + time * 0.37
        direction = wp.vec2f(wp.cos(angle), wp.sin(angle))
        warped_uv = uv * frequency + direction * time * 0.05

        sine_term = wp.sin(wp.dot(warped_uv, direction * 3.0))
        perp_direction = wp.vec2f(-direction.y, direction.x)
        cosine_term = wp.cos(wp.dot(warped_uv, perp_direction * 3.0))

        value += amplitude * 0.7 * sine_term
        value += amplitude * 0.3 * cosine_term
        total_weight += amplitude

        frequency *= lacunarity
        amplitude *= gain

    normalized = value / total_weight
    return 0.5 + 0.5 * normalized


@wp.func
def worley_noise(
    uv: wp.vec2f,
    time: float,
    cell_frequency: float,
    jitter: float,
    softness: float,
) -> float:
    frequency = wp.max(cell_frequency, 0.0001)
    softness_clamped = wp.max(softness, 0.001)
    jitter_clamped = wp.clamp(jitter, 0.0, 1.0)

    cell_uv = uv * frequency
    base_cell = wp.vec2f(wp.floor(cell_uv.x), wp.floor(cell_uv.y))

    accum = 0.0
    for offset_x in range(-1, 2):
        for offset_y in range(-1, 2):
            neighbor_cell = base_cell + wp.vec2f(float(offset_x), float(offset_y))
            hash_seed = neighbor_cell + wp.vec2f(time * 0.23, time * 0.41)
            random_offset = _hash_vec2(hash_seed) - wp.vec2f(0.5, 0.5)
            feature = neighbor_cell + random_offset * jitter_clamped
            delta = feature - cell_uv
            distance = wp.length(delta)
            accum += wp.exp(-distance * softness_clamped)

    soft_min = -wp.log(accum + 1.0e-6) / softness_clamped
    contrast = 1.5 + 0.5 * softness_clamped
    value = wp.exp(-soft_min * contrast)
    return value

@wp.kernel
def generate_noise_kernel(
    width: int,
    height: int,
    inv_width: float,
    inv_height: float,
    scale: float,
    time: float,
    base_frequency: float,
    gain: float,
    lacunarity: float,
    worley_weight: float,
    worley_frequency: float,
    worley_jitter: float,
    worley_softness: float,
    perlin_weight: float,
    perlin_frequency: float,
    perlin_seed: int,
    out_image: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    x = tid % width
    y = tid // width

    uv = wp.vec2f((float(x) + 0.5) * inv_width, (float(y) + 0.5) * inv_height)
    uv = (uv - wp.vec2f(0.5, 0.5)) * scale

    fbm_value = fbm_noise(uv, time, base_frequency, gain, lacunarity)
    worley_value = worley_noise(uv, time, worley_frequency, worley_jitter, worley_softness)

    worley_mix = fbm_value + (worley_value - fbm_value) * wp.clamp(worley_weight, 0.0, 1.0)

    perlin_rng = wp.rand_init(perlin_seed, tid)
    perlin_pos = wp.vec3f(uv.x * perlin_frequency, uv.y * perlin_frequency, time)
    perlin_value = 0.5 + 0.5 * wp.noise(perlin_rng, perlin_pos)

    blended = worley_mix + (perlin_value - worley_mix) * wp.clamp(perlin_weight, 0.0, 1.0)
    out_image[tid] = blended


@dataclass
class NoiseConfig:
    width: int = 512
    height: int = 512
    scale: float = 3.0
    time: float = 0.0
    base_frequency: float = 1.0
    gain: float = 0.5
    lacunarity: float = 2.0
    worley_weight: float = 0.0
    worley_frequency: float = 3.0
    worley_jitter: float = 0.75
    worley_softness: float = 4.0
    perlin_weight: float = 0.0
    perlin_frequency: float = 2.0
    perlin_seed: int = 1337
    device: Optional[str] = None
    show: bool = True
    save_path: Optional[str] = None
    interactive: bool = True


def _launch_noise_kernel(config: NoiseConfig, buffer: wp.array) -> None:
    wp.launch(
        kernel=generate_noise_kernel,
        dim=config.width * config.height,
        inputs=[
            config.width,
            config.height,
            1.0 / float(config.width),
            1.0 / float(config.height),
            config.scale,
            config.time,
            config.base_frequency,
            config.gain,
            config.lacunarity,
            config.worley_weight,
            config.worley_frequency,
            config.worley_jitter,
            config.worley_softness,
            config.perlin_weight,
            config.perlin_frequency,
            int(config.perlin_seed),
            buffer,
        ],
    )


def _generate_noise(
    config: NoiseConfig, buffer: Optional[wp.array] = None
) -> tuple[np.ndarray, wp.array]:
    expected_size = config.width * config.height

    with wp.ScopedDevice(config.device):
        if buffer is None or buffer.shape[0] != expected_size:
            buffer = wp.empty(expected_size, dtype=wp.float32)

        _launch_noise_kernel(config, buffer)
        wp.synchronize()

    image = buffer.numpy().reshape((config.height, config.width))
    return image, buffer


def generate_noise_image(config: NoiseConfig) -> np.ndarray:
    image, _ = _generate_noise(config)
    return image


def _static_display(image: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        print(f"matplotlib is not available ({exc}); skipping display.")
        return

    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    ax.axis("off")

    try:
        plt.show()
    except Exception as exc:  # pragma: no cover - backend specific
        print(f"Could not open a display window ({exc}); consider using --save.")


def _interactive_display(
    initial_image: np.ndarray, config: NoiseConfig, buffer: Optional[wp.array]
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        print(f"matplotlib is not available ({exc}); falling back to static display.")
        _static_display(initial_image)
        return

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    plt.subplots_adjust(bottom=0.68)
    image_artist = ax.imshow(initial_image, cmap="gray", vmin=0.0, vmax=1.0)
    ax.axis("off")

    slider_specs = [
        ("Scale", "scale", 0.2, 10.0, config.scale),
        ("Time", "time", -5.0, 5.0, config.time),
        ("Base Freq", "base_frequency", 0.1, 5.0, config.base_frequency),
        ("Gain", "gain", 0.1, 0.9, config.gain),
        ("Lacunarity", "lacunarity", 1.0, 4.0, config.lacunarity),
        ("W Weight", "worley_weight", 0.0, 1.0, config.worley_weight),
        ("W Freq", "worley_frequency", 0.5, 15.0, config.worley_frequency),
        ("W Jitter", "worley_jitter", 0.0, 1.0, config.worley_jitter),
        ("W Soft", "worley_softness", 1.0, 12.0, config.worley_softness),
        ("P Weight", "perlin_weight", 0.0, 1.0, config.perlin_weight),
        ("P Freq", "perlin_frequency", 0.1, 12.0, config.perlin_frequency),
    ]

    sliders: dict[str, Slider] = {}
    slider_top = 0.62
    slider_height = 0.028
    slider_gap = 0.038

    for index, (label, attr, vmin, vmax, vinit) in enumerate(slider_specs):
        ax_slider = fig.add_axes([0.15, slider_top - index * slider_gap, 0.7, slider_height])
        slider = Slider(ax_slider, label, vmin, vmax, valinit=vinit)
        sliders[attr] = slider

    noise_buffer = buffer
    if noise_buffer is None or noise_buffer.shape[0] != config.width * config.height:
        updated_image, noise_buffer = _generate_noise(config, noise_buffer)
        image_artist.set_data(updated_image)
    else:
        image_artist.set_data(initial_image)

    def refresh(_value: float) -> None:
        nonlocal noise_buffer

        for attr, slider in sliders.items():
            setattr(config, attr, slider.val)

        updated_image, noise_buffer = _generate_noise(config, noise_buffer)
        image_artist.set_data(updated_image)
        fig.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(refresh)

    try:
        plt.show()
    except Exception as exc:  # pragma: no cover - backend specific
        print(f"Could not open a display window ({exc}); consider using --save.")


def display_image(image: np.ndarray, config: NoiseConfig, buffer: Optional[wp.array] = None) -> None:
    if config.save_path is not None:
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "Saving requires pillow; install it or omit --save"
            ) from exc

        image_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(image_uint8, mode="L").save(config.save_path)
        print(f"Saved noise image to {config.save_path}")

    if not config.show:
        return

    if config.interactive:
        _interactive_display(image, config, buffer)
    else:
        _static_display(image)


def parse_args() -> NoiseConfig:
    parser = argparse.ArgumentParser(description="Generate differentiable procedural noise using Warp")
    parser.add_argument("--width", type=int, default=512, help="Output image width in pixels")
    parser.add_argument("--height", type=int, default=512, help="Output image height in pixels")
    parser.add_argument("--scale", type=float, default=3.0, help="Domain scale applied to UV coordinates")
    parser.add_argument("--time", type=float, default=0.0, help="Animation time parameter for the noise field")
    parser.add_argument("--base-frequency", type=float, default=1.0, help="Base frequency multiplier for the noise")
    parser.add_argument("--gain", type=float, default=0.5, help="Amplitude falloff between octaves")
    parser.add_argument("--lacunarity", type=float, default=2.0, help="Frequency growth between octaves")
    parser.add_argument("--worley-weight", type=float, default=0.0, help="Blend factor for Worley noise (0 disables it)")
    parser.add_argument("--worley-frequency", type=float, default=3.0, help="Cell density for Worley noise")
    parser.add_argument("--worley-jitter", type=float, default=0.75, help="Jitter strength for Worley feature points")
    parser.add_argument("--worley-softness", type=float, default=4.0, help="Soft-min sharpness for Worley distance aggregation")
    parser.add_argument("--perlin-weight", type=float, default=0.0, help="Blend factor for Perlin noise (0 disables it)")
    parser.add_argument("--perlin-frequency", type=float, default=2.0, help="Frequency multiplier for Perlin noise sampling")
    parser.add_argument("--perlin-seed", type=int, default=1337, help="Seed for Perlin noise RNG initialization")
    parser.add_argument("--device", type=str, default=None, help="Override Warp device (cpu, cuda:0, etc.)")
    parser.add_argument("--no-show", action="store_true", help="Skip displaying the generated image")

    interactive_group = parser.add_mutually_exclusive_group()
    interactive_group.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        help="Enable interactive sliders (default when showing)",
    )
    interactive_group.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="Disable interactive sliders and show a static image",
    )
    parser.set_defaults(interactive=True)

    parser.add_argument("--save", type=str, default=None, help="Optional path to save the image")

    args = parser.parse_args()

    return NoiseConfig(
        width=args.width,
        height=args.height,
        scale=args.scale,
        time=args.time,
        base_frequency=args.base_frequency,
        gain=args.gain,
        lacunarity=args.lacunarity,
        worley_weight=args.worley_weight,
        worley_frequency=args.worley_frequency,
        worley_jitter=args.worley_jitter,
        worley_softness=args.worley_softness,
        perlin_weight=args.perlin_weight,
        perlin_frequency=args.perlin_frequency,
        perlin_seed=args.perlin_seed,
        device=args.device,
        show=not args.no_show,
        save_path=args.save,
        interactive=args.interactive,
    )


def main() -> None:
    config = parse_args()
    image, buffer = _generate_noise(config)
    display_image(image, config, buffer)


if __name__ == "__main__":
    main()
