#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_optic_flow_benchmark as optic  # noqa: E402


def apply_difficulty(args: argparse.Namespace) -> argparse.Namespace:
    preset = optic.DIFFICULTY_PRESETS[args.difficulty]
    for key, value in preset.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, value)
    return args


def spec_from_args(args: argparse.Namespace) -> optic.OpticFlowSpec:
    return optic.OpticFlowSpec(
        hex_rings=args.hex_rings,
        timesteps=args.timesteps,
        panorama_width=args.panorama_width,
        panorama_height=args.panorama_height,
        fov_azimuth_deg=args.fov_azimuth_deg,
        fov_elevation_deg=args.fov_elevation_deg,
        acceptance_angle_deg=args.acceptance_angle_deg,
        blur_samples=args.blur_samples,
        contrast=args.contrast,
        sensor_noise_std=args.sensor_noise_std,
        texture_mode=args.texture_mode,
        max_yaw_rate=args.max_yaw_rate,
        max_forward=args.max_forward,
        max_lateral=args.max_lateral,
        motion_scale=args.motion_scale,
    )


def _target_text(target: np.ndarray, step: int) -> str:
    current_yaw = float(target[0] * step)
    current_forward = float(target[1] * step)
    current_lateral = float(target[2] * step)
    return (
        f"per-step target\n"
        f"yaw rate: {target[0]:+.3f}\n"
        f"forward:  {target[1]:+.3f}\n"
        f"lateral:  {target[2]:+.3f}\n"
        f"\n"
        f"current cumulative\n"
        f"yaw:     {current_yaw:+.3f}\n"
        f"forward: {current_forward:+.3f}\n"
        f"lateral: {current_lateral:+.3f}"
    )


def _writer_for(path: Path, fps: int) -> animation.AbstractMovieWriter:
    suffix = path.suffix.lower()
    if suffix == ".gif":
        return animation.PillowWriter(fps=fps)
    if suffix in {".mp4", ".m4v", ".mov"}:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "Saving MP4/MOV requires ffmpeg. Install ffmpeg or use --output sample.gif."
            )
        return animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=1800,
            extra_args=["-pix_fmt", "yuv420p"],
        )
    raise ValueError("Output must end in .mp4, .m4v, .mov, or .gif")


def make_visualization(args: argparse.Namespace) -> Path:
    spec = spec_from_args(args)
    rng = np.random.default_rng(args.seed)
    batch = optic.generate_optic_flow_batch(spec, batch_size=1, rng=rng)
    frames = batch.inputs[0]
    targets = batch.targets[0]
    target = targets[0]
    lattice = optic.hex_lattice(spec.hex_rings)
    steps = np.arange(spec.timesteps, dtype=np.float32)
    cumulative_yaw = targets[:, 0] * steps
    cumulative_forward = targets[:, 1] * steps
    cumulative_lateral = targets[:, 2] * steps

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10.5, 5.8), dpi=args.dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0])
    ax_eye = fig.add_subplot(gs[:, 0])
    ax_translation = fig.add_subplot(gs[0, 1])
    ax_yaw = fig.add_subplot(gs[1, 1])

    marker_size = max(120, 900 / max(spec.hex_rings + 1, 1))
    scatter = ax_eye.scatter(
        lattice[:, 0],
        lattice[:, 1],
        c=frames[0],
        s=marker_size,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        edgecolors="#1f2937",
        linewidths=0.45,
    )
    ax_eye.set_aspect("equal")
    ax_eye.set_xticks([])
    ax_eye.set_yticks([])
    ax_eye.set_title("Input: fly-like hex retinal samples")
    time_text = ax_eye.text(
        0.02,
        0.98,
        "",
        transform=ax_eye.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 4},
    )
    cbar = fig.colorbar(scatter, ax=ax_eye, fraction=0.046, pad=0.02)
    cbar.set_label("sampled luminance")

    ax_translation.axhline(0, color="#9ca3af", linewidth=1)
    ax_translation.axvline(0, color="#9ca3af", linewidth=1)
    lim = 1.15 * max(
        float(np.max(np.abs(cumulative_forward))),
        float(np.max(np.abs(cumulative_lateral))),
        spec.max_forward,
        spec.max_lateral,
        1e-3,
    )
    ax_translation.set_xlim(-lim, lim)
    ax_translation.set_ylim(-lim, lim)
    ax_translation.set_aspect("equal")
    ax_translation.set_xlabel("cumulative lateral displacement")
    ax_translation.set_ylabel("cumulative forward displacement")
    ax_translation.set_title("Cumulative translation target")
    translation_arrow = ax_translation.quiver(
        [0.0],
        [0.0],
        [cumulative_lateral[0]],
        [cumulative_forward[0]],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#2563eb",
        width=0.018,
    )
    cumulative_trace, = ax_translation.plot([], [], color="#60a5fa", linewidth=2, alpha=0.75)

    max_yaw = max(float(np.max(np.abs(cumulative_yaw))), abs(float(target[0])), 1e-3)
    ax_yaw.set_xlim(-max_yaw * 1.15, max_yaw * 1.15)
    ax_yaw.set_ylim(-0.5, 0.5)
    ax_yaw.axvline(0, color="#9ca3af", linewidth=1)
    ax_yaw.set_yticks([])
    ax_yaw.set_xlabel("cumulative yaw")
    ax_yaw.set_title("Cumulative yaw target")
    yaw_bar = ax_yaw.barh(
        [0.0],
        [cumulative_yaw[0]],
        height=0.35,
        color="#dc2626" if target[0] >= 0 else "#7c3aed",
    )
    target_label = ax_yaw.text(
        0.02,
        -0.15,
        _target_text(target, 0),
        transform=ax_yaw.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9, "pad": 5},
    )

    fig.suptitle(
        f"Synthetic optic-flow training example ({args.difficulty}, seed={args.seed})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    def update(t: int):
        scatter.set_array(frames[t])
        cumulative_trace.set_data(cumulative_lateral[: t + 1], cumulative_forward[: t + 1])
        translation_arrow.set_UVC(
            np.asarray([cumulative_lateral[t]], dtype=np.float32),
            np.asarray([cumulative_forward[t]], dtype=np.float32),
        )
        time_text.set_text(
            f"timestep {t + 1}/{spec.timesteps}\n"
            f"input_dim={spec.input_dim}\n"
            f"contrast={spec.contrast:.2f}, noise={spec.sensor_noise_std:.2f}"
        )
        yaw_bar.patches[0].set_width(float(cumulative_yaw[t]))
        target_label.set_text(_target_text(targets[t], t))
        return scatter, translation_arrow, cumulative_trace, time_text, yaw_bar.patches[0], target_label

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=spec.timesteps,
        interval=1000 / args.fps,
        blit=False,
        repeat=True,
    )
    writer = _writer_for(args.output, args.fps)
    anim.save(args.output, writer=writer)
    plt.close(fig)
    return args.output


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a video/GIF visualizing synthetic optic-flow training data."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/optic_flow_training_sample.mp4"),
        help="Output .mp4/.mov/.m4v or .gif path.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", choices=tuple(optic.DIFFICULTY_PRESETS), default="medium")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--hex-rings", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument("--panorama-width", type=int, default=256)
    parser.add_argument("--panorama-height", type=int, default=96)
    parser.add_argument("--fov-azimuth-deg", type=float, default=150.0)
    parser.add_argument("--fov-elevation-deg", type=float, default=95.0)
    parser.add_argument("--acceptance-angle-deg", type=float, default=None)
    parser.add_argument("--blur-samples", type=int, default=None)
    parser.add_argument("--contrast", type=float, default=None)
    parser.add_argument("--sensor-noise-std", type=float, default=None)
    parser.add_argument("--texture-mode", choices=("naturalistic", "checker", "mixed"), default="mixed")
    parser.add_argument("--max-yaw-rate", type=float, default=0.25)
    parser.add_argument("--max-forward", type=float, default=0.55)
    parser.add_argument("--max-lateral", type=float, default=0.35)
    parser.add_argument("--motion-scale", type=float, default=0.55)
    args = apply_difficulty(parser.parse_args(argv))
    if args.timesteps < 2:
        parser.error("--timesteps must be at least 2")
    if args.hex_rings < 0:
        parser.error("--hex-rings must be non-negative")
    if args.fps < 1:
        parser.error("--fps must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output = make_visualization(args)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
