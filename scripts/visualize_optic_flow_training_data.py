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


def target_flow_field(
    lattice: np.ndarray,
    target: np.ndarray,
    spec: optic.OpticFlowSpec,
    arrow_length: float,
) -> np.ndarray:
    """Clean target-implied optic-flow sketch in normalized lattice coordinates."""
    yaw_rate, forward, lateral = map(float, target)
    az = lattice[:, 0] * math.radians(spec.fov_azimuth_deg) / 2.0
    el = lattice[:, 1] * math.radians(spec.fov_elevation_deg) / 2.0
    depth_gain = 0.65 + 0.35 * np.cos(el)
    horizontal = (
        yaw_rate
        + spec.motion_scale * (forward * np.sin(az) - lateral * np.cos(az)) * depth_gain
    )
    vertical = spec.motion_scale * forward * np.sin(el) * np.cos(az)
    flow = np.stack([horizontal, vertical], axis=1).astype(np.float32)
    denom = float(np.percentile(np.linalg.norm(flow, axis=1), 90))
    if denom <= 1e-8:
        return np.zeros_like(flow)
    return (flow / denom * arrow_length).astype(np.float32)


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
    analytic_flow = target_flow_field(
        lattice,
        target,
        spec,
        arrow_length=args.flow_arrow_length,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.2, 5.9), dpi=args.dpi, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0])
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
    flow_arrows = None
    if args.show_flow_arrows:
        flow_arrows = ax_eye.quiver(
            lattice[:, 0],
            lattice[:, 1],
            analytic_flow[:, 0],
            analytic_flow[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#2563eb",
            alpha=0.78,
            width=0.006,
            headwidth=4.0,
            headlength=5.0,
            headaxislength=4.2,
            zorder=5,
        )
    ax_eye.set_aspect("equal")
    ax_eye.set_xlim(-1.28, 1.28)
    ax_eye.set_ylim(-1.28, 1.28)
    ax_eye.set_xticks([])
    ax_eye.set_yticks([])
    ax_eye.set_title("Hex retinal input", fontsize=13, pad=8)
    time_text = ax_eye.text(
        0.02,
        0.98,
        "",
        transform=ax_eye.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
    )

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
    ax_translation.set_xlabel("lateral")
    ax_translation.set_ylabel("forward")
    translation_title = ax_translation.set_title("Translation target", fontsize=13)
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
    ax_yaw.set_xlabel("yaw")
    yaw_title = ax_yaw.set_title("Yaw target", fontsize=13)
    yaw_bar = ax_yaw.barh(
        [0.0],
        [cumulative_yaw[0]],
        height=0.35,
        color="#dc2626" if target[0] >= 0 else "#7c3aed",
    )

    fig.suptitle(
        f"Synthetic optic-flow training example ({args.difficulty}, seed={args.seed})",
        fontsize=13,
    )

    def update(t: int):
        scatter.set_array(frames[t])
        cumulative_trace.set_data(cumulative_lateral[: t + 1], cumulative_forward[: t + 1])
        translation_arrow.set_UVC(
            np.asarray([cumulative_lateral[t]], dtype=np.float32),
            np.asarray([cumulative_forward[t]], dtype=np.float32),
        )
        time_text.set_text(
            f"frame {t + 1}/{spec.timesteps}\n"
            f"samples {spec.input_dim}\n"
            f"contrast {spec.contrast:.2f}, noise {spec.sensor_noise_std:.2f}\n"
            f"blue arrows: target-implied flow"
        )
        yaw_bar.patches[0].set_width(float(cumulative_yaw[t]))
        translation_title.set_text(
            f"Translation: fwd {cumulative_forward[t]:+.2f}, lat {cumulative_lateral[t]:+.2f}\n"
            f"per step: fwd {target[1]:+.2f}, lat {target[2]:+.2f}"
        )
        yaw_title.set_text(f"Yaw: {cumulative_yaw[t]:+.2f} rad | rate {target[0]:+.2f}")
        return (
            scatter,
            flow_arrows,
            translation_arrow,
            cumulative_trace,
            time_text,
            yaw_bar.patches[0],
            translation_title,
            yaw_title,
        )

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
    parser.add_argument(
        "--show-flow-arrows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay a clean optic-flow sketch implied by the target motion.",
    )
    parser.add_argument("--flow-arrow-length", type=float, default=0.16)
    args = apply_difficulty(parser.parse_args(argv))
    if args.timesteps < 2:
        parser.error("--timesteps must be at least 2")
    if args.hex_rings < 0:
        parser.error("--hex-rings must be non-negative")
    if args.fps < 1:
        parser.error("--fps must be positive")
    if args.flow_arrow_length <= 0:
        parser.error("--flow-arrow-length must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output = make_visualization(args)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
