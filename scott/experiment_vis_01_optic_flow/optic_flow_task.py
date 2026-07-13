#!/usr/bin/env python3
"""Experiment vis-01 -- the naturalistic optic-flow / self-motion estimation task (the scientific
heart of the experiment; reimplemented fresh + self-contained -- it does NOT import anything under
scripts/flow/).

WHAT THE TASK IS
----------------
A fly-like hexagonal ommatidial eye moves through a 3D scene under smoothly time-varying ego-motion.
At every timestep the network sees the eye's luminance image (one scalar per ommatidium) and must
regress the *instantaneous* 5-DOF self-motion vector

    target[t] = [ yaw_rate, forward, lateral, roll_rate, pitch_rate ]   (body frame, at time t)

This is a per-timestep regression (MSE loss). Because the motion evolves over the clip, a single
frame is not enough -- the estimate must be read out of the *temporal* pattern of image motion
(optic flow), which is exactly what makes the recurrence load-bearing (see the verifier ablations in
run_experiment.py: time-shuffle and single-frame both collapse).

WHY THE SCENE HAS REAL DEPTH (the parallax that separates translation from rotation)
-----------------------------------------------------------------------------------
Rotational flow is depth-independent (everything sweeps by at the same angular rate); translational
flow is depth-dependent (near things move more -- motion parallax). A network can only disambiguate
"I turned" from "I strafed" if the scene carries depth. So the scene is genuinely 3D:
  * a far panoramic BACKGROUND at infinite depth (1/f naturalistic texture) -> pure rotational flow,
  * a textured GROUND PLANE below the eye (1/f texture) -> depth that grows with distance to horizon,
  * a set of near textured OBJECTS at sampled depths -> strong parallax + (optionally) their own
    independent motion, injecting local flow INCONSISTENT with ego-motion (distractors to suppress).
The `no_parallax` ablation flattens ground+objects to infinite depth (direction-only sampling): under
it, translation becomes unreadable while rotation survives -- the physical check that depth is doing
real work.

SENSOR (fly-like, with acceptance-angle blur)
---------------------------------------------
Ommatidia sit on a hexagonal lattice over the eye's field of view; each integrates light over a small
Gaussian acceptance cone (optical blur), approximated by a weighted set of sub-rays per ommatidium.
`input_dim = #ommatidia` (a function of `hex_rings`).

PUBLIC SURFACE (kept parallel to Exp-5/6's task modules so the shared engine reuses it by import):
  * ``EpisodeSpec``          -- every knob, as a frozen dataclass (calibration sweeps these).
  * ``make_scene_bank``      -- the FIXED world statistics (1/f Fourier features) shared by all
                                episodes (the analogue of Exp-5/6's fixed odor bank).
  * ``generate_batch`` / ``Batch`` -- one batch of episodes (+ the ablation hooks the verifier uses).
  * ``batch_to_torch``       -- numpy episode -> (inputs, targets, loss_mask) device tensors.
  * ``masked_mse`` / ``dof_rmse_r2`` / ``mean_r2`` -- the regression loss + per-DOF RMSE/R² metrics.
  * ``naive_baseline_r2``    -- the achievable floor: a least-squares frame-difference linear decoder.
  * ``N_DOF`` / ``DOF_NAMES``.
Video: ``render_episode_video`` (mp4 via imageio, gif fallback) and ``render_sanity_clips``
(single-DOF pure-yaw / pure-forward / ... clips) so a human can WATCH the stimuli and confirm the
physics (correct parallax, correct flow direction per DOF, objects moving independently).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

# --- the physical BODY 5-DOF (drives rendering + retinal flow; NOT all directly scored) ------------
BODY_DOF = ("yaw_rate", "forward", "lateral", "roll_rate", "pitch_rate")
N_BODY = 5
BODY_SCALE = np.array([1.2, 0.9, 0.6, 0.7, 0.7], dtype=np.float32)  # yaw, fwd, lat, roll, pitch (rad/s, m/s)

# --- the REGRESSION TARGET = candidate scored channels (per timestep) -------------------------------
# CANDIDATE SET (the strong-model gate + object-density sweep decide which actually clear the floor):
#   * yaw_rate / roll_rate / pitch_rate -- depth-independent rotational rates, directly observable.
#   * forward_v / lateral_v            -- ABSOLUTE translational velocity (m/s). Not recoverable from
#     a single flow frame (flow = velocity/depth), BUT recoverable STATISTICALLY under DENSE static
#     clutter with a FIXED depth prior: the net learns p(Z) and reads v off the flow-field statistics
#     (near-object / high-flow tail), with estimator variance shrinking as clutter density rises. This
#     LEANS ON the learned depth prior -- it breaks if the depth distribution changes (stated honestly).
#   * heading_az   -- azimuth of the focus of expansion (travel direction); observable, scale-free.
#   * ventral_flow -- ground optic-flow magnitude |v_ground|/h (h = height above ground); the classic
#     insect flow-regulation variable (David 1982; Srinivasan 1996; Baird 2005/2013). Observable
#     regardless of the depth prior (it IS the ground image-flow rate), unlike absolute v.
TARGET_NAMES = ("yaw_rate", "roll_rate", "pitch_rate", "forward_v", "lateral_v", "heading_az",
                "ventral_flow")
N_TARGETS = 7
# back-compat aliases (common.py / model / metrics operate on the SCORED TARGET vector):
DOF_NAMES = TARGET_NAMES
N_DOF = N_TARGETS
DOF_SCALE = BODY_SCALE     # legacy alias for the "ou" mode + single-DOF sanity clips (physical 5-DOF)

# --- TRIAL-TYPE split (continuous mode) --------------------------------------------------------------
# Turning produces large whole-field image motion that swamps the small motion from translating, so
# mixing them makes translation unreadable. Separating trials lets each be measured cleanly (and mirrors
# real flies: turn in bursts, translate in between). Each episode is tagged with a trial type; each
# channel is scored ONLY on the trials where it actually varies.
#   * "turn"      -- rotational rates vary (yaw/roll/pitch OU), translation ~ 0.
#   * "translate" -- translation varies (cruise forward + lateral sideslip), rotational rates ~ 0.
#   * "mixed"     -- both vary (the legacy regime; used by saccade/ou modes and any mixed fraction).
TRIAL_TYPE_NAMES = ("turn", "translate", "mixed")
TRIAL_TURN, TRIAL_TRANSLATE, TRIAL_MIXED = 0, 1, 2
# default channels scored per trial type (rotation on turn; the observable translation cues on translate)
DEFAULT_SCORED_TURN = ("yaw_rate", "roll_rate", "pitch_rate")
DEFAULT_SCORED_TRANSLATE = ("ventral_flow", "heading_az")
# body-frame trajectory columns (DOF_NAMES order of _continuous_trajectory / BODY_DOF):
_ROT_BODY_COLS = (0, 3, 4)     # yaw_rate, roll_rate, pitch_rate
_TRANS_BODY_COLS = (1, 2)      # forward, lateral


# --------------------------------------------------------------------------------------
# episode geometry + difficulty knobs  (the calibration ladder lives here)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EpisodeSpec:
    # --- sensor (fly hex eye) ---
    hex_rings: int = 6                 # #ommatidia = 1 + 3R(R+1); R=6 -> 127, R=8 -> 217
    fov_az_deg: float = 150.0          # horizontal field of view
    fov_el_deg: float = 100.0          # vertical field of view
    accept_sigma_deg: float = 3.5      # Gaussian acceptance half-angle (optical blur)
    # blur_rings=0 (default): acceptance blur applied ANALYTICALLY in the frequency domain (the optical
    # MTF is a Gaussian in spatial frequency) -> center-ray sampling, ~7x faster, the training path.
    # blur_rings>=1: additionally cast geometric sub-rays (accurate at depth edges) -- used for video.
    blur_rings: int = 0
    # --- clip / integration ---
    # dt=0.02 (50 fps) resolves the ~50-80 ms body-yaw saccades over a few frames; seq_len=64 -> a
    # ~1.3 s clip carrying ~1-2 saccades separated by translation-dominated intersaccadic intervals.
    seq_len: int = 64                  # T timesteps (frames) per episode
    dt: float = 0.02                   # seconds per frame (50 fps; resolves saccades)
    substeps: int = 3                  # motion sub-integration steps per frame (physical smoothness)
    warmup: int = 4                    # initial frames excluded from the loss (flow needs history)
    # --- ego-motion mode ---
    # "continuous" (DEFAULT): the classic OPTOMOTOR regime -- smooth continuously time-varying rotation
    # on ALL THREE axes (yaw/roll/pitch) at comparable per-axis variance (bounded OU rotational rates),
    # concurrent with a translating cruise. No saccades, no gaze transform (retinal = body): removes the
    # saccade-detection degeneracy of the earlier design. "saccade_fixate" = the Drosophila saccade +
    # gaze-stabilization mode (kept available, OFF by default). "ou" = legacy low-pass mode.
    motion_mode: str = "continuous"
    ou_tau: float = 0.35               # ("ou" mode) OU correlation time (s)
    rot_trans_balance: float = 1.0     # ("ou" mode) rotation/translation scale trade
    motion_gain: float = 1.0           # global multiplier on the ego-motion scales (difficulty)
    # --- TRIAL-TYPE split (continuous mode): per-batch fraction of turn-only vs translate-only trials.
    # turn-only zeroes translation; translate-only zeroes rotation; the remainder (1 - turn - translate)
    # are "mixed" (both vary). Default = half turn / half translate, no mixed. Non-continuous modes are
    # always "mixed" (no split). ---
    trial_frac_turn: float = 0.5       # fraction of episodes that are turn-only (rotation varies)
    trial_frac_translate: float = 0.5  # fraction of episodes that are translate-only (translation varies)
    # --- continuous-rotation (optomotor) kinematics: smooth OU rotational rates, biologically bounded,
    # COMPARABLE variance per axis so no rotational channel dominates the loss/gradient. ---
    rot_rate_dps: float = 60.0         # OU std of EACH rotational rate (yaw/roll/pitch), deg/s
    rot_tau: float = 0.30              # rotational-rate OU correlation time (s)
    # Which rotational axes actually VARY in continuous mode. "all" = yaw+roll+pitch (default, the full
    # 3-axis optomotor regime). "yaw" = yaw-only (roll & pitch held at 0): the reduced 1-D de-risk task
    # with a measured strong-model ceiling. Only affects motion_mode="continuous".
    rot_axes: str = "all"
    # --- saccade-fixate body kinematics (defaults cite SACCADE_STATS.md §3, Drosophila free cruise) ---
    saccade_rate_hz: float = 1.2       # ~1 saccade/s open cruise (Censi 2013: 1.37/s); ISI ~0.7-1 s
    saccade_dur_s: float = 0.08        # ~50-80 ms (Muijres 2015 free 49+/-18 ms; tethered ~80-100 ms)
    saccade_amp_deg: float = 90.0      # modal ~90 deg (Tammero & Dickinson 2002; Muijres 2015 93+/-27)
    saccade_amp_jitter_deg: float = 30.0
    roll_bank_deg: float = 30.0        # transient bank per saccade, roll->counter-roll (Muijres 2015)
    forward_speed: float = 0.5         # cruise ~0.5 m/s (Medici & Fry 2012 range 0.2-0.9)
    forward_speed_jitter: float = 0.2  # slow OU variation of the cruise speed
    sideslip_speed: float = 0.06       # sideslip small/transient, actively suppressed (Muijres 2015)
    residual_yaw_dps: float = 20.0     # LOAD-BEARING knob: intersaccadic residual body yaw rate (deg/s).
                                       # Unpublished for D. melanogaster (SACCADE_STATS §6.1); blowfly
                                       # proxy 0-100 deg/s head yaw. Lower -> cleaner translational flow.
    # PITCH DYNAMICS: give pitch genuine variance (climbs/dives) so it is OBSERVABLE (it failed before
    # purely for lack of variance). OU pitch-rate std; body pitch stays biologically bounded ~45-55 deg
    # cruising (Medici & Fry 2012; Ristroph et al. 2013 correct ~20 deg perturbations in ~60 ms).
    pitch_rate_dps: float = 45.0       # OU std of body pitch-rate (deg/s) -- modest climb/dive excursions
    pitch_tau: float = 0.4             # pitch-rate correlation time (s)
    # --- gaze stabilization -> the RETINAL rotation (target stays BODY self-motion) ---
    # Per-axis SUB-UNITY stabilization gains applied to the SMOOTH INTERSACCADIC rotational slip ONLY;
    # saccadic transients pass through un-stabilized (the head saccade). retinal_slip = (1-gain)*body.
    # Translation is NOT attenuated. LOCKED defaults (pitch is a placeholder, calibration-swept):
    gaze_gain_yaw: float = 0.70        # ~70% intersaccadic yaw slip reduction (Cellini/Salem/Mongeau 2021)
    gaze_gain_roll: float = 0.90       # roll near-fully compensated (van Hateren & Schilstra 1999; Beatus 2015)
    gaze_gain_pitch: float = 0.65      # PLACEHOLDER: pitch gaze gain unmeasured in Drosophila (swept in calibration)
    # --- scene: ground + background + objects ---
    # ALTITUDE is drawn per episode in [altitude_lo, altitude_hi] (flight height above the ground). This
    # is what makes the ventral-flow surrogate genuinely OBSERVABLE: v/h is read straight off the ground
    # image flow, while absolute v (= (v/h)*h) needs the unknown, per-episode-varying h. ground_height is
    # the fallback constant altitude when altitude_lo == altitude_hi.
    ground_height: float = 1.2         # eye height above the ground plane (m); fallback / mean altitude
    altitude_lo: float = 0.6           # per-episode altitude range (m) -- varied so v/h != scaled v
    altitude_hi: float = 2.0
    ground_tex_scale: float = 0.7      # spatial-frequency scale of the ground 1/f texture
    bg_tex_scale: float = 1.1          # spatial-frequency scale of the background 1/f texture
    tex_octaves: int = 5               # #Fourier components in each 1/f texture
    tex_beta: float = 1.0              # 1/f exponent (amp ~ 1/|f|^beta); 1.0 = pink
    contrast: float = 1.0              # global luminance contrast (difficulty: lower = harder)
    # --- DENSE STATIC foreground clutter (near-field, NON-moving) drawn each episode from a FIXED depth
    # distribution. This is the mechanism that makes absolute translational velocity recoverable
    # STATISTICALLY: many static objects at a constant depth prior let the net learn p(Z) and read v off
    # the flow-field statistics. Rendered with correct occlusion (nearer occludes farther) + motion
    # boundaries (flow discontinuities at object edges) via depth compositing. ---
    n_clutter: int = 48                # number of STATIC near-field clutter objects (the density knob)
    clutter_depth_lo: float = 0.3      # FIXED clutter depth distribution (m); uniform in [lo, hi]
    clutter_depth_hi: float = 3.0
    obj_phys_radius: float = 0.12      # physical radius of a clutter object (m) -> angular size = r/depth
    #   (so nearer objects subtend a LARGER solid angle -- correct occlusion coverage & parallax).
    # --- independently-moving distractors: a SEPARATE knob (default OFF for vis_01; reserved vis_02) ---
    n_moving_distractors: int = 0      # objects with their OWN velocity (inconsistent flow to suppress)
    n_objects: int = 4                 # LEGACY moving-object count (only used by motion_mode!=continuous)
    obj_ang_radius_deg: float = 9.0    # LEGACY fixed angular radius (saccade/ou modes)
    obj_depth_lo: float = 0.6          # LEGACY object depth range (saccade/ou modes)
    obj_depth_hi: float = 3.0
    obj_speed: float = 0.5             # self-motion speed (m/s) of moving distractors / legacy objects
    # --- sensor noise ---
    sensor_noise_std: float = 0.03     # additive Gaussian noise on ommatidial luminance (difficulty)

    # ---- derived ----
    @property
    def n_ommatidia(self) -> int:
        R = self.hex_rings
        return 1 + 3 * R * (R + 1)

    @property
    def input_dim(self) -> int:
        return self.n_ommatidia

    @property
    def timesteps(self) -> int:
        return self.seq_len


@dataclass(frozen=True)
class Batch:
    inputs: np.ndarray     # [B, T, n_ommatidia]  luminance movie
    targets: np.ndarray    # [B, T, N_DOF]        instantaneous 5-DOF self-motion (body frame)
    loss_mask: np.ndarray  # [B, T]               1 after warmup (scored steps)
    trial_type: np.ndarray = None  # [B] int trial-type code (0=turn, 1=translate, 2=mixed); tags which
    #                                channels are scored (see dof_score_mask / resolve_scored_map)


# --------------------------------------------------------------------------------------
# sensor geometry: hex ommatidial lattice + acceptance-angle blur sub-rays
# --------------------------------------------------------------------------------------
def _hex_axial_coords(rings: int) -> np.ndarray:
    """Axial (q, r) hex coordinates within `rings` rings of the origin (pointy-top layout)."""
    coords = []
    for q in range(-rings, rings + 1):
        r_lo = max(-rings, -q - rings)
        r_hi = min(rings, -q + rings)
        for r in range(r_lo, r_hi + 1):
            coords.append((q, r))
    return np.asarray(coords, dtype=np.float64)


def _dir_from_azel(az: np.ndarray, el: np.ndarray) -> np.ndarray:
    """Unit viewing directions from azimuth/elevation (radians). Camera frame: +Z forward, +Y up,
    +X right. az>0 -> right, el>0 -> up."""
    ce = np.cos(el)
    return np.stack([ce * np.sin(az), np.sin(el), ce * np.cos(az)], axis=-1)


def build_sensor(spec: EpisodeSpec) -> dict:
    """Return the ommatidial sensor: center directions [N,3] and the acceptance-blur sub-ray
    directions [N, S, 3] with Gaussian weights [S]. Deterministic (geometry only)."""
    axial = _hex_axial_coords(spec.hex_rings)
    # axial -> planar hex pixel coords, then scale into the FOV (angular) rectangle.
    px = 1.5 * axial[:, 0]
    py = np.sqrt(3.0) * (axial[:, 1] + axial[:, 0] / 2.0)
    px /= (np.abs(px).max() + 1e-9)
    py /= (np.abs(py).max() + 1e-9)
    az = np.deg2rad(px * spec.fov_az_deg / 2.0)
    el = np.deg2rad(py * spec.fov_el_deg / 2.0)
    centers = _dir_from_azel(az, el).astype(np.float64)          # [N,3]
    N = centers.shape[0]

    # acceptance-angle blur: center + rings of offset sub-rays, Gaussian-weighted by angular offset.
    sig = np.deg2rad(spec.accept_sigma_deg)
    offsets = [(0.0, 0.0)]                                       # (radius_angle, phi)
    for ring in range(1, spec.blur_rings + 1):
        rad = sig * ring
        n_phi = 6 * ring
        for k in range(n_phi):
            offsets.append((rad, 2.0 * np.pi * k / n_phi))
    radii = np.asarray([o[0] for o in offsets])
    phis = np.asarray([o[1] for o in offsets])
    weights = np.exp(-0.5 * (radii / max(sig, 1e-9)) ** 2)
    weights /= weights.sum()
    S = len(offsets)

    # build an orthonormal tangent basis per center direction, then rotate the center toward the
    # tangent offset by the ring angle -> true small-angle acceptance cone.
    up = np.tile(np.array([0.0, 1.0, 0.0]), (N, 1))
    ref = np.where(np.abs(centers[:, 1:2]) > 0.9, np.tile([1.0, 0.0, 0.0], (N, 1)), up)
    e1 = np.cross(centers, ref); e1 /= (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12)
    e2 = np.cross(centers, e1); e2 /= (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-12)
    subdirs = np.zeros((N, S, 3), dtype=np.float64)
    for s in range(S):
        tang = np.cos(phis[s]) * e1 + np.sin(phis[s]) * e2       # [N,3] unit tangent
        d = np.cos(radii[s]) * centers + np.sin(radii[s]) * tang
        subdirs[:, s, :] = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    return {"centers": centers.astype(np.float32),               # [N,3]
            "az": az.astype(np.float32), "el": el.astype(np.float32),
            "subdirs": subdirs.astype(np.float32),               # [N,S,3]
            "weights": weights.astype(np.float32)}               # [S]


# --------------------------------------------------------------------------------------
# world textures: fixed 1/f statistics (the "scene bank"), sampled by 2D coordinate
# --------------------------------------------------------------------------------------
def make_scene_bank(spec: EpisodeSpec, seed: int) -> dict:
    """FIXED world statistics shared by every episode (analogue of Exp-5/6's fixed odor bank): the
    1/f Fourier-feature frequencies + amplitudes for the background and ground textures. Per-episode
    variety comes from fresh random phases + object placements (drawn in generate_batch), so all
    episodes share the same spatial-frequency statistics but are distinct worlds."""
    rng = np.random.default_rng(seed)

    sig = np.deg2rad(spec.accept_sigma_deg)     # acceptance half-angle (rad) -> frequency-domain MTF
    # BLUR IS APPLIED EXACTLY ONCE (the two mechanisms are mutually exclusive):
    #   * blur_rings == 0 (training path): acceptance blur is baked ANALYTICALLY here (optical MTF =
    #     Gaussian in spatial frequency), sampling one center ray per ommatidium.
    #   * blur_rings  > 0 (video path): geometric sub-ray averaging in build_sensor provides the blur,
    #     so the analytic MTF is SKIPPED here to avoid double-blurring.
    apply_mtf = int(spec.blur_rings) == 0

    def _feats(scale: float, angular: bool) -> dict:
        K = int(spec.tex_octaves) * 6
        # log-spaced spatial frequencies (1/f content), random 2D orientations, amp ~ 1/|f|^beta.
        mag = np.exp(rng.uniform(np.log(0.3), np.log(3.5), size=K)) * scale
        ang = rng.uniform(0, 2 * np.pi, size=K)
        freqs = np.stack([mag * np.cos(ang), mag * np.sin(ang)], axis=1)
        amps = (1.0 / np.maximum(mag, 1e-3) ** spec.tex_beta)
        amps /= np.sqrt(np.sum(amps ** 2)) + 1e-9
        if apply_mtf:
            # For the background (angular coords, radians) the cutoff is |f|*sigma directly; for
            # coordinate-space textures (ground/objects) the angular acceptance maps to a coordinate
            # blur at a representative scale, approximated with the same Gaussian cutoff.
            cutoff = mag * sig if angular else mag * sig * 6.0
            amps = amps * np.exp(-0.5 * cutoff ** 2)
        return {"freqs": freqs.astype(np.float32), "amps": amps.astype(np.float32)}

    return {"bg": _feats(spec.bg_tex_scale, angular=True),
            "ground": _feats(spec.ground_tex_scale, angular=False),
            "seed": int(seed)}


def _sample_texture(feats: dict, u: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """1/f texture value at 2D coords u[...,2], with per-component phase[K]. Returns luminance in
    ~[0,1] (mean 0.5). Vectorized over any leading shape."""
    proj = u @ feats["freqs"].T                     # [..., K]
    vals = np.cos(proj + phase) * feats["amps"]     # [..., K]
    lum = vals.sum(axis=-1)                          # [...]
    return 0.5 + 0.35 * lum


# --------------------------------------------------------------------------------------
# ego-motion: smooth time-varying 5-DOF trajectories
# --------------------------------------------------------------------------------------
def _ou_trajectory(spec: EpisodeSpec, B: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """[B, T, 5] smooth ego-motion. OU (low-pass noise) or saccade-and-fixate yaw. Units = physical
    (rad/s, m/s); this IS the per-timestep regression target."""
    scale = (DOF_SCALE * spec.motion_gain).astype(np.float32).copy()
    rot_idx = [0, 3, 4]   # yaw, roll, pitch
    trans_idx = [1, 2]    # forward, lateral
    scale[rot_idx] *= spec.rot_trans_balance
    scale[trans_idx] /= max(spec.rot_trans_balance, 1e-3)

    dt = spec.dt
    a = np.exp(-dt / max(spec.ou_tau, 1e-3))          # OU decay per frame
    innov = np.sqrt(1.0 - a * a)
    x = np.zeros((B, T, N_BODY), dtype=np.float32)
    x[:, 0, :] = rng.normal(0, 1, size=(B, N_BODY)).astype(np.float32)
    for t in range(1, T):
        x[:, t, :] = a * x[:, t - 1, :] + innov * rng.normal(0, 1, size=(B, N_BODY)).astype(np.float32)
    traj = x * scale

    if spec.motion_mode == "saccade":
        # fly-like: forward drive stays smooth (OU above); yaw becomes fast transients between
        # near-zero fixation intervals. Poisson-timed saccades of alternating sign.
        yaw = np.zeros((B, T), dtype=np.float32)
        p = spec.saccade_rate_hz * dt
        for b in range(B):
            sign = 1.0
            t = 0
            while t < T:
                if rng.random() < p:
                    dur = max(1, int(round(0.08 / dt)))       # ~80 ms saccade
                    amp = sign * scale[0] * rng.uniform(2.0, 4.0)
                    yaw[b, t:t + dur] = amp
                    sign *= -1.0
                    t += dur
                else:
                    t += 1
        traj[:, :, 0] = yaw
    return traj.astype(np.float32)


def _ou_1d(n: int, tau: float, dt: float, std: float, rng: np.random.Generator) -> np.ndarray:
    """A single Ornstein-Uhlenbeck (low-pass-noise) trace of length n with stationary std `std`."""
    a = np.exp(-dt / max(tau, 1e-3))
    innov = np.sqrt(1.0 - a * a) * std
    x = np.zeros(n, dtype=np.float32)
    x[0] = rng.normal(0, std)
    for t in range(1, n):
        x[t] = a * x[t - 1] + innov * rng.normal()
    return x


def _continuous_trajectory(spec: EpisodeSpec, B: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """CONTINUOUS optomotor BODY trajectory [B,T,5]: smooth, continuously time-varying rotation on all
    three axes (yaw/roll/pitch) at COMPARABLE per-axis variance (bounded OU rotational rates), conc.
    with a translating cruise (forward OU around cruise speed + lateral sideslip OU). No saccades, no
    gaze transform -- retinal = body. Columns = [yaw_rate, forward, lateral, roll_rate, pitch_rate]
    (rad/s + m/s); this IS the source of the regression target (via _compute_targets)."""
    dt = spec.dt; g = spec.motion_gain
    body = np.zeros((B, T, N_BODY), dtype=np.float32)
    rot_std = np.deg2rad(spec.rot_rate_dps) * g              # same std on yaw/roll/pitch -> comparable variance
    yaw_only = str(spec.rot_axes).lower() == "yaw"           # 1-D de-risk: roll & pitch stay 0
    for b in range(B):
        body[b, :, 0] = _ou_1d(T, spec.rot_tau, dt, rot_std, rng)                 # yaw_rate
        if not yaw_only:
            body[b, :, 3] = _ou_1d(T, spec.rot_tau, dt, rot_std, rng)             # roll_rate
            body[b, :, 4] = _ou_1d(T, spec.rot_tau, dt, rot_std, rng)             # pitch_rate
        fwd = spec.forward_speed + _ou_1d(T, 0.5, dt, spec.forward_speed_jitter, rng)
        body[b, :, 1] = np.clip(fwd, 0.05, 1.2) * g                               # forward (cruise)
        body[b, :, 2] = _ou_1d(T, 0.4, dt, spec.sideslip_speed, rng) * g          # lateral sideslip
    return body


def assign_trial_types(B: int, spec: EpisodeSpec, rng: np.random.Generator) -> np.ndarray:
    """Assign each of B episodes a trial-type code (TRIAL_TURN / TRIAL_TRANSLATE / TRIAL_MIXED) using a
    FIXED per-batch split from spec.trial_frac_turn / spec.trial_frac_translate (so both types are
    guaranteed present in every batch, which the per-trial-type scoring needs). Remainder -> mixed."""
    f_turn = float(np.clip(spec.trial_frac_turn, 0.0, 1.0))
    f_trans = float(np.clip(spec.trial_frac_translate, 0.0, 1.0))
    if f_turn + f_trans > 1.0:                                  # renormalize an over-specified split
        s = f_turn + f_trans
        f_turn, f_trans = f_turn / s, f_trans / s
    n_turn = int(round(B * f_turn))
    n_trans = int(round(B * f_trans))
    n_mixed = max(0, B - n_turn - n_trans)
    if n_turn + n_trans + n_mixed != B:                        # rounding fixups -> keep total = B
        n_turn = B - n_trans - n_mixed
    codes = np.array([TRIAL_TURN] * n_turn + [TRIAL_TRANSLATE] * n_trans + [TRIAL_MIXED] * n_mixed,
                     dtype=np.int64)
    rng.shuffle(codes)
    return codes


def _apply_trial_type(body: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Zero the non-varying DOF per trial type (in place on a copy). turn-only -> translation ~ 0;
    translate-only -> rotational rates ~ 0; mixed -> unchanged. body is [B,T,5] in BODY_DOF order."""
    body = body.copy()
    turn = codes == TRIAL_TURN
    trans = codes == TRIAL_TRANSLATE
    for c in _TRANS_BODY_COLS:                                 # turn-only: no translation
        body[turn, :, c] = 0.0
    for c in _ROT_BODY_COLS:                                   # translate-only: no rotation
        body[trans, :, c] = 0.0
    return body


def _saccade_fixate_trajectory(spec: EpisodeSpec, B: int, T: int, rng: np.random.Generator
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Drosophila free-cruising BODY trajectory (SACCADE_STATS.md §3): brief fast yaw saccades (raised-
    cosine yaw-rate bump integrating to ~90 deg, sign random) punctuating long translation-dominated
    intersaccadic intervals, each saccade banked ~30 deg (roll -> counter-roll), with LOW residual
    intersaccadic body rotation. Forward cruise ~0.5 m/s; sideslip small/transient; pitch ~constant.

    Returns (body_traj [B,T,5], saccade_mask [B,T] bool). body_traj columns = the 5-DOF instantaneous
    self-motion in the DOF_NAMES order [yaw_rate, forward, lateral, roll_rate, pitch_rate], rad/s + m/s;
    it is the REGRESSION TARGET. saccade_mask marks the saccade frames (used by gaze stabilization)."""
    dt = spec.dt
    g = spec.motion_gain
    dur_n = max(1, int(round(spec.saccade_dur_s / dt)))
    body = np.zeros((B, T, N_BODY), dtype=np.float32)
    mask = np.zeros((B, T), dtype=bool)
    refractory = dur_n + max(1, int(round(0.15 / dt)))       # min frames between saccade onsets

    for b in range(B):
        # --- intersaccadic (smooth) components ---
        fwd = spec.forward_speed + _ou_1d(T, 0.5, dt, spec.forward_speed_jitter, rng)
        fwd = np.clip(fwd, 0.15, 0.95) * g
        lat = _ou_1d(T, 0.3, dt, spec.sideslip_speed, rng) * g
        res_yaw = _ou_1d(T, 0.25, dt, np.deg2rad(spec.residual_yaw_dps), rng) * g   # LOW residual body yaw
        # pitch now carries genuine variance (climbs/dives) so it is observable, not a dead channel:
        res_pitch = _ou_1d(T, spec.pitch_tau, dt, np.deg2rad(spec.pitch_rate_dps), rng) * g
        yaw = res_yaw.copy()
        roll = np.zeros(T, dtype=np.float32)
        pitch = res_pitch.copy()

        # --- place saccades (Poisson onsets with a refractory floor) ---
        t = int(rng.integers(0, max(1, refractory)))
        while t < T - 1:
            if rng.random() < spec.saccade_rate_hz * dt:
                n = min(dur_n, T - t)
                idx = np.arange(n)
                shape = 0.5 * (1.0 - np.cos(2 * np.pi * (idx + 0.5) / dur_n))       # raised-cosine bump
                amp = np.deg2rad(spec.saccade_amp_deg + rng.normal(0, spec.saccade_amp_jitter_deg))
                amp *= (1.0 if rng.random() < 0.5 else -1.0) * g
                denom = max(float(shape.sum()) * dt, 1e-6)
                yaw_bump = amp * shape / denom                                       # integrates to amp
                # banked turn: roll angle theta(tau)=bank*sin^2(pi tau/dur) -> roll_rate = dtheta/dtau
                bank = np.deg2rad(spec.roll_bank_deg) * np.sign(amp) * g
                roll_rate = bank * (np.pi / spec.saccade_dur_s) * np.sin(2 * np.pi * (idx + 0.5) / dur_n)
                yaw[t:t + n] = yaw_bump
                roll[t:t + n] = roll_rate
                mask[b, t:t + n] = True
                t += refractory
            else:
                t += 1
        body[b, :, 0] = yaw; body[b, :, 1] = fwd; body[b, :, 2] = lat
        body[b, :, 3] = roll; body[b, :, 4] = pitch
    return body, mask


def _gaze_stabilize(body: np.ndarray, mask: np.ndarray, spec: EpisodeSpec) -> np.ndarray:
    """Map the BODY trajectory to the RETINAL trajectory that actually drives the eye (SACCADE_STATS
    §3, step 2-4). Per-axis sub-unity gaze gains attenuate the SMOOTH intersaccadic rotational slip
    ONLY (retinal = (1-gain)*body during intervals); saccadic transients pass through un-stabilized
    (the head saccade). Translation is passed through unattenuated. The TARGET stays the BODY motion,
    so the network must learn this FIXED-gain gaze transform to recover body self-motion from the
    (cleaner) retinal flow -- and, crucially, the intersaccadic flow the eye sees is now
    translation-dominated (depth-carrying)."""
    retinal = body.copy()
    interval = ~mask                                          # [B,T] intersaccadic frames
    gains = {0: spec.gaze_gain_yaw, 3: spec.gaze_gain_roll, 4: spec.gaze_gain_pitch}
    for axis, gain in gains.items():
        r = retinal[:, :, axis]
        r[interval] = (1.0 - gain) * r[interval]              # attenuate intersaccadic rotational slip
        retinal[:, :, axis] = r
    return retinal                                            # translation cols 1,2 unchanged


def _compute_targets(body: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Map the physical BODY 5-DOF trajectory [B,T,5] (+ per-episode altitude h [B]) to the candidate
    TARGET vector [B,T,N_TARGETS] = [yaw_rate, roll_rate, pitch_rate, forward_v, lateral_v, heading_az,
    ventral_flow], EXACTLY from the scene geometry:
      * yaw/roll/pitch  -- body rotational rates (rad/s), depth-independent.
      * forward_v/lateral_v -- absolute body translational velocity (m/s); recoverable only
        statistically under dense fixed-depth clutter (leans on the learned depth prior).
      * heading_az = atan2(v_lateral, v_forward)  -- FOE azimuth / travel direction (rad), scale-free.
      * ventral_flow = |v_ground| / h  -- ground optic-flow magnitude (rad/s), observable regardless of
        the depth prior (h drawn per episode so it decouples from absolute v)."""
    B, T, _ = body.shape
    hh = h[:, None]                                          # [B,1]
    yaw, fwd, lat, roll, pitch = (body[:, :, i] for i in range(N_BODY))
    heading_az = np.arctan2(lat, fwd)                        # FOE azimuth (rad)
    ventral_flow = np.sqrt(fwd ** 2 + lat ** 2) / hh         # ground image-flow magnitude v/h
    return np.stack([yaw, roll, pitch, fwd, lat, heading_az, ventral_flow], axis=-1).astype(np.float32)


def _expmap_so3(w: np.ndarray) -> np.ndarray:
    """Batched SO(3) exponential of axis-angle vectors w[...,3] (Rodrigues). Returns R[...,3,3]."""
    theta = np.linalg.norm(w, axis=-1, keepdims=True)               # [...,1]
    k = w / np.maximum(theta, 1e-9)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zeros = np.zeros_like(kx)
    K = np.stack([zeros, -kz, ky, kz, zeros, -kx, -ky, kx, zeros], axis=-1)
    K = K.reshape(w.shape[:-1] + (3, 3))
    th = theta[..., None]
    eye = np.broadcast_to(np.eye(3), w.shape[:-1] + (3, 3))
    return eye + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


# --------------------------------------------------------------------------------------
# render one batch of episodes: integrate poses, cast ommatidial rays, sample the 3D scene
# --------------------------------------------------------------------------------------
def _render(spec: EpisodeSpec, bank: dict, sensor: dict, traj: np.ndarray,
            objects: dict, phases: dict, no_parallax: bool, ground_h=None) -> np.ndarray:
    """Given ego-motion traj [B,T,5], produce the luminance movie [B,T,N] by integrating each
    camera pose (with `substeps`), casting every ommatidium's blur sub-rays into the world, and
    compositing objects (nearest) over ground (below horizon) over background (infinite depth)."""
    B, T, _ = traj.shape
    subdirs = sensor["subdirs"]                       # [N,S,3]
    N, S, _ = subdirs.shape
    wts = sensor["weights"]                            # [S]

    # --- integrate poses (R[B,T,3,3], pos[B,T,3]) from the body-frame 5-DOF rates ---
    R = np.zeros((B, T, 3, 3), dtype=np.float64)
    pos = np.zeros((B, T, 3), dtype=np.float64)
    R[:, 0] = np.eye(3)
    dt_sub = spec.dt / max(spec.substeps, 1)
    for t in range(1, T):
        Rc = R[:, t - 1].copy()
        pc = pos[:, t - 1].copy()
        yaw, fwd, lat, roll, pitch = [traj[:, t, i] for i in range(5)]
        w_body = np.stack([pitch, yaw, roll], axis=-1)      # about X(pitch), Y(yaw), Z(roll)
        v_body = np.stack([lat, np.zeros_like(lat), fwd], axis=-1)  # X=lateral, Z=forward
        for _ in range(max(spec.substeps, 1)):
            dR = _expmap_so3(w_body * dt_sub)               # [B,3,3]
            Rc = Rc @ dR
            pc = pc + np.einsum("bij,bj->bi", Rc, v_body) * dt_sub
        R[:, t] = Rc
        pos[:, t] = pc

    # --- world sub-ray directions: [B,T,N,S,3] = R applied to the fixed camera-frame sub-rays ---
    # heavy ray tensors are kept float32 (cos/matmul over ~1e8 elements is the cost driver); pose
    # integration above stayed float64 for accuracy but is tiny ([B,T,3,3]).
    wdirs = np.einsum("btij,nsj->btnsi", R.astype(np.float32), subdirs.astype(np.float32)).astype(np.float32)
    origins = pos[:, :, None, None, :].astype(np.float32)    # [B,T,1,1,3]

    # ============ BACKGROUND (infinite depth: sample by world direction) ============
    d = wdirs
    az = np.arctan2(d[..., 0], d[..., 2])
    el = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    u_bg = np.stack([az, el], axis=-1)                       # angular coords -> rotational flow
    lum = _sample_texture(bank["bg"], u_bg, phases["bg"]).astype(np.float32)    # [B,T,N,S]
    depth = np.full(lum.shape, np.inf, dtype=np.float32)

    # ============ GROUND PLANE y = -h (translation -> parallax via world-coord sampling) ============
    wy = d[..., 1]
    below = wy < -1e-4
    if no_parallax:
        # flatten: sample ground as a pure-direction texture (no translation dependence, depth inf-ish)
        u_g = np.stack([az, el * 1.3], axis=-1)
        g_lum = _sample_texture(bank["ground"], u_g, phases["ground"])
        g_depth = np.where(below, 50.0, np.inf)
    else:
        # per-episode altitude h_b: ground plane at y = -h_b (h broadcast over T,N,S)
        if ground_h is None:
            h_b = spec.ground_height
        else:
            h_b = np.asarray(ground_h, dtype=np.float32).reshape(-1, 1, 1, 1)   # [B,1,1,1] -> broadcast [B,T,N,S]
        t_hit = (-h_b - origins[..., 1]) / np.where(below, wy, -1.0)
        hit = below & (t_hit > 0)
        wx = origins[..., 0] + t_hit * d[..., 0]
        wz = origins[..., 2] + t_hit * d[..., 2]
        u_g = np.stack([wx, wz], axis=-1)
        g_lum = _sample_texture(bank["ground"], u_g, phases["ground"])
        g_depth = np.where(hit, t_hit, np.inf)
    take = g_depth < depth
    lum = np.where(take, g_lum, lum)
    depth = np.where(take, g_depth, depth)

    # ============ NEAR OBJECTS (nearest wins -> correct occlusion + motion boundaries) ============
    # Each object is a sphere of PHYSICAL radius radii[o]: its angular radius per frame = arctan(r/dist),
    # so nearer objects subtend a larger solid angle. Depth compositing (o_depth < depth) makes nearer
    # objects occlude farther ones and the ground/background, and the inside/outside boundary is a flow
    # discontinuity (object flow = its own parallax, background flow = ego-flow). Ground-truth targets
    # are computed analytically from geometry, so they are exact regardless of the render.
    radii = objects.get("radii")
    fixed_ang_r = np.deg2rad(spec.obj_ang_radius_deg)         # fallback angular radius (no phys radii given)
    for o in range(objects["centers"].shape[0]):
        c0 = objects["centers"][o]                           # [3] initial world position
        v = objects["vels"][o]                               # [3] independent velocity (0 for static clutter)
        tvec = np.arange(T) * spec.dt
        c_t = (c0[None, :] + v[None, :] * tvec[:, None]).astype(np.float32)   # [T,3] (f32: object render is the cost driver)
        c_full = c_t[None, :, None, None, :]                  # [1,T,1,1,3]
        rel = c_full - origins                                # camera -> object
        dist = np.linalg.norm(rel, axis=-1)                   # [B,T,1,1]
        rel_dir = rel / np.maximum(dist[..., None], 1e-6)
        cos_ang = np.sum(rel_dir * d, axis=-1)                # [B,T,N,S] alignment ray<->object
        if radii is not None and o < len(radii):
            ang_r = np.arctan(float(radii[o]) / np.maximum(dist[..., 0], 1e-6))   # [B,T,1] per-frame size
            ang_r = ang_r[..., None]                          # [B,T,1,1] -> broadcast over N,S
        else:
            ang_r = fixed_ang_r
        inside = cos_ang > np.cos(ang_r)
        if no_parallax:
            # object appears at a fixed direction texture, no translation parallax
            u_o = np.stack([az, el], axis=-1)
            o_lum = _sample_texture(bank["bg"], u_o * 2.0, phases["obj"][o])
            o_depth = np.where(inside, 40.0 + o, np.inf)
        else:
            # local object coords from the two tangent angles -> its own texture patch
            off = d - cos_ang[..., None] * rel_dir            # tangential component
            scale = 8.0 / np.maximum(ang_r, 1e-3)             # scalar or [B,T,1,1]
            u_o = np.stack([off[..., 0], off[..., 1]], axis=-1) * scale[..., None] \
                if not np.isscalar(scale) else np.stack([off[..., 0], off[..., 1]], axis=-1) * scale
            o_lum = _sample_texture(bank["bg"], u_o, phases["obj"][o])
            o_depth = np.where(inside, np.broadcast_to(dist, inside.shape), np.inf)
        take = inside & (o_depth < depth)
        lum = np.where(take, o_lum, lum)
        depth = np.where(take, o_depth, depth)

    # --- acceptance-angle blur: Gaussian-weighted sum over the S sub-rays ---
    img = np.einsum("btns,s->btn", lum, wts)                 # [B,T,N]
    img = (img - 0.5) * spec.contrast + 0.5
    return img.astype(np.float32)


# --------------------------------------------------------------------------------------
# generate one batch of episodes (+ verifier ablation hooks)
# --------------------------------------------------------------------------------------
def _build_objects(spec: EpisodeSpec, rng: np.random.Generator, no_objects: bool = False) -> dict:
    """Place the scene objects for one episode as (centers[K,3], vels[K,3], radii[K] physical radius m):
      * DENSE STATIC CLUTTER (continuous mode): spec.n_clutter non-moving objects at depths drawn from
        the FIXED distribution uniform[clutter_depth_lo, clutter_depth_hi], each a sphere of physical
        radius spec.obj_phys_radius (angular size = r/depth, so nearer = bigger -> correct occlusion).
        This dense fixed-depth field is what makes absolute v recoverable statistically.
      * MOVING DISTRACTORS: spec.n_moving_distractors objects with their OWN velocity (default 0 for
        vis_01; reserved for vis_02).
      * LEGACY (motion_mode != 'continuous'): spec.n_objects moving objects at obj_depth_lo..hi,
        angular radius obj_ang_radius_deg (unchanged saccade/ou behaviour)."""
    az_hw = np.deg2rad(spec.fov_az_deg / 2); el_hw = np.deg2rad(spec.fov_el_deg / 2)

    def _place(n, depth_lo, depth_hi, phys_radius, moving):
        if n <= 0:
            return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0,)))
        depths = rng.uniform(depth_lo, depth_hi, size=n)
        oaz = rng.uniform(-az_hw, az_hw, size=n)
        oel = rng.uniform(-el_hw, el_hw, size=n)
        centers = _dir_from_azel(oaz, oel) * depths[:, None]
        if moving:
            vdir = rng.normal(0, 1, size=(n, 3)); vdir /= (np.linalg.norm(vdir, axis=1, keepdims=True) + 1e-9)
            vels = vdir * spec.obj_speed
        else:
            vels = np.zeros((n, 3))
        radii = np.full(n, float(phys_radius))
        return centers, vels, radii

    if no_objects:
        return {"centers": np.zeros((0, 3)), "vels": np.zeros((0, 3)), "radii": np.zeros((0,))}

    parts = []
    if spec.motion_mode == "continuous":
        parts.append(_place(spec.n_clutter, spec.clutter_depth_lo, spec.clutter_depth_hi,
                            spec.obj_phys_radius, moving=False))
        # moving distractors: physical radius set so they subtend ~obj_ang_radius_deg at their mid-depth
        mid = 0.5 * (spec.obj_depth_lo + spec.obj_depth_hi)
        r_move = np.tan(np.deg2rad(spec.obj_ang_radius_deg)) * mid
        parts.append(_place(spec.n_moving_distractors, spec.obj_depth_lo, spec.obj_depth_hi,
                            r_move, moving=True))
    else:                                                    # legacy moving objects (saccade/ou modes)
        mid = 0.5 * (spec.obj_depth_lo + spec.obj_depth_hi)
        r_leg = np.tan(np.deg2rad(spec.obj_ang_radius_deg)) * mid
        parts.append(_place(spec.n_objects, spec.obj_depth_lo, spec.obj_depth_hi, r_leg, moving=True))

    centers = np.concatenate([p[0] for p in parts], axis=0) if parts else np.zeros((0, 3))
    vels = np.concatenate([p[1] for p in parts], axis=0) if parts else np.zeros((0, 3))
    radii = np.concatenate([p[2] for p in parts], axis=0) if parts else np.zeros((0,))
    return {"centers": centers.astype(np.float64), "vels": vels.astype(np.float64),
            "radii": radii.astype(np.float64)}


def generate_batch(bank: dict, spec: EpisodeSpec, batch_size: int, rng: np.random.Generator, *,
                   sensor: dict | None = None,
                   time_shuffle: bool = False, single_frame: bool = False,
                   no_objects: bool = False, no_parallax: bool = False,
                   static_noise: bool = True) -> Batch:
    """One episode-batch. Default path = the normal training stimulus. Ablation hooks (default off,
    used by the verifier -- prove the task needs motion/temporal/depth computation):
      * time_shuffle  -- permute the FRAMES in time (targets permuted identically). Destroys optic
                         flow while preserving the single-frame marginal -> a temporal-processing
                         (recurrence) network must collapse; a static per-frame regressor need not.
      * single_frame  -- freeze the movie to frame 0 (repeat it), targets kept -> no motion at all,
                         so nothing but chance is recoverable (RMSE -> target std).
      * no_objects    -- remove the moving distractors (difficulty drops; cleaner ego-flow).
      * no_parallax   -- flatten ground+objects to infinite depth: rotation stays readable, TRANSLATION
                         becomes unreadable (the physical check that depth carries the translation).
    """
    spec_eff = replace(spec, n_objects=0) if no_objects else spec
    if sensor is None:
        sensor = build_sensor(spec_eff)
    B, T = batch_size, spec.seq_len

    # per-episode ALTITUDE h (flight height above ground): varied so v/h (observable) decouples from v.
    h_ep = rng.uniform(spec_eff.altitude_lo, spec_eff.altitude_hi, size=B).astype(np.float32)

    # BODY trajectory drives rendering; the RETINAL trajectory is what the eye sees (= body except in
    # saccade_fixate, where gaze stabilization attenuates the intersaccadic rotational slip); the TARGET
    # is the candidate channel vector derived from the body motion + altitude.
    if spec.motion_mode == "continuous":                    # DEFAULT: continuous optomotor rotation
        body_traj = _continuous_trajectory(spec_eff, B, T, rng)
        trial_type = assign_trial_types(B, spec_eff, rng)    # turn-only / translate-only / mixed split
        body_traj = _apply_trial_type(body_traj, trial_type)  # zero the non-varying DOF per trial type
        retinal_traj = body_traj
    elif spec.motion_mode == "saccade_fixate":
        body_traj, saccade_mask = _saccade_fixate_trajectory(spec_eff, B, T, rng)
        retinal_traj = _gaze_stabilize(body_traj, saccade_mask, spec_eff)
        trial_type = np.full(B, TRIAL_MIXED, dtype=np.int64)  # no split in saccade mode (both vary)
    else:                                                    # legacy "ou"/"saccade": no gaze transform
        body_traj = _ou_trajectory(spec_eff, B, T, rng)
        retinal_traj = body_traj
        trial_type = np.full(B, TRIAL_MIXED, dtype=np.int64)
    traj = _compute_targets(body_traj, h_ep)                 # [B,T,N_TARGETS] -- the candidate target

    # per-episode fresh phases (same 1/f statistics, distinct worlds) + object placements
    Kbg = bank["bg"]["amps"].shape[0]
    Kg = bank["ground"]["amps"].shape[0]
    objects = _build_objects(spec_eff, rng, no_objects=no_objects)
    n_obj = objects["centers"].shape[0]
    phases = {"bg": rng.uniform(0, 2 * np.pi, size=Kbg).astype(np.float32),
              "ground": rng.uniform(0, 2 * np.pi, size=Kg).astype(np.float32),
              "obj": [rng.uniform(0, 2 * np.pi, size=Kbg).astype(np.float32) for _ in range(n_obj)]}

    img = _render(spec_eff, bank, sensor, retinal_traj, objects, phases, no_parallax,
                  ground_h=h_ep)                              # render RETINAL, per-episode altitude

    if single_frame:
        img = np.repeat(img[:, :1, :], T, axis=1)            # freeze -> no motion
    if time_shuffle:
        perm = rng.permutation(T)                            # permute frames AND targets identically
        img = img[:, perm, :]
        traj = traj[:, perm, :]

    if spec.sensor_noise_std > 0:
        img = img + rng.normal(0, spec.sensor_noise_std, size=img.shape).astype(np.float32)

    loss_mask = np.zeros((B, T), dtype=np.float32)
    loss_mask[:, spec.warmup:] = 1.0
    if single_frame:                                         # nothing is learnable -> nothing scored
        loss_mask[:] = 0.0
        loss_mask[:, spec.warmup:] = 1.0                     # keep shape; RMSE will sit at target std
    return Batch(img.astype(np.float32), traj.astype(np.float32), loss_mask, trial_type)


# --------------------------------------------------------------------------------------
# torch bridge + regression metric layer
# --------------------------------------------------------------------------------------
def batch_to_torch(batch: Batch, device):
    import torch
    return (torch.from_numpy(batch.inputs).to(device),
            torch.from_numpy(batch.targets).to(device),
            torch.from_numpy(batch.loss_mask).to(device))


def masked_mse(pred, targets, mask, dof_weights=None, dof_mask=None):
    """Masked MSE over scored timesteps. pred/targets [B,T,N_DOF], mask [B,T].

    PER-DOF-NORMALIZED LOSS (MUST-FIX 3). Unnormalized, the huge-variance yaw channel dominated the
    gradient (74% yaw / 24% roll / 1% pitch) and starved low-variance channels (pitch). `dof_weights`
    [N_DOF] (typically 1/target_variance per DOF) rescales each channel's squared error so every scored
    channel contributes comparable gradient. Default (None) = equal weights (legacy). R²/RMSE are
    reported per DOF regardless (scale-invariant), so this only rebalances what the model optimizes.

    TRIAL-TYPE-AWARE SCORING. `dof_mask` [B, N_DOF] (0/1) gates which channels are scored for each
    EPISODE by its trial type (dof_score_mask): rotation channels only on turn trials, ground-flow +
    heading only on translate trials. Loss is the mean weighted squared error over the scored
    (episode, timestep, channel) entries only."""
    import torch
    sq = (pred - targets) ** 2                               # [B,T,N_DOF]
    if dof_weights is not None:
        w = dof_weights.to(sq.dtype).to(sq.device) if torch.is_tensor(dof_weights) \
            else torch.as_tensor(dof_weights, dtype=sq.dtype, device=sq.device)
        sq = sq * w
    full = mask.unsqueeze(-1)                                 # [B,T,1] time (post-warmup) mask
    if dof_mask is not None:
        dm = dof_mask.to(sq.dtype).to(sq.device) if torch.is_tensor(dof_mask) \
            else torch.as_tensor(dof_mask, dtype=sq.dtype, device=sq.device)
        full = full * dm.unsqueeze(1)                         # [B,T,N_DOF] per-episode scored channels
    else:
        full = full.expand_as(sq)
    return (sq * full).sum() / full.sum().clamp_min(1.0)


def dof_rmse_r2(pred, targets, mask):
    """Per-DOF (rmse, r2) over scored steps. Returns (rmse[5], r2[5]) as numpy arrays.
    R² = 1 - SS_res/SS_tot with SS_tot about the per-DOF target mean over scored steps."""
    import torch
    m = mask.unsqueeze(-1)                                    # [B,T,1]
    n = m.sum().clamp_min(1.0)
    resid = (pred - targets) * m
    ss_res = (resid ** 2).sum(dim=(0, 1))                    # [5]
    tmean = (targets * m).sum(dim=(0, 1)) / n                # [5]
    ss_tot = (((targets - tmean) * m) ** 2).sum(dim=(0, 1))  # [5]
    rmse = torch.sqrt(ss_res / n)
    r2 = 1.0 - ss_res / ss_tot.clamp_min(1e-8)
    return rmse.detach().cpu().numpy(), r2.detach().cpu().numpy()


def dof_indices(names) -> list:
    """Resolve a DOF selector to integer indices. Accepts 'all', a DOF name, or a list of names/ints.
    Used to compute the primary metric over only the SCORED/learnable DOF (so dead channels can't
    inject null-channel noise into the connectome-vs-control contrast)."""
    if names is None or names == "all" or names == ["all"]:
        return list(range(N_DOF))
    if isinstance(names, str):
        names = [names]
    out = []
    for n in names:
        if isinstance(n, (int, np.integer)):
            out.append(int(n))
        else:
            out.append(DOF_NAMES.index(n))
    return out


def resolve_scored_map(cfg) -> dict:
    """Build the TRIAL-TYPE -> scored-DOF-indices mapping (the trial-type-aware `scored_dofs`). Reads
    cfg.scored_turn / cfg.scored_translate / cfg.scored_mixed (lists of DOF names or 'all'); defaults =
    rotation on turn trials, ground-flow (ventral_flow) + heading (heading_az) on translate trials, and
    the UNION on mixed trials. Returns {trial_type_code: [dof_idx, ...]}."""
    turn = getattr(cfg, "scored_turn", None) or list(DEFAULT_SCORED_TURN)
    trans = getattr(cfg, "scored_translate", None) or list(DEFAULT_SCORED_TRANSLATE)
    mixed = getattr(cfg, "scored_mixed", None)
    turn_idx = dof_indices(turn)
    trans_idx = dof_indices(trans)
    mixed_idx = dof_indices(mixed) if mixed else sorted(set(turn_idx) | set(trans_idx))
    return {TRIAL_TURN: turn_idx, TRIAL_TRANSLATE: trans_idx, TRIAL_MIXED: mixed_idx}


def scored_union(scored_map: dict) -> list:
    """Sorted union of all DOF indices scored by any trial type in `scored_map` (the DOF the primary
    scalar metric averages over -- each computed on the trials where it is scored)."""
    u: set = set()
    for idxs in scored_map.values():
        u.update(idxs)
    return sorted(u)


def dof_score_mask(trial_type: np.ndarray, scored_map: dict) -> np.ndarray:
    """[B, N_DOF] boolean: True where episode b's trial type scores DOF j (per `scored_map`). Combined
    with the time mask, this is what restricts each channel to the trials where it actually varies."""
    B = int(np.asarray(trial_type).shape[0])
    out = np.zeros((B, N_DOF), dtype=bool)
    tt = np.asarray(trial_type)
    for code, idxs in scored_map.items():
        rows = tt == code
        if not rows.any():
            continue
        for j in idxs:
            out[rows, j] = True
    return out


def scored_mean_r2(pred, targets, mask, scored=None) -> float:
    """Mean R² over the SCORED DOF only (default: all 5) -- the scalar primary metric (higher is
    better; used by the permutation-rank stat + best-by-val selection). Restrict `scored` to the
    learnable subset (e.g. rotation-only ['yaw_rate','roll_rate','pitch_rate']) so two dead channels
    do not dilute / add noise to the connectome-vs-control comparison."""
    _rmse, r2 = dof_rmse_r2(pred, targets, mask)
    idx = dof_indices(scored)
    return float(np.mean([r2[i] for i in idx]))


def mean_r2(pred, targets, mask) -> float:
    """Mean R² across all 5 DOF (kept for back-compat; the primary uses scored_mean_r2)."""
    _rmse, r2 = dof_rmse_r2(pred, targets, mask)
    return float(np.mean(r2))


# --------------------------------------------------------------------------------------
# naive baseline: least-squares frame-difference linear decoder (the achievable floor)
# --------------------------------------------------------------------------------------
def naive_baseline_r2(bank: dict, spec: EpisodeSpec, rng: np.random.Generator,
                      n_train: int = 12, n_test: int = 8, batch_size: int = 16) -> dict:
    """Fit a linear map from [frame_t, frame_t - frame_{t-1}] (per-timestep temporal-contrast
    features) to the 5-DOF target by least squares on n_train batches, evaluate on n_test fresh
    batches. This is a memoryless local-flow decoder -- the floor a recurrent network should beat if
    integration over time helps. Returns per-DOF R² + mean R²."""
    sensor = build_sensor(spec)

    def feats(b: Batch):
        x = b.inputs                                         # [B,T,N]
        dx = np.zeros_like(x); dx[:, 1:] = x[:, 1:] - x[:, :-1]
        f = np.concatenate([x, dx, np.ones(x.shape[:2] + (1,), np.float32)], axis=-1)
        m = b.loss_mask.astype(bool)
        return f[m], b.targets[m]                            # [(scored), 2N+1], [(scored),5]

    Xs, Ys = [], []
    for _ in range(n_train):
        X, Y = feats(generate_batch(bank, spec, batch_size, rng, sensor=sensor))
        Xs.append(X); Ys.append(Y)
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)                # [2N+1, 5]

    ss_res = np.zeros(N_DOF); ss_tot = np.zeros(N_DOF); n = 0
    ymean = Y.mean(0)
    for _ in range(n_test):
        Xt, Yt = feats(generate_batch(bank, spec, batch_size, rng, sensor=sensor))
        pred = Xt @ W
        ss_res += ((Yt - pred) ** 2).sum(0)
        ss_tot += ((Yt - ymean) ** 2).sum(0)
        n += Yt.shape[0]
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-8)
    return {"per_dof_r2": {DOF_NAMES[i]: round(float(r2[i]), 4) for i in range(N_DOF)},
            "mean_r2": round(float(np.mean(r2)), 4)}


# --------------------------------------------------------------------------------------
# VIDEO: watch the stimuli -- hex luminance movie + synchronized 5-DOF traces
# --------------------------------------------------------------------------------------
def _write_video(frames: list, out_path: Path, fps: int) -> Path:
    """Write RGB frames to mp4 (imageio-ffmpeg), falling back to gif if no ffmpeg is available."""
    import imageio.v2 as imageio
    out_path = Path(out_path)
    try:
        with imageio.get_writer(out_path.with_suffix(".mp4"), fps=fps, codec="libx264",
                                quality=8, macro_block_size=None) as w:
            for fr in frames:
                w.append_data(fr)
        return out_path.with_suffix(".mp4")
    except Exception as e:                                    # no ffmpeg -> gif fallback
        gif = out_path.with_suffix(".gif")
        imageio.mimsave(gif, frames, fps=fps)
        print(f"  (mp4 unavailable: {type(e).__name__}; wrote gif fallback {gif})")
        return gif


def render_episode_video(spec: EpisodeSpec, out_path: Path, seed: int = 0, fps: int = 8,
                         title: str = "optic-flow episode") -> Path:
    """Render ONE sample episode to video: (top) hex ommatidial luminance movie, (bottom) the
    synchronized 5-DOF ground-truth traces with a moving time cursor. Lets a human confirm the
    physics is right (parallax, per-DOF flow direction, independent object motion)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    sensor = build_sensor(spec)
    rng = np.random.default_rng(seed)
    b = generate_batch(bank=make_scene_bank(spec, seed=seed), spec=spec, batch_size=1, rng=rng,
                       sensor=sensor)
    img = b.inputs[0]                                         # [T,N]
    traj = b.targets[0]                                       # [T,5]
    az = np.rad2deg(sensor["az"]); el = np.rad2deg(sensor["el"])
    T = img.shape[0]
    vmin, vmax = float(img.min()), float(img.max())
    tt = np.arange(T) * spec.dt

    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(6.4, 6.0), dpi=100)
        ax0 = fig.add_axes([0.08, 0.46, 0.86, 0.48])
        ax0.scatter(az, el, c=img[t], cmap="gray", s=90, vmin=vmin, vmax=vmax,
                    marker="h", edgecolors="none")
        ax0.set_title(f"{title}  (frame {t+1}/{T})", fontsize=10)
        ax0.set_xlabel("azimuth (deg)"); ax0.set_ylabel("elevation (deg)")
        ax0.set_aspect("equal"); ax0.set_facecolor("#202020")
        ax1 = fig.add_axes([0.08, 0.07, 0.86, 0.30])
        for i in range(N_DOF):
            ax1.plot(tt, traj[:, i], lw=1.4, label=DOF_NAMES[i])
        ax1.axvline(tt[t], color="k", lw=1.2)
        ax1.set_xlabel("time (s)"); ax1.set_ylabel("5-DOF target")
        ax1.legend(fontsize=6, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.28))
        ax1.grid(alpha=0.3)
        canvas = FigureCanvasAgg(fig); canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[..., :3].copy()
        frames.append(buf)
        plt.close(fig)
    return _write_video(frames, Path(out_path), fps=fps)


def render_saccade_demo(spec: EpisodeSpec, out_path: Path, seed: int = 0, fps: int = 12,
                        title: str = "saccade-fixate flight") -> Path:
    """Render a saccade-fixate demo: (top) the hex retinal luminance movie, (bottom) the BODY 5-DOF
    traces with saccade windows shaded and the RETINAL (gaze-stabilized) yaw overlaid, so the fast
    yaw saccades punctuating straight translational intervals -- and the intersaccadic yaw/roll
    stabilization -- are visible. Only meaningful for motion_mode='saccade_fixate'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    sensor = build_sensor(spec)
    rng = np.random.default_rng(seed)
    body, mask = _saccade_fixate_trajectory(spec, 1, spec.seq_len, rng)
    retinal = _gaze_stabilize(body, mask, spec)
    bank = make_scene_bank(spec, seed=seed)
    Kbg = bank["bg"]["amps"].shape[0]; Kg = bank["ground"]["amps"].shape[0]
    phases = {"bg": rng.uniform(0, 2 * np.pi, size=Kbg).astype(np.float32),
              "ground": rng.uniform(0, 2 * np.pi, size=Kg).astype(np.float32),
              "obj": [rng.uniform(0, 2 * np.pi, size=Kbg).astype(np.float32) for _ in range(spec.n_objects)]}
    n_obj = spec.n_objects
    if n_obj > 0:
        depths = rng.uniform(spec.obj_depth_lo, spec.obj_depth_hi, size=n_obj)
        oaz = rng.uniform(-np.deg2rad(spec.fov_az_deg / 2), np.deg2rad(spec.fov_az_deg / 2), size=n_obj)
        oel = rng.uniform(-np.deg2rad(spec.fov_el_deg / 2), np.deg2rad(spec.fov_el_deg / 2), size=n_obj)
        centers = _dir_from_azel(oaz, oel) * depths[:, None]
        vdir = rng.normal(0, 1, (n_obj, 3)); vdir /= (np.linalg.norm(vdir, axis=1, keepdims=True) + 1e-9)
        objects = {"centers": centers.astype(np.float64), "vels": (vdir * spec.obj_speed).astype(np.float64)}
    else:
        objects = {"centers": np.zeros((0, 3)), "vels": np.zeros((0, 3))}
    img = _render(spec, bank, sensor, retinal, objects, phases, no_parallax=False)[0]
    if spec.sensor_noise_std > 0:
        img = img + rng.normal(0, spec.sensor_noise_std, img.shape).astype(np.float32)

    az = np.rad2deg(sensor["az"]); el = np.rad2deg(sensor["el"])
    T = img.shape[0]; tt = np.arange(T) * spec.dt
    vmin, vmax = float(img.min()), float(img.max())
    b0 = body[0]; r0 = retinal[0]; m0 = mask[0]
    tgt0 = _compute_targets(body, np.array([spec.ground_height], np.float32))[0]   # observable channels
    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(6.6, 6.2), dpi=100)
        ax0 = fig.add_axes([0.08, 0.46, 0.86, 0.48])
        ax0.scatter(az, el, c=img[t], cmap="gray", s=85, vmin=vmin, vmax=vmax, marker="h", edgecolors="none")
        ax0.set_title(f"{title}  (frame {t+1}/{T}, t={tt[t]:.2f}s)", fontsize=10)
        ax0.set_xlabel("azimuth (deg)"); ax0.set_ylabel("elevation (deg)")
        ax0.set_aspect("equal"); ax0.set_facecolor("#202020")
        ax1 = fig.add_axes([0.08, 0.07, 0.86, 0.30])
        # shade saccade windows
        in_sac = False; start = 0
        for i in range(T):
            if m0[i] and not in_sac:
                in_sac = True; start = i
            elif not m0[i] and in_sac:
                ax1.axvspan(tt[start], tt[i], color="#eb6834", alpha=0.15); in_sac = False
        if in_sac:
            ax1.axvspan(tt[start], tt[-1], color="#eb6834", alpha=0.15)
        ax1.plot(tt, np.rad2deg(b0[:, 0]), lw=1.5, color="#2a78d6", label="body yaw_rate")
        ax1.plot(tt, np.rad2deg(r0[:, 0]), lw=1.0, color="#7fb0e8", ls="--", label="retinal yaw_rate")
        ax1.plot(tt, np.rad2deg(b0[:, 3]), lw=1.2, color="#1baf7a", label="body roll_rate")
        ax1.plot(tt, np.rad2deg(b0[:, 4]), lw=1.0, color="#d68a2a", label="body pitch_rate")
        ax1.plot(tt, np.rad2deg(tgt0[:, 3]), lw=1.2, color="#a349a4", label="ventral flow v/h (deg/s)")
        ax1.plot(tt, np.rad2deg(tgt0[:, 5]), lw=1.0, color="#555555", label="heading az (deg)")
        ax1.axvline(tt[t], color="k", lw=1.2)
        ax1.set_xlabel("time (s)"); ax1.set_ylabel("rate (deg/s) / speed")
        ax1.legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.32)); ax1.grid(alpha=0.3)
        canvas = FigureCanvasAgg(fig); canvas.draw()
        frames.append(np.asarray(canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    return _write_video(frames, Path(out_path), fps=fps)


def render_flow_field_demo(spec: EpisodeSpec, out_path: Path, seed: int = 0, fps: int = 8,
                           title: str = "continuous optomotor + dense clutter") -> Path:
    """Render ONE episode of the DEFAULT (continuous-rotation + dense fixed-depth clutter) stimulus,
    with the ANALYTIC optic-flow (motion-field) vectors overlaid on the hex luminance frame so the
    clutter/parallax and the flow field are both visible. The motion field per ommatidium is the exact
    textbook expression from the ego-motion (rotation omega, translation v) and the per-ray scene depth
    Z:  d_dot = -(I - d d^T) v / Z  -  omega x d   (translational term depth-dependent -> parallax;
    rotational term depth-independent), projected onto the (az, el) tangent basis. Ground-truth targets
    (bottom panel) are exact from geometry."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    sensor = build_sensor(spec)
    rng = np.random.default_rng(seed)
    bank = make_scene_bank(spec, seed=seed)
    B, T = 1, spec.seq_len
    h_ep = rng.uniform(spec.altitude_lo, spec.altitude_hi, size=B).astype(np.float32)
    body = (_continuous_trajectory(spec, B, T, rng) if spec.motion_mode == "continuous"
            else _ou_trajectory(spec, B, T, rng))
    objects = _build_objects(spec, rng)
    Kbg = bank["bg"]["amps"].shape[0]; Kg = bank["ground"]["amps"].shape[0]
    n_obj = objects["centers"].shape[0]
    phases = {"bg": rng.uniform(0, 2 * np.pi, size=Kbg).astype(np.float32),
              "ground": rng.uniform(0, 2 * np.pi, size=Kg).astype(np.float32),
              "obj": [rng.uniform(0, 2 * np.pi, size=Kbg).astype(np.float32) for _ in range(n_obj)]}
    img = _render(spec, bank, sensor, body, objects, phases, no_parallax=False, ground_h=h_ep)[0]
    if spec.sensor_noise_std > 0:
        img = img + rng.normal(0, spec.sensor_noise_std, img.shape).astype(np.float32)
    tgt = _compute_targets(body, h_ep)[0]                    # [T, N_TARGETS]

    # analytic motion field per ommatidium: need per-ray world direction + depth each frame.
    centers = sensor["centers"]                              # [Nomm, 3] camera-frame center dirs
    az = np.rad2deg(sensor["az"]); el = np.rad2deg(sensor["el"])
    # integrate poses (reuse the renderer's convention) to get camera R, pos and depth of the ground hit
    R = np.zeros((T, 3, 3)); R[0] = np.eye(3); pos = np.zeros((T, 3))
    dt_sub = spec.dt / max(spec.substeps, 1)
    for t in range(1, T):
        Rc = R[t - 1].copy(); pc = pos[t - 1].copy()
        yaw, fwd, lat, roll, pitch = body[0, t]
        w = np.array([pitch, yaw, roll]); vb = np.array([lat, 0.0, fwd])
        for _ in range(max(spec.substeps, 1)):
            th = np.linalg.norm(w * dt_sub)
            if th > 1e-9:
                k = (w * dt_sub) / th
                K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                dR = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
            else:
                dR = np.eye(3)
            Rc = Rc @ dR; pc = pc + Rc @ vb * dt_sub
        R[t] = Rc; pos[t] = pc

    frames = []
    for t in range(T):
        yaw, fwd, lat, roll, pitch = body[0, t]
        omega = np.array([pitch, yaw, roll]); v = np.array([lat, 0.0, fwd])
        d = centers                                          # [Nomm,3] camera-frame view dirs
        wy = (R[t] @ d.T).T[:, 1]                            # world y of each ray -> ground depth
        # ground range along each ray (else far background): Z = h / (-wy) for downward rays
        with np.errstate(divide="ignore", invalid="ignore"):
            Zg = np.where(wy < -1e-3, float(h_ep[0]) / (-wy), 60.0)
        Z = np.clip(Zg, 0.2, 60.0)
        # motion field: d_dot = -(I - d d^T) v / Z - omega x d
        proj = v[None, :] - (d * (d @ v)[:, None])           # (I - d d^T) v
        trans = -proj / Z[:, None]
        rot = -np.cross(np.tile(omega, (d.shape[0], 1)), d)
        ddot = trans + rot                                   # [Nomm,3] camera-frame retinal velocity
        # project onto (az, el) tangent basis: e_az ~ d/d az, e_el ~ d/d el
        e_az = np.stack([np.cos(sensor["az"]), np.zeros_like(sensor["az"]), -np.sin(sensor["az"])], 1)
        e_el = np.stack([-np.sin(sensor["el"]) * np.sin(sensor["az"]), np.cos(sensor["el"]),
                         -np.sin(sensor["el"]) * np.cos(sensor["az"])], 1)
        u = np.sum(ddot * e_az, 1); w_ = np.sum(ddot * e_el, 1)

        fig = plt.figure(figsize=(6.6, 6.4), dpi=100)
        ax0 = fig.add_axes([0.09, 0.44, 0.86, 0.50])
        ax0.scatter(az, el, c=img[t], cmap="gray", s=70, vmin=float(img.min()), vmax=float(img.max()),
                    marker="h", edgecolors="none")
        sc = 8.0
        ax0.quiver(az, el, u * sc, w_ * sc, color="#22d3aa", width=0.003, scale=60,
                   headwidth=3, alpha=0.9)
        ax0.set_title(f"{title}  (frame {t+1}/{T})", fontsize=10)
        ax0.set_xlabel("azimuth (deg)"); ax0.set_ylabel("elevation (deg)")
        ax0.set_aspect("equal"); ax0.set_facecolor("#202020")
        ax1 = fig.add_axes([0.09, 0.06, 0.86, 0.30])
        tt = np.arange(T) * spec.dt
        for i in range(N_TARGETS):
            ax1.plot(tt, tgt[:, i], lw=1.2, label=TARGET_NAMES[i])
        ax1.axvline(tt[t], color="k", lw=1.1)
        ax1.set_xlabel("time (s)"); ax1.set_ylabel("target")
        ax1.legend(fontsize=5, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.26)); ax1.grid(alpha=0.3)
        canvas = FigureCanvasAgg(fig); canvas.draw()
        frames.append(np.asarray(canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    return _write_video(frames, Path(out_path), fps=fps)


def render_sanity_clips(spec: EpisodeSpec, out_dir: Path, fps: int = 8) -> list:
    """Render single-DOF 'sanity' clips (pure yaw, pure forward, pure lateral, pure roll, pure pitch)
    so each channel's flow can be visually verified in isolation. Motion is a constant unit rate on
    one DOF, zero on the rest."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, name in enumerate(BODY_DOF):                       # single-DOF clips are per PHYSICAL body-DOF
        written.append(_render_constant_dof(spec, i, out_dir / f"sanity_{name}", fps=fps))
    return written


def _render_constant_dof(spec: EpisodeSpec, dof: int, out_path: Path, fps: int = 8) -> Path:
    """Helper: a clip driven by a constant unit rate on ONE dof (others zero) -> pure single-channel
    flow. Uses the same renderer as training but with a hand-set trajectory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    sensor = build_sensor(spec)
    bank = make_scene_bank(spec, seed=7)
    rng = np.random.default_rng(dof + 100)
    T = spec.seq_len
    traj = np.zeros((1, T, N_BODY), dtype=np.float32)
    traj[0, :, dof] = BODY_SCALE[dof]                        # constant unit-scale rate on this body-DOF
    Kbg = bank["bg"]["amps"].shape[0]; Kg = bank["ground"]["amps"].shape[0]
    phases = {"bg": rng.uniform(0, 2 * np.pi, size=Kbg).astype(np.float32),
              "ground": rng.uniform(0, 2 * np.pi, size=Kg).astype(np.float32), "obj": []}
    objects = {"centers": np.zeros((0, 3)), "vels": np.zeros((0, 3))}
    img = _render(replace(spec, n_objects=0), bank, sensor, traj, objects, phases, no_parallax=False)[0]
    az = np.rad2deg(sensor["az"]); el = np.rad2deg(sensor["el"])
    vmin, vmax = float(img.min()), float(img.max())
    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(5.2, 4.6), dpi=100)
        ax = fig.add_axes([0.1, 0.1, 0.85, 0.82])
        ax.scatter(az, el, c=img[t], cmap="gray", s=100, vmin=vmin, vmax=vmax, marker="h",
                   edgecolors="none")
        ax.set_title(f"pure {BODY_DOF[dof]}  (frame {t+1}/{T})", fontsize=11)
        ax.set_xlabel("azimuth (deg)"); ax.set_ylabel("elevation (deg)")
        ax.set_aspect("equal"); ax.set_facecolor("#202020")
        canvas = FigureCanvasAgg(fig); canvas.draw()
        frames.append(np.asarray(canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    return _write_video(frames, Path(out_path), fps=fps)


if __name__ == "__main__":
    # tiny self-check: shapes, metric sanity, and a naive-baseline read on a small spec.
    spec = EpisodeSpec(hex_rings=4, seq_len=16, n_objects=3)
    bank = make_scene_bank(spec, seed=0)
    rng = np.random.default_rng(0)
    b = generate_batch(bank, spec, 8, rng)
    print("input_dim", spec.input_dim, "inputs", b.inputs.shape, "targets", b.targets.shape,
          "mask_frac", float(b.loss_mask.mean()))
    print("naive baseline:", naive_baseline_r2(bank, spec, np.random.default_rng(1),
                                                n_train=6, n_test=4, batch_size=8))
