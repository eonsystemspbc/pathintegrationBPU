#!/usr/bin/env python3
"""Experiment 6 -- the odor->evidence *temporal-integration* task.

This is the NEW generative task for Experiment 6, kept behind the SAME public surface as
Experiment 5's ``odor_valence_task.py`` so the shared engine (``common.py`` +
``MatrixEpisodicRNN``) needs only an import swap and an ``output_dim`` change. What is
preserved byte-for-byte in interface (so the Exp-5 training loop / metric layer are reused
unchanged):

  * ``EpisodeSpec``   -- episode geometry (odors, presentations, drift, evidence noise, sparsity).
  * ``make_odor_bank``-- the fixed sparse, unit-norm odor prototypes (identical to Exp 5).
  * ``generate_batch``/``Batch`` -- one episode-batch (STREAM of interleaved presentations, then a
                         QUERY phase), plus the ablation hooks the verifier uses.
  * ``batch_to_torch``-- numpy episode -> device tensors (the SAME 5-tuple the Exp-5 loop expects).
  * ``masked_ce_ov`` / ``ov_correct_total`` -- the uniform 3-way loss + accuracy at query steps.
  * ``N_VALENCE`` / ``ROLE_DIMS`` / ``CHANCE`` -- 3 / 3 / 1/3.

WHY A NEW TASK (vs Exp 5). Exp 5's odor->valence task is SINGLE-SHOT: each odor is shown once
with its reinforcement, so recall is a binding/lookup, not an integration. Experiment 6 tests
whether connectome topology helps when the task *requires temporal integration*: each odor's
latent category must be read out from the running MEAN of several noisy scalar evidence samples
spread across an interleaved stream. Finite K + per-sample noise => an irreducible mid-band error
that is the difficulty knob. The Bayes-optimal decoder is the thresholded sample mean with
boundaries at +/- m/2 ("valence = category of the average signal").

TASK STRUCTURE (per episode)
----------------------------
Draw O odors from a bank of ``num_odors``. Each odor i gets a latent category c_i drawn BALANCED
from 3 classes -- {attract:+m, neutral:0, repulse:-m}, chance 1/3. Each PRESENTATION of odor i
emits a fresh scalar evidence sample e_{i,t} = mu(c_i) + eta, eta ~ N(0, sigma^2). Odor i is
presented K times; the N_pres = O*K presentation steps are a RANDOM interleave of the multiset
{odor_i x K}.

  STREAM  (N_pres = O*K steps, random interleave):
      step vector = [ odor_i(+odor noise) | e+ | e- | query=0 ]
      where e+ = relu(+e_{i,t}), e- = relu(-e_{i,t})  (two rectified evidence channels, parallel
      to Exp-5's reward/punish one-hot). Odor & evidence co-occur at every presentation step.
  QUERY   (O steps, one per odor, random order):
      step vector = [ odor_i(+odor noise) | 0 | 0 | query=1 ]  ->  target c_i, SCORED.

Supervision is END-OF-STREAM ONLY: the loss + accuracy are evaluated only at the O query steps;
every presentation step is masked out. REVERSAL is DROPPED (pure stationary integration).

INPUT LAYOUT (per timestep): ``[ odor(odor_dim) | e+(1) | e-(1) | query(1) ]``.
``ROLE_DIMS = 3`` (e+, e-, query); ``input_dim = odor_dim + ROLE_DIMS``; ``output_dim = 3``.

TWO DECOUPLED NOISE SOURCES (the design's confound defusal):
  * ``odor_noise_std`` -- LOW (0.03): odor identity stays reliably recognizable, so routing is NOT
    the bottleneck (removes the odor-recognition confound).
  * ``evidence_noise_std`` = sigma -- the PRIMARY difficulty/cap knob. Per-presentation SNR = m/sigma;
    integrated SNR = (m/sigma)*sqrt(K). Lowering m/sigma lowers the achievable plateau WITHOUT an
    optimization stall (unlike raising O, which stalls -- Exp-5 subrun-01's item-count cliff).

THE 5-TUPLE / MASK OVERLOAD (so Exp-5's ``train_one_run_ov`` is reused UNCHANGED). The Exp-5 loop
expects ``batch_to_torch`` to return (inputs, targets, query_mask, initial_query_mask,
reversed_query_mask) and reports an "initial"/"reversed" accuracy split. Reversal is gone here, so
those two spare mask channels are repurposed into a meaningful per-category secondary at zero cost:
  * ``initial_query_mask`` := the NEUTRAL-class query steps  -> reported as "neutral recall"
    (the hardest 3-way class, straddling both decision boundaries -- most integration-sensitive).
  * ``reversed_query_mask`` := the POLAR-class query steps (attract UNION repulse) -> "polar recall".
The pooled ``query_mask`` (all O queries) remains the primary metric (pooled 3-way accuracy). The
FULL 3-way per-category split {attract,neutral,repulse}, the integration curve, and the two
ablations are produced by the verifier eval-modes in ``run_experiment.py`` (run at pre-flight).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

# 3-way latent valence category: 0 = attract (+m), 1 = neutral (0), 2 = repulse (-m).
N_VALENCE = 3
ROLE_DIMS = 3  # e+, e-, query  (appended after the odor vector)
CHANCE = 1.0 / N_VALENCE

# category -> mean-signal direction (multiplied by drift m); index matches the target class id.
_MU_DIR = np.array([+1.0, 0.0, -1.0], dtype=np.float32)   # attract / neutral / repulse
NEUTRAL_CLASS = 1


@dataclass(frozen=True)
class EpisodeSpec:
    num_odors: int
    odor_dim: int
    odors_per_episode: int            # O
    presentations_per_odor: int       # K
    drift: float                      # m  (attract mean +m, repulse mean -m)
    evidence_noise_std: float         # sigma  (the primary difficulty / cap knob)
    odor_sparsity: float
    odor_noise_std: float             # LOW -- keeps odor identity recognizable

    @property
    def input_dim(self) -> int:
        return self.odor_dim + ROLE_DIMS

    @property
    def n_presentation_steps(self) -> int:
        return self.odors_per_episode * self.presentations_per_odor    # N_pres = O*K

    @property
    def timesteps(self) -> int:
        return self.n_presentation_steps + self.odors_per_episode      # STREAM + QUERY


@dataclass(frozen=True)
class Batch:
    inputs: np.ndarray               # [B, T, odor_dim + ROLE_DIMS]
    targets: np.ndarray              # [B, T]  category in {0,1,2} at query steps (0 elsewhere; masked)
    query_mask: np.ndarray           # [B, T]  1 at ALL O query steps -> loss + pooled 3-way acc
    initial_query_mask: np.ndarray   # [B, T]  1 at the NEUTRAL-class query steps (overloaded slot)
    reversed_query_mask: np.ndarray  # [B, T]  1 at the POLAR-class (attract|repulse) query steps


def make_odor_bank(spec: EpisodeSpec, seed: int) -> np.ndarray:
    """Fixed bank of sparse, unit-norm odor prototypes [num_odors, odor_dim].
    IDENTICAL to Exp 5's make_odor_bank (comparable geometry across experiments)."""
    rng = np.random.default_rng(seed)
    bank = rng.normal(0.0, 1.0, size=(spec.num_odors, spec.odor_dim)).astype(np.float32)
    mask = rng.random(bank.shape) < spec.odor_sparsity
    bank *= mask.astype(np.float32)
    norms = np.linalg.norm(bank, axis=1, keepdims=True)
    empty = norms.squeeze(-1) == 0
    if np.any(empty):
        cols = rng.integers(0, spec.odor_dim, size=int(empty.sum()))
        bank[empty, cols] = 1.0
        norms = np.linalg.norm(bank, axis=1, keepdims=True)
    return (bank / np.maximum(norms, 1e-6)).astype(np.float32)


def _write_odor(dest: np.ndarray, odor: np.ndarray, noise_std: float,
                rng: np.random.Generator) -> None:
    dest[: odor.shape[0]] = odor
    if noise_std > 0:
        dest[: odor.shape[0]] += rng.normal(0.0, noise_std, size=odor.shape).astype(np.float32)


def _balanced_categories(O: int, rng: np.random.Generator) -> np.ndarray:
    """O category labels drawn BALANCED across the 3 classes (exactly O/3 each when divisible;
    the remainder filled by a random class subset), then shuffled -> no class prior advantage."""
    base = np.tile(np.arange(N_VALENCE), O // N_VALENCE).astype(np.int64)
    rem = O % N_VALENCE
    if rem:
        base = np.concatenate([base, rng.choice(N_VALENCE, size=rem, replace=False)])
    rng.shuffle(base)
    return base


def generate_batch(odor_bank: np.ndarray, spec: EpisodeSpec, batch_size: int,
                   rng: np.random.Generator, *,
                   first_only: bool = False, shuffle_evidence: bool = False) -> Batch:
    """One episode-batch. Phase boundaries are identical across the batch (same spec); the odors,
    categories, evidence samples, and step orderings are drawn independently per sample.

    Ablation hooks (default off -> the normal training path is unchanged; used by the verifier):
      * first_only        -- zero the evidence channels on all but each odor's FIRST presentation
                             (a real integrator drops toward single-shot ~chance-of-K=1).
      * shuffle_evidence  -- permute the per-step evidence samples across presentation steps, breaking
                             the odor<->evidence correspondence (accuracy must collapse toward 1/3).
    """
    T = spec.timesteps
    O, K = spec.odors_per_episode, spec.presentations_per_odor
    inputs = np.zeros((batch_size, T, spec.input_dim), dtype=np.float32)
    targets = np.zeros((batch_size, T), dtype=np.float32)
    query_mask = np.zeros((batch_size, T), dtype=np.float32)
    initial_query_mask = np.zeros((batch_size, T), dtype=np.float32)   # NEUTRAL-class queries
    reversed_query_mask = np.zeros((batch_size, T), dtype=np.float32)  # POLAR-class queries

    eplus_col = spec.odor_dim
    eminus_col = spec.odor_dim + 1
    query_col = spec.odor_dim + 2
    mu = (spec.drift * _MU_DIR).astype(np.float32)                     # per-class mean signal
    sigma = float(spec.evidence_noise_std)

    for b in range(batch_size):
        odor_ids = rng.choice(spec.num_odors, size=O, replace=False)
        categories = _balanced_categories(O, rng)                     # [O] class id per local odor

        # --- STREAM: random interleave of the multiset {local_odor x K} ------------------------
        order = np.repeat(np.arange(O), K)                            # [N_pres]
        rng.shuffle(order)
        # fresh evidence sample per presentation step
        evid = mu[categories[order]] + rng.normal(0.0, sigma, size=order.shape[0]).astype(np.float32)
        if first_only:                                                # keep only each odor's 1st show
            keep = np.zeros(order.shape[0], dtype=bool)
            seen = set()
            for j, loc in enumerate(order):
                if int(loc) not in seen:
                    keep[j] = True
                    seen.add(int(loc))
            evid = np.where(keep, evid, 0.0).astype(np.float32)
        if shuffle_evidence:                                          # break odor<->evidence link
            evid = evid[rng.permutation(order.shape[0])]

        step = 0
        for j in range(order.shape[0]):
            loc = int(order[j])
            _write_odor(inputs[b, step], odor_bank[odor_ids[loc]], spec.odor_noise_std, rng)
            e = float(evid[j])
            inputs[b, step, eplus_col] = max(e, 0.0)                  # relu(+e)
            inputs[b, step, eminus_col] = max(-e, 0.0)                # relu(-e)
            step += 1

        # --- QUERY: one step per odor, random order -> recall the integrated category ----------
        for loc in rng.permutation(O):
            _write_odor(inputs[b, step], odor_bank[odor_ids[loc]], spec.odor_noise_std, rng)
            inputs[b, step, query_col] = 1.0
            c = int(categories[loc])
            targets[b, step] = float(c)
            query_mask[b, step] = 1.0
            if c == NEUTRAL_CLASS:
                initial_query_mask[b, step] = 1.0                     # neutral recall (secondary)
            else:
                reversed_query_mask[b, step] = 1.0                    # polar recall (secondary)
            step += 1

        if step != T:
            raise AssertionError(f"internal timestep mismatch: {step} != {T}")

    return Batch(inputs, targets, query_mask, initial_query_mask, reversed_query_mask)


def batch_to_torch(batch: Batch, device) -> tuple[torch.Tensor, ...]:
    """(inputs, targets_long, query_mask, neutral_mask, polar_mask) on `device`.
    Returned in exactly the 5-tuple order Exp-5's train_one_run_ov / _ov_eval consume."""
    return (
        torch.from_numpy(batch.inputs).to(device),
        torch.from_numpy(batch.targets).to(device).long(),
        torch.from_numpy(batch.query_mask).to(device),
        torch.from_numpy(batch.initial_query_mask).to(device),
        torch.from_numpy(batch.reversed_query_mask).to(device),
    )


# --------------------------------------------------------------------------------------
# uniform metric layer -- one loss + one accuracy for the engine (3-way category)
# --------------------------------------------------------------------------------------
def masked_ce_ov(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked cross-entropy over N_VALENCE classes at query steps.
    logits [B,T,N_VALENCE], targets [B,T] (long class idx), mask [B,T] (1 at query steps)."""
    B, T, Kc = logits.shape
    raw = nn.functional.cross_entropy(
        logits.reshape(B * T, Kc), targets.reshape(B * T).long(), reduction="none"
    ).reshape(B, T)
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


def ov_correct_total(logits: torch.Tensor, targets: torch.Tensor,
                     mask: torch.Tensor) -> tuple[float, float]:
    """(#correct, #scored) over the masked query steps: argmax(logits)==target."""
    pred = logits.argmax(dim=-1)
    correct = ((pred == targets.long()).float() * mask).sum()
    total = mask.sum()
    return float(correct.item()), float(total.item())


def per_category_correct_total(logits: torch.Tensor, targets: torch.Tensor,
                               query_mask: torch.Tensor) -> dict[int, tuple[float, float]]:
    """Per-class (#correct, #scored) at query steps, keyed by class id {0:attract,1:neutral,2:repulse}.
    Used by the verifier / analysis for the full 3-way per-category secondary."""
    pred = logits.argmax(dim=-1)
    out: dict[int, tuple[float, float]] = {}
    for c in range(N_VALENCE):
        cls_mask = query_mask * (targets.long() == c).float()
        correct = ((pred == targets.long()).float() * cls_mask).sum()
        out[c] = (float(correct.item()), float(cls_mask.sum().item()))
    return out


def bayes_accuracy(spec: EpisodeSpec, K: int | None = None) -> dict:
    """Analytic Bayes-optimal 3-way accuracy: the thresholded sample-mean decoder with boundaries
    at +/- m/2. sample-mean ~ N(mu(c), sigma^2/K); a = (m/2)*sqrt(K)/sigma. Balanced classes:
      attract/repulse acc = Phi(a);  neutral acc = 2*Phi(a) - 1;  overall = mean of the three.
    This is the oracle ceiling the verifier reports (no network can beat it)."""
    from math import erf, sqrt
    Ke = spec.presentations_per_odor if K is None else int(K)
    sigma = max(float(spec.evidence_noise_std), 1e-9)
    a = (spec.drift / 2.0) * sqrt(Ke) / sigma
    phi = 0.5 * (1.0 + erf(a / sqrt(2.0)))                 # Phi(a)
    polar = phi
    neutral = max(2.0 * phi - 1.0, 0.0)
    overall = (2.0 * polar + neutral) / 3.0
    return {"K": Ke, "attract": round(polar, 4), "neutral": round(neutral, 4),
            "repulse": round(polar, 4), "overall": round(overall, 4),
            "per_presentation_snr": round(spec.drift / sigma, 4),
            "integrated_snr": round((spec.drift / sigma) * sqrt(Ke), 4)}
