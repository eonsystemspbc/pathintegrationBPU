#!/usr/bin/env python3
"""Experiment 5 -- the odor->valence associative-reversal task (Phase 2).

This is a COPY of the task-generation half of
``scripts/associative/run_mb_associative_learning.py`` (episode structure, odor bank,
batch generator), kept self-contained inside Experiment 5's folder so the experiment's
frozen record does not depend on a shared script that could change (build-experiment
data rule). What is copied verbatim (structure preserved for comparability with the
original benchmark):

  * ``EpisodeSpec``            -- episode geometry (odors, reversal, sparsity, noise).
  * ``make_odor_bank``         -- the fixed sparse odor prototypes.
  * ``generate_batch``         -- one episode-batch: a LEARN phase (each odor shown once
                                  with reward XOR punishment), an INITIAL-QUERY phase (odor
                                  + query gate -> recall its valence), a REVERSAL phase (a
                                  subset re-paired with the flipped valence), and a
                                  FINAL-QUERY phase (recall the updated valence).
  * ``batch_to_torch``         -- numpy episode -> device tensors.

What is NEW here (the uniform metric layer so all four Exp-5 paradigms are scored
identically, exactly as Exp 4 scored MQAR with one shared logits+accuracy):

  * The task is cast as a **2-class** problem (valence in {reward=0, punish=1}); every
    paradigm emits ``logits[B,T,N_VALENCE]`` and is scored by argmax at query steps.
    Chance = 1/N_VALENCE = 0.5.
  * ``masked_ce_ov``           -- masked cross-entropy at query steps (the training loss).
  * ``ov_correct_total``       -- (correct, total) at masked query steps (recall accuracy);
                                  evaluated separately with the initial / final query masks to
                                  report initial-recall vs after-reversal accuracy.

INPUT LAYOUT (per timestep): ``[ odor(odor_dim) | reward(1) | punish(1) | query(1) ]``.
The model routes odor->ALPN (cue), [reward,punish]->DAN teaching (=the valence class
one-hot + the reinforcement gate), query->recall<-MBON. reward and punish are mutually
exclusive at a learn step; both zero at a query step (query gate = 1 there).

DIFFERENCE FROM THE ORIGINAL SCRIPT'S MODEL HALF: the original trains a fully-generic
recurrent net (dense W_in / dense readout / trainable recurrence, whole-matrix controls).
Experiment 5 does NOT use that model -- it drives this task through the biological-I/O
four-paradigm engine ported from Experiment 4 (arm_bptt / arm_plasticity). Only the task
above is shared.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

# 2-class valence: column 0 = reward (appetitive), column 1 = punishment (aversive).
N_VALENCE = 2
ROLE_DIMS = 3  # reward, punishment, query appended after the odor vector
CHANCE = 1.0 / N_VALENCE


@dataclass(frozen=True)
class EpisodeSpec:
    num_odors: int
    odor_dim: int
    odors_per_episode: int
    reversal_count: int
    reversal_repeats: int
    odor_sparsity: float
    odor_noise_std: float

    @property
    def input_dim(self) -> int:
        return self.odor_dim + ROLE_DIMS

    @property
    def timesteps(self) -> int:
        return (
            self.odors_per_episode                              # LEARN
            + self.odors_per_episode                            # INITIAL QUERY
            + self.reversal_count * self.reversal_repeats       # REVERSAL (subset re-paired)
            + self.odors_per_episode                            # FINAL QUERY
        )


@dataclass(frozen=True)
class Batch:
    inputs: np.ndarray            # [B, T, odor_dim + ROLE_DIMS]
    targets: np.ndarray           # [B, T]  valence class in {0,1} (0 elsewhere; masked out)
    query_mask: np.ndarray        # [B, T]  1 at ALL query steps (initial + final) -> loss + pooled acc
    initial_query_mask: np.ndarray  # [B, T]  1 at the INITIAL query steps (all odors, pre-reversal)
    reversed_query_mask: np.ndarray  # [B, T]  1 at the FINAL query steps for the REVERSED odors only
                                     #         (the clean overwrite/update test; Q4 delta-vs-Hebbian)


def make_odor_bank(spec: EpisodeSpec, seed: int) -> np.ndarray:
    """Fixed bank of sparse, unit-norm odor prototypes [num_odors, odor_dim]."""
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


def _write_stimulus(dest: np.ndarray, odor: np.ndarray, noise_std: float,
                    rng: np.random.Generator) -> None:
    dest[: odor.shape[0]] = odor
    if noise_std > 0:
        dest[: odor.shape[0]] += rng.normal(0.0, noise_std, size=odor.shape).astype(np.float32)


def generate_batch(odor_bank: np.ndarray, spec: EpisodeSpec, batch_size: int,
                   rng: np.random.Generator) -> Batch:
    """One episode-batch. Phase boundaries are identical across the batch (same spec); the
    odors, valences, and step orderings are drawn independently per sample."""
    T = spec.timesteps
    inputs = np.zeros((batch_size, T, spec.input_dim), dtype=np.float32)
    targets = np.zeros((batch_size, T), dtype=np.float32)
    query_mask = np.zeros((batch_size, T), dtype=np.float32)
    initial_query_mask = np.zeros((batch_size, T), dtype=np.float32)
    reversed_query_mask = np.zeros((batch_size, T), dtype=np.float32)

    reward_col = spec.odor_dim
    punishment_col = spec.odor_dim + 1
    query_col = spec.odor_dim + 2

    for b in range(batch_size):
        odor_ids = rng.choice(spec.num_odors, size=spec.odors_per_episode, replace=False)
        initial_valence = rng.integers(0, 2, size=spec.odors_per_episode, dtype=np.int64)
        reversal_local = rng.choice(spec.odors_per_episode, size=spec.reversal_count, replace=False)
        reversed_set = set(int(x) for x in reversal_local)
        final_valence = initial_valence.copy()
        final_valence[reversal_local] = 1 - final_valence[reversal_local]

        step = 0
        # --- LEARN: each odor shown once with its initial valence (odor + reinforcement co-occur) ---
        for local_idx in rng.permutation(spec.odors_per_episode):
            _write_stimulus(inputs[b, step], odor_bank[odor_ids[local_idx]], spec.odor_noise_std, rng)
            if initial_valence[local_idx] == 1:
                inputs[b, step, punishment_col] = 1.0
            else:
                inputs[b, step, reward_col] = 1.0
            step += 1

        # --- INITIAL QUERY: odor + query gate -> recall initial valence ---
        for local_idx in rng.permutation(spec.odors_per_episode):
            _write_stimulus(inputs[b, step], odor_bank[odor_ids[local_idx]], spec.odor_noise_std, rng)
            inputs[b, step, query_col] = 1.0
            targets[b, step] = float(initial_valence[local_idx])
            query_mask[b, step] = 1.0
            initial_query_mask[b, step] = 1.0
            step += 1

        # --- REVERSAL: the reversed subset re-paired with the flipped valence ---
        for _ in range(spec.reversal_repeats):
            for local_idx in rng.permutation(reversal_local):
                _write_stimulus(inputs[b, step], odor_bank[odor_ids[local_idx]], spec.odor_noise_std, rng)
                if final_valence[local_idx] == 1:
                    inputs[b, step, punishment_col] = 1.0
                else:
                    inputs[b, step, reward_col] = 1.0
                step += 1

        # --- FINAL QUERY: odor + query gate -> recall the (possibly updated) valence ---
        for local_idx in rng.permutation(spec.odors_per_episode):
            _write_stimulus(inputs[b, step], odor_bank[odor_ids[local_idx]], spec.odor_noise_std, rng)
            inputs[b, step, query_col] = 1.0
            targets[b, step] = float(final_valence[local_idx])
            query_mask[b, step] = 1.0
            if int(local_idx) in reversed_set:                 # reversed odors only -> the update test
                reversed_query_mask[b, step] = 1.0
            step += 1

        if step != T:
            raise AssertionError(f"internal timestep mismatch: {step} != {T}")

    return Batch(inputs, targets, query_mask, initial_query_mask, reversed_query_mask)


def batch_to_torch(batch: Batch, device) -> tuple[torch.Tensor, ...]:
    """(inputs, targets_long, query_mask, initial_mask, reversed_mask) on `device`.
    targets are cast to long class indices for cross-entropy / argmax scoring."""
    return (
        torch.from_numpy(batch.inputs).to(device),
        torch.from_numpy(batch.targets).to(device).long(),
        torch.from_numpy(batch.query_mask).to(device),
        torch.from_numpy(batch.initial_query_mask).to(device),
        torch.from_numpy(batch.reversed_query_mask).to(device),
    )


# --------------------------------------------------------------------------------------
# uniform metric layer -- one loss + one accuracy for ALL four paradigms (2-class valence)
# --------------------------------------------------------------------------------------
def masked_ce_ov(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked cross-entropy over N_VALENCE classes at query steps.
    logits [B,T,N_VALENCE], targets [B,T] (long class idx), mask [B,T] (1 at query steps)."""
    B, T, K = logits.shape
    raw = nn.functional.cross_entropy(
        logits.reshape(B * T, K), targets.reshape(B * T).long(), reduction="none"
    ).reshape(B, T)
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


def ov_correct_total(logits: torch.Tensor, targets: torch.Tensor,
                     mask: torch.Tensor) -> tuple[float, float]:
    """(#correct, #scored) over the masked query steps: argmax(logits)==target."""
    pred = logits.argmax(dim=-1)                          # [B,T]
    correct = ((pred == targets.long()).float() * mask).sum()
    total = mask.sum()
    return float(correct.item()), float(total.item())
