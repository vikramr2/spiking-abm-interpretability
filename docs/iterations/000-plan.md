# Iteration 000: plan for M0 and M1 (ANN reference circuit on induction)

**Date:** 2026-10-29.

- **Milestones:** M0 and M1 (SPEC §9).
- **Status:** plan. Nothing has been run. This record is updated with commands, results and decisions as the iteration runs.
- **Compute:** CPU. Model A at this size trains in minutes, so no cluster job is planned.

## Problem

Every later claim (H1–H4) is a comparison against model A's circuit. So model A has to be trained first, and its circuit identified with fixed, pre-registered criteria, before any spiking model is built. If model A does not form an induction circuit reliably, the project's ground truth is missing and the plan changes.

## M0: setup

| Item | Content |
| --- | --- |
| `environment.yml` | Python 3.11, torch (CPU), snntorch, transformer_lens, infomap, numpy, scipy, pytest, ruff, with exact versions pinned. SuperNeuroABM is left out until M4. |
| Layout | `sai/` is the library (`data.py`, `models.py`, `scores.py`, `graph.py`), `experiments/` holds the milestone scripts, and results go to `experiments/results/`, committed as JSON. |
| CLI | `python -m experiments.m1 --seed S`, and `python -m experiments.m1 --summary`. |
| Determinism | One seeded `torch.Generator` for the data and one for the initialisation, `torch.use_deterministic_algorithms(True)`, and one thread. |
| Behaviour lock | `tests/golden.json` holds the final loss and the head scores of a 200-step run at seed 0, matched to 1e-6. |

**Exit:** `pytest -q tests` passes, and two runs of the golden case give identical results.

## M1: model A on T1

**Data.** For each sequence, draw a prefix of 32 tokens uniformly from a vocabulary of 64, and repeat it once, so the context is 64. A fresh batch is drawn each step from a seeded generator. The repeated-half loss is the mean cross-entropy on positions 33–63, where the target is predictable only by induction. The first-half loss is the control: it should stay near $\log_2 64 = 6$ bits.

**Model.** The SPEC §2 architecture: attention-only, 2 layers, 4 heads, $d_{model} = 64$, no MLP, no LayerNorm. It is built as a TransformerLens `HookedTransformer` with `attn_only=True` and `normalization_type=None`. **Proposal:** if training is unstable without LayerNorm, rerun with LayerNorm, and record it as a deviation.

**Training.** AdamW, learning rate 1e-3, weight decay 0.01, batch 64, 6000 steps, with a full evaluation every 100 steps on a fixed held-out batch of 512 sequences. Seeds 0–4.

**Measurements**, saved per evaluation step:

1. Repeated-half and first-half loss.
2. Previous-token score per head: mean attention from position $t$ to $t - 1$ over the second half.
3. Induction score per head: mean attention from position $t$ in the second half to position $t - 31$, the token after the first occurrence of the current token.
4. At the end only, a mean ablation of each head, measured as the change in repeated-half loss.
5. At the end only, K-composition by path patching: replace the layer-0 head outputs that feed the keys of layer 1 with their means, and measure the change in repeated-half loss per (layer-0 head, layer-1 head) pair.

## Pre-registered criteria

These are fixed before any run. A change to them is logged in the CHANGELOG with its reason.

A seed **has an induction circuit** if all of the following hold:

- **Learning.** The final repeated-half loss is below 1.0 bit, and the first-half loss is above 5.5 bits.
- **Induction head.** At least one layer-1 head has an induction score above 0.4, and its mean ablation raises the repeated-half loss by at least 1.0 bit.
- **Previous-token head.** At least one layer-0 head has a previous-token score above 0.4.
- **Composition.** Path-patching that layer-0 head into the keys of that layer-1 head raises the repeated-half loss by at least 0.5 bits.
- **Phase change.** The repeated-half loss drops by at least 2 bits within a window of at most 500 steps, and the rise of the induction score falls within the same window.

**M1 passes** if at least 4 of 5 seeds have an induction circuit. These head labels (previous-token, induction, other) are the ground truth for H1 and H2.

**If M1 fails:** first check the training budget (12 000 steps) and the prefix length (16 and 48). If it still fails, add LayerNorm. If it still fails, stop and redesign the task before building any spiking model.

## Next (iteration 001, M2)

These steps are listed here so that M1 records what M2 needs:

1. **Model B**: convert model A for each seed (SPEC §3.1), and report the conversion gap in repeated-half loss.
2. **Model C**: train the spiking transformer directly on the same data and seeds, at $T = 8$, with spike-driven attention (SPEC §2.1).
3. **Scoring**: apply the M1 criteria to B and C unchanged, with attention measured as normalised coincidence counts. This decides H1.

**Things to keep from M1 for M2:** the trained checkpoints of A, the held-out evaluation batch, and the per-seed head labels. All three are committed.

## Risks

- **No LayerNorm.** Attention-only models without LayerNorm may train more slowly. The fallback is in the M1 failure path above.
- **Score thresholds.** The 0.4 thresholds for head scores are a judgement call. The full score distributions are saved, so a threshold can be revisited openly.
- **TransformerLens versions.** Its default configuration can change between versions. The version is pinned in M0, and the configuration is written out in full in the code.
