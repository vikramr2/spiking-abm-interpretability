# Iteration 000: plan for M0 and M1 (ANN reference circuit on induction)

**Date:** 2026-10-29.

- **Milestones:** M0 and M1 (SPEC §9).
- **Status:** M0 done; M1 failed its original criteria, and the role-level amendment is being confirmed on seeds 10–14. This record is updated with commands, results and decisions as the iteration runs.
- **Compute:** CPU. Model A at this size trains in minutes, so no cluster job is planned.

## Problem

Every later claim (H1–H4) is a comparison against model A's circuit. So model A has to be trained first, and its circuit identified with fixed, pre-registered criteria, before any spiking model is built. If model A does not form an induction circuit reliably, the project's ground truth is missing and the plan changes.

## M0: setup

| Item | Content |
| --- | --- |
| `environment.yml` | Python 3.11, torch (CPU), snntorch, infomap, numpy, scipy, pytest, ruff, with exact versions pinned. SuperNeuroABM is left out until M4. |
| Layout | `sai/` is the library (`data.py`, `models.py`, `scores.py`; `graph.py` comes at M3), `experiments/` holds the milestone scripts, and results go to `experiments/results/`, committed as JSON. |
| CLI | `python -m experiments.m1 --seed S`, and `python -m experiments.m1 --summary`. A run writes `m1_s<seed>.json` (config and history) and `m1_s<seed>.pt` (weights). |
| Determinism | One seeded `torch.Generator` for the data and one for the initialisation, `torch.use_deterministic_algorithms(True)`, and one thread. |
| Behaviour lock | `tests/golden.json` holds the final loss and the head scores of a 200-step run at seed 0, matched to 1e-6. |

**Exit:** `pytest -q tests` passes, and two runs of the golden case give identical results.

## M0 record

**Environment.** On this machine (a cloud container), a venv with Python 3.11 and the pins in `environment.yml`: torch 2.15.0, snntorch 1.0.0, infomap 2.16.0, numpy 2.4.6, scipy 1.17.1, pytest 9.2.0, ruff 0.17.3. The CPU-only torch index was unreachable here, so torch came from PyPI with its CUDA libraries, and runs on CPU.

**Built.**

| File | Contents |
| --- | --- |
| `sai/data.py` | T1 batches with a per-sequence period, the fixed held-out batch (seed 12345, shared by all training seeds), target masks. |
| `sai/models.py` | `Config` and `AttnOnly` (model A). A `cache` dict receives each layer's attention pattern and head outputs. |
| `sai/scores.py` | Losses in bits on the repeated and first-copy targets; previous-token and induction scores per head. |
| `experiments/m1.py` | `train()`, the CLI and `--summary`. Initialisation and data use separate seeded generators (seed, seed + 10⁶); deterministic algorithms, one thread. |
| `tests/test_regression.py`, `tests/golden.json` | Data and score checks on ideal patterns, a determinism check (two 20-step runs identical), and the golden 200-step run at seed 0, matched to 1e-6. |

**Exit check.** `pytest -q tests` passes (4 tests, about 30 s), and `ruff check .` is clean.

**Change found by the smoke test: the period now varies.** The planned data repeated a 32-token prefix at a fixed offset. A 3000-step smoke run at seed 0 reached a repeated-half loss of 0.01 bits with a layer-1 "induction" score of 1.0, but no layer-0 head ever exceeded a previous-token score of 0.04. With learned positional embeddings and a fixed period, the head can attend 31 positions back by position alone, so the task does not require an induction circuit. The period is now drawn per sequence from 8 to 32. A 4000-step smoke run at seed 0 with the new data:

| Step | Repeated-half loss (bits) | First-half loss (bits) | Max previous-token score, layer 0 | Max induction score, layer 1 |
| --- | --- | --- | --- | --- |
| 1500 | 5.28 | 6.18 | 0.09 | 0.12 |
| 1750 | 2.56 | 6.48 | 0.24 | 0.64 |
| 2000 | 1.41 | 6.40 | 0.35 | 0.79 |
| 4000 | 0.91 | 6.30 | 0.44 | 0.83 |

A previous-token head and an induction head now form together, at a drop of 2.7 bits within 250 steps. These are smoke numbers, not M1 results: ablation and path patching have not been run. **Disclosure:** seed 0 is an M1 seed, so its trajectory has been seen. The M1 thresholds below were not changed after seeing it; only the score definitions were rewritten for the variable period.

## M1: model A on T1

**Data** (revised at M0, see below). Each sequence has 64 tokens drawn uniformly from a vocabulary of 64. A period $m$ is drawn per sequence from 8 to 32, and tokens $m$ to $2m - 1$ are overwritten with a copy of tokens 0 to $m - 1$; the tokens after the copy stay random. A fresh batch is drawn each step from a seeded generator. The repeated-half loss is the mean cross-entropy on targets $m + 1$ to $2m - 1$, which are predictable only by induction. The first-half loss is on targets 1 to $m - 1$; it is the control and should stay near $\log_2 64 = 6$ bits.

**Model.** The SPEC §2 architecture: attention-only, 2 layers, 4 heads, $d_{model} = 64$, no MLP, no LayerNorm. It is written directly in PyTorch (`sai/models.py`), with no biases and initialisation standard deviation $0.8 / \sqrt{d_{model}} = 0.1$. **Proposal:** if training is unstable without LayerNorm, rerun with LayerNorm, and record it as a deviation.

**Training.** AdamW, learning rate 1e-3, weight decay 0.01, batch 64, 6000 steps, with a full evaluation every 100 steps on a fixed held-out batch of 512 sequences. Seeds 0–4.

**Measurements**, saved per evaluation step:

1. Repeated-half and first-half loss.
2. Previous-token score per head: mean attention from position $t$ to $t - 1$, over all $t \ge 1$.
3. Induction score per head: mean attention from position $t$ in the copy ($m \le t \le 2m - 2$) to position $t - m + 1$, the token after the first occurrence of the current token.
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

**How the criteria are coded** (`experiments/m1.py`, `CRIT`, `analyse`, `verdict`). Written down before the M1 results were read:

- **Means** for ablation and patching are per position, over the 512 held-out sequences. Every loss in the criteria is measured on that batch.
- **Several candidate heads.** Every layer-0 head above the previous-token threshold, and every layer-1 head that passes both induction conditions, is a candidate. Composition passes if any (previous-token head, induction head) pair passes.
- **Phase change.** The "rise of the induction score" is placed at the first evaluation step where the largest layer-1 induction score reaches half of its total rise from step 0 to the end. A pair of evaluation steps at most 500 apart, with a loss drop of at least 2 bits between them, must contain that step. Evaluations are every 100 steps.
- **Head labels** for H1 and H2 come from `verdict()`, recomputed from the saved run: "prev" for the layer-0 candidates, "induction" for the layer-1 candidates, "other" for the rest.

**M1 passes** if at least 4 of 5 seeds have an induction circuit. These head labels (previous-token, induction, other) are the ground truth for H1 and H2.

**If M1 fails:** first check the training budget (12 000 steps) and the period range (8–16 and 16–32). If it still fails, add LayerNorm. If it still fails, stop and redesign the task before building any spiking model.

## M1 results

**Runs.** `python -m experiments.m1 --seed S` for S = 0–4, run in parallel on this machine (4 CPUs, about 13 min each). Results: `experiments/results/m1_s<S>.json` and `.pt`. `python -m experiments.m1 --summary` gives the verdicts.

**Verdict under the pre-registered criteria: M1 fails, 3 of 5 seeds (1, 2, 4) pass.** Seeds 0 and 3 fail the induction-head and composition criteria.

| Seed | Repeated / first loss (bits) | Previous-token scores, layer 0 | Induction scores, layer 1 | Single-head ablation, layer 1 (bits) | Passing pair | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.81 / 6.28 | 0.46, 0.45, 0.03, 0.06 | 0.82, 0.83, 0.83, 0.81 | 0.88, 0.91, 0.88, 0.99 | none | fail |
| 1 | 0.82 / 6.29 | 0.02, 0.05, 0.03, 0.47 | 0.85, 0.85, 0.85, 0.85 | 1.03, 0.76, 0.91, 0.90 | (3, 0) | pass |
| 2 | 0.85 / 6.29 | 0.04, 0.04, 0.03, 0.47 | 0.80, 0.79, 0.80, 0.78 | 0.92, 0.92, 0.92, 1.03 | (3, 3) | pass |
| 3 | 0.86 / 6.28 | 0.35, 0.40, 0.41, 0.04 | 0.84, 0.84, 0.84, 0.83 | 0.85, 0.79, 0.93, 0.97 | none | fail |
| 4 | 0.81 / 6.27 | 0.04, 0.46, 0.05, 0.02 | 0.83, 0.83, 0.84, 0.83 | 0.99, 0.86, 0.78, 1.01 | (1, 3) | pass |

Learning and the phase change pass on every seed. The half-rise step is 1600–1800, inside a drop from about 5 bits at step 1500 to about 1.3 bits at step 2000.

**Diagnosis: the circuit is redundant, not missing.** Every layer-1 head is an induction head (scores 0.78–0.85 on every seed). The induction work is shared, so ablating one head costs about 0.9 bits, and the passing seeds clear the 1.0-bit threshold only by margins of 0.01–0.03. In seeds 0 and 3, the previous-token role is also split, across 2 and 3 layer-0 heads; each one then carries only part of the key path, so no single pair reaches the 0.5-bit patching threshold. The criteria assumed one head per role.

**Post-hoc check (exploratory, not pre-registered).** Group mean ablation on the saved weights, with "previous-token-like" relaxed to a score above 0.3 so that seed 3's head 0 (0.35) is included:

| Seed | Previous-token-like heads, layer 0 | Ablate them together (bits) | Ablate all 4 layer-1 heads (bits) | Ablate the other layer-0 heads (bits) |
| --- | --- | --- | --- | --- |
| 0 | 0, 1 | 5.00 | 5.21 | 1.28 |
| 1 | 3 | 4.78 | 5.20 | 1.48 |
| 2 | 3 | 5.03 | 5.17 | 2.01 |
| 3 | 0, 1, 2 | 4.73 | 5.16 | 0.59 |
| 4 | 1 | 4.99 | 5.21 | 1.63 |

Removing either role as a group sends the repeated-half loss back to near chance (about 6 bits) on every seed. So all 5 seeds have a previous-token → induction circuit at the level of roles; only how the roles are spread across heads differs. This bears on H2: the ground-truth unit is a group of heads, which suits community detection better than single-head labels.

**Decision (user): amend to role-level criteria, confirm on fresh seeds.** The pre-registered failure path (a longer budget, other period ranges, LayerNorm) does not address redundancy, so it was not followed. The amended criteria below were fixed and committed before seeds 10–14 were trained. Seeds 5–9 stay held out for SPEC §5.2.

### Amended criteria (role level)

The original criteria are kept and still reported. A seed **has an induction circuit at role level** if:

- **Learning** and **phase change:** unchanged.
- **Roles.** The previous-token role is every layer-0 head with a previous-token score above 0.3. The induction role is every layer-1 head with an induction score above 0.4.
- **Previous-token role.** It is non-empty, and ablating all its heads together raises the repeated-half loss by at least 2.0 bits.
- **Induction role.** The same, at least 2.0 bits.
- **Composition.** Replacing the previous-token role's outputs by their means, only in the input to the induction role's keys, raises the repeated-half loss by at least 1.0 bit.

**M1 passes** if at least 4 of the 5 confirmation seeds (10–14) pass. Head labels for M2 come from `role_verdict()`.

**Seeds 0–4 under the amended criteria** (not confirmatory, because the amendment was designed on them): all 5 pass. Group ablation of the previous-token role costs 4.73–5.03 bits, the induction role 5.16–5.21 bits, and the group key patch 4.83–5.07 bits. `--reanalyse` on the saved weights reproduced the original single-head numbers exactly.

**Why these thresholds.** Chance is about 6 bits and trained models reach about 0.8, so removing a whole role should cost about 5 bits. Thresholds of 2.0 and 1.0 bits ask for well under half of that, so they test for the circuit's presence, not for how cleanly it is split. The 0.3 previous-token threshold (down from 0.4) is the one change chosen after seeing seed 3's head 0 at 0.35, and it is disclosed as such.

## Next (iteration 001, M2)

These steps are listed here so that M1 records what M2 needs:

1. **Model B**: convert model A for each seed (SPEC §3.1), and report the conversion gap in repeated-half loss.
2. **Model C**: train the spiking transformer directly on the same data and seeds, at $T = 8$, with spike-driven attention (SPEC §2.1).
3. **Scoring**: apply the M1 criteria to B and C unchanged, with attention measured as normalised coincidence counts. This decides H1.

**Things to keep from M1 for M2:** the trained checkpoints of A, the held-out evaluation batch, and the per-seed head labels. All three are committed.

## Risks

- **No LayerNorm.** Attention-only models without LayerNorm may train more slowly. The fallback is in the M1 failure path above.
- **Score thresholds.** The 0.4 thresholds for head scores are a judgement call. The full score distributions are saved, so a threshold can be revisited openly.
- **Period-dependent difficulty.** Short periods give fewer predictable targets per sequence, and a long period leaves little context after the copy. The loss is averaged over targets, so long periods weigh more.
