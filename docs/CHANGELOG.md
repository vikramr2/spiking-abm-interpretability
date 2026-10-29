# Changelog

This file follows the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. The project is research, so entries are dated rather than versioned. Record decisions and reversals here as well as file changes.

## 2026-10-29

### Added (M1 confirmation)

- **Results:** `experiments/results/m1_s{10..14}.json` and `.pt`.
- **Finding:** M1 **passes** at role level, 5 of 5 confirmation seeds; the original criteria pass 1 of 5. All four layer-1 heads are induction heads on every seed; the previous-token role is 1–3 layer-0 heads. Group key patching costs 4.8–5.1 bits, close to ablating the whole role, so the circuit runs through K-composition. SPEC status updated; M2 scoring uses the role-level criteria, and H1's falsification criterion is restated at role level (before any M2 run).

### Changed (M1 criteria amendment)

- **Decision (user):** amend M1 to role-level criteria (group ablation per role, group key patching), confirmed on fresh seeds 10–14; seeds 5–9 stay held out for SPEC §5.2. The original criteria are kept and still reported. The previous-token role threshold is 0.3 (was 0.4 per head), chosen after seeing seed 3; disclosed in iteration 000.
- **Code:** `key_patch` takes lists of source and destination heads (single-pair results unchanged); `experiments/m1.py` adds `ROLE_CRIT`, `role_heads`, `role_verdict`, role-level analysis, `--reanalyse` and the role summary. Seeds 0–4 reanalysed (original fields bit-identical): all pass at role level, not confirmatory.

### Added (M1 results)

- **Results:** `experiments/results/m1_s{0..4}.json` and `.pt` (6000 steps each).
- **Finding:** M1 **fails** under the pre-registered criteria, 3 of 5 seeds. The circuit is redundant, not missing: all four layer-1 heads are induction heads, so single-head ablation costs about 0.9 bits (threshold 1.0), and in seeds 0 and 3 the previous-token role is split across 2–3 layer-0 heads. A post-hoc group ablation (exploratory) removes either role at 4.7–5.2 bits on every seed. Iteration 000 records both.
- **Decision pending:** whether to amend the criteria to role level (confirmed on fresh seeds), change the architecture, or follow the pre-registered failure path.

### Added (M1 code)

- **Code:** `AttnOnly.forward` takes `means`, `ablate` and `key_patch` (mean ablation and K-composition path patching; the default path is unchanged, golden bit-identical). `experiments/m1.py` adds `CRIT` (the pre-registered thresholds), `analyse()` (per-head ablation, 4 × 4 key-patching deltas) and `verdict()` (per-criterion checks and head labels); `--summary` prints verdicts. New test: interventions are local.
- **Docs:** iteration 000 records how the criteria are coded (per-position means, candidate pairs, the half-rise rule for the phase change), written before the M1 results were read.

### Added (M0)

- **Code:** `sai/` (`data.py`, `models.py`, `scores.py`), `experiments/m1.py` (training, CLI, summary), `tests/test_regression.py` with `tests/golden.json` (behaviour lock), `environment.yml` with pinned versions, `ruff.toml`, `.gitignore`, `CLAUDE.md` (commands and traps).
- **Docs:** iteration 000 M0 record (environment, files, exit check, smoke test); SPEC status, T1 row, head-score definition and environment.
- **Change (task T1):** the repeat period is drawn per sequence from 8 to 32 instead of fixed at 32. With a fixed period, a smoke run reached near-zero loss with a positional induction head and no previous-token head, so the task did not require the circuit. Score definitions in iteration 000 are rewritten for the variable period; the thresholds are unchanged. Seed 0's smoke trajectory has been seen, as disclosed in the record.
- **Reversal:** model A is written in plain PyTorch, not TransformerLens, so that model C shares its layout and hook points; TransformerLens is dropped from the environment.

### Changed

- **`docs/SPEC.md`:** learning is stated explicitly. Models A and C are trained by backpropagation (C with surrogate gradients); no Map-STDP or other local plasticity is used. Community objectives are analysis only.
- **`docs/SPEC.md` §5.2 (new):** the map equation becomes one of five candidate objectives (first-order and second-order map equations, Markov stability, nested SBM with MDL, information bottleneck clustering), with a selection protocol: select on seeds 0–4, confirm on held-out seeds 5–9. §5.1 adds task-weighted edges (a Proposal) and fitting across training checkpoints. H2 now also covers formation at the phase change. Milestones, open questions, environment and references are updated to match.
- **Decision (user):** the project looks for the objective that best explains task-oriented community formation, rather than assuming the map equation.

### Added

- **`docs/SPEC.md`:** goal, hypotheses H1–H4 with falsification criteria, model variants A–D (ANN, converted SNN as a null control, directly trained spiking transformer, SuperNeuroABM rebuild), attention in spikes (§2.1), simulators, tasks T1–T3, analyses including the spike-transfer graph and Infomap communities (§5.1), controls, relation to known results, environment, milestones M0–M5, open questions and references.
- **`docs/iterations/000-plan.md`:** the M0 setup and the M1 plan for model A on the induction task, with pre-registered criteria for an induction circuit.
- **`README.md`:** links to the docs.
- **Decision:** model D (SuperNeuroABM) is built only if M3 names a question that model C's recorded spikes cannot answer. SuperNeuroABM needs a GPU, and has no import path from PyTorch.
