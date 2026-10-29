# Changelog

This file follows the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. The project is research, so entries are dated rather than versioned. Record decisions and reversals here as well as file changes.

## 2026-10-29

### Changed

- **`docs/SPEC.md`:** learning is stated explicitly. Models A and C are trained by backpropagation (C with surrogate gradients); no Map-STDP or other local plasticity is used. Community objectives are analysis only.
- **`docs/SPEC.md` §5.2 (new):** the map equation becomes one of five candidate objectives (first-order and second-order map equations, Markov stability, nested SBM with MDL, information bottleneck clustering), with a selection protocol: select on seeds 0–4, confirm on held-out seeds 5–9. §5.1 adds task-weighted edges (a Proposal) and fitting across training checkpoints. H2 now also covers formation at the phase change. Milestones, open questions, environment and references are updated to match.
- **Decision (user):** the project looks for the objective that best explains task-oriented community formation, rather than assuming the map equation.

### Added

- **`docs/SPEC.md`:** goal, hypotheses H1–H4 with falsification criteria, model variants A–D (ANN, converted SNN as a null control, directly trained spiking transformer, SuperNeuroABM rebuild), attention in spikes (§2.1), simulators, tasks T1–T3, analyses including the spike-transfer graph and Infomap communities (§5.1), controls, relation to known results, environment, milestones M0–M5, open questions and references.
- **`docs/iterations/000-plan.md`:** the M0 setup and the M1 plan for model A on the induction task, with pre-registered criteria for an induction circuit.
- **`README.md`:** links to the docs.
- **Decision:** model D (SuperNeuroABM) is built only if M3 names a question that model C's recorded spikes cannot answer. SuperNeuroABM needs a GPU, and has no import path from PyTorch.
