# Changelog

This file follows the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. The project is research, so entries are dated rather than versioned. Record decisions and reversals here as well as file changes.

## 2026-10-29

### Added

- **`docs/SPEC.md`:** goal, hypotheses H1–H4 with falsification criteria, model variants A–D (ANN, converted SNN as a null control, directly trained spiking transformer, SuperNeuroABM rebuild), attention in spikes (§2.1), simulators, tasks T1–T3, analyses including the spike-transfer graph and Infomap communities (§5.1), controls, relation to known results, environment, milestones M0–M5, open questions and references.
- **`docs/iterations/000-plan.md`:** the M0 setup and the M1 plan for model A on the induction task, with pre-registered criteria for an induction circuit.
- **`README.md`:** links to the docs.
- **Decision:** model D (SuperNeuroABM) is built only if M3 names a question that model C's recorded spikes cannot answer. SuperNeuroABM needs a GPU, and has no import path from PyTorch.
