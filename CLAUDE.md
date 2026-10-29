# CLAUDE.md

Guidance for Claude Code in this repository.

## Project

Research. The question: does modelling a small transformer as a spiking network (and later as a SuperNeuroABM agent model) teach anything about its mechanisms beyond standard interpretability? Sibling project: `vikramr2/map-stdp`. Here nothing learns with Map-STDP: model A trains by backpropagation, model C by surrogate gradients, and community objectives (map equation and alternatives, SPEC §5.2) are analysis only.

## Docs

- `docs/SPEC.md`: goals, hypotheses H1–H4 with falsification criteria, models A–D, analyses, milestones. **Proposal** = undecided, **Open** = unresolved.
- `docs/iterations/NNN-*.md`: one record per iteration (problem, runs, results, decisions). Pre-registered criteria change only with a CHANGELOG entry and a reason.
- `docs/CHANGELOG.md`: a dated entry for every change to docs or code, including decisions and reversals.

## Code

- `sai/`: the library (`data.py` task T1, `models.py` model A, `scores.py` losses in bits and head scores).
- `experiments/m1.py`: model A training and CLI. Results go to `experiments/results/m1_s<seed>[_st<steps>].json` (+ `.pt` weights), committed.

## Commands

```bash
python -m experiments.m1 --seed 0       # one M1 run (about 7 min on one CPU thread)
python -m experiments.m1 --summary
pytest -q tests                         # behaviour lock
python -m tests.test_regression --update   # only for an intended behaviour change; log it
ruff check .
```

## Traps

- **RNG order defines every run.** Initialisation draws parameters in the order of `AttnOnly`'s `shapes` dict; data draws the periods, then the tokens. Moving or adding a draw breaks `tests/golden.json`. Put new features behind flags whose default path draws nothing extra.
- **T1 needs a variable period.** A fixed repeat offset lets a positional head fake induction (iteration 000).
- **Seeds 10–14 confirm the M1 role-level criteria**, fixed before they ran (iteration 000).
- **Seeds 5–9 are held out** for confirming the SPEC §5.2 objective selection. Do not train or look at them before that selection.
