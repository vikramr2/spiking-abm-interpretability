"""Behaviour lock (iteration 000, M0).

    pytest -q tests
    python -m tests.test_regression --update     # only for an intended behaviour change
"""

import json
import sys
from pathlib import Path

import pytest
import torch

from experiments.m1 import train
from sai.data import induction_batch, target_masks
from sai.models import AttnOnly, Config
from sai.scores import head_scores

GOLDEN = Path(__file__).with_name("golden.json")
GOLDEN_RUN = dict(seed=0, steps=200, eval_every=100)
TOL = 1e-6


def fingerprint():
    run, _ = train(**GOLDEN_RUN)
    return run["history"]


def test_data_copies_first_period():
    seq, m = induction_batch(16, 64, 64, torch.Generator().manual_seed(0))
    for row, mb in zip(seq, m.tolist(), strict=True):
        assert torch.equal(row[mb : 2 * mb], row[:mb])
    rep, first = target_masks(m, 64)
    assert rep.sum(1).tolist() == (m - 1).tolist()
    assert first.sum(1).tolist() == (m - 1).tolist()


def test_head_scores_on_ideal_patterns():
    _, m = induction_batch(8, 64, 64, torch.Generator().manual_seed(1))
    Q = 63
    q = torch.arange(Q)
    prev = torch.zeros(8, 1, Q, Q)
    prev[:, 0, q[1:], q[1:] - 1] = 1
    ind = torch.zeros(8, 1, Q, Q)
    for b, mb in enumerate(m.tolist()):
        ind[b, 0, q, (q - mb + 1).clamp(min=0)] = 1
    assert head_scores(prev, m)[0] == [pytest.approx(1.0)]
    assert head_scores(ind, m)[1] == [pytest.approx(1.0)]


def test_interventions_are_local():
    """Key patching changes only the destination head's pattern; ablating a head changes only
    its own output; no intervention changes layer 0's patterns."""
    model = AttnOnly(Config(), torch.Generator().manual_seed(0))
    seq, _ = induction_batch(4, 64, 64, torch.Generator().manual_seed(2))
    tokens = seq[:, :-1]
    base = {}
    with torch.no_grad():
        model(tokens, base)
        means = [base[f"head_out{layer}"].mean(0) for layer in range(2)]
        patched, ablated = {}, {}
        model(tokens, patched, means=means, key_patch=(0, 1, 1, 2))
        model(tokens, ablated, means=means, ablate=[(0, 3)])
    assert torch.equal(patched["pattern0"], base["pattern0"])
    for h in range(4):
        same = torch.equal(patched["pattern1"][:, h], base["pattern1"][:, h])
        assert same == (h != 2)
        same = torch.equal(ablated["head_out0"][:, h], base["head_out0"][:, h])
        assert same == (h != 3)


def test_deterministic():
    a, _ = train(seed=3, steps=20, eval_every=10)
    b, _ = train(seed=3, steps=20, eval_every=10)
    assert a["history"] == b["history"]


def test_golden():
    golden = json.loads(GOLDEN.read_text())
    got = fingerprint()
    assert len(got) == len(golden)
    for g, h in zip(got, golden, strict=True):
        assert g["step"] == h["step"]
        for key in ("loss_rep", "loss_first"):
            assert g[key] == pytest.approx(h[key], abs=TOL), (g["step"], key)
        for key in ("prev", "ind"):
            for gl, hl in zip(g[key], h[key], strict=True):
                assert gl == pytest.approx(hl, abs=TOL), (g["step"], key)


if __name__ == "__main__":
    if "--update" in sys.argv:
        GOLDEN.write_text(json.dumps(fingerprint(), indent=1) + "\n")
        print(f"wrote {GOLDEN}")
