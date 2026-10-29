"""Task T1, induction (SPEC §4; iteration 000).

Each sequence has ``n_ctx`` uniform random tokens over ``vocab``. A period ``m`` is drawn per
sequence from ``[m_min, m_max]``, and tokens ``m .. 2m - 1`` are overwritten with a copy of
tokens ``0 .. m - 1``; tokens from ``2m`` on stay random. The period varies so that a fixed
positional offset cannot stand in for induction (iteration 000, smoke test).

The model reads tokens ``0 .. n_ctx - 2`` and predicts tokens ``1 .. n_ctx - 1``. Target ``k``
is predictable by induction for ``m + 1 <= k <= 2m - 1``: the current token ``x[k-1]`` is in
the copy, and the token after its first occurrence is ``x[k - m]``.
"""

import torch

EVAL_SEED = 12345  # one held-out batch shared by all training seeds
M_MIN, M_MAX = 8, 32


def induction_batch(batch, n_ctx, vocab, generator, m_min=M_MIN, m_max=M_MAX):
    """Return ``(seq, m)``: a ``(batch, n_ctx)`` long tensor and the ``(batch,)`` periods."""
    m = torch.randint(m_min, m_max + 1, (batch,), generator=generator)
    seq = torch.randint(vocab, (batch, n_ctx), generator=generator)
    pos = torch.arange(n_ctx)
    copy = (pos >= m[:, None]) & (pos < 2 * m[:, None])
    src = (pos - m[:, None]).clamp(min=0)
    return torch.where(copy, seq.gather(1, src), seq), m


def eval_batch(batch, n_ctx, vocab):
    """The fixed held-out batch (iteration 000: 512 sequences)."""
    return induction_batch(batch, n_ctx, vocab, torch.Generator().manual_seed(EVAL_SEED))


def target_masks(m, n_ctx):
    """``(batch, n_ctx - 1)`` masks over targets: (repeated, first copy). Index ``i`` is token
    ``i + 1``."""
    k = torch.arange(1, n_ctx)[None]
    m = m[:, None]
    return (k >= m + 1) & (k <= 2 * m - 1), k <= m - 1
