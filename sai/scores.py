"""Losses and head scores (SPEC §5; iteration 000 measurements 1-3). Losses are in bits."""

import math

import torch
import torch.nn.functional as F

from sai.data import target_masks


def split_losses(logits, seq, m):
    """Mean cross-entropy in bits on the (repeated, first copy) targets."""
    nll = F.cross_entropy(logits.transpose(1, 2), seq[:, 1:], reduction="none") / math.log(2)
    rep, first = target_masks(m, seq.shape[1])
    return nll[rep].mean().item(), nll[first].mean().item()


def head_scores(pattern, m):
    """Previous-token and induction scores per head for one layer's ``(B, H, Q, K)`` pattern.

    Previous-token is the attention ``q -> q - 1``, averaged over all ``q >= 1``. Induction is
    ``q -> q - m + 1``, the token after the first occurrence of the current token, averaged over
    queries in the copy (``m <= q <= 2m - 2``).
    """
    B, H, Q, _ = pattern.shape
    q = torch.arange(Q)
    prev = pattern[:, :, q[1:], q[1:] - 1].mean((0, 2))
    in_copy = (q[None] >= m[:, None]) & (q[None] <= 2 * m[:, None] - 2)
    key = (q[None] - m[:, None] + 1).clamp(min=0)
    att = pattern.gather(3, key[:, None, :, None].expand(B, H, Q, 1))[..., 0]
    ind = (att * in_copy[:, None]).sum((0, 2)) / in_copy.sum()
    return prev.tolist(), ind.tolist()
