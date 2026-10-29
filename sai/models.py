"""Model A: attention-only transformer (SPEC §2).

No MLPs, no LayerNorm, no biases, learned positional embeddings. Written directly rather than
through TransformerLens so that model C can share its layout and hook points.
"""

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 64
    d_head: int = 16
    n_ctx: int = 64
    vocab: int = 64
    init_std: float = 0.1  # 0.8 / sqrt(d_model)

    def to_dict(self):
        return asdict(self)


class AttnOnly(nn.Module):
    """Logits from the residual stream ``x = W_E[t] + W_pos + sum of head outputs``."""

    def __init__(self, cfg, generator):
        super().__init__()
        self.cfg = cfg
        L, H, d, dh = cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_head
        shapes = {
            "W_E": (cfg.vocab, d),
            "W_pos": (cfg.n_ctx, d),
            "W_Q": (L, H, d, dh),
            "W_K": (L, H, d, dh),
            "W_V": (L, H, d, dh),
            "W_O": (L, H, dh, d),
            "W_U": (d, cfg.vocab),
        }
        for name, shape in shapes.items():  # dict order fixes the RNG order
            p = nn.Parameter(torch.empty(shape))
            nn.init.normal_(p, 0.0, cfg.init_std, generator=generator)
            self.register_parameter(name, p)

    def forward(self, tokens, cache=None, means=None, ablate=(), key_patch=None):
        """``tokens`` is ``(B, Q)``. If ``cache`` is a dict, it receives ``pattern{l}``
        ``(B, H, Q, K)`` and ``head_out{l}`` ``(B, H, Q, d)`` for each layer ``l``.

        Interventions (iteration 000, measurements 4-5) use ``means``, a list over layers of
        ``(H, Q, d)`` mean head outputs. ``ablate`` holds ``(layer, head)`` pairs whose output is
        replaced by its mean everywhere. ``key_patch = (src_layer, src_heads, dst_layer,
        dst_heads)`` replaces the source heads' outputs by their means only in the input to the
        destination heads' keys (path patching for K-composition; heads are sequences).
        """
        Q = tokens.shape[1]
        x = self.W_E[tokens] + self.W_pos[:Q]
        causal = torch.ones(Q, Q, dtype=torch.bool).tril()
        scale = self.cfg.d_head**-0.5
        outs = []
        for layer in range(self.cfg.n_layers):
            q = torch.einsum("bqd,hde->bhqe", x, self.W_Q[layer])
            k = torch.einsum("bkd,hde->bhke", x, self.W_K[layer])
            v = torch.einsum("bkd,hde->bhke", x, self.W_V[layer])
            if key_patch is not None and key_patch[2] == layer:
                sl, src, _, dst = key_patch
                x_k = x
                for sh in src:
                    x_k = x_k - outs[sl][:, sh] + means[sl][sh][:Q]
                k = k.clone()
                for dh in dst:
                    k[:, dh] = x_k @ self.W_K[layer, dh]
            scores = (q @ k.transpose(-1, -2) * scale).masked_fill(~causal, float("-inf"))
            pattern = scores.softmax(-1)
            head_out = torch.einsum("bhqe,hed->bhqd", pattern @ v, self.W_O[layer])
            for al, ah in ablate:
                if al == layer:
                    head_out = head_out.clone()
                    head_out[:, ah] = means[layer][ah][:Q]
            if cache is not None:
                cache[f"pattern{layer}"] = pattern
                cache[f"head_out{layer}"] = head_out
            outs.append(head_out)
            x = x + head_out.sum(1)
        return x @ self.W_U
