"""M1: model A on task T1 (iteration 000).

    python -m experiments.m1 --seed 0            # one run -> experiments/results/m1_s<seed>.json
    python -m experiments.m1 --summary           # final losses and best head scores per seed
"""

import argparse
import glob
import json
import os

import torch

from sai.data import eval_batch, induction_batch
from sai.models import AttnOnly, Config
from sai.scores import head_scores, split_losses

RESULTS = "experiments/results"
STEPS = 6000


def deterministic():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


@torch.no_grad()
def evaluate(model, seq, m):
    cache = {}
    logits = model(seq[:, :-1], cache)
    rep, first = split_losses(logits, seq, m)
    scores = [head_scores(cache[f"pattern{layer}"], m) for layer in range(model.cfg.n_layers)]
    return {
        "loss_rep": rep,
        "loss_first": first,
        "prev": [s[0] for s in scores],
        "ind": [s[1] for s in scores],
    }


def train(seed, cfg=None, steps=STEPS, eval_every=100, batch=64, eval_size=512, lr=1e-3,
          wd=0.01):
    """Train model A; return the run record (config, history) and the model."""
    cfg = cfg or Config()
    deterministic()
    g_init = torch.Generator().manual_seed(seed)
    g_data = torch.Generator().manual_seed(seed + 1_000_000)
    model = AttnOnly(cfg, g_init)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    held_out, m_held = eval_batch(eval_size, cfg.n_ctx, cfg.vocab)
    history = [{"step": 0, **evaluate(model, held_out, m_held)}]
    for step in range(1, steps + 1):
        seq, _ = induction_batch(batch, cfg.n_ctx, cfg.vocab, g_data)
        logits = model(seq[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), seq[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0:
            history.append({"step": step, **evaluate(model, held_out, m_held)})
    run = dict(seed=seed, steps=steps, eval_every=eval_every, batch=batch, eval_size=eval_size,
               lr=lr, wd=wd, config=cfg.to_dict(), history=history)
    return run, model


def result_path(seed, steps):
    tag = "" if steps == STEPS else f"_st{steps}"
    return f"{RESULTS}/m1_s{seed}{tag}.json"


def summary():
    for path in sorted(glob.glob(f"{RESULTS}/m1_*.json")):
        with open(path) as f:
            last = json.load(f)["history"][-1]
        print(f"{os.path.basename(path)}  step {last['step']}  rep {last['loss_rep']:.3f}  "
              f"first {last['loss_first']:.3f}  max prev L0 {max(last['prev'][0]):.2f}  "
              f"max ind L1 {max(last['ind'][1]):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary()
        return
    run, model = train(args.seed, steps=args.steps)
    os.makedirs(RESULTS, exist_ok=True)
    path = result_path(args.seed, args.steps)
    with open(path, "w") as f:
        json.dump(run, f)
    torch.save(model.state_dict(), path.replace(".json", ".pt"))
    print(path)


if __name__ == "__main__":
    main()
