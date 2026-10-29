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


# Pre-registered criteria (iteration 000). Changing one needs a CHANGELOG entry and a reason.
CRIT = dict(rep_max=1.0, first_min=5.5, score_min=0.4, ablate_min=1.0, patch_min=0.5,
            drop_min=2.0, window=500)
# Role-level amendment (iteration 000, after the M1 failure; confirmed on seeds 10-14).
ROLE_CRIT = dict(prev_min=0.3, ind_min=0.4, group_ablate_min=2.0, group_patch_min=1.0)
CONFIRM_SEEDS = range(10, 15)


@torch.no_grad()
def analyse(model, eval_size=512, scores=None):
    """Mean ablation per head and K-composition path patching (measurements 4-5).

    Means are over the held-out batch, per position. Deltas are in repeated-half loss (bits).
    With ``scores`` (the final evaluation), also the role-level group ablations and the group
    key patch of ``role_heads``.
    """
    cfg = model.cfg
    seq, m = eval_batch(eval_size, cfg.n_ctx, cfg.vocab)
    tokens = seq[:, :-1]
    cache = {}
    base = split_losses(model(tokens, cache), seq, m)[0]
    means = [cache[f"head_out{layer}"].mean(0) for layer in range(cfg.n_layers)]

    def delta(**kw):
        return split_losses(model(tokens, means=means, **kw), seq, m)[0] - base

    H = cfg.n_heads
    ablation = [[delta(ablate=[(layer, h)]) for h in range(H)] for layer in range(cfg.n_layers)]
    patch = [[delta(key_patch=(0, [h0], 1, [h1])) for h1 in range(H)] for h0 in range(H)]
    out = dict(base_rep=base, ablation=ablation, patch_0to1_keys=patch)
    if scores is not None:
        prev, ind = role_heads(scores)
        out["roles"] = dict(prev=prev, induction=ind,
                            ablate_prev=delta(ablate=[(0, h) for h in prev]) if prev else 0.0,
                            ablate_induction=delta(ablate=[(1, h) for h in ind]) if ind else 0.0,
                            patch=delta(key_patch=(0, prev, 1, ind)) if prev and ind else 0.0)
    return out


def role_heads(last):
    """Role-level heads from the final evaluation: layer-0 previous-token, layer-1 induction."""
    c = ROLE_CRIT
    prev = [h for h, s in enumerate(last["prev"][0]) if s > c["prev_min"]]
    ind = [h for h, s in enumerate(last["ind"][1]) if s > c["ind_min"]]
    return prev, ind


def verdict(run):
    """Apply the iteration-000 criteria to one run. Returns per-criterion results and labels."""
    c, hist, an = CRIT, run["history"], run["analysis"]
    last = hist[-1]
    prev0, ind1, abl1 = last["prev"][0], last["ind"][1], an["ablation"][1]
    prev_heads = [h for h, s in enumerate(prev0) if s > c["score_min"]]
    ind_heads = [h for h, s in enumerate(ind1) if s > c["score_min"] and abl1[h] >= c["ablate_min"]]
    pairs = [(h0, h1) for h0 in prev_heads for h1 in ind_heads
             if an["patch_0to1_keys"][h0][h1] >= c["patch_min"]]
    # Phase change: a drop of >= drop_min bits within <= window steps, containing the step at
    # which the max layer-1 induction score first reaches half of its total rise.
    top = [max(h["ind"][1]) for h in hist]
    half = top[0] + (top[-1] - top[0]) / 2
    s_half = next(h["step"] for h, t in zip(hist, top, strict=True) if t >= half)
    phase = any(a["step"] <= s_half <= b["step"]
                and b["step"] - a["step"] <= c["window"]
                and a["loss_rep"] - b["loss_rep"] >= c["drop_min"]
                for i, a in enumerate(hist) for b in hist[i + 1:])
    checks = dict(learning=last["loss_rep"] < c["rep_max"] and last["loss_first"] > c["first_min"],
                  induction_head=bool(ind_heads), prev_token_head=bool(prev_heads),
                  composition=bool(pairs), phase_change=phase)
    labels = [["other"] * len(prev0) for _ in last["prev"]]
    for h in prev_heads:
        labels[0][h] = "prev"
    for h in ind_heads:
        labels[1][h] = "induction"
    return dict(checks=checks, passed=all(checks.values()), pairs=pairs, s_half=s_half,
                labels=labels)


def role_verdict(run):
    """Role-level criteria (amendment): learning and phase change as in ``verdict``; group
    ablation of each role and the group key patch replace the single-head checks."""
    c, r = ROLE_CRIT, run["analysis"]["roles"]
    v = verdict(run)
    checks = dict(learning=v["checks"]["learning"], phase_change=v["checks"]["phase_change"],
                  prev_role=bool(r["prev"]) and r["ablate_prev"] >= c["group_ablate_min"],
                  induction_role=bool(r["induction"])
                  and r["ablate_induction"] >= c["group_ablate_min"],
                  composition=r["patch"] >= c["group_patch_min"])
    labels = [["other"] * len(run["history"][-1]["prev"][0]) for _ in range(2)]
    for h in r["prev"]:
        labels[0][h] = "prev"
    for h in r["induction"]:
        labels[1][h] = "induction"
    return dict(checks=checks, passed=all(checks.values()), labels=labels)


def result_path(seed, steps):
    tag = "" if steps == STEPS else f"_st{steps}"
    return f"{RESULTS}/m1_s{seed}{tag}.json"


def summary():
    n_pass = n = n_role_pass = n_role = 0
    for path in sorted(glob.glob(f"{RESULTS}/m1_*.json")):
        with open(path) as f:
            run = json.load(f)
        last = run["history"][-1]
        line = (f"{os.path.basename(path)}  rep {last['loss_rep']:.3f}  "
                f"first {last['loss_first']:.3f}  max prev L0 {max(last['prev'][0]):.2f}  "
                f"max ind L1 {max(last['ind'][1]):.2f}")
        if "analysis" in run:
            v = verdict(run)
            failed = [k for k, ok in v["checks"].items() if not ok]
            line += f"  pairs {v['pairs']}  {'PASS' if v['passed'] else 'FAIL ' + ','.join(failed)}"
            n, n_pass = n + 1, n_pass + v["passed"]
        if "roles" in run.get("analysis", {}):
            r, rv = run["analysis"]["roles"], role_verdict(run)
            failed = [k for k, ok in rv["checks"].items() if not ok]
            line += (f"  | roles P{r['prev']} I{r['induction']} abl {r['ablate_prev']:.2f}/"
                     f"{r['ablate_induction']:.2f} patch {r['patch']:.2f} "
                     f"{'PASS' if rv['passed'] else 'FAIL ' + ','.join(failed)}")
            if run["seed"] in CONFIRM_SEEDS:
                n_role, n_role_pass = n_role + 1, n_role_pass + rv["passed"]
        print(line)
    if n:
        print(f"{n_pass}/{n} seeds pass the original criteria")
    if n_role:
        print(f"{n_role_pass}/{n_role} confirmation seeds (10-14) pass the role-level criteria "
              "(M1 passes at >= 4 of 5)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--reanalyse", action="store_true",
                    help="recompute the analysis of an existing run from its saved weights")
    args = ap.parse_args()
    if args.summary:
        summary()
        return
    path = result_path(args.seed, args.steps)
    if args.reanalyse:
        with open(path) as f:
            run = json.load(f)
        model = AttnOnly(Config(**run["config"]), torch.Generator())
        model.load_state_dict(torch.load(path.replace(".json", ".pt")))
        deterministic()
        run["analysis"] = analyse(model, run["eval_size"], run["history"][-1])
        with open(path, "w") as f:
            json.dump(run, f)
        print(path)
        return
    run, model = train(args.seed, steps=args.steps)
    run["analysis"] = analyse(model, run["eval_size"], run["history"][-1])
    run["crit"] = CRIT
    os.makedirs(RESULTS, exist_ok=True)
    with open(path, "w") as f:
        json.dump(run, f)
    torch.save(model.state_dict(), path.replace(".json", ".pt"))
    print(path)


if __name__ == "__main__":
    main()
