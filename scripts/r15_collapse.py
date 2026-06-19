"""R15: does raising the late-training exploration FLOOR (explore_std_end 0.05 -> 0.2) reduce
the 3D bimodal COLLAPSE RATE? (R12/R13/R14 refuted capacity/representation/planner/eval-smoothness;
the only lever left is training-side gait-acquisition reliability = initial-condition robustness.)

Controlled, all-fresh-retrain design (isolates the one knob; controls for retrain variance):
  base_s{0,1}  = reward-free raw quad, explore_std_end 0.05 (the R11 recipe), 200k
  treat_s{0,1} = identical EXCEPT explore_std_end 0.20, 200k

Eval = the SAME protocol as R14 base: BASE MPPI (h3 i6 s512 e64), THRESH=150, 20 eps/cell.
Reports collapse_rate + good_basin_mean per cell and a Fisher-exact test of the collapse COUNT
(base vs treat) per seed — the headline significance test (red-team: never a bare mean).

Usage: python scripts/r15_collapse.py --episodes 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import fisher_exact

from scripts.diag_dof import eval_return, load_model

THRESH = 150.0  # identical to R14: splits the bimodal gait/collapse clusters

# (label, dir, seed, explore_std_end) — all reward-free raw (latent_norm none, dim 256)
CELLS = [
    ("base_s0", "runs/R15_explore/base_s0", 0, 0.05),
    ("base_s1", "runs/R15_explore/base_s1", 1, 0.05),
    ("treat_s0", "runs/R15_explore/treat_s0", 0, 0.20),
    ("treat_s1", "runs/R15_explore/treat_s1", 1, 0.20),
]


def summarize(rets: list[float]) -> dict:
    a = np.asarray(rets, dtype=np.float64)
    good = a[a >= THRESH]
    return {
        "mean": round(float(a.mean()), 1),
        "median": round(float(np.median(a)), 1),
        "collapse_rate": round(float((a < THRESH).mean()), 3),
        "n_collapse": int((a < THRESH).sum()),
        "n_good": int(good.size),
        "good_basin_mean": round(float(good.mean()), 1) if good.size else None,
        "eps": [round(x, 1) for x in rets],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", default="runs/R15_explore/collapse.json")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # BASE MPPI = R14 eval config (one source of truth)
    from scripts.collapse_rate import BASE

    res: dict = {}
    for label, d, _seed, std_end in CELLS:
        if not (Path(d) / "model.pt").exists():
            print(f"[skip] {label}: no checkpoint at {d}")
            continue
        wm, env = load_model(d, "quadruped-walk", "none", 256, device)
        _, raw = eval_return(wm, env, BASE, args.episodes, device)
        s = summarize(raw)
        s["explore_std_end"] = std_end
        res[label] = s
        print(f"[{label}] std_end={std_end} mean={s['mean']:6.1f} "
              f"collapse={s['collapse_rate']:.2f} ({s['n_collapse']}/{args.episodes}) "
              f"good={s['good_basin_mean']}")
        del wm, env
        torch.cuda.empty_cache()

    # Fisher-exact: collapse count base vs treat, per seed (headline significance)
    n = args.episodes
    sig: dict = {}
    for seed in (0, 1):
        b, t = res.get(f"base_s{seed}"), res.get(f"treat_s{seed}")
        if b and t:
            table = [[t["n_collapse"], n - t["n_collapse"]],
                     [b["n_collapse"], n - b["n_collapse"]]]
            odds, pval = fisher_exact(table, alternative="less")  # H1: treat collapses LESS
            sig[f"seed{seed}"] = {
                "base_collapse": b["n_collapse"], "treat_collapse": t["n_collapse"],
                "fisher_p_treat_less": round(float(pval), 4),
            }
            print(f"[fisher seed{seed}] base {b['n_collapse']}/{n} vs treat "
                  f"{t['n_collapse']}/{n}  p(treat<base)={pval:.4f}")
    res["_significance"] = sig

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
