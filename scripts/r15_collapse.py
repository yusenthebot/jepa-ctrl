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

from jepa_ctrl.collapse_stats import fisher_both_sided, seed_verdict, threshold_in_valley
from jepa_ctrl.render import contact_sheet
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


def render_good_and_collapsed(label, d, device, outdir, max_steps=1000) -> dict:
    """Eyes-on (red-team #1): roll episodes until we capture one good (>=THRESH) and one
    collapsed (<THRESH), save a keyframe contact sheet for each. The human-readable proof
    that THRESH actually separates a walking gait from a fallen/frozen robot."""
    from scripts.collapse_rate import BASE

    from jepa_ctrl.model.jepa_controller import JepaController

    wm, env = load_model(d, "quadruped-walk", "none", 256, device)
    ctrl = JepaController(wm, env.act_low, env.act_high, BASE, device)
    captured: dict = {}
    for _ in range(12):
        if "good" in captured and "collapsed" in captured:
            break
        ctrl.reset()
        obs = env.reset()
        frames, step_r, total, done, steps = [], [], 0.0, False, 0
        while not done and steps < max_steps:
            frames.append(env.render())
            obs, r, done = env.step(ctrl.act(obs))
            total += r
            step_r.append(total)
            steps += 1
        kind = "good" if total >= THRESH else "collapsed"
        if kind not in captured:
            path = Path(outdir) / f"{label}_{kind}_{total:.0f}.png"
            contact_sheet(frames, path, title=f"{label} {kind} R={total:.0f}",
                          step_returns=step_r)
            captured[kind] = {"return": round(total, 1), "sheet": str(path)}
            print(f"[render] {label}: {kind} R={total:.0f} -> {path}")
    del wm, env, ctrl
    torch.cuda.empty_cache()
    return captured


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", default="runs/R15_explore/collapse.json")
    p.add_argument("--render", action="store_true",
                   help="also save good+collapsed contact sheets per cell (eyes-on red-team #1)")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # BASE MPPI = R14 eval config (one source of truth)
    from scripts.collapse_rate import BASE

    res: dict = {}
    all_eps: list[float] = []
    for label, d, _seed, std_end in CELLS:
        if not (Path(d) / "model.pt").exists():
            print(f"[skip] {label}: no checkpoint at {d}")
            continue
        wm, env = load_model(d, "quadruped-walk", "none", 256, device)
        _, raw = eval_return(wm, env, BASE, args.episodes, device)
        s = summarize(raw)
        s["explore_std_end"] = std_end
        res[label] = s
        all_eps.extend(raw)
        print(f"[{label}] std_end={std_end} mean={s['mean']:6.1f} "
              f"collapse={s['collapse_rate']:.2f} ({s['n_collapse']}/{args.episodes}) "
              f"good={s['good_basin_mean']}")
        del wm, env
        torch.cuda.empty_cache()

    # red-team #1: confirm THRESH sits in the bimodal valley of the POOLED returns
    if all_eps:
        valley = threshold_in_valley(all_eps, THRESH)
        res["_threshold_valley"] = valley
        print(f"[valley] THRESH={THRESH} bimodal={valley['bimodal']} "
              f"valley_ok={valley['valley_ok']} "
              f"(below={valley['mass_below']} above={valley['mass_above']} "
              f"valley_count={valley['valley_count']})")
        if not valley["valley_ok"]:
            print("[valley][WARN] THRESH does NOT sit in a clean valley -> "
                  "collapse_rate may be a thresholding artefact, not bimodality.")

    # red-team #4: Fisher BOTH tails + observed direction (never a single hidden tail)
    n = args.episodes
    sig: dict = {}
    per_seed: list[tuple[int, int]] = []
    for seed in (0, 1):
        b, t = res.get(f"base_s{seed}"), res.get(f"treat_s{seed}")
        if b and t:
            f = fisher_both_sided(b["n_collapse"], t["n_collapse"], n)
            sig[f"seed{seed}"] = f
            per_seed.append((b["n_collapse"], t["n_collapse"]))
            print(f"[fisher seed{seed}] base {b['n_collapse']}/{n} vs treat "
                  f"{t['n_collapse']}/{n}  dir={f['observed_direction']} "
                  f"p(less)={f['p_treat_less']} p(more)={f['p_treat_more']} "
                  f"p(2sided)={f['p_two_sided']}")
    res["_significance"] = sig

    # red-team #3: cross-seed agreement -> INCONCLUSIVE if seeds disagree on sign
    if per_seed:
        v = seed_verdict(per_seed)
        res["_seed_verdict"] = v
        print(f"[verdict] {v['verdict']} ({v['direction']}): {v['reason']} "
              f"effects(base-treat)={v['effects']}")

    if args.render:
        rdir = Path(args.out).parent / "render"
        res["_render"] = {}
        for label, d, _seed, _std in CELLS:
            if (Path(d) / "model.pt").exists():
                res["_render"][label] = render_good_and_collapsed(label, d, device, rdir)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
