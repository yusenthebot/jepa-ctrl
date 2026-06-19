"""R14: 3D quadruped control is BIMODAL (R13). Properly characterize the COLLAPSE RATE
over many episodes, then test whether an EVAL-TIME action-smoothness lever reduces it on
the FIXED trained models (no retrain).

R13 swept SEARCH-BREADTH knobs (samples/iters/horizon/elites) and found no clean lever.
This script tests the ACTION-SMOOTHNESS / EXPLOITATION knobs R13 did NOT touch, which
directly govern how jerky/exploratory the executed action is — the plausible mechanism
for early tip-over collapse:
  corr        temporal AR(1) correlation of the sampled action noise (0 = white, ->1 smooth)
  std_min     floor on the executed action std (residual exploration jitter at eval)
  temperature softmax sharpness over elites (lower = greedier/more exploitative)
  momentum    mu/std EMA across MPPI iters (higher = smoother plan updates)

Metric (never a single 3-ep mean — all prior quad numbers were underpowered):
  collapse_rate = frac(return < THRESH);  good_basin_mean = mean(return >= THRESH).

Usage: python scripts/collapse_rate.py --episodes 20 --part base
       python scripts/collapse_rate.py --episodes 20 --part lever
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from jepa_ctrl.model.mppi import MPPIConfig
from scripts.diag_dof import eval_return, load_model

THRESH = 150.0  # R13: RF gait ~250-470 / collapse ~15-125; RW gait ~830 — 150 splits both

# primary = the degrading reward-free arm (s0,s1); RW_s0 = positive control
MODELS = [
    ("quad_RF_s0", "runs/R11_quad200k/rawcons_s0", "quadruped-walk", "none", 256),
    ("quad_RF_s1", "runs/R11_quad200k/rawcons_s1", "quadruped-walk", "none", 256),
    ("quad_RW_s0", "runs/R11_quad200k/reward_s0", "quadruped-walk", "simnorm", 256),
]

BASE = MPPIConfig(horizon=3, iters=6, num_samples=512, num_elites=64)

# one knob at a time off BASE (R13 lesson: never co-vary knobs)
LEVERS = {
    "base": BASE,
    "corr0.3": replace(BASE, corr=0.3),
    "corr0.6": replace(BASE, corr=0.6),
    "corr0.85": replace(BASE, corr=0.85),
    "stdmin0.15": replace(BASE, std_min=0.15),
    "stdmin0.3": replace(BASE, std_min=0.3),
    "temp0.1": replace(BASE, temperature=0.1),
    "temp0.05": replace(BASE, temperature=0.05),
    "mom0.3": replace(BASE, momentum=0.3),
    "mom0.5": replace(BASE, momentum=0.5),
}


def summarize(rets: list[float]) -> dict:
    a = np.asarray(rets, dtype=np.float64)
    good = a[a >= THRESH]
    return {
        "mean": round(float(a.mean()), 1),
        "median": round(float(np.median(a)), 1),
        "collapse_rate": round(float((a < THRESH).mean()), 3),
        "n_good": int(good.size),
        "good_basin_mean": round(float(good.mean()), 1) if good.size else None,
        "eps": [round(x, 1) for x in rets],
    }


def run(levers: dict, episodes: int, out: str, device: str) -> None:
    res: dict = {}
    for label, d, task, ln, ld in MODELS:
        wm, env = load_model(d, task, ln, ld, device)
        res[label] = {}
        for lname, cfg in levers.items():
            _, raw = eval_return(wm, env, cfg, episodes, device)
            s = summarize(raw)
            res[label][lname] = s
            print(f"[{label}] {lname:<11} mean={s['mean']:6.1f} "
                  f"collapse={s['collapse_rate']:.2f} good={s['good_basin_mean']} "
                  f"(n_good={s['n_good']}/{episodes})")
        del wm, env
        torch.cuda.empty_cache()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--part", default="base", choices=["base", "lever"])
    p.add_argument("--out", default="")
    args = p.parse_args()
    levers = {"base": BASE} if args.part == "base" else LEVERS
    out = args.out or f"runs/R14_collapse/{args.part}.json"
    run(levers, args.episodes, out, "cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
