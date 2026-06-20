"""R17 authoritative red-team: does reset-curriculum cut the reward-free RAW-latent quadruped
collapse_rate vs the MATCHED base, under ONE identical eval protocol?

Pre-registered guards (all must hold to claim a win):
  - direction: treat collapses LESS than base
  - significance: fisher_both_sided pooled (n = episodes x n_seeds), p_treat_less < 0.05
  - seed agreement: seed_verdict over per-seed (base_n_collapse, treat_n_collapse) == consistent/treat_helps
  - no good-basin damage: treat good_basin_mean must NOT be materially below base (a knob that lowers
    collapse but craters the gait is not a win)
  - valley: THRESH sits in the bimodal valley of the POOLED returns (else collapse_rate is an artefact)

TREAT = runs/R17_resetcurr/seed{0,1}/model.pt   (reset-curriculum p=0.3)
BASE  = runs/R11_quad200k/rawcons_s{0,1}/model.pt (same config, NO reset-curriculum)
Both reward-free RAW (latent-norm none, dim 256), quadruped-walk, 200k. Eval seeds are fixed
(1000+sd) inside collapse_eval, so base/treat see the SAME 20 start conditions per seed.

SIM-TOUCHING: run serialized AFTER both trainings finish (one sim at a time).
"""
from __future__ import annotations

import json
from pathlib import Path

from jepa_ctrl.collapse_stats import fisher_both_sided, seed_verdict, threshold_in_valley
from scripts.collapse_eval import collapse_eval

SEEDS = [0, 1]
EPISODES = 20
THRESH = 150.0
TREAT = "runs/R17_resetcurr/seed{s}/model.pt"
BASE = "runs/R11_quad200k/rawcons_s{s}/model.pt"
OUT = Path("runs/R17_resetcurr/REDTEAM.json")


def _eval(path: str) -> dict:
    return collapse_eval(path, "quadruped-walk", "none", 256, EPISODES, THRESH, "cuda")


def main() -> None:
    per_seed_counts: list[tuple[int, int]] = []  # (base_n_collapse, treat_n_collapse)
    pooled_eps: list[float] = []
    rows = []
    base_collapse_total = treat_collapse_total = 0
    base_good, treat_good = [], []

    for s in SEEDS:
        b = _eval(BASE.format(s=s))
        t = _eval(TREAT.format(s=s))
        per_seed_counts.append((b["n_collapse"], t["n_collapse"]))
        base_collapse_total += b["n_collapse"]
        treat_collapse_total += t["n_collapse"]
        if b["n_good"]:
            base_good.append(b["good_basin_mean"])
        if t["n_good"]:
            treat_good.append(t["good_basin_mean"])
        pooled_eps += b["eps"] + t["eps"]
        rows.append({"seed": s, "base": b, "treat": t})
        print(f"seed{s}: base collapse {b['n_collapse']}/{EPISODES} "
              f"(good_basin {b['good_basin_mean']:.0f}) | "
              f"treat collapse {t['n_collapse']}/{EPISODES} (good_basin {t['good_basin_mean']:.0f})")

    n = EPISODES * len(SEEDS)
    fisher = fisher_both_sided(base_collapse_total, treat_collapse_total, n)
    verdict = seed_verdict(per_seed_counts)
    valley = threshold_in_valley(pooled_eps, THRESH)
    base_gb = sum(base_good) / len(base_good) if base_good else 0.0
    treat_gb = sum(treat_good) / len(treat_good) if treat_good else 0.0
    # guardrail: treat good-basin must be >= 90% of base good-basin (no >10% gait damage)
    good_basin_ok = treat_gb >= 0.90 * base_gb if base_gb else True

    win = (
        fisher["observed_direction"] == "treat_helps"
        and fisher["p_treat_less"] < 0.05
        and verdict["verdict"] == "consistent"
        and verdict["direction"] == "treat_helps"
        and good_basin_ok
        and valley["valley_ok"]
    )
    out = {
        "claim": "reset-curriculum cuts reward-free RAW quad collapse_rate vs matched base",
        "n_per_arm": n,
        "base_collapse_total": base_collapse_total,
        "treat_collapse_total": treat_collapse_total,
        "base_collapse_rate": round(base_collapse_total / n, 3),
        "treat_collapse_rate": round(treat_collapse_total / n, 3),
        "fisher": fisher,
        "seed_verdict": verdict,
        "good_basin": {"base": round(base_gb, 1), "treat": round(treat_gb, 1), "ok": good_basin_ok},
        "valley": {k: valley[k] for k in ("bimodal", "valley_ok", "mass_below", "mass_above")},
        "VERDICT": "WIN" if win else "NOT_CONFIRMED",
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print("\n==== R17 RED-TEAM VERDICT ====")
    print(f"base collapse_rate {out['base_collapse_rate']} vs treat {out['treat_collapse_rate']} "
          f"(n={n}/arm)")
    print(f"fisher p_treat_less={fisher['p_treat_less']} dir={fisher['observed_direction']}")
    print(f"seed_verdict={verdict['verdict']}/{verdict['direction']} effects={verdict['effects']}")
    print(f"good_basin base={base_gb:.0f} treat={treat_gb:.0f} ok={good_basin_ok}")
    print(f"valley bimodal={valley['bimodal']} valley_ok={valley['valley_ok']}")
    print(f">>> {out['VERDICT']} <<<   -> {OUT}")


if __name__ == "__main__":
    main()
