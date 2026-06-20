"""Pure, dependency-light statistics for the R15 collapse-knob eval.

Separated from scripts/r15_collapse.py so the three pre-registered red-team
guards can be unit-tested on CPU before any checkpoint exists:
  (1) threshold_in_valley  — the collapse THRESH must sit in the bimodal valley,
                             not mid-cluster, or collapse_rate is an artefact.
  (3) seed_verdict         — if the 2 seeds disagree on the sign of the effect,
                             the result is INCONCLUSIVE (R14 levers reversed on s1).
  (4) fisher_both_sided    — report BOTH tails + the observed direction, never a
                             single pre-chosen tail that hides "treat collapses MORE".
"""
from __future__ import annotations

import numpy as np
from scipy.stats import fisher_exact


def threshold_in_valley(
    eps: list[float], thresh: float, n_bins: int = 20, valley_ratio: float = 0.5
) -> dict:
    """Confirm `thresh` splits a genuinely bimodal pooled-return distribution at its
    valley. Returns the pooled histogram so the caller can eyeball it too.

    bimodal: there is mass on both sides AND two separated density peaks.
    valley_ok: the density at `thresh` is below `valley_ratio` x the smaller of the
    two flanking modes — i.e. the threshold lands in the trough, not on a peak.
    """
    a = np.asarray(eps, dtype=np.float64)
    counts, edges = np.histogram(a, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    below = centers < thresh
    above = ~below
    mass_below = int(counts[below].sum())
    mass_above = int(counts[above].sum())

    peak_below = int(counts[below].max()) if mass_below else 0
    peak_above = int(counts[above].max()) if mass_above else 0
    bimodal = mass_below > 0 and mass_above > 0 and peak_below > 0 and peak_above > 0

    # density in the bin straddling the threshold
    thr_bin = int(np.clip(np.searchsorted(edges, thresh) - 1, 0, n_bins - 1))
    valley_count = int(counts[thr_bin])
    flank = min(peak_below, peak_above) if bimodal else 0
    valley_ok = bool(bimodal and valley_count <= valley_ratio * flank)

    return {
        "bimodal": bool(bimodal),
        "valley_ok": valley_ok,
        "mass_below": mass_below,
        "mass_above": mass_above,
        "peak_below": peak_below,
        "peak_above": peak_above,
        "valley_count": valley_count,
        "thresh": float(thresh),
        "hist_counts": counts.astype(int).tolist(),
        "hist_centers": [round(float(c), 1) for c in centers],
    }


def seed_verdict(per_seed: list[tuple[int, int]]) -> dict:
    """per_seed = [(base_collapse, treat_collapse), ...] one entry per seed.

    effect = base - treat (positive => treat collapses LESS => treat helps).
    Consistent only if every seed agrees on a non-zero sign. Any sign disagreement
    (incl. a flat seed against a moving one) => INCONCLUSIVE, the R14-style trap.
    """
    effects = [b - t for (b, t) in per_seed]
    signs = {int(np.sign(e)) for e in effects}
    if len(per_seed) < 2:
        return {"verdict": "single_seed", "direction": "unknown",
                "reason": "need >=2 seeds", "effects": effects}
    if signs == {1}:
        return {"verdict": "consistent", "direction": "treat_helps",
                "reason": "all seeds: treat collapses less", "effects": effects}
    if signs == {-1}:
        return {"verdict": "consistent", "direction": "treat_hurts",
                "reason": "all seeds: treat collapses more", "effects": effects}
    return {"verdict": "inconclusive", "direction": "mixed",
            "reason": "seeds disagree on sign of effect -> add seed 2", "effects": effects}


def fisher_both_sided(base_collapse: int, treat_collapse: int, n: int) -> dict:
    """Fisher-exact on the 2x2 collapse-count table, reporting BOTH one-sided tails,
    the two-sided p, and the observed direction. Row order [treat; base] so
    `alternative='less'` is H1: treat collapses LESS than base.
    """
    table = [[treat_collapse, n - treat_collapse],
             [base_collapse, n - base_collapse]]
    _, p_less = fisher_exact(table, alternative="less")
    _, p_more = fisher_exact(table, alternative="greater")
    _, p_two = fisher_exact(table, alternative="two-sided")
    if treat_collapse < base_collapse:
        direction = "treat_helps"
    elif treat_collapse > base_collapse:
        direction = "treat_hurts"
    else:
        direction = "no_difference"
    return {
        "base_collapse": int(base_collapse),
        "treat_collapse": int(treat_collapse),
        "n": int(n),
        "p_treat_less": round(float(p_less), 4),
        "p_treat_more": round(float(p_more), 4),
        "p_two_sided": round(float(p_two), 4),
        "observed_direction": direction,
    }
