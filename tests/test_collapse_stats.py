"""R15 pre-registered red-team checks, as pure CPU-testable functions.

These guard the collapse-knob eval (scripts/r15_collapse.py) against the three
self-deception modes flagged in STATUS before any checkpoint lands:
  (1) THRESH placed mid-cluster instead of in the bimodal valley  -> threshold_in_valley
  (3) 2 seeds silently disagree, reported as a clean win          -> seed_verdict
  (4) Fisher run 1-sided, hiding a treat-collapses-MORE result    -> fisher_both_sided
"""
from __future__ import annotations

import numpy as np

from jepa_ctrl.collapse_stats import (
    fisher_both_sided,
    seed_verdict,
    threshold_in_valley,
)


# ---- (1) THRESH-in-valley --------------------------------------------------
def test_valley_ok_for_clear_bimodal():
    rng = np.random.default_rng(0)
    low = rng.normal(60, 15, 200)   # collapse cluster
    high = rng.normal(380, 25, 200)  # gait cluster
    eps = np.concatenate([low, high]).tolist()
    r = threshold_in_valley(eps, thresh=150.0)
    assert r["bimodal"] is True
    assert r["valley_ok"] is True
    assert r["mass_below"] > 0 and r["mass_above"] > 0


def test_valley_rejects_threshold_inside_a_mode():
    rng = np.random.default_rng(1)
    low = rng.normal(60, 15, 200)
    high = rng.normal(380, 25, 200)
    eps = np.concatenate([low, high]).tolist()
    # 380 sits in the middle of the gait cluster, not the valley
    r = threshold_in_valley(eps, thresh=380.0)
    assert r["valley_ok"] is False


def test_valley_unimodal_is_not_bimodal():
    rng = np.random.default_rng(2)
    eps = rng.normal(300, 20, 300).tolist()
    r = threshold_in_valley(eps, thresh=150.0)
    assert r["bimodal"] is False


# ---- (3) seed agreement ----------------------------------------------------
def test_seed_verdict_consistent_win():
    # treat collapses less than base in BOTH seeds -> consistent improvement
    v = seed_verdict([(12, 4), (10, 5)])  # (base_collapse, treat_collapse)
    assert v["verdict"] == "consistent"
    assert v["direction"] == "treat_helps"


def test_seed_verdict_disagreement_is_inconclusive():
    # seed0: treat helps (12->4); seed1: treat hurts (5->11) -> must flag inconclusive
    v = seed_verdict([(12, 4), (5, 11)])
    assert v["verdict"] == "inconclusive"
    assert "seed" in v["reason"].lower()


def test_seed_verdict_consistent_harm():
    v = seed_verdict([(3, 9), (4, 10)])
    assert v["verdict"] == "consistent"
    assert v["direction"] == "treat_hurts"


# ---- (4) two-sided Fisher honesty ------------------------------------------
def test_fisher_reports_both_directions_and_observed():
    # treat collapses LESS (4 vs 12 of 20)
    r = fisher_both_sided(base_collapse=12, treat_collapse=4, n=20)
    assert 0.0 <= r["p_treat_less"] <= 1.0
    assert 0.0 <= r["p_treat_more"] <= 1.0
    assert 0.0 <= r["p_two_sided"] <= 1.0
    assert r["observed_direction"] == "treat_helps"
    # the supported tail should be the smaller p
    assert r["p_treat_less"] < r["p_treat_more"]


def test_fisher_honest_when_treat_collapses_more():
    r = fisher_both_sided(base_collapse=3, treat_collapse=14, n=20)
    assert r["observed_direction"] == "treat_hurts"
    assert r["p_treat_more"] < r["p_treat_less"]
