from __future__ import annotations

import numpy as np

from jepa_ctrl.metrics import (
    aggregate_returns,
    collapse_diagnostics,
    fidelity_ok,
    is_collapsed,
    latent_rollout_fidelity,
)


def test_aggregate_returns_basic():
    agg = aggregate_returns([10.0, 20.0, 30.0])
    assert agg["mean"] == 20.0
    assert agg["n_seeds"] == 3
    assert agg["low_confidence"] is False
    assert abs(agg["std"] - 10.0) < 1e-9  # sample std (ddof=1), consistent with ci95


def test_aggregate_single_seed_is_low_confidence():
    agg = aggregate_returns([42.0])
    assert agg["low_confidence"] is True
    assert agg["std"] == 0.0 and agg["ci95"] == 0.0


def test_collapse_diagnostics_healthy():
    rng = np.random.default_rng(0)
    h = collapse_diagnostics(rng.standard_normal((512, 32)))
    assert h["dead_dim_fraction"] == 0.0
    assert h["effective_rank"] > 20
    assert h["participation_ratio"] > 20
    assert h["mean_pairwise_dist"] > 1.0
    assert is_collapsed(h) is False


def test_point_collapse_detected_and_flagged():
    # latents collapse to ~a constant point: caught by scale (std + pairwise), not eff_rank.
    rng = np.random.default_rng(0)
    collapsed = np.ones((512, 32)) + rng.standard_normal((512, 32)) * 1e-4
    c = collapse_diagnostics(collapsed)
    assert c["dead_dim_fraction"] > 0.9
    assert c["mean_pairwise_dist"] < 0.1
    assert is_collapsed(c) is True


def test_dimensional_collapse_detected_and_flagged():
    # latents on a 1-D subspace (large scale): caught by eff_rank + participation ratio.
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((512, 1)) @ rng.standard_normal((1, 32))
    d = collapse_diagnostics(Z)
    assert d["effective_rank"] < 2.0
    assert d["participation_ratio"] < 2.0
    assert is_collapsed(d) is True


def test_fidelity_identical_passes():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((4, 256, 32))  # K=4, N=256, d=32, healthy targets
    fid = latent_rollout_fidelity(z, z)
    for k in fid:
        assert fid[k]["centered_cosine"] > 0.999
        assert fid[k]["nrmse"] < 1e-6
        assert fid[k]["target_collapsed"] is False
    assert fidelity_ok(fid) is True


def test_fidelity_random_fails():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((2, 256, 32))
    b = rng.standard_normal((2, 256, 32))
    fid = latent_rollout_fidelity(a, b)
    assert abs(fid[1]["centered_cosine"]) < 0.2
    assert fidelity_ok(fid) is False


def test_fidelity_offset_fooling_defeated():
    # REGRESSION: a constant (per-k mean) predictor scores HIGH raw cosine on offset data,
    # but centered cosine ~0 and fidelity_ok must reject it.
    rng = np.random.default_rng(2)
    z_true = 5.0 + rng.standard_normal((2, 256, 32))  # large shared positive offset
    z_hat = np.broadcast_to(z_true.mean(axis=1, keepdims=True), z_true.shape).copy()
    fid = latent_rollout_fidelity(z_hat, z_true)
    assert fid[1]["cosine"] > 0.8  # raw cosine is fooled by the offset...
    assert abs(fid[1]["centered_cosine"]) < 0.2  # ...centered cosine is not
    assert fidelity_ok(fid) is False


def test_fidelity_target_collapse_defeated():
    # REGRESSION: perfect copy of a COLLAPSED target looks flawless (nrmse~0) but must fail.
    rng = np.random.default_rng(3)
    base = rng.standard_normal((1, 1, 32))
    z_true = np.broadcast_to(base, (2, 256, 32)) + rng.standard_normal((2, 256, 32)) * 1e-3
    z_hat = z_true.copy()
    fid = latent_rollout_fidelity(z_hat, z_true)
    assert fid[1]["nrmse"] < 1e-6  # looks perfect on raw error...
    assert fid[1]["target_collapsed"] is True  # ...but the target encoder collapsed
    assert fidelity_ok(fid) is False
