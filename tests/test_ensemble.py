"""R18 — PredictorEnsemble for JEPA epistemic-disagreement. N independent latent predictors
sharing the (frozen, external) encoder + EMA target; disagreement = cross-head variance of the
predicted next latent. These are sim-free CPU unit tests of the ensemble mechanics; the
calibration diagnostic (disagreement vs true prediction error) lives in scripts/r18_calib.py.
"""
from __future__ import annotations

import torch

from jepa_ctrl.model.nets import PredictorEnsemble

torch.manual_seed(0)

LD, AD, N, B = 64, 4, 5, 8


def _mk(latent_norm="none"):
    return PredictorEnsemble(N, LD, AD, hidden=128, action_head_dim=32,
                             simnorm_groups=8, latent_norm=latent_norm)


def test_forward_shape_is_n_batch_latent():
    ens = _mk()
    z_pre = torch.randn(B, LD)
    a = torch.randn(B, AD)
    out = ens(z_pre, a)
    assert out.shape == (N, B, LD)


def test_heads_are_distinct_under_independent_init():
    ens = _mk()
    z_pre = torch.randn(B, LD)
    a = torch.randn(B, AD)
    out = ens(z_pre, a)  # (N, B, LD)
    # independently-initialised heads must NOT produce identical predictions
    pairwise_max_diff = max(
        (out[i] - out[j]).abs().max().item()
        for i in range(N) for j in range(i + 1, N)
    )
    assert pairwise_max_diff > 1e-3, "ensemble heads collapsed to identical functions"


def test_disagreement_nonneg_and_batch_shaped():
    ens = _mk()
    z_pre = torch.randn(B, LD)
    a = torch.randn(B, AD)
    d = ens.disagreement(z_pre, a)
    assert d.shape == (B,)
    assert torch.all(d >= 0)


def test_disagreement_zero_when_heads_identical():
    """Calibration sanity: clone one head into all N -> zero cross-head variance; independent
    init -> strictly positive. This is the signal the diagnostic relies on."""
    ens = _mk()
    # force all heads identical
    import copy
    h0 = ens.heads[0]
    for i in range(1, N):
        ens.heads[i].load_state_dict(copy.deepcopy(h0.state_dict()))
    z_pre = torch.randn(B, LD)
    a = torch.randn(B, AD)
    d_identical = ens.disagreement(z_pre, a)
    assert torch.all(d_identical < 1e-6), "identical heads must have ~zero disagreement"
    ens2 = _mk()  # fresh independent heads
    assert ens2.disagreement(z_pre, a).mean() > 1e-5, "independent heads must disagree"


def test_mean_prediction_shape_and_matches_manual_mean():
    ens = _mk()
    z_pre = torch.randn(B, LD)
    a = torch.randn(B, AD)
    out = ens(z_pre, a)
    mean = ens.mean_prediction(z_pre, a)
    assert mean.shape == (B, LD)
    assert torch.allclose(mean, out.mean(dim=0), atol=1e-6)


def test_heads_have_independent_parameters():
    ens = _mk()
    ids = {id(p) for p in ens.heads[0].parameters()}
    for i in range(1, N):
        assert ids.isdisjoint({id(p) for p in ens.heads[i].parameters()}), \
            "heads must own independent parameters for independent SGD"
