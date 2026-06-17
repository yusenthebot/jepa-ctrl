from __future__ import annotations

import torch

from jepa_ctrl.model import (
    DistHead,
    Encoder,
    ModelConfig,
    Predictor,
    SimNorm,
    WorldModel,
    symexp,
    symlog,
)

torch.manual_seed(0)


# --- (1) SimNorm: groups are a valid simplex --------------------------------------
def test_simnorm_groups_are_simplex():
    sn = SimNorm(group_size=8)
    x = torch.randn(4, 128) * 5.0
    y = sn(x)
    assert y.shape == x.shape
    assert (y >= 0).all()
    groups = y.reshape(4, 16, 8)
    sums = groups.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_simnorm_rejects_bad_dim():
    sn = SimNorm(group_size=8)
    try:
        sn(torch.randn(2, 10))
        raise AssertionError("expected ValueError on non-divisible dim")
    except ValueError:
        pass


# --- (2) Encoder: shape + valid simplex, pre activation exposed -------------------
def test_encoder_shape_and_simplex():
    enc = Encoder(obs_dim=17, latent_dim=128, hidden=256, simnorm_groups=8)
    obs = torch.randn(5, 17)
    z = enc(obs)
    assert z.shape == (5, 128)
    assert (z >= 0).all()
    sums = z.reshape(5, 16, 8).sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    pre = enc.pre_simnorm(obs)
    assert pre.shape == (5, 128)
    # pre activation is genuinely pre-normalization (not already a simplex)
    pre_sums = pre.reshape(5, 16, 8).sum(-1)
    assert not torch.allclose(pre_sums, torch.ones_like(pre_sums), atol=1e-3)


# --- (3) EMA update moves target toward online; target frozen ---------------------
def test_ema_moves_target_and_target_frozen():
    cfg = ModelConfig(obs_dim=17, act_dim=6, latent_dim=128)
    wm = WorldModel(cfg)

    # target params start equal to online (deepcopy) and are frozen
    for p in wm.target_encoder.parameters():
        assert p.requires_grad is False
    for p in wm.target_value_head.parameters():
        assert p.requires_grad is False

    # perturb the online encoder so EMA has somewhere to move
    with torch.no_grad():
        for p in wm.encoder.parameters():
            p.add_(torch.randn_like(p))

    before = [t.clone() for t in wm.target_encoder.parameters()]
    wm.ema_update()
    after = list(wm.target_encoder.parameters())

    moved_any = False
    for b, o, a in zip(before, wm.encoder.parameters(), after, strict=True):
        # target moved toward online: |a - o| < |b - o| where they differ
        d_before = (b - o).abs().sum()
        d_after = (a - o).abs().sum()
        assert d_after <= d_before + 1e-6
        if (a - b).abs().sum() > 0:
            moved_any = True
    assert moved_any


def test_target_gets_no_grad_after_backward():
    cfg = ModelConfig(obs_dim=17, act_dim=6, latent_dim=128, horizon=3)
    wm = WorldModel(cfg)
    h, b = cfg.horizon, 4
    obs_seq = torch.randn(h + 1, b, cfg.obs_dim)
    act_seq = torch.randn(h, b, cfg.act_dim)

    loss = wm.consistency_loss(obs_seq, act_seq)
    loss.backward()

    for p in wm.target_encoder.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0


# --- (4) consistency_loss: finite, grads reach encoder+predictor, not target ------
def test_consistency_loss_grad_flow():
    cfg = ModelConfig(obs_dim=17, act_dim=6, latent_dim=128, horizon=4)
    wm = WorldModel(cfg)
    h, b = cfg.horizon, 3
    obs_seq = torch.randn(h + 1, b, cfg.obs_dim)
    act_seq = torch.randn(h, b, cfg.act_dim)

    loss = wm.consistency_loss(obs_seq, act_seq)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    loss.backward()

    enc_grad = sum(
        p.grad.abs().sum() for p in wm.encoder.parameters() if p.grad is not None
    )
    pred_grad = sum(
        p.grad.abs().sum() for p in wm.predictor.parameters() if p.grad is not None
    )
    assert enc_grad > 0
    assert pred_grad > 0
    # target encoder strictly receives no gradient
    for p in wm.target_encoder.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0


def test_predictor_residual_simplex_output():
    pred = Predictor(latent_dim=128, act_dim=6, simnorm_groups=8)
    b = 3
    z_pre = torch.randn(b, 128)
    a = torch.randn(b, 6)
    z_next = pred(z_pre, a)
    assert z_next.shape == (b, 128)
    sums = z_next.reshape(b, 16, 8).sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_rollout_open_loop_shapes_and_simplex():
    cfg = ModelConfig(obs_dim=17, act_dim=6, latent_dim=128, horizon=3)
    wm = WorldModel(cfg)
    b = 2
    z0_pre = wm.encode_pre(torch.randn(b, cfg.obs_dim))
    act_seq = torch.randn(cfg.horizon, b, cfg.act_dim)
    preds = wm.rollout(z0_pre, act_seq)
    assert len(preds) == cfg.horizon
    for z in preds:
        assert z.shape == (b, cfg.latent_dim)
        sums = z.reshape(b, cfg.latent_dim // cfg.simnorm_groups, cfg.simnorm_groups).sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# --- (5) DistHead symlog two-hot round-trip ---------------------------------------
def test_disthead_symlog_twohot_roundtrip():
    head = DistHead(latent_dim=4, act_dim=2, bins=101, vmin=-10.0, vmax=10.0)
    # targets across a wide magnitude range, well inside symlog([-10,10]) coverage
    x = torch.tensor([-50.0, -3.2, -0.5, 0.0, 0.7, 4.0, 123.0])
    label = head.twohot_encode(symlog(x))  # (N, bins)
    # two-hot rows are valid distributions
    assert torch.allclose(label.sum(-1), torch.ones_like(x), atol=1e-5)
    assert (label >= 0).all()
    # decode: log(prob)->logits round-trips through to_scalar back to x
    logits = torch.log(label.clamp_min(1e-9))
    recon = head.to_scalar(logits)
    assert torch.allclose(recon, x, rtol=1e-2, atol=1e-2)


def test_symlog_symexp_inverse():
    x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-4)


# --- (6) full WorldModel param count ~1M for cheetah dims -------------------------
def test_param_count_cheetah_under_3m():
    cfg = ModelConfig(obs_dim=17, act_dim=6, latent_dim=256)
    wm = WorldModel(cfg)
    # count only the online (trainable) params; target nets are frozen copies
    trainable = sum(p.numel() for p in wm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wm.parameters())
    assert trainable < 3_000_000, f"trainable params {trainable} exceed 3M"
    assert total < 6_000_000, f"total params {total} exceed 6M"


def test_reward_value_losses_finite_and_grad():
    cfg = ModelConfig(obs_dim=17, act_dim=6, latent_dim=128, num_q=5)
    wm = WorldModel(cfg)
    b = 4
    z = wm.encode(torch.randn(b, cfg.obs_dim))
    a = torch.randn(b, cfg.act_dim)
    r_tgt = torch.randn(b)
    v_tgt = torch.randn(b) * 3.0
    r_loss, v_loss = wm.reward_value_losses(z, a, r_tgt, v_tgt)
    assert torch.isfinite(r_loss) and torch.isfinite(v_loss)
    (r_loss + v_loss).backward()
    rg = sum(p.grad.abs().sum() for p in wm.reward_head.parameters() if p.grad is not None)
    assert rg > 0
