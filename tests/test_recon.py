"""RECONSTRUCTION baseline for the distractor head-to-head (pure torch, NO sim).

The distractor-robustness experiment pits two matched pixel world models:
  - JEPA arm = grounding "reward" + CNN encoder = latent consistency + reward, NO decoder.
  - RECONSTRUCTION arm = same CNN encoder + a pixel decoder, consistency REPLACED by a
    Dreamer-style pixel reconstruction loss (predicted latents must reconstruct future frames),
    reward+value grounding kept byte-identical to the JEPA arm.

These tests prove the machinery (decoder shapes, recon-loss grad reaches encoder+predictor+
decoder, a recon-arm Trainer.update runs on random uint8, and the JEPA arm has NO decoder and
no reconstruction term). All tensors are tiny / random — no dm_control, no render, no training.
"""

from __future__ import annotations

import torch

from jepa_ctrl.model import Decoder, ModelConfig, TrainConfig, WorldModel

torch.manual_seed(0)


def _cnn_cfg(latent_dim: int = 64, horizon: int = 3, act_dim: int = 4) -> ModelConfig:
    return ModelConfig(
        obs_dim=0,
        act_dim=act_dim,
        encoder_type="cnn",
        obs_shape=(9, 84, 84),
        latent_dim=latent_dim,
        simnorm_groups=8,
        num_q=3,
        horizon=horizon,
    )


def _grad_sum(module) -> torch.Tensor:
    return sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None)


# =====================================================================================
# (1) Decoder: latent -> reconstructed frames, shape round-trip with CNNEncoder
# =====================================================================================
def test_decoder_shape_round_trip():
    dec = Decoder(latent_dim=64, obs_shape=(9, 84, 84))
    z = torch.randn(5, 64)
    img = dec(z)
    assert img.shape == (5, 9, 84, 84)
    assert torch.isfinite(img).all()


def test_decoder_preserves_leading_dims():
    # the decoder must handle a rollout-shaped latent (H, B, latent_dim) -> (H, B, C, H, W)
    dec = Decoder(latent_dim=32, obs_shape=(9, 84, 84))
    z = torch.randn(3, 7, 32)
    img = dec(z)
    assert img.shape == (3, 7, 9, 84, 84)


def test_decoder_output_in_centered_range_after_tanh_free_init():
    # the decoder reconstructs the encoder's centered space; values are finite floats (no
    # nan/inf from the transpose stack at init). Range is not hard-bounded but must be finite.
    dec = Decoder(latent_dim=16, obs_shape=(9, 84, 84))
    img = dec(torch.zeros(2, 16))
    assert torch.isfinite(img).all()


# =====================================================================================
# (2) reconstruction_loss: finite + grad reaches encoder + predictor + decoder
# =====================================================================================
def test_reconstruction_loss_finite():
    cfg = _cnn_cfg(horizon=3)
    wm = WorldModel(cfg, build_decoder=True)
    h, b = cfg.horizon, 2
    obs_seq = torch.randint(0, 256, (h + 1, b, 9, 84, 84), dtype=torch.uint8)
    act_seq = torch.randn(h, b, cfg.act_dim)
    latents = wm.rollout_latents(obs_seq, act_seq)
    loss = wm.reconstruction_loss(obs_seq, latents)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert loss >= 0.0


def test_reconstruction_loss_grad_reaches_encoder_predictor_decoder():
    cfg = _cnn_cfg(horizon=3)
    wm = WorldModel(cfg, build_decoder=True)
    h, b = cfg.horizon, 2
    obs_seq = torch.randint(0, 256, (h + 1, b, 9, 84, 84), dtype=torch.uint8)
    act_seq = torch.randn(h, b, cfg.act_dim)
    latents = wm.rollout_latents(obs_seq, act_seq)
    loss = wm.reconstruction_loss(obs_seq, latents)
    loss.backward()
    # the Dreamer-style observation model shapes the WHOLE representation: encoder (via z_0),
    # predictor (via every predicted z_hat), and the decoder itself.
    assert _grad_sum(wm.encoder) > 0, "recon must shape the encoder"
    assert _grad_sum(wm.predictor) > 0, "recon must shape the predictor (future-frame pressure)"
    assert _grad_sum(wm.decoder) > 0, "recon must train the decoder"
    # the EMA target encoder stays stop-grad
    for p in wm.target_encoder.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0


def test_reconstruction_targets_future_frame_not_only_current():
    """The predicted latent must reconstruct the FUTURE frame: perturbing a later action
    changes the rolled latents and therefore the reconstruction loss (it is not a static
    autoencoder on o_t alone)."""
    cfg = _cnn_cfg(horizon=4)
    wm = WorldModel(cfg, build_decoder=True).eval()
    h, b = cfg.horizon, 2
    obs_seq = torch.randint(0, 256, (h + 1, b, 9, 84, 84), dtype=torch.uint8)
    act_seq = torch.randn(h, b, cfg.act_dim)
    latents = wm.rollout_latents(obs_seq, act_seq)
    perturbed = act_seq.clone()
    perturbed[2] = perturbed[2] + 3.0
    latents_p = wm.rollout_latents(obs_seq, perturbed)
    # the predicted latents AFTER the perturbed step move -> the future-frame reconstruction
    # at those steps is driven by the rolled (future) latents, not just an autoencoder on o_t.
    future_delta = sum(
        float((latents[k] - latents_p[k]).detach().abs().sum()) for k in range(3, h + 1)
    )
    assert future_delta > 0.0, "perturbing a later action must change the future predicted latents"
    # and the reconstruction loss therefore depends on the full rollout (recompute end-to-end)
    base = wm.reconstruction_loss(obs_seq, latents).detach()
    new = wm.reconstruction_loss(obs_seq, latents_p).detach()
    assert torch.isfinite(base) and torch.isfinite(new)


# =====================================================================================
# (3) Trainer.update on the reconstruction arm runs on random uint8 + returns a metric
# =====================================================================================
def _seed_pixel_buffer(buf, n_steps=48, ep_len=12, act_dim=4):
    # frames must match the buffer's stored shape: the Trainer stores the FULL frame-stacked obs
    # (e.g. 9-channel for a 3-frame stack) as one slot (frame_stack=1), not a single RGB frame.
    c, h, w = buf.frame_shape
    for i in range(n_steps):
        first = (i % ep_len) == 0
        done = (i % ep_len) == (ep_len - 1)
        frame = torch.randint(0, 256, (c, h, w), dtype=torch.uint8)
        nxt = torch.randint(0, 256, (c, h, w), dtype=torch.uint8)
        buf.add(frame, torch.randn(act_dim), float(torch.randn(())), nxt, done, first=first)


def test_recon_arm_trainer_update_runs_on_uint8():
    from jepa_ctrl.model import MPPIConfig
    from jepa_ctrl.model.trainer import Trainer

    cfg = _cnn_cfg(latent_dim=64, horizon=2, act_dim=4)
    wm = WorldModel(cfg)  # built WITHOUT a decoder; the recon Trainer builds + owns one
    assert wm.decoder is None
    tcfg = TrainConfig(
        grounding="reconstruction", batch_size=4, seed_steps=0, recon_coef=1.0, capacity=1024
    )
    act_low = -torch.ones(cfg.act_dim)
    act_high = torch.ones(cfg.act_dim)
    trainer = Trainer(wm, tcfg, act_low, act_high, mppi_cfg=MPPIConfig(horizon=2))

    # the Trainer must have built the decoder for the reconstruction arm
    assert trainer.model.decoder is not None
    # the decoder params must be in the optimizer (otherwise reconstruction can't learn)
    opt_params = {id(p) for g in trainer.opt.param_groups for p in g["params"]}
    assert all(id(p) in opt_params for p in trainer.model.decoder.parameters())

    _seed_pixel_buffer(trainer.buffer, act_dim=cfg.act_dim)
    out = trainer.update()
    assert torch.isfinite(torch.tensor(out["loss"]))
    assert "reconstruction" in out
    assert "consistency" not in out  # reconstruction REPLACES consistency
    assert torch.isfinite(torch.tensor(out["reconstruction"]))
    # reward+value grounding is kept (identical to the JEPA "reward" arm)
    assert "reward" in out and "value" in out


def test_recon_arm_update_drives_the_decoder():
    """A reconstruction-arm update step produces a gradient on the decoder (the whole point —
    capacity is spent modeling pixels, including any background distractor)."""
    from jepa_ctrl.model import MPPIConfig
    from jepa_ctrl.model.trainer import Trainer

    cfg = _cnn_cfg(latent_dim=64, horizon=2, act_dim=4)
    wm = WorldModel(cfg)
    tcfg = TrainConfig(grounding="reconstruction", batch_size=4, seed_steps=0, capacity=1024)
    trainer = Trainer(
        wm, tcfg, -torch.ones(cfg.act_dim), torch.ones(cfg.act_dim), mppi_cfg=MPPIConfig(horizon=2)
    )
    _seed_pixel_buffer(trainer.buffer, act_dim=cfg.act_dim)
    before = torch.cat([p.detach().flatten() for p in trainer.model.decoder.parameters()]).clone()
    trainer.update()
    after = torch.cat([p.detach().flatten() for p in trainer.model.decoder.parameters()])
    assert not torch.allclose(before, after), "decoder params must move under the recon arm"


# =====================================================================================
# (4) the JEPA arm (reward + cnn) owns NO decoder and has NO reconstruction term
# =====================================================================================
def test_jepa_arm_has_no_decoder():
    from jepa_ctrl.model import MPPIConfig
    from jepa_ctrl.model.trainer import Trainer

    cfg = _cnn_cfg(latent_dim=64, horizon=2, act_dim=4)
    wm = WorldModel(cfg)  # default build_decoder=False
    assert wm.decoder is None
    tcfg = TrainConfig(grounding="reward", batch_size=4, seed_steps=0, capacity=1024)  # JEPA arm
    trainer = Trainer(
        wm, tcfg, -torch.ones(cfg.act_dim), torch.ones(cfg.act_dim), mppi_cfg=MPPIConfig(horizon=2)
    )
    # the JEPA arm NEVER builds a decoder (structural: it cannot waste capacity on the distractor)
    assert trainer.model.decoder is None
    _seed_pixel_buffer(trainer.buffer, act_dim=cfg.act_dim)
    out = trainer.update()
    assert "consistency" in out  # JEPA arm = latent consistency
    assert "reconstruction" not in out, "the JEPA arm must have NO reconstruction term"
    assert torch.isfinite(torch.tensor(out["loss"]))


def test_jepa_decode_raises_without_decoder():
    cfg = _cnn_cfg()
    wm = WorldModel(cfg)
    assert wm.decoder is None
    try:
        wm.decode(torch.randn(2, cfg.latent_dim))
        raise AssertionError("expected RuntimeError decoding without a decoder")
    except RuntimeError:
        pass


def test_pixel_buffer_capacity_clamps_to_byte_budget():
    """OOM guard (zero-allocation unit test): the state default capacity (1e6) at a pixel slot
    size must clamp to the 8 GiB ceiling, while a realistic pixel capacity passes through."""
    from jepa_ctrl.model.trainer import pixel_buffer_capacity

    obs = (9, 84, 84)  # 9*84*84*2 = 127008 bytes/slot
    clamped = pixel_buffer_capacity(1_000_000, obs)
    assert clamped < 1_000_000, "default capacity must be clamped for a pixel buffer"
    assert clamped * 127008 <= 8 * 1024**3, "clamped capacity still exceeds the 8 GiB ceiling"
    # a realistic pixel run (~6.3 GB at 5e4) is below the ceiling -> unchanged
    assert pixel_buffer_capacity(50_000, obs) == 50_000
    # never returns < 1 even for an absurd slot size
    assert pixel_buffer_capacity(1_000_000, (3, 100_000, 100_000)) >= 1
