from __future__ import annotations

import numpy as np
import torch

from jepa_ctrl.model import CNNEncoder, ModelConfig, PixelReplayBuffer, WorldModel
from jepa_ctrl.pixels import ProceduralDistractor, composite_distractor, mask_background

torch.manual_seed(0)
RNG = np.random.default_rng(0)


# --- (1) compositing: background replaced, robot untouched, dtype/shape preserved -------
def _synthetic_scene(h=24, w=32):
    """A synthetic rendered frame + seg mask: a robot rectangle (id 0) on background (id -1)."""
    rgb = RNG.integers(0, 256, (h, w, 3), dtype=np.uint8)
    seg = np.full((h, w), -1, dtype=np.int32)  # all background
    seg[6:14, 8:20] = 0  # a robot blob with object id 0
    return rgb, seg


def test_composite_replaces_background_keeps_robot():
    rgb, seg = _synthetic_scene()
    distractor = RNG.integers(0, 256, rgb.shape, dtype=np.uint8)
    out = composite_distractor(rgb, seg, distractor)

    assert out.shape == rgb.shape
    assert out.dtype == np.uint8
    bg = seg == -1
    robot = ~bg
    # background pixels now equal the distractor
    assert np.array_equal(out[bg], distractor[bg])
    # robot pixels are untouched (identical to the original render)
    assert np.array_equal(out[robot], rgb[robot])
    # something actually changed (the synthetic distractor differs from the render on bg)
    assert (out != rgb).any()
    assert rgb.dtype == np.uint8


def test_mask_background_zeros_bg_keeps_robot():
    """R20 masked-target: background -> 0, robot pixels kept verbatim."""
    rgb, seg = _synthetic_scene()
    out = mask_background(rgb, seg)
    assert out.shape == rgb.shape and out.dtype == np.uint8
    bg = seg == -1
    robot = ~bg
    assert np.array_equal(out[robot], rgb[robot])   # robot kept exactly
    assert np.all(out[bg] == 0)                     # background zeroed
    assert out[robot].any()                         # robot signal survives (not all-zero)


def test_mask_background_does_not_mutate_input():
    rgb, seg = _synthetic_scene()
    rgb_copy = rgb.copy()
    _ = mask_background(rgb, seg)
    assert np.array_equal(rgb, rgb_copy)


def test_mask_background_accepts_hwc_seg_channel0():
    rgb, seg2d = _synthetic_scene()
    seg = np.stack([seg2d, np.zeros_like(seg2d)], axis=-1)  # (H,W,2) dm_control raw
    out = mask_background(rgb, seg)
    assert np.array_equal(out[seg2d != -1], rgb[seg2d != -1])
    assert np.all(out[seg2d == -1] == 0)


def test_mask_background_all_background_is_zero():
    rgb = RNG.integers(1, 256, (16, 16, 3), dtype=np.uint8)  # nonzero so the test is meaningful
    seg = np.full((16, 16), -1, dtype=np.int32)
    assert np.all(mask_background(rgb, seg) == 0)


def test_composite_does_not_mutate_inputs():
    rgb, seg = _synthetic_scene()
    rgb_copy = rgb.copy()
    distractor = RNG.integers(0, 256, rgb.shape, dtype=np.uint8)
    distractor_copy = distractor.copy()
    _ = composite_distractor(rgb, seg, distractor)
    assert np.array_equal(rgb, rgb_copy)
    assert np.array_equal(distractor, distractor_copy)


def test_composite_accepts_hwc_seg_channel0():
    # dm_control raw segmentation is (H, W, 2): [object_id, geom_type]. channel 0 is the id.
    rgb, seg2d = _synthetic_scene()
    seg = np.stack([seg2d, np.zeros_like(seg2d)], axis=-1)  # (H, W, 2)
    distractor = RNG.integers(0, 256, rgb.shape, dtype=np.uint8)
    out = composite_distractor(rgb, seg, distractor)
    bg = seg2d == -1
    assert np.array_equal(out[bg], distractor[bg])
    assert np.array_equal(out[~bg], rgb[~bg])


def test_composite_all_background():
    rgb = RNG.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    seg = np.full((16, 16), -1, dtype=np.int32)
    distractor = RNG.integers(0, 256, rgb.shape, dtype=np.uint8)
    out = composite_distractor(rgb, seg, distractor)
    assert np.array_equal(out, distractor)  # everything is background


# --- (2) ProceduralDistractor: shape/dtype, determinism, temporal coherence -------------
def test_distractor_shape_dtype_and_range():
    d = ProceduralDistractor(40, 50, seed=3)
    f = d.frame(0)
    assert f.shape == (40, 50, 3)
    assert f.dtype == np.uint8
    assert f.min() >= 0 and f.max() <= 255


def test_distractor_deterministic_per_seed():
    a = ProceduralDistractor(32, 32, seed=7).frame(5)
    b = ProceduralDistractor(32, 32, seed=7).frame(5)
    c = ProceduralDistractor(32, 32, seed=8).frame(5)
    assert np.array_equal(a, b)  # same seed + t -> identical
    assert not np.array_equal(a, c)  # different seed -> different field


def test_distractor_temporally_coherent_not_iid():
    d = ProceduralDistractor(48, 48, seed=1)
    f0 = d.frame(0).astype(np.float64).ravel()
    f1 = d.frame(1).astype(np.float64).ravel()
    f_far = d.frame(400).astype(np.float64).ravel()

    def corr(x, y):
        return float(np.corrcoef(x, y)[0, 1])

    adjacent = corr(f0, f1)
    # consecutive frames are highly correlated (smooth drift), not iid noise
    assert adjacent > 0.9, f"consecutive frames not coherent: corr={adjacent}"
    # the field genuinely evolves over time (not a frozen image): a far frame differs
    assert not np.array_equal(d.frame(0), d.frame(400))
    # iid noise would have ~0 adjacent correlation; coherence is the point
    noise_a = RNG.integers(0, 256, f0.shape).astype(np.float64)
    noise_b = RNG.integers(0, 256, f0.shape).astype(np.float64)
    assert adjacent > abs(corr(noise_a, noise_b)) + 0.5
    # the distractor still moves (far frame less correlated than adjacent)
    assert corr(f0, f_far) < adjacent


# --- (3) CNNEncoder: output shape + simplex-or-not under latent_norm, on uint8 ----------
def test_cnn_encoder_shape_and_simplex():
    enc = CNNEncoder((9, 84, 84), latent_dim=128, simnorm_groups=8, latent_norm="simnorm")
    obs = torch.randint(0, 256, (5, 9, 84, 84), dtype=torch.uint8)
    z = enc(obs)
    assert z.shape == (5, 128)
    assert (z >= 0).all()
    sums = z.reshape(5, 16, 8).sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    pre = enc.pre_simnorm(obs)
    assert pre.shape == (5, 128)
    pre_sums = pre.reshape(5, 16, 8).sum(-1)
    assert not torch.allclose(pre_sums, torch.ones_like(pre_sums), atol=1e-3)


def test_cnn_encoder_latent_norm_none_is_raw():
    enc = CNNEncoder((9, 84, 84), latent_dim=128, latent_norm="none")
    obs = torch.randint(0, 256, (3, 9, 84, 84), dtype=torch.uint8)
    z = enc(obs)
    assert z.shape == (3, 128)
    # raw latent: not constrained to a simplex (some negatives, groups don't sum to 1)
    sums = z.reshape(3, 16, 8).sum(-1)
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3)
    # forward == pre_simnorm under Identity norm
    assert torch.allclose(z, enc.pre_simnorm(obs))


def test_cnn_encoder_normalizes_uint8_internally():
    enc = CNNEncoder((9, 84, 84), latent_dim=64)
    obs_u8 = torch.randint(0, 256, (2, 9, 84, 84), dtype=torch.uint8)
    # passing the same data as float should yield the same latent (normalization is internal)
    z_u8 = enc(obs_u8)
    z_f = enc(obs_u8.float())
    assert torch.allclose(z_u8, z_f, atol=1e-5)


# --- (4) CNN WorldModel builds + encode/rollout on a random uint8 batch -----------------
def test_cnn_world_model_builds_and_encodes():
    cfg = ModelConfig(
        obs_dim=0, act_dim=6, encoder_type="cnn", obs_shape=(9, 84, 84), latent_dim=128, horizon=3
    )
    wm = WorldModel(cfg)
    assert isinstance(wm.encoder, CNNEncoder)
    assert isinstance(wm.target_encoder, CNNEncoder)
    for p in wm.target_encoder.parameters():
        assert p.requires_grad is False

    b = 4
    obs = torch.randint(0, 256, (b, 9, 84, 84), dtype=torch.uint8)
    z = wm.encode(obs)
    assert z.shape == (b, cfg.latent_dim)
    sums = z.reshape(b, cfg.latent_dim // cfg.simnorm_groups, cfg.simnorm_groups).sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_cnn_world_model_rollout_latents_on_uint8():
    cfg = ModelConfig(
        obs_dim=0, act_dim=2, encoder_type="cnn", obs_shape=(9, 84, 84), latent_dim=128, horizon=3
    )
    wm = WorldModel(cfg)
    h, b = cfg.horizon, 2
    obs_seq = torch.randint(0, 256, (h + 1, b, 9, 84, 84), dtype=torch.uint8)
    act_seq = torch.randn(h, b, cfg.act_dim)
    latents = wm.rollout_latents(obs_seq, act_seq)
    assert len(latents) == h + 1
    for z in latents:
        assert z.shape == (b, cfg.latent_dim)
    # consistency loss runs end-to-end on pixels and grads reach the CNN encoder + predictor
    loss = wm.consistency_loss_from(obs_seq, act_seq, latents)
    assert torch.isfinite(loss)
    loss.backward()
    enc_grad = sum(p.grad.abs().sum() for p in wm.encoder.parameters() if p.grad is not None)
    assert enc_grad > 0


def test_config_cnn_requires_obs_shape():
    try:
        ModelConfig(obs_dim=0, act_dim=2, encoder_type="cnn")  # missing obs_shape
        raise AssertionError("expected ValueError for cnn without obs_shape")
    except ValueError:
        pass


# --- (5) PixelReplayBuffer: uint8 individual frames, frame-stack on sample --------------
def _fill_pixel_buffer(buf, n_steps=40, ep_len=10, h=84, w=84):
    """Add n_steps transitions across episodes of length ep_len; frames are tiny uint8."""
    for i in range(n_steps):
        first = (i % ep_len) == 0
        done = (i % ep_len) == (ep_len - 1)
        frame = torch.randint(0, 256, (3, h, w), dtype=torch.uint8)
        nxt = torch.randint(0, 256, (3, h, w), dtype=torch.uint8)
        buf.add(frame, torch.randn(buf.act_dim), float(i), nxt, done, first=first)


def test_pixel_buffer_add_sample_shapes_and_dtype():
    buf = PixelReplayBuffer(capacity=200, frame_shape=(3, 84, 84), act_dim=6, frame_stack=3)
    assert buf.obs_shape == (9, 84, 84)
    _fill_pixel_buffer(buf, n_steps=40)
    assert len(buf) == 40

    length = 3
    batch = 5
    out = buf.sample_subtraj(batch, length)
    obs_seq, action_seq, reward = out["obs_seq"], out["action_seq"], out["reward"]
    assert obs_seq.shape == (length + 1, batch, 9, 84, 84)
    assert obs_seq.dtype == torch.uint8  # stored + sampled as uint8, NOT float32
    assert action_seq.shape == (length, batch, 6)
    assert reward.shape == (length, batch)


def test_pixel_buffer_single_transition_sample():
    buf = PixelReplayBuffer(capacity=50, frame_shape=(3, 16, 16), act_dim=2, frame_stack=3)
    _fill_pixel_buffer(buf, n_steps=20, ep_len=20, h=16, w=16)
    out = buf.sample_subtraj(batch=4, length=2)
    assert out["obs_seq"].shape == (3, 4, 9, 16, 16)
    assert out["obs_seq"].dtype == torch.uint8


def test_pixel_buffer_memory_under_budget():
    # 1e5 frames of 84x84x3 uint8 (frame + next_frame) must fit comfortably (< a few GB).
    # Use the allocation-free estimator so the budget check never faults in the 4.2 GB it sizes
    # (and never trips a virtual-address cap on a torch/CUDA process).
    expected = 100_000 * 84 * 84 * 3 * 2  # 1e5 * 84*84*3 * 1 byte * 2 stores
    est = PixelReplayBuffer.estimate_nbytes(100_000, (3, 84, 84))
    assert est == expected
    assert est / (1024**3) < 6.0, f"pixel buffer is {est / (1024**3):.2f} GiB, expected < 6 GiB"
    # the estimator must match a small real buffer's nbytes (formula sanity)
    small = PixelReplayBuffer(capacity=200, frame_shape=(3, 84, 84), act_dim=6, frame_stack=3)
    assert small.nbytes == PixelReplayBuffer.estimate_nbytes(200, (3, 84, 84))


def test_pixel_buffer_stack_clamps_at_episode_start():
    # at an episode's first frame, the stack should repeat that frame (no straddle into a prior
    # episode). Build a buffer where every frame is a constant value == its step index.
    buf = PixelReplayBuffer(capacity=50, frame_shape=(3, 4, 4), act_dim=1, frame_stack=3)
    for i in range(12):
        first = i == 0 or i == 6  # two episodes starting at 0 and 6
        done = i == 5 or i == 11
        frame = torch.full((3, 4, 4), i % 6, dtype=torch.uint8)
        nxt = torch.full((3, 4, 4), (i % 6) + 1, dtype=torch.uint8)
        buf.add(frame, torch.zeros(1), float(i), nxt, done, first=first)
    # stack ending at flat index 6 (the start of episode 2): all 3 frames must be value 0
    stack = buf._stack_at(torch.tensor([6]))  # (1, 9, 4, 4)
    assert stack.shape == (1, 9, 4, 4)
    # channels 0:3 (oldest) clamped to the first frame == channels 6:9 (newest) == 0
    assert torch.all(stack == 0)
