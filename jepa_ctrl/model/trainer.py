from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from .buffer import PixelReplayBuffer, ReplayBuffer
from .mppi import MPPIConfig, MPPIPlanner, train_mppi
from .sigreg import sigreg_loss
from .world_model import WorldModel


def pixel_buffer_capacity(requested: int, obs_shape, max_bytes: int = 8 * 1024**3) -> int:
    """Clamp a pixel replay-buffer capacity to a byte budget. A pixel slot is ~100x heavier than
    a state vector, so reusing the state-buffer default capacity (1e6) for a pixel model is a
    60+ GB OOM bomb. Bound it so no caller can fault in tens of GB (OOM rule: estimate before
    allocate). Real pixel runs pass ~5e4 (~6.3 GB) and stay below the ceiling unchanged."""
    bytes_per_slot = 2 * int(np.prod(obs_shape))  # frame + next_frame, uint8
    return max(1, min(int(requested), max_bytes // bytes_per_slot))


class NearFallBank:
    """R17 reset-curriculum bank of physics states harvested from LOW-return (collapsing)
    episodes. sample() returns a banked state + small Gaussian pose noise so a new episode can
    START near a fall, teaching recovery. Only LOW-return episodes (the caller gates on a
    return threshold) contribute, so the bank concentrates on the bad-start coverage that the
    standard reset distribution under-samples.

    Reservoir over states (not episodes): when full, a uniformly-random existing slot is
    overwritten, so the bank stays an unbiased sample of all low-return states seen so far.
    """

    def __init__(self, capacity: int, pose_noise: float, rng_seed: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.pose_noise = float(pose_noise)
        self._rng = np.random.default_rng(rng_seed)
        self._states: list[np.ndarray] = []
        self._seen = 0  # total low-return states offered (for reservoir replacement)

    def add_episode(self, states: list[np.ndarray], ep_return: float) -> None:
        """Add a full low-return episode's states to the reservoir. The caller decides the
        episode is low-return; ep_return is accepted for symmetry/logging only."""
        for s in states:
            s = np.asarray(s, np.float64).copy()
            self._seen += 1
            if len(self._states) < self.capacity:
                self._states.append(s)
            else:
                j = int(self._rng.integers(0, self._seen))
                if j < self.capacity:
                    self._states[j] = s

    def sample(self) -> np.ndarray:
        """A banked state + bounded zero-mean Gaussian pose noise (fresh copy each call)."""
        if not self._states:
            raise IndexError("sample() from an empty NearFallBank")
        i = int(self._rng.integers(0, len(self._states)))
        base = self._states[i]
        noise = self._rng.normal(0.0, self.pose_noise, size=base.shape)
        return (base + noise).astype(np.float64)

    def __len__(self) -> int:
        return len(self._states)


@dataclass(frozen=True)
class TrainConfig:
    """Immutable training config. Coefficients follow the locked architecture: consistency
    dominates (20), reward/value grounding is subordinate (0.1 each).

    `grounding` selects the anti-collapse signal for the GROUNDLESS ablation + the distractor
    reconstruction head-to-head:
      - "reward" (default, R3 positive control): reward/value gradients reach the encoder. With
        the CNN encoder and NO decoder this is the JEPA arm of the distractor head-to-head =
        latent consistency + reward grounding (the model the reconstruction arm is matched to).
      - "inverse_dynamics": id_coef * inverse-dynamics on the shared rolled latents grounds the
        encoder/predictor; reward & value heads still train but on DETACHED latents (no gradient
        to encoder/predictor) -> a genuinely REWARD-FREE representation, still plannable by MPPI.
      - "sigreg": sigreg_coef * SIGReg on the RAW online latents (use latent_norm="none"); reward
        & value likewise trained on DETACHED latents. SIGReg replaces SimNorm+EMA as the pressure.
      - "reconstruction": consistency is REPLACED by recon_coef * pixel reconstruction (a
        Dreamer-style decoder; predicted latents must reconstruct future frames). reward+value
        grounding is byte-identical to the "reward" arm, so the ONLY difference vs the JEPA arm
        is consistency <-> reconstruction. The decoder wastes capacity modeling a time-varying
        background distractor — the failure a reconstruction model structurally cannot avoid.
    """

    grounding: str = "reward"  # "reward" | "inverse_dynamics" | "sigreg" | "reconstruction"
    id_coef: float = 1.0  # inverse-dynamics grounding weight (inverse_dynamics arm)
    sigreg_coef: float = 1.0  # SIGReg grounding weight (sigreg arm)
    recon_coef: float = 1.0  # pixel reconstruction weight (reconstruction arm)
    freeze_repr: bool = False  # red-team: freeze repr at random init; only rv heads learn
    lr: float = 3e-4
    enc_lr_scale: float = 0.3
    batch_size: int = 256
    grad_clip: float = 10.0
    explore_std: float = 0.3  # gaussian action noise added to the MPPI behaviour action (start)
    explore_std_end: float = 0.05  # annealed exploration floor — reduce late-training thrashing
    explore_anneal_steps: int = 100_000  # linear anneal explore_std -> explore_std_end over this
    # hybrid objective (explore-then-exploit): linearly anneal the intrinsic weight beta on the
    # collection planner. beta_start dominates with disagreement early (discover the sparse reward
    # on EVERY seed), beta_end~0 hands over to reward exploitation late (dwell/hold => coherent data).
    explore_beta_start: float = 1.0
    explore_beta_end: float = 0.0
    explore_beta_anneal_steps: int = 60_000
    seed_steps: int = 1000  # uniform-random warmup before MPPI takes over collection
    ema_tau_start: float = 0.99
    ema_tau_end: float = 0.996
    ema_anneal_steps: int = 100_000
    eval_every: int = 10_000
    gamma: float = 0.99
    capacity: int = 1_000_000
    # R17 reset-curriculum (cover-to-recover probe of the bad-start-coverage diagnosis). Default
    # OFF -> additive, every existing run is byte-identical. When on, a fraction reset_p of resets
    # start from a banked near-fall state (+ pose noise) instead of the standard reset.
    reset_curriculum: bool = False
    reset_p: float = 0.3  # probability a reset draws from the bank (when non-empty)
    # R20 masked-target distractor robustness (pixel only): the EMA consistency target is computed
    # on a parallel ROBOT-ONLY (background-zeroed) frame stream while the online encoder/planner see
    # the full distractor-composited obs. Requires a PixelDMCEnv with masked_target=True.
    masked_target: bool = False
    bank_capacity: int = 5000  # max physics states held in the near-fall reservoir
    bank_pose_noise: float = 0.02  # Gaussian std added to a sampled state at reset
    bank_return_thresh: float = 150.0  # episodes with return < this feed the bank


def _param_groups(model: WorldModel, cfg: TrainConfig) -> list[dict]:
    """Encoder gets lr * enc_lr_scale (slower encoder per the locked architecture); the
    target nets are frozen and excluded."""
    enc_ids = {id(p) for p in model.encoder.parameters()}
    enc, rest = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (enc if id(p) in enc_ids else rest).append(p)
    groups = [
        {"params": enc, "lr": cfg.lr * cfg.enc_lr_scale},
        {"params": rest, "lr": cfg.lr},
    ]
    return [g for g in groups if g["params"]]  # drop empty groups (e.g. frozen encoder)


class Trainer:
    """Trains a WorldModel: collect with an MPPI behaviour policy (+ exploration noise),
    update with consistency_coef * consistency + rv_coef * (reward + value), step the EMA
    targets, and run a periodic eval hook. The collect loop drives an env via the same
    (reset/step) protocol as the harness DMCEnv; the caller owns the env (the trainer never
    constructs or closes a simulator).
    """

    def __init__(
        self,
        model: WorldModel,
        cfg: TrainConfig,
        act_low: torch.Tensor,
        act_high: torch.Tensor,
        mppi_cfg: MPPIConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.cfg = cfg
        if cfg.grounding == "reconstruction" and self.model.decoder is None:
            # the reconstruction arm OWNS a decoder; the Trainer builds it (and includes its
            # params in the optimizer below) so the JEPA arm can stay decoder-free by default.
            mcfg = self.model.cfg
            if mcfg.encoder_type != "cnn" or mcfg.obs_shape is None:
                raise ValueError("reconstruction grounding requires a CNN encoder + obs_shape")
            from .nets import Decoder

            self.model.decoder = Decoder(mcfg.latent_dim, mcfg.obs_shape).to(self.device)
        if cfg.freeze_repr:  # red-team: representation stays at random init; only rv heads learn
            for mod in (self.model.encoder, self.model.predictor):
                for p in mod.parameters():
                    p.requires_grad_(False)
        self.mppi_cfg = mppi_cfg or train_mppi()
        self.act_low = torch.as_tensor(act_low, dtype=torch.float32, device=self.device)
        self.act_high = torch.as_tensor(act_high, dtype=torch.float32, device=self.device)
        self.planner = MPPIPlanner(
            self.model, self.mppi_cfg, self.act_low, self.act_high, self.device
        )
        # pixel encoder -> uint8 frame buffer (frame-stacked on sample); state encoder -> flat
        # buffer. update() only consumes sample_subtraj(), whose interface is identical.
        if model.cfg.encoder_type == "cnn":
            mc = model.cfg
            # the env already frame-stacks; store the FULL stacked obs as one frame (frame_stack=1)
            # to avoid a redundant re-stack and the env/buffer single-frame interface mismatch.
            # Bound the pixel buffer to a byte budget (see pixel_buffer_capacity): the state
            # default capacity (1e6) would be a 60+ GB OOM bomb at pixel slot sizes.
            # masked-target stores a 2nd robot-only stream -> 2x per-slot bytes; budget for it.
            slot_mult = 2 if cfg.masked_target else 1
            pix_cap = pixel_buffer_capacity(cfg.capacity, mc.obs_shape, 8 * 1024**3 // slot_mult)
            self.buffer = PixelReplayBuffer(
                pix_cap, tuple(mc.obs_shape), mc.act_dim, 1, self.device,
                masked=cfg.masked_target,
            )
        else:
            self.buffer = ReplayBuffer(
                cfg.capacity, model.cfg.obs_dim, model.cfg.act_dim, self.device
            )
        self.opt = torch.optim.Adam(_param_groups(model, cfg))
        self.step = 0
        # R19 leg3 instrumentation: count collection steps that hit reward>0 (= reached the reward
        # region). Task-agnostic exploration metric — directly tests "did exploration discover the
        # sparse reward", independent of whether the eval planner can exploit it.
        self.reward_hits = 0
        # R17 reset-curriculum state (inert unless cfg.reset_curriculum). The bank harvests
        # physics states from low-return episodes; _ep_states / _ep_return track the CURRENT
        # episode as collect feeds note_step(). _rc_rng gates the reset draw deterministically.
        self.bank = NearFallBank(cfg.bank_capacity, cfg.bank_pose_noise, rng_seed=0)
        self._ep_states: list[np.ndarray] = []
        self._ep_return = 0.0
        self._rc_rng = np.random.default_rng(0)

    # --- EMA schedule -------------------------------------------------------------
    def _enc_tau(self) -> float:
        c = self.cfg
        frac = min(1.0, self.step / max(1, c.ema_anneal_steps))
        return c.ema_tau_start + (c.ema_tau_end - c.ema_tau_start) * frac

    def _explore_std(self) -> float:
        """Linearly anneal exploration noise explore_std -> explore_std_end. Constant high noise
        thrashes a near-solved policy (R6 finding: reacher oscillated under fixed std=0.3)."""
        c = self.cfg
        frac = min(1.0, self.step / max(1, c.explore_anneal_steps))
        return c.explore_std + (c.explore_std_end - c.explore_std) * frac

    def _explore_beta(self) -> float:
        """Linearly anneal the hybrid intrinsic weight beta_start -> beta_end (explore->exploit)."""
        c = self.cfg
        frac = min(1.0, self.step / max(1, c.explore_beta_anneal_steps))
        return c.explore_beta_start + (c.explore_beta_end - c.explore_beta_start) * frac

    # --- behaviour policy ---------------------------------------------------------
    @torch.no_grad()
    def behaviour_action(self, obs: torch.Tensor) -> torch.Tensor:
        """MPPI action + annealed gaussian exploration noise (uniform random during seed steps)."""
        if self.step < self.cfg.seed_steps:
            u = torch.rand(self.model.cfg.act_dim, device=self.device)
            return self.act_low + u * (self.act_high - self.act_low)
        self.planner.explore_beta = self._explore_beta()  # hybrid explore->exploit schedule
        a = self.planner.plan(obs)
        a = a + self._explore_std() * torch.randn_like(a)
        return a.clamp(self.act_low, self.act_high)

    # --- value/reward targets -----------------------------------------------------
    @torch.no_grad()
    def _value_target(
        self,
        next_obs: torch.Tensor,
        next_action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        """SARSA TD target r + gamma * (1-done) * min_subset q_target(z', a') with the
        target value net. z' = encode(next_obs) (no grad) and a' = next_action — the action
        ACTUALLY taken next in the sub-trajectory (not a zero-action proxy), so the value head
        learns the on-policy return rather than the value of doing nothing."""
        z_next = self.model.encode(next_obs)  # SimNorm latent (no grad)
        v_logits = self.model.target_value_head.logits(z_next, next_action)  # (num_q,B,bins)
        v = self.model.target_value_head.to_scalar(v_logits)  # (num_q, B)
        k = min(self.mppi_cfg.min_q_subset, v.shape[0])
        v_min = v.topk(k, dim=0, largest=False).values.amin(0)  # (B,)
        return reward + self.cfg.gamma * (~done).float() * v_min

    # --- one gradient step --------------------------------------------------------
    def update(self) -> dict[str, float]:
        """Sample a sub-trajectory, compute the arm-gated combined loss, step, EMA.

        Common to every arm: latent consistency on the shared rolled latents. The GROUNDING and
        the gradient path of the reward/value heads depend on cfg.grounding:
          - "reward": rv on the SHARED (grad-coupled) latents -> rv grounds the encoder (R3).
          - "inverse_dynamics"/"sigreg": rv on DETACHED latents -> rv NEVER touches the
            encoder/predictor (reward-free representation), while the task-agnostic grounding
            (inverse-dynamics or SIGReg) is the anti-collapse pressure. rv heads still train.
        """
        cfg, mcfg = self.cfg, self.model.cfg
        h = mcfg.horizon
        traj = self.buffer.sample_subtraj(cfg.batch_size, h)
        obs_seq, action_seq, reward_seq = traj["obs_seq"], traj["action_seq"], traj["reward"]
        done0 = torch.zeros(cfg.batch_size, dtype=torch.bool, device=self.device)

        # one open-loop rollout shared by consistency/reconstruction + grounding (rolls ONCE)
        latents = self.model.rollout_latents(obs_seq, action_seq)  # [z_0, z_hat_1..z_hat_H]
        recon_arm = cfg.grounding == "reconstruction"
        # reconstruction arm REPLACES latent consistency with pixel reconstruction; both are
        # skipped under the frozen-repr red-team control (no representation learning at all).
        if cfg.freeze_repr:
            repr_loss = latents[0].new_zeros(())
        elif recon_arm:
            repr_loss = self.model.reconstruction_loss(obs_seq, latents)
        else:
            # R20 masked-target: route the stop-grad consistency target through the robot-only
            # masked stream (if the buffer provides it) while the online latents used the full obs.
            target_obs_seq = traj.get("obs_masked_seq")
            repr_loss = self.model.consistency_loss_from(
                obs_seq, action_seq, latents, target_obs_seq=target_obs_seq
            )
        repr_coef = mcfg.consistency_coef if not recon_arm else cfg.recon_coef

        reward_free = cfg.grounding in ("inverse_dynamics", "sigreg")
        # reward-free arms detach the latents under the rv heads so NO rv gradient reaches the
        # encoder/predictor; the reward arm keeps them coupled (rv IS the grounding there).
        rv_latents = [z.detach() for z in latents] if reward_free else latents

        r_loss = self.model.reward_grounding_loss(action_seq, reward_seq, rv_latents)
        v_tgt = self._value_target(obs_seq[1], action_seq[1], reward_seq[0], done0)
        v_loss = self.model.value_loss(rv_latents[0], action_seq[0], v_tgt)

        loss = repr_coef * repr_loss + mcfg.rv_coef * (r_loss + v_loss)
        metrics = {
            "reward": float(r_loss.detach()),
            "value": float(v_loss.detach()),
        }
        metrics["reconstruction" if recon_arm else "consistency"] = float(repr_loss.detach())

        if not cfg.freeze_repr and cfg.grounding == "inverse_dynamics":
            id_loss = self.model.inverse_dynamics_loss(action_seq, latents)
            loss = loss + cfg.id_coef * id_loss
            metrics["inverse_dynamics"] = float(id_loss.detach())
        elif not cfg.freeze_repr and cfg.grounding == "sigreg":
            online = torch.cat(latents, dim=0)  # ((H+1)*B, d) RAW online latents (grad-coupled)
            sr_loss = sigreg_loss(online, lam=1.0)
            loss = loss + cfg.sigreg_coef * sr_loss
            metrics["sigreg"] = float(sr_loss.detach())

        # R18+ disagreement ensemble: trained on DETACHED latents, so it adds an independent loss
        # term that updates ONLY the ensemble params (the representation is unchanged vs the
        # no-ensemble run). Skipped under freeze_repr (no learning at all).
        if not cfg.freeze_repr and self.model.predictor_ensemble is not None:
            ens_loss = self.model.ensemble_consistency_loss_from(obs_seq, action_seq, latents)
            loss = loss + mcfg.consistency_coef * ens_loss
            metrics["ensemble"] = float(ens_loss.detach())
            # Plan2Explore intrinsic value: disagreement-as-reward SARSA TD on DETACHED latents
            # (measure-don't-reshape — the intrinsic value head never touches the representation).
            if self.model.explore_value_head is not None:
                z0d = latents[0].detach()
                with torch.no_grad():
                    disag0 = self.model.ensemble_disagreement(z0d, action_seq[0])  # (B,)
                    self.model.update_disag_scale(disag0)
                    r_int = disag0 / (self.model.disag_scale + 1e-8)  # normalized intrinsic reward
                    zn = self.model.encode(obs_seq[1])
                    ev_t = self.model.target_explore_value_head.logits(zn, action_seq[1])
                    ev_t = self.model.target_explore_value_head.to_scalar(ev_t).mean(0)  # optimistic
                    iv_tgt = r_int + cfg.gamma * ev_t
                iv_loss = self.model.intrinsic_value_loss(z0d, action_seq[0], iv_tgt)
                loss = loss + mcfg.rv_coef * iv_loss
                metrics["intrinsic_value"] = float(iv_loss.detach())
                metrics["disag_scale"] = float(self.model.disag_scale)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad), cfg.grad_clip
        )
        self.opt.step()
        self.model.ema_update(self._enc_tau())

        metrics["loss"] = float(loss.detach())
        return metrics

    # --- reset-curriculum hooks (R17) --------------------------------------------
    def reset_env(self, env) -> torch.Tensor:
        """Reset `env` and return the obs tensor; the SINGLE reset entry point shared by
        train() and scripts/train.py so the curriculum drives BOTH collect loops identically.

        Default (cfg.reset_curriculum=False): a plain env.reset() — byte-identical to the
        pre-R17 path. With the curriculum on and a non-empty bank, with prob reset_p the
        episode starts from a banked near-fall state (+ pose noise); otherwise a standard
        reset. Always re-arms the per-episode tracker.
        """
        cfg = self.cfg
        if (
            cfg.reset_curriculum
            and len(self.bank) > 0
            and float(self._rc_rng.random()) < cfg.reset_p
        ):
            obs_np = env.reset(from_state=self.bank.sample())
        else:
            obs_np = env.reset()
        self._ep_states = []
        self._ep_return = 0.0
        return torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)

    def note_step(self, state: np.ndarray | None, reward: float, done: bool) -> None:
        """Feed one collected transition into the per-episode tracker. `state` is the
        env physics state BEFORE the step (env.get_state()); pass None to skip banking
        (e.g. curriculum off). On episode end, if the episode is low-return, harvest its
        states into the bank."""
        if state is not None:
            self._ep_states.append(np.asarray(state, np.float64).copy())
        self._ep_return += float(reward)
        if done:
            if (
                self.cfg.reset_curriculum
                and self._ep_states
                and self._ep_return < self.cfg.bank_return_thresh
            ):
                self.bank.add_episode(self._ep_states, self._ep_return)
            self._ep_states = []
            self._ep_return = 0.0

    # --- collect + train loop -----------------------------------------------------
    def collect_step(self, env, obs: torch.Tensor) -> tuple[torch.Tensor, float, bool]:
        """Take one behaviour action in `env`, store the transition. Returns (next_obs, r, done).
        `env` follows the DMCEnv protocol (reset/step). The trainer never builds the env.

        When cfg.reset_curriculum, snapshots env.get_state() BEFORE the step and feeds the
        transition to note_step() so the near-fall bank is harvested without the caller having
        to thread the tracking itself."""
        pre_state = (
            env.get_state()
            if self.cfg.reset_curriculum and hasattr(env, "get_state")
            else None
        )
        # R20 masked-target: snapshot the robot-only masked obs for o_t (before) and o_{t+1} (after).
        masked_t = (
            torch.as_tensor(env.masked_obs(), dtype=torch.uint8, device=self.device)
            if self.buffer.masked else None
        )
        a = self.behaviour_action(obs)
        next_obs_np, reward, done = env.step(a.cpu().numpy())
        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=self.device)
        if self.buffer.masked:
            masked_tp1 = torch.as_tensor(env.masked_obs(), dtype=torch.uint8, device=self.device)
            self.buffer.add(obs, a, reward, next_obs, done,
                            masked_frame=masked_t, masked_next_frame=masked_tp1)
        else:
            self.buffer.add(obs, a, reward, next_obs, done)
        self.step += 1
        if reward > 0.0:
            self.reward_hits += 1
        self.note_step(pre_state, reward, done)
        return next_obs, float(reward), bool(done)

    def train(
        self,
        env,
        total_steps: int,
        updates_per_step: int = 1,
        eval_hook: Callable[[int], None] | None = None,
    ) -> None:  # pragma: no cover - drives a live sim; not exercised in unit tests
        """Full collect+update loop. NOT run in unit tests (it drives a real simulator)."""
        obs = self.reset_env(env)
        self.planner.reset()
        while self.step < total_steps:
            next_obs, _, done = self.collect_step(env, obs)
            obs = next_obs
            if done:
                obs = self.reset_env(env)
                self.planner.reset()
            if len(self.buffer) > self.model.cfg.horizon + 1 and self.step >= self.cfg.seed_steps:
                for _ in range(updates_per_step):
                    self.update()
            if eval_hook is not None and self.step % self.cfg.eval_every == 0:
                eval_hook(self.step)
