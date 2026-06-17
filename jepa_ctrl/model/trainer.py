from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .buffer import ReplayBuffer
from .mppi import MPPIConfig, MPPIPlanner, train_mppi
from .world_model import WorldModel


@dataclass(frozen=True)
class TrainConfig:
    """Immutable training config. Coefficients follow the locked architecture: consistency
    dominates (20), reward/value grounding is subordinate (0.1 each)."""

    lr: float = 3e-4
    enc_lr_scale: float = 0.3
    batch_size: int = 256
    grad_clip: float = 10.0
    explore_std: float = 0.3  # gaussian action noise added to the MPPI behaviour action
    seed_steps: int = 1000  # uniform-random warmup before MPPI takes over collection
    ema_tau_start: float = 0.99
    ema_tau_end: float = 0.996
    ema_anneal_steps: int = 100_000
    eval_every: int = 10_000
    gamma: float = 0.99
    capacity: int = 1_000_000


def _param_groups(model: WorldModel, cfg: TrainConfig) -> list[dict]:
    """Encoder gets lr * enc_lr_scale (slower encoder per the locked architecture); the
    target nets are frozen and excluded."""
    enc_ids = {id(p) for p in model.encoder.parameters()}
    enc, rest = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (enc if id(p) in enc_ids else rest).append(p)
    return [
        {"params": enc, "lr": cfg.lr * cfg.enc_lr_scale},
        {"params": rest, "lr": cfg.lr},
    ]


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
        self.mppi_cfg = mppi_cfg or train_mppi()
        self.act_low = torch.as_tensor(act_low, dtype=torch.float32, device=self.device)
        self.act_high = torch.as_tensor(act_high, dtype=torch.float32, device=self.device)
        self.planner = MPPIPlanner(
            self.model, self.mppi_cfg, self.act_low, self.act_high, self.device
        )
        self.buffer = ReplayBuffer(cfg.capacity, model.cfg.obs_dim, model.cfg.act_dim, self.device)
        self.opt = torch.optim.Adam(_param_groups(model, cfg))
        self.step = 0

    # --- EMA schedule -------------------------------------------------------------
    def _enc_tau(self) -> float:
        c = self.cfg
        frac = min(1.0, self.step / max(1, c.ema_anneal_steps))
        return c.ema_tau_start + (c.ema_tau_end - c.ema_tau_start) * frac

    # --- behaviour policy ---------------------------------------------------------
    @torch.no_grad()
    def behaviour_action(self, obs: torch.Tensor) -> torch.Tensor:
        """MPPI action + gaussian exploration noise (uniform random during seed steps)."""
        if self.step < self.cfg.seed_steps:
            u = torch.rand(self.model.cfg.act_dim, device=self.device)
            return self.act_low + u * (self.act_high - self.act_low)
        a = self.planner.plan(obs)
        a = a + self.cfg.explore_std * torch.randn_like(a)
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
        """Sample a sub-trajectory + transitions, compute the combined loss, step, EMA."""
        cfg, mcfg = self.cfg, self.model.cfg
        h = mcfg.horizon
        traj = self.buffer.sample_subtraj(cfg.batch_size, h)
        obs_seq, action_seq, reward_seq = traj["obs_seq"], traj["action_seq"], traj["reward"]

        # one open-loop rollout shared by consistency + reward grounding
        latents = self.model.rollout_latents(obs_seq, action_seq)  # [z_0, z_hat_1..z_hat_H]
        consistency = self.model.consistency_loss_from(obs_seq, action_seq, latents)

        # reward grounding over the FULL rollout: r(o_{t+k}, a_{t+k}) for k=0..H-1
        r_loss = self.model.reward_grounding_loss(action_seq, reward_seq, latents)

        # value grounding at step 0 with a SARSA target using the REAL next action a_{t+1}
        done0 = torch.zeros(cfg.batch_size, dtype=torch.bool, device=self.device)
        v_tgt = self._value_target(obs_seq[1], action_seq[1], reward_seq[0], done0)
        v_loss = self.model.value_loss(latents[0], action_seq[0], v_tgt)

        loss = mcfg.consistency_coef * consistency + mcfg.rv_coef * (r_loss + v_loss)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad), cfg.grad_clip
        )
        self.opt.step()
        self.model.ema_update(self._enc_tau())

        return {
            "loss": float(loss.detach()),
            "consistency": float(consistency.detach()),
            "reward": float(r_loss.detach()),
            "value": float(v_loss.detach()),
        }

    # --- collect + train loop -----------------------------------------------------
    def collect_step(self, env, obs: torch.Tensor) -> tuple[torch.Tensor, float, bool]:
        """Take one behaviour action in `env`, store the transition. Returns (next_obs, r, done).
        `env` follows the DMCEnv protocol (reset/step). The trainer never builds the env."""
        a = self.behaviour_action(obs)
        next_obs_np, reward, done = env.step(a.cpu().numpy())
        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=self.device)
        self.buffer.add(obs, a, reward, next_obs, done)
        self.step += 1
        return next_obs, float(reward), bool(done)

    def train(
        self,
        env,
        total_steps: int,
        updates_per_step: int = 1,
        eval_hook: Callable[[int], None] | None = None,
    ) -> None:  # pragma: no cover - drives a live sim; not exercised in unit tests
        """Full collect+update loop. NOT run in unit tests (it drives a real simulator)."""
        obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=self.device)
        self.planner.reset()
        while self.step < total_steps:
            next_obs, _, done = self.collect_step(env, obs)
            obs = next_obs
            if done:
                obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=self.device)
                self.planner.reset()
            if len(self.buffer) > self.model.cfg.horizon + 1 and self.step >= self.cfg.seed_steps:
                for _ in range(updates_per_step):
                    self.update()
            if eval_hook is not None and self.step % self.cfg.eval_every == 0:
                eval_hook(self.step)
