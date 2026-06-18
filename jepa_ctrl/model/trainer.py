from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .buffer import ReplayBuffer
from .mppi import MPPIConfig, MPPIPlanner, train_mppi
from .sigreg import sigreg_loss
from .world_model import WorldModel


@dataclass(frozen=True)
class TrainConfig:
    """Immutable training config. Coefficients follow the locked architecture: consistency
    dominates (20), reward/value grounding is subordinate (0.1 each).

    `grounding` selects the anti-collapse signal for the 3-arm GROUNDLESS ablation:
      - "reward" (default, R3 positive control): reward/value gradients reach the encoder.
      - "inverse_dynamics": id_coef * inverse-dynamics on the shared rolled latents grounds the
        encoder/predictor; reward & value heads still train but on DETACHED latents (no gradient
        to encoder/predictor) -> a genuinely REWARD-FREE representation, still plannable by MPPI.
      - "sigreg": sigreg_coef * SIGReg on the RAW online latents (use latent_norm="none"); reward
        & value likewise trained on DETACHED latents. SIGReg replaces SimNorm+EMA as the pressure.
    """

    grounding: str = "reward"  # "reward" | "inverse_dynamics" | "sigreg"
    id_coef: float = 1.0  # inverse-dynamics grounding weight (inverse_dynamics arm)
    sigreg_coef: float = 1.0  # SIGReg grounding weight (sigreg arm)
    freeze_repr: bool = False  # red-team: freeze repr at random init; only rv heads learn
    lr: float = 3e-4
    enc_lr_scale: float = 0.3
    batch_size: int = 256
    grad_clip: float = 10.0
    explore_std: float = 0.3  # gaussian action noise added to the MPPI behaviour action (start)
    explore_std_end: float = 0.05  # annealed exploration floor — reduce late-training thrashing
    explore_anneal_steps: int = 100_000  # linear anneal explore_std -> explore_std_end over this
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
        self.buffer = ReplayBuffer(cfg.capacity, model.cfg.obs_dim, model.cfg.act_dim, self.device)
        self.opt = torch.optim.Adam(_param_groups(model, cfg))
        self.step = 0

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

    # --- behaviour policy ---------------------------------------------------------
    @torch.no_grad()
    def behaviour_action(self, obs: torch.Tensor) -> torch.Tensor:
        """MPPI action + annealed gaussian exploration noise (uniform random during seed steps)."""
        if self.step < self.cfg.seed_steps:
            u = torch.rand(self.model.cfg.act_dim, device=self.device)
            return self.act_low + u * (self.act_high - self.act_low)
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

        # one open-loop rollout shared by consistency + grounding (the predictor rolls ONCE)
        latents = self.model.rollout_latents(obs_seq, action_seq)  # [z_0, z_hat_1..z_hat_H]
        consistency = (
            latents[0].new_zeros(())
            if cfg.freeze_repr  # frozen-repr control: no representation learning at all
            else self.model.consistency_loss_from(obs_seq, action_seq, latents)
        )

        reward_free = cfg.grounding in ("inverse_dynamics", "sigreg")
        # reward-free arms detach the latents under the rv heads so NO rv gradient reaches the
        # encoder/predictor; the reward arm keeps them coupled (rv IS the grounding there).
        rv_latents = [z.detach() for z in latents] if reward_free else latents

        r_loss = self.model.reward_grounding_loss(action_seq, reward_seq, rv_latents)
        v_tgt = self._value_target(obs_seq[1], action_seq[1], reward_seq[0], done0)
        v_loss = self.model.value_loss(rv_latents[0], action_seq[0], v_tgt)

        loss = mcfg.consistency_coef * consistency + mcfg.rv_coef * (r_loss + v_loss)
        metrics = {
            "consistency": float(consistency.detach()),
            "reward": float(r_loss.detach()),
            "value": float(v_loss.detach()),
        }

        if not cfg.freeze_repr and cfg.grounding == "inverse_dynamics":
            id_loss = self.model.inverse_dynamics_loss(action_seq, latents)
            loss = loss + cfg.id_coef * id_loss
            metrics["inverse_dynamics"] = float(id_loss.detach())
        elif not cfg.freeze_repr and cfg.grounding == "sigreg":
            online = torch.cat(latents, dim=0)  # ((H+1)*B, d) RAW online latents (grad-coupled)
            sr_loss = sigreg_loss(online, lam=1.0)
            loss = loss + cfg.sigreg_coef * sr_loss
            metrics["sigreg"] = float(sr_loss.detach())

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad), cfg.grad_clip
        )
        self.opt.step()
        self.model.ema_update(self._enc_tau())

        metrics["loss"] = float(loss.detach())
        return metrics

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
