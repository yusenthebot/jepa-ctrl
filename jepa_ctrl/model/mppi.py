from __future__ import annotations

from dataclasses import dataclass

import torch

from .world_model import WorldModel


@dataclass(frozen=True)
class MPPIConfig:
    """Immutable MPPI planning config (latent receding-horizon, BLUEPRINT defaults).

    Train preset: horizon=3, iters=4, num_samples=256, num_elites=64.
    Eval preset:  horizon=3, iters=6, num_samples=512, num_elites=64.
    """

    horizon: int = 3
    iters: int = 4
    num_samples: int = 256
    num_elites: int = 64
    gamma: float = 0.99
    temperature: float = 0.5  # softmax temperature over elite scores
    momentum: float = 0.1  # mu/std EMA across iters (1-momentum on the new estimate)
    std_min: float = 0.05
    std_max: float = 2.0
    init_std: float = 1.0
    corr: float = 0.0  # temporal correlation of the action noise in [0,1)
    min_q_subset: int = 2  # min-of-subset size from the value ensemble (pessimism)
    objective: str = "reward"  # "reward" (extrinsic + terminal value) | "disagreement" (intrinsic
    # Plan2Explore exploration: maximise discounted ensemble disagreement; task reward unused) |
    # "hybrid" (extrinsic + beta*intrinsic; beta annealed by the trainer => explore-early exploit-late)

    def __post_init__(self) -> None:
        if self.num_elites > self.num_samples:
            raise ValueError("num_elites cannot exceed num_samples")
        if not 0.0 <= self.corr < 1.0:
            raise ValueError("corr must be in [0, 1)")
        if self.objective not in ("reward", "disagreement", "hybrid", "goal"):
            raise ValueError(
                f"objective must be 'reward'|'disagreement'|'hybrid'|'goal', got {self.objective!r}"
            )


def train_mppi() -> MPPIConfig:
    return MPPIConfig(horizon=3, iters=4, num_samples=256, num_elites=64)


def eval_mppi() -> MPPIConfig:
    return MPPIConfig(horizon=3, iters=6, num_samples=512, num_elites=64)


class MPPIPlanner:
    """Latent-space receding-horizon MPPI over a trained WorldModel.

    Per call: encode obs -> sample N action sequences ~N(mu,std) (clamped, temporally
    correlated) -> roll the predictor open-loop -> score sum_h gamma^h r(z_h,a_h) +
    gamma^H min_ensemble q(z_H,a_H) -> softmax-weight elites -> update mu,std; iterate.
    Returns the first action of the receding-horizon plan; `mu` is warm-started across
    calls (policy prior). Random-shooting fallback = a single iteration from a fresh
    zero-mean prior (no prior carried).
    """

    def __init__(
        self,
        model: WorldModel,
        cfg: MPPIConfig,
        act_low: torch.Tensor,
        act_high: torch.Tensor,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device)
        self.act_dim = model.cfg.act_dim
        self.act_low = torch.as_tensor(act_low, dtype=torch.float32, device=self.device)
        self.act_high = torch.as_tensor(act_high, dtype=torch.float32, device=self.device)
        self._mu: torch.Tensor | None = None  # warm-start prior (H, act_dim)
        # hybrid objective: weight on the intrinsic (disagreement) term. The trainer anneals this
        # 1->0 over training so collection explores early and exploits the task reward late. Inert
        # unless cfg.objective=="hybrid".
        self.explore_beta: float = 1.0
        # goal objective (R21): SimNorm latent of the goal obs, set via set_goal(). The planner scores
        # trajectories by NEGATIVE latent distance to it — reward-free, decoder-free goal-reaching.
        self.goal_z: torch.Tensor | None = None

    @torch.no_grad()
    def set_goal(self, goal_obs: torch.Tensor) -> None:
        """Set the goal as the SimNorm latent of `goal_obs` (state vec or pixel frame). The goal
        objective then drives the rollout toward this latent. Reward-free / decoder-free."""
        g = torch.as_tensor(goal_obs, dtype=torch.float32, device=self.device)
        if g.ndim in (1, 3):
            g = g.unsqueeze(0)
        self.goal_z = self.model.encode(g)  # (1, latent) SimNorm — same space as rollout latents

    def reset(self) -> None:
        """Drop the warm-start prior (call between episodes)."""
        self._mu = None

    @torch.no_grad()
    def _score(self, z0_pre: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """actions: (H, N, act_dim). Returns (N,) trajectory score under cfg.objective.
          reward       -> extrinsic only
          disagreement -> intrinsic only
          hybrid       -> extrinsic + explore_beta * intrinsic (beta annealed => explore->exploit)"""
        obj = self.cfg.objective
        if obj == "disagreement":
            return self._score_disagreement(z0_pre, actions)
        if obj == "hybrid":
            return self._score_reward(z0_pre, actions) + self.explore_beta * self._score_disagreement(
                z0_pre, actions
            )
        if obj == "goal":
            return self._score_goal(z0_pre, actions)
        return self._score_reward(z0_pre, actions)

    @torch.no_grad()
    def _score_goal(self, z0_pre: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Reward-free GOAL-reaching score: NEGATIVE discounted latent distance to goal_z along the
        rollout (closer to the goal latent each step = higher score). SimNorm latents, MSE distance.
        No reward/value/decoder consulted. actions: (H, N, act_dim) -> (N,). Requires set_goal()."""
        if self.goal_z is None:
            raise RuntimeError("goal objective requires set_goal() first")
        cfg = self.cfg
        qm = self.model.quasimetric_head  # R22: learned reachability metric if attached, else L2
        preds = self.model.rollout(z0_pre, actions)  # H x (N, latent) SimNorm
        score = z0_pre.new_zeros(actions.shape[1])
        for h, z_h in enumerate(preds):
            if qm is not None:
                dist = qm(z_h, self.goal_z.expand_as(z_h))  # (N,) quasimetric steps-to-go
            else:
                dist = ((z_h - self.goal_z) ** 2).mean(dim=-1)  # (N,) latent-L2 fallback
            score = score - (cfg.gamma**h) * dist
        return score

    @torch.no_grad()
    def _score_reward(self, z0_pre: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Extrinsic score: discounted reward + terminal value (min-of-subset pessimism)."""
        cfg = self.cfg
        preds = self.model.rollout(z0_pre, actions)  # H x (N, latent)
        ret = z0_pre.new_zeros(actions.shape[1])
        for h, z_h in enumerate(preds):
            r_logits = self.model.reward_head.logits(z_h, actions[h])[0]
            ret = ret + (cfg.gamma**h) * self.model.reward_head.to_scalar(r_logits)
        # terminal value bootstrap: min-of-subset over the value ensemble (pessimism)
        z_term = preds[-1]
        v_logits = self.model.value_head.logits(z_term, actions[-1])  # (num_q, N, bins)
        v_scalar = self.model.value_head.to_scalar(v_logits)  # (num_q, N)
        k = min(cfg.min_q_subset, v_scalar.shape[0])
        v_term = v_scalar.topk(k, dim=0, largest=False).values.amin(0)
        return ret + (cfg.gamma ** cfg.horizon) * v_term

    @torch.no_grad()
    def _score_disagreement(self, z0_pre: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Intrinsic Plan2Explore score over the SAME shared predictor (fair vs the reward arm);
        only the objective differs — seek states the ensemble disagrees on (high model uncertainty
        = novel). Task reward/value are NOT consulted.

        Score = sum_h gamma^h * normalized_disagreement(z_h, a_h)  [+ gamma^H * intrinsic_value(z_H,
        a_H) when an explore value head exists]. The intrinsic value bootstrap turns the MYOPIC
        H-step novelty sum into LONG-HORIZON exploration ("expected future novelty"), which is what
        a temporally-extended hard-exploration task (e.g. swing-up) needs. OPTIMISTIC: mean over the
        intrinsic value ensemble (exploration seeks uncertainty, so no min-of-subset pessimism).
        actions: (H, N, act_dim) -> (N,)."""
        cfg = self.cfg
        ev = self.model.explore_value_head
        score = z0_pre.new_zeros(actions.shape[1])
        z_in = z0_pre
        for h in range(cfg.horizon):
            score = score + (cfg.gamma**h) * self.model.normalized_disagreement(z_in, actions[h])
            z_in = self.model.predictor(z_in, actions[h])  # open-loop, shared dynamics
        if ev is not None:
            v_term = ev.to_scalar(ev.logits(z_in, actions[-1])).mean(0)  # (N,) optimistic mean
            score = score + (cfg.gamma ** cfg.horizon) * v_term
        return score

    @torch.no_grad()
    def plan(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: unbatched state (obs_dim,) or pixel (C,H,W) — returns first action (act_dim,)."""
        cfg = self.cfg
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs.ndim in (1, 3):  # add batch dim; keep (C,H,W) intact for the CNN encoder
            obs = obs.unsqueeze(0)
        z0_pre = self.model.encode_pre(obs).expand(cfg.num_samples, -1).contiguous()

        h, ad = cfg.horizon, self.act_dim
        mu = (
            torch.zeros(h, ad, device=self.device)
            if self._mu is None
            else self._mu.clone()
        )
        std = torch.full((h, ad), cfg.init_std, device=self.device)

        for _ in range(cfg.iters):
            actions = self._sample(mu, std)  # (H, N, act_dim)
            scores = self._score(z0_pre, actions)  # (N,)
            elite_idx = scores.topk(min(cfg.num_elites, cfg.num_samples)).indices
            elites = actions[:, elite_idx]  # (H, E, act_dim)
            e_scores = scores[elite_idx]
            w = torch.softmax((e_scores - e_scores.max()) / cfg.temperature, dim=0)
            w = w.view(1, -1, 1)
            new_mu = (w * elites).sum(1)
            new_std = (w * (elites - new_mu.unsqueeze(1)) ** 2).sum(1).sqrt()
            mu = cfg.momentum * mu + (1.0 - cfg.momentum) * new_mu
            std = (cfg.momentum * std + (1.0 - cfg.momentum) * new_std).clamp(
                cfg.std_min, cfg.std_max
            )

        self._mu = torch.cat([mu[1:], mu[-1:].clone()], dim=0)  # shift for receding horizon
        return mu[0].clamp(self.act_low, self.act_high)

    def _sample(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Temporally-correlated Gaussian samples, clamped to bounds. -> (H, N, act_dim)."""
        cfg = self.cfg
        h, ad, n = cfg.horizon, self.act_dim, cfg.num_samples
        noise = torch.randn(h, n, ad, device=self.device)
        if cfg.corr > 0.0:  # AR(1) smoothing along the horizon
            for t in range(1, h):
                noise[t] = cfg.corr * noise[t - 1] + (1.0 - cfg.corr) * noise[t]
        actions = mu.unsqueeze(1) + std.unsqueeze(1) * noise
        return actions.clamp(self.act_low, self.act_high)
