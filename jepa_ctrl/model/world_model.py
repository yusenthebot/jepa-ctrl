from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .nets import DistHead, Encoder, InverseDynamicsHead, Predictor, symlog


def _ema_update_(target: nn.Module, online: nn.Module, tau: float) -> None:
    """In-place Polyak update: target <- tau*target + (1-tau)*online. No grad."""
    with torch.no_grad():
        for t, o in zip(target.parameters(), online.parameters(), strict=True):
            t.mul_(tau).add_(o.detach(), alpha=1.0 - tau)
        for t, o in zip(target.buffers(), online.buffers(), strict=True):
            t.copy_(o)


class WorldModel(nn.Module):
    """Action-conditioned JEPA world model (state input, no decoder).

    Bundles: online encoder f_theta, EMA target encoder f_xi (stop-grad), residual predictor
    g_phi, distributional reward head, distributional value-ensemble head + its EMA target net.
    The target encoder and target value net never receive gradient; they track the online nets
    via `ema_update`.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        ld, ad = cfg.latent_dim, cfg.act_dim
        g = cfg.simnorm_groups

        ln = cfg.latent_norm
        self.encoder = Encoder(cfg.obs_dim, ld, cfg.enc_hidden, g, latent_norm=ln)
        self.target_encoder = copy.deepcopy(self.encoder)
        self._freeze(self.target_encoder)

        self.predictor = Predictor(ld, ad, cfg.pred_hidden, cfg.action_head_dim, g, latent_norm=ln)
        self.inverse_dynamics_head = InverseDynamicsHead(ld, ad, cfg.pred_hidden)

        self.reward_head = DistHead(ld, ad, cfg.pred_hidden, cfg.bins, cfg.vmin, cfg.vmax, num_q=1)
        self.value_head = DistHead(
            ld, ad, cfg.pred_hidden, cfg.bins, cfg.vmin, cfg.vmax, num_q=cfg.num_q
        )
        self.target_value_head = copy.deepcopy(self.value_head)
        self._freeze(self.target_value_head)

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad_(False)

    # --- encoding -----------------------------------------------------------------
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Online SimNorm latent (gradient flows)."""
        return self.encoder(obs)

    def encode_pre(self, obs: torch.Tensor) -> torch.Tensor:
        """Online pre-SimNorm activation (the predictor's residual space)."""
        return self.encoder.pre_simnorm(obs)

    @torch.no_grad()
    def encode_target(self, obs: torch.Tensor) -> torch.Tensor:
        """EMA target SimNorm latent, stop-grad (the JEPA consistency target)."""
        return self.target_encoder(obs)

    # --- rollout ------------------------------------------------------------------
    def rollout(self, z0_pre: torch.Tensor, action_seq: torch.Tensor) -> list[torch.Tensor]:
        """Open-loop autoregressive rollout. z0_pre = pre-SimNorm latent of o_t.
        action_seq: (H, batch, act_dim). Returns [z_hat_1, ..., z_hat_H] as SimNorm latents,
        feeding the predictor its OWN output each step (pre-SimNorm carried via the residual).
        """
        z_pre = z0_pre
        out: list[torch.Tensor] = []
        for k in range(action_seq.shape[0]):
            z_next = self.predictor(z_pre, action_seq[k])
            out.append(z_next)
            # open-loop: feed the predictor its own SimNorm output as next residual space.
            z_pre = z_next
        return out

    # --- losses -------------------------------------------------------------------
    def rollout_latents(
        self, obs_seq: torch.Tensor, action_seq: torch.Tensor
    ) -> list[torch.Tensor]:
        """Online rollout latents shared by consistency + reward grounding.

        Returns [z_0, z_hat_1, .., z_hat_H] (length H+1) as SimNorm latents:
        z_0 = encode(obs_seq[0]) (online, grad) and z_hat_k = the open-loop predicted
        latent of o_{t+k} for k=1..H. Index alignment: out[k] is the latent for o_{t+k}.
        Rolls the predictor ONCE; consistency and reward grounding consume the same list.
        """
        z0_pre = self.encode_pre(obs_seq[0])
        z0 = self.encoder.simnorm(z0_pre)
        return [z0, *self.rollout(z0_pre, action_seq)]

    def consistency_loss_from(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        latents: list[torch.Tensor],
    ) -> torch.Tensor:
        """Consistency loss reusing pre-rolled latents (shared with reward grounding).

        latents: [z_0, z_hat_1, .., z_hat_H] from rollout_latents. Uses latents[k] (= the
        predicted latent of o_{t+k}) for k=1..H; latents[0] is the encoding of o_t and is not
        scored against a target here. L = sum_{k=1..H} rho^k * smooth_l1(latents[k], f_xi(o_{t+k})).
        """
        cfg = self.cfg
        h = action_seq.shape[0]
        loss = obs_seq.new_zeros(())
        for k in range(1, h + 1):
            target = self.encode_target(obs_seq[k])  # stop-grad inside
            loss = loss + (cfg.rho**k) * F.smooth_l1_loss(latents[k], target)
        return loss

    def consistency_loss(
        self, obs_seq: torch.Tensor, action_seq: torch.Tensor
    ) -> torch.Tensor:
        """Multi-step latent consistency over horizon H with discount rho^k.

        obs_seq: (H+1, batch, obs_dim) — o_t .. o_{t+H}. action_seq: (H, batch, act_dim).
        L = sum_{k=1..H} rho^k * smooth_l1(z_hat_{t+k}, stopgrad f_xi(o_{t+k})).
        z_hat rolled open-loop from the online encoding of o_t.
        """
        latents = self.rollout_latents(obs_seq, action_seq)  # [z_0, z_hat_1..z_hat_H]
        return self.consistency_loss_from(obs_seq, action_seq, latents)

    def reward_grounding_loss(
        self,
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor,
        latents: list[torch.Tensor],
    ) -> torch.Tensor:
        """Two-hot reward cross-entropy summed over the FULL open-loop rollout.

        action_seq: (H, batch, act_dim). reward_seq: (H, batch) = r(o_{t+k}, a_{t+k}).
        latents: [z_0, z_hat_1, .., z_hat_H] from rollout_latents (length H+1) — z_0 is the
        online encoding (grad through encoder); z_hat_k>=1 are predicted (grad through the
        predictor + encoder). For each step k=0..H-1 the reward head predicts reward_seq[k]
        from (latents[k], action_seq[k]); the gradient therefore reaches EVERY rolled latent
        (the anti-collapse grounding signal), not just step 0. Returns the mean over k.
        """
        h = action_seq.shape[0]
        loss = action_seq.new_zeros(())
        for k in range(h):
            r_logits = self.reward_head.logits(latents[k], action_seq[k])[0]
            r_label = self.reward_head.twohot_encode(symlog(reward_seq[k]))
            loss = loss + -(r_label * F.log_softmax(r_logits, dim=-1)).sum(-1).mean()
        return loss / h

    def inverse_dynamics_loss(
        self, action_seq: torch.Tensor, latents: list[torch.Tensor]
    ) -> torch.Tensor:
        """Task-AGNOSTIC inverse-dynamics grounding over the SHARED rolled latents.

        action_seq: (H, batch, act_dim). latents: [z_0, z_hat_1, .., z_hat_H] from
        rollout_latents (length H+1) — latents[k] is the predicted latent of o_{t+k}. For each
        step k=0..H-1 the head predicts action_seq[k] (= a_{t+k}) from (latents[k], latents[k+1]);
        the MSE gradient therefore flows into BOTH the encoder (via z_0) and the predictor (via
        every z_hat), making the latent action-discriminative on the PREDICTED dynamics. This is
        the anti-collapse signal that REPLACES task reward in the inverse-dynamics arm.
        Returns the mean over k.
        """
        h = action_seq.shape[0]
        loss = action_seq.new_zeros(())
        for k in range(h):
            a_hat = self.inverse_dynamics_head(latents[k], latents[k + 1])
            loss = loss + F.mse_loss(a_hat, action_seq[k])
        return loss / h

    def value_loss(
        self, z0: torch.Tensor, action0: torch.Tensor, value_target: torch.Tensor
    ) -> torch.Tensor:
        """Two-hot value cross-entropy over the full ensemble at step 0.

        z0, action0: (batch, *). value_target: (batch,) — the SARSA TD target. Grounds the
        value head at (z_0, a_0); the target is computed grad-free by the trainer.
        """
        v_logits = self.value_head.logits(z0, action0)  # (num_q, batch, bins)
        v_label = self.value_head.twohot_encode(symlog(value_target))
        return -(v_label.unsqueeze(0) * F.log_softmax(v_logits, dim=-1)).sum(-1).mean()

    def reward_value_losses(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        reward_target: torch.Tensor,
        value_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-entropy of two-hot(symlog(target)) vs predicted bin logits.
        z, action: (batch, *). targets: (batch,). Returns (reward_loss, value_loss)."""
        r_logits = self.reward_head.logits(z, action)[0]
        r_label = self.reward_head.twohot_encode(symlog(reward_target))
        r_loss = -(r_label * F.log_softmax(r_logits, dim=-1)).sum(-1).mean()

        v_logits = self.value_head.logits(z, action)  # (num_q, batch, bins)
        v_label = self.value_head.twohot_encode(symlog(value_target))
        v_loss = -(v_label.unsqueeze(0) * F.log_softmax(v_logits, dim=-1)).sum(-1).mean()
        return r_loss, v_loss

    # --- EMA ----------------------------------------------------------------------
    def ema_update(self, enc_tau: float | None = None) -> None:
        """Polyak-update the target encoder (enc_tau, default cfg.ema_tau) and the target
        value net (cfg.value_tau). Targets never see gradient; this is their only update path.
        """
        tau = self.cfg.ema_tau if enc_tau is None else enc_tau
        _ema_update_(self.target_encoder, self.encoder, tau)
        _ema_update_(self.target_value_head, self.value_head, 1.0 - self.cfg.value_tau)
