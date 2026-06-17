from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .nets import DistHead, Encoder, Predictor, symlog


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

        self.encoder = Encoder(cfg.obs_dim, ld, cfg.enc_hidden, g)
        self.target_encoder = copy.deepcopy(self.encoder)
        self._freeze(self.target_encoder)

        self.predictor = Predictor(ld, ad, cfg.pred_hidden, cfg.action_head_dim, g)

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
    def consistency_loss(
        self, obs_seq: torch.Tensor, action_seq: torch.Tensor
    ) -> torch.Tensor:
        """Multi-step latent consistency over horizon H with discount rho^k.

        obs_seq: (H+1, batch, obs_dim) — o_t .. o_{t+H}. action_seq: (H, batch, act_dim).
        L = sum_{k=1..H} rho^k * smooth_l1(z_hat_{t+k}, stopgrad f_xi(o_{t+k})).
        z_hat rolled open-loop from the online encoding of o_t.
        """
        cfg = self.cfg
        h = action_seq.shape[0]
        z0_pre = self.encode_pre(obs_seq[0])
        preds = self.rollout(z0_pre, action_seq)
        loss = obs_seq.new_zeros(())
        for k in range(1, h + 1):
            target = self.encode_target(obs_seq[k])  # stop-grad inside
            loss = loss + (cfg.rho**k) * F.smooth_l1_loss(preds[k - 1], target)
        return loss

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
