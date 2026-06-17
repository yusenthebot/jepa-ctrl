from __future__ import annotations

import numpy as np
import torch

from ..controllers import Controller
from .mppi import MPPIConfig, MPPIPlanner, eval_mppi
from .world_model import WorldModel


class JepaController(Controller):
    """Plugs a trained WorldModel + latent MPPI into the harness `evaluate_controller`.

    `act(obs)` encodes obs, runs receding-horizon MPPI, and returns the first action as a
    np.float32 vector clamped to the env bounds. `reset()` drops the planner's warm-start
    prior between episodes. Pure inference (the model is put in eval mode, no grad).
    """

    def __init__(
        self,
        model: WorldModel,
        act_low: np.ndarray,
        act_high: np.ndarray,
        mppi_cfg: MPPIConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.act_low = np.asarray(act_low, np.float32)
        self.act_high = np.asarray(act_high, np.float32)
        self.planner = MPPIPlanner(
            self.model,
            mppi_cfg or eval_mppi(),
            torch.as_tensor(self.act_low, device=self.device),
            torch.as_tensor(self.act_high, device=self.device),
            self.device,
        )

    def reset(self) -> None:
        self.planner.reset()

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs, np.float32), device=self.device)
        action = self.planner.plan(obs_t).cpu().numpy().astype(np.float32)
        return np.clip(action, self.act_low, self.act_high).astype(np.float32)
