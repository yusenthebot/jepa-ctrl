from __future__ import annotations

import torch


class ReplayBuffer:
    """Flat transition store with episode-boundary tracking for sub-trajectory sampling.

    Stores (obs, action, reward, done) on a ring. `sample_subtraj(batch, length)` draws
    contiguous windows that DO NOT straddle an episode boundary, so the JEPA consistency
    loss always sees real H-step rollouts. obs is stored with one extra slot per step so a
    length-L window yields L obs (o_t..o_{t+L-1}) plus the trailing next-obs; consistency
    needs (H+1) observations, so request length = horizon + 1.

    All tensors live on `device` (cpu is the default; the trainer moves the model to GPU and
    samples are .to(device) at use). Pure-torch, no env / sim dependency.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = torch.device(device)
        self._obs = torch.zeros(self.capacity, obs_dim, device=self.device)
        self._next_obs = torch.zeros(self.capacity, obs_dim, device=self.device)
        self._action = torch.zeros(self.capacity, act_dim, device=self.device)
        self._reward = torch.zeros(self.capacity, device=self.device)
        self._done = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        self._pos = 0
        self._full = False

    def __len__(self) -> int:
        return self.capacity if self._full else self._pos

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_obs: torch.Tensor,
        done: bool,
    ) -> None:
        """Append one transition (o_t, a_t, r_t, o_{t+1}, done_t)."""
        i = self._pos
        self._obs[i] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self._next_obs[i] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        self._action[i] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self._reward[i] = float(reward)
        self._done[i] = bool(done)
        self._pos = (i + 1) % self.capacity
        if self._pos == 0:
            self._full = True

    def sample(self, batch: int) -> dict[str, torch.Tensor]:
        """Uniformly sample `batch` single transitions (for reward/value grounding)."""
        n = len(self)
        if n == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = torch.randint(0, n, (batch,), device=self.device)
        return {
            "obs": self._obs[idx],
            "action": self._action[idx],
            "reward": self._reward[idx],
            "next_obs": self._next_obs[idx],
            "done": self._done[idx],
        }

    def sample_subtraj(self, batch: int, length: int) -> dict[str, torch.Tensor]:
        """Sample `batch` contiguous windows of `length` steps that do not cross a done.

        Returns obs_seq (length+1, batch, obs_dim) = o_t..o_{t+length}, action_seq
        (length, batch, act_dim), reward (length, batch). Pass length=horizon for a
        (H+1)-observation rollout window.
        """
        n = len(self)
        if n <= length:
            raise ValueError(f"buffer has {n} transitions, need > {length} for a window")
        starts = self._valid_starts(length, batch)
        offs = torch.arange(length, device=self.device)
        rows = (starts.unsqueeze(1) + offs.unsqueeze(0)) % self.capacity  # (batch, length)

        obs = self._obs[rows].transpose(0, 1)  # (length, batch, obs_dim)
        last_next = self._next_obs[rows[:, -1]].unsqueeze(0)  # (1, batch, obs_dim)
        obs_seq = torch.cat([obs, last_next], dim=0)  # (length+1, batch, obs_dim)
        action_seq = self._action[rows].transpose(0, 1)  # (length, batch, act_dim)
        reward = self._reward[rows].transpose(0, 1)  # (length, batch)
        return {"obs_seq": obs_seq, "action_seq": action_seq, "reward": reward}

    def _valid_starts(self, length: int, batch: int) -> torch.Tensor:
        """Rejection-sample start indices whose first length-1 steps contain no `done`
        (a done within the window would splice two episodes)."""
        n = len(self)
        max_start = n - length  # inclusive upper bound on a flat (non-wrapped) start
        out = torch.empty(batch, dtype=torch.long, device=self.device)
        filled = 0
        for _ in range(64):  # bounded retries; falls through with whatever is valid
            cand = torch.randint(0, max_start + 1, (batch * 2,), device=self.device)
            offs = torch.arange(length - 1, device=self.device)
            windows = cand.unsqueeze(1) + offs.unsqueeze(0)  # (M, length-1)
            ok = ~self._done[windows].any(dim=1)
            good = cand[ok]
            take = min(good.numel(), batch - filled)
            if take > 0:
                out[filled : filled + take] = good[:take]
                filled += take
            if filled >= batch:
                return out
        # extremely fragmented buffer: top up with raw candidates (best effort)
        if filled < batch:
            out[filled:] = torch.randint(
                0, max_start + 1, (batch - filled,), device=self.device
            )
        return out
