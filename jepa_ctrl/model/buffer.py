from __future__ import annotations

import torch


class PixelReplayBuffer:
    """Frame-efficient pixel replay: stores INDIVIDUAL RGB frames as uint8 (not float32, not
    pre-stacked) and reconstructs the k-frame stack on sample.

    Memory math (why this design): an 84x84x3 frame is 84*84*3 = 21168 bytes (uint8). At
    capacity 1e5 that is 1e5 * 21168 ~= 2.1 GB; storing the trailing next-frame too doubles it
    to ~4.2 GB, comfortably inside 64 GB. By contrast a NAIVE float32, PRE-STACKED buffer at
    1e6 capacity would be 1e6 * (9*84*84) * 4 bytes ~= 254 GB (and even the 6.3 GB pre-stacked
    uint8 variant the brief cites wastes 3x by duplicating frames across stacks) — it does NOT
    fit. Storing single uint8 frames and stacking on read is the only laptop-feasible option.

    Frame-stacking: a stacked obs at step t is [frame_{t-k+1}, .., frame_t] concatenated on the
    channel axis -> (k*3, H, W). At an episode start (or near the ring/episode boundary) earlier
    frames are clamped to the first in-episode frame (standard frame-stack edge handling), so a
    stack never splices two episodes. sample_subtraj returns uint8 obs_seq the encoder
    normalizes internally. Pure-torch, no env / sim dependency.
    """

    def __init__(
        self,
        capacity: int,
        frame_shape: tuple[int, int, int],
        act_dim: int,
        frame_stack: int = 3,
        device: torch.device | str = "cpu",
    ) -> None:
        self.capacity = int(capacity)
        self.frame_shape = tuple(int(x) for x in frame_shape)  # (3, H, W)
        self.act_dim = int(act_dim)
        self.frame_stack = int(frame_stack)
        self.device = torch.device(device)
        c, h, w = self.frame_shape
        self.obs_shape = (c * self.frame_stack, h, w)
        self._frame = torch.zeros(self.capacity, c, h, w, dtype=torch.uint8, device=self.device)
        self._next_frame = torch.zeros(
            self.capacity, c, h, w, dtype=torch.uint8, device=self.device
        )
        self._action = torch.zeros(self.capacity, act_dim, device=self.device)
        self._reward = torch.zeros(self.capacity, device=self.device)
        self._done = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        # `first` marks an episode's first stored transition, so frame-stack clamping never
        # walks back past a reset into the previous episode.
        self._first = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        self._pos = 0
        self._full = False

    def __len__(self) -> int:
        return self.capacity if self._full else self._pos

    @property
    def nbytes(self) -> int:
        """Total bytes of the frame stores (the dominant term) — for the memory budget check."""
        return self._frame.element_size() * self._frame.nelement() * 2  # frame + next_frame

    @staticmethod
    def estimate_nbytes(capacity: int, frame_shape: tuple[int, int, int]) -> int:
        """Frame-store bytes for a (capacity, frame_shape) WITHOUT allocating — so a memory-budget
        check (or capacity planner) never has to fault in the buffer it is sizing."""
        c, h, w = (int(x) for x in frame_shape)
        return int(capacity) * c * h * w * 2  # uint8, frame + next_frame

    def add(
        self,
        frame: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_frame: torch.Tensor,
        done: bool,
        first: bool = False,
    ) -> None:
        """Append one transition. `frame`/`next_frame` are single (3,H,W) uint8 RGB frames
        (o_t, o_{t+1}); `first` flags the episode's opening transition."""
        i = self._pos
        self._frame[i] = torch.as_tensor(frame, dtype=torch.uint8, device=self.device)
        self._next_frame[i] = torch.as_tensor(next_frame, dtype=torch.uint8, device=self.device)
        self._action[i] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self._reward[i] = float(reward)
        self._done[i] = bool(done)
        self._first[i] = bool(first)
        self._pos = (i + 1) % self.capacity
        if self._pos == 0:
            self._full = True

    def _stack_at(self, idx: torch.Tensor) -> torch.Tensor:
        """Build (B, k*3, H, W) uint8 stacks ending at flat indices `idx` (no episode straddle).

        Walks back up to k-1 steps; once a `first` (episode start) is hit the remaining older
        frames are clamped to that first frame.
        """
        k = self.frame_stack
        b = idx.shape[0]
        c, h, w = self.frame_shape
        rows = torch.empty(b, k, dtype=torch.long, device=self.device)
        rows[:, k - 1] = idx
        cur = idx.clone()
        stop = self._first[idx].clone()  # already at an episode start -> clamp everything older
        for back in range(1, k):
            prev = (cur - 1) % self.capacity
            cur = torch.where(stop, cur, prev)  # frozen rows keep their (clamped) index
            rows[:, k - 1 - back] = cur
            stop = stop | self._first[cur]  # once we step ONTO a first frame, clamp from here on
        frames = self._frame[rows.reshape(-1)].reshape(b, k, c, h, w)
        return frames.reshape(b, k * c, h, w)

    def sample_subtraj(self, batch: int, length: int) -> dict[str, torch.Tensor]:
        """Sample `batch` contiguous windows of `length` steps that do not cross a done.

        Returns obs_seq (length+1, batch, k*3, H, W) uint8 = stacked o_t..o_{t+length},
        action_seq (length, batch, act_dim), reward (length, batch). Each obs in the sequence
        is a freshly frame-stacked (k*3,H,W) uint8 tensor. Pass length=horizon.
        """
        n = len(self)
        if n <= length:
            raise ValueError(f"buffer has {n} transitions, need > {length} for a window")
        starts = self._valid_starts(length, batch)
        offs = torch.arange(length, device=self.device)
        rows = (starts.unsqueeze(1) + offs.unsqueeze(0)) % self.capacity  # (batch, length)

        obs_steps = [self._stack_at(rows[:, j]) for j in range(length)]  # each (B, kc, H, W)
        obs = torch.stack(obs_steps, dim=0)  # (length, B, kc, H, W)
        # trailing next-obs: stack ending at the next_frame of the last window step. The stored
        # next_frame equals the following frame's frame, so stacking at the next flat index works.
        last_next_idx = (rows[:, -1] + 1) % self.capacity
        last_next = self._stack_at(last_next_idx).unsqueeze(0)  # (1, B, kc, H, W)
        obs_seq = torch.cat([obs, last_next], dim=0)  # (length+1, B, kc, H, W)
        action_seq = self._action[rows].transpose(0, 1)
        reward = self._reward[rows].transpose(0, 1)
        return {"obs_seq": obs_seq, "action_seq": action_seq, "reward": reward}

    def _valid_starts(self, length: int, batch: int) -> torch.Tensor:
        """Rejection-sample non-wrapped start indices whose first `length` steps contain no
        `done` (a done within the window would splice two episodes). Leaves room for the
        trailing next-obs stack by capping starts at n-length-1 when possible."""
        n = len(self)
        max_start = max(0, n - length - 1)
        out = torch.empty(batch, dtype=torch.long, device=self.device)
        filled = 0
        for _ in range(64):
            cand = torch.randint(0, max_start + 1, (batch * 2,), device=self.device)
            offs = torch.arange(length, device=self.device)
            windows = cand.unsqueeze(1) + offs.unsqueeze(0)  # (M, length)
            ok = ~self._done[windows].any(dim=1)
            good = cand[ok]
            take = min(good.numel(), batch - filled)
            if take > 0:
                out[filled : filled + take] = good[:take]
                filled += take
            if filled >= batch:
                return out
        if filled < batch:
            out[filled:] = torch.randint(
                0, max_start + 1, (batch - filled,), device=self.device
            )
        return out


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
