#!/usr/bin/env python
"""R21 reward-free GOAL-REACHING eval. Loads a trained JEPA world model and tests whether latent-
distance MPPI (objective='goal', NO reward/value/decoder) can drive the REAL sim from a start state
to a HINDSIGHT goal state (a state actually reached K steps later in a reference rollout — guaranteed
reachable). Compares the goal-planner to a random-action baseline from the same start.

Metric (true physics-state L2 to the goal state): final/initial ratio (lower=better) + success rate
(final < frac*initial). REAL-VERIFY: every episode is driven in the live dm_control sim via
set_state + step. Usage:
  MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python scripts/r21_goal_eval.py --ckpt <dir>/model.pt \
      --task point_mass-easy [--n-goals 20 --horizon-gap 30 --plan-steps 40 --seed 1]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from jepa_ctrl.envs import DMCEnv
from jepa_ctrl.model import ModelConfig, WorldModel
from jepa_ctrl.model.mppi import MPPIConfig, MPPIPlanner
from jepa_ctrl.seeding import set_seed


def load_wm(ckpt: str, obs_dim: int, act_dim: int, device: str) -> WorldModel:
    sd = torch.load(ckpt, map_location=device)
    ld = sd["encoder.proj.weight"].shape[0]  # infer latent_dim from the encoder projection
    cfg = ModelConfig(obs_dim=obs_dim, act_dim=act_dim, latent_dim=ld)  # simnorm default
    wm = WorldModel(cfg).to(device)
    wm.load_state_dict(sd)
    wm.eval()
    return wm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task", default="point_mass-easy")
    p.add_argument("--n-goals", type=int, default=20)
    p.add_argument("--horizon-gap", type=int, default=30, help="goal = state reached this many steps later")
    p.add_argument("--plan-steps", type=int, default=40, help="max sim steps allowed to reach the goal")
    p.add_argument("--action-repeat", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--success-frac", type=float, default=0.33, help="success if final < frac*initial")
    p.add_argument("--quasimetric-head", default="", help="R22: path to a trained IQE head (.pt) to "
                   "attach as the goal metric (else latent-L2)")
    p.add_argument("--shuffle-goals", action="store_true", help="red-flag control: permute goals vs "
                   "starts; a real goal-reacher must collapse to ~random under this")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    set_seed(a.seed)

    env = DMCEnv(a.task, seed=a.seed, action_repeat=a.action_repeat)
    wm = load_wm(a.ckpt, env.obs_dim, env.act_dim, device)
    if a.quasimetric_head:  # R22: attach the trained IQE head so goal-MPPI uses it instead of L2
        from jepa_ctrl.model.nets import QuasimetricHead
        hd = torch.load(a.quasimetric_head, map_location=device)
        qm = QuasimetricHead(hd["latent_dim"], k=hd["k"]).to(device)
        qm.load_state_dict(hd["state_dict"]); qm.eval()
        wm.quasimetric_head = qm
        print(f"[goal_eval] attached quasimetric head {a.quasimetric_head} (k={hd['k']})")
    cfg = MPPIConfig(horizon=5, iters=6, num_samples=512, num_elites=64, objective="goal")
    planner = MPPIPlanner(wm, cfg, env.act_low, env.act_high, device=device)
    rng = np.random.default_rng(a.seed)

    # reference rollout (random actions) -> (obs, physics_state) sequence; hindsight goals from it.
    def cur_obs():
        return env._flat_obs(env._env.task.get_observation(env._env.physics))

    env.reset()
    traj_obs, traj_state = [], []
    for _ in range(a.n_goals + a.horizon_gap + 5):
        traj_state.append(env.get_state())
        traj_obs.append(cur_obs())
        u = rng.uniform(env.act_low, env.act_high).astype(np.float32)
        _, _, done = env.step(u)
        if done:
            env.reset()

    def dist(s_a, s_b):
        return float(np.linalg.norm(np.asarray(s_a) - np.asarray(s_b)))

    def pdist(s_a, s_b):  # position-only: first half of the physics state (qpos), drops velocity
        s_a, s_b = np.asarray(s_a), np.asarray(s_b)
        nq = len(s_a) // 2
        return float(np.linalg.norm(s_a[:nq] - s_b[:nq]))

    goal_ratios, rand_ratios, succ, rand_succ = [], [], 0, 0
    goal_pratios, rand_pratios, psucc, rand_psucc = [], [], 0, 0
    n = min(a.n_goals, len(traj_state) - a.horizon_gap - 1)
    # shuffle-goals red-flag control: pair each start with ANOTHER start's hindsight goal, breaking the
    # start->goal reachability correspondence. A real goal-reacher must collapse toward random here.
    gperm = rng.permutation(n) if a.shuffle_goals else np.arange(n)
    for i in range(n):
        start_state = traj_state[i]
        gidx = gperm[i] + a.horizon_gap
        goal_state = traj_state[gidx]
        goal_obs = traj_obs[gidx]
        d0 = dist(start_state, goal_state)
        p0 = pdist(start_state, goal_state)
        if d0 < 1e-6 or p0 < 1e-6:
            continue
        # --- goal-MPPI rollout in the real sim ---
        env.reset(from_state=start_state)
        planner.reset()
        planner.set_goal(torch.as_tensor(goal_obs, dtype=torch.float32, device=device))
        for _ in range(a.plan_steps):
            act = planner.plan(torch.as_tensor(cur_obs(), dtype=torch.float32, device=device))
            env.step(act.cpu().numpy())
        end_goal = env.get_state()
        d_goal = dist(end_goal, goal_state)
        # --- random-action baseline from the same start ---
        env.reset(from_state=start_state)
        for _ in range(a.plan_steps):
            env.step(rng.uniform(env.act_low, env.act_high).astype(np.float32))
        end_rand = env.get_state()
        d_rand = dist(end_rand, goal_state)

        goal_ratios.append(d_goal / d0)
        rand_ratios.append(d_rand / d0)
        succ += int(d_goal < a.success_frac * d0)
        rand_succ += int(d_rand < a.success_frac * d0)
        # position-only (qpos): the spatial-reach signal, free of the velocity-match confound
        pg = pdist(end_goal, goal_state); pr = pdist(end_rand, goal_state)
        goal_pratios.append(pg / p0); rand_pratios.append(pr / p0)
        psucc += int(pg < a.success_frac * p0); rand_psucc += int(pr < a.success_frac * p0)

    m = len(goal_ratios)
    out = {
        "task": a.task, "n": m, "horizon_gap": a.horizon_gap, "plan_steps": a.plan_steps,
        "goal_ratio_median": float(np.median(goal_ratios)) if m else None,
        "rand_ratio_median": float(np.median(rand_ratios)) if m else None,
        "goal_success_rate": succ / m if m else None,
        "rand_success_rate": rand_succ / m if m else None,
        "goal_pos_ratio_median": float(np.median(goal_pratios)) if m else None,
        "rand_pos_ratio_median": float(np.median(rand_pratios)) if m else None,
        "goal_pos_success_rate": psucc / m if m else None,
        "rand_pos_success_rate": rand_psucc / m if m else None,
        "success_frac": a.success_frac, "seed": a.seed,
    }
    print(json.dumps(out, indent=2))
    import os
    os.makedirs("runs/R21_goaleval", exist_ok=True)
    with open(f"runs/R21_goaleval/{a.task.replace('-', '_')}_s{a.seed}.json", "w") as f:
        json.dump(out, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
