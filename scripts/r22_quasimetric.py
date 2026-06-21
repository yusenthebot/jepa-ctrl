#!/usr/bin/env python
"""R22 — make the JEPA latent a GOAL METRIC via a frozen-encoder QUASIMETRIC head (diagnosed R21 fix;
R21: latent-L2 vs true-dist rho=0.23). This script: (1) collect a reacher buffer (obs+qpos) on the
FROZEN R6 control encoder; (2) train the IQE quasimetric head (local hinge d<=gap on real within-episode
pairs + contrastive spread on random pairs); (3) RHO PRE-CHECK GATE — spearman(d_phi, ||qpos_a-qpos_b||)
vs the latent-L2 baseline; (4) SAVE the head. Goal-reaching itself is run by r21_goal_eval.py
--quasimetric-head (correct set_state hindsight harness, position metric, + shuffled-goal control).
Usage: MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python scripts/r22_quasimetric.py \
   --ckpt runs/R06_stability/reacher-easy_s0_r6/model.pt --task reacher-easy [--seed 1]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from jepa_ctrl.envs import DMCEnv
from jepa_ctrl.model import ModelConfig, WorldModel
from jepa_ctrl.model.nets import QuasimetricHead
from jepa_ctrl.seeding import set_seed


def load_encoder(ckpt, obs_dim, act_dim, device):
    sd = torch.load(ckpt, map_location=device)
    ld = sd["encoder.proj.weight"].shape[0]
    wm = WorldModel(ModelConfig(obs_dim=obs_dim, act_dim=act_dim, latent_dim=ld)).to(device)
    wm.load_state_dict(sd, strict=False)  # old R6 ckpt predates disag_scale/inverse_dynamics_head
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    return wm, ld


def collect(env, n_steps, rng):
    obs, qpos, ep = [], [], []
    env.reset(); e = 0
    for _ in range(n_steps):
        s = env.get_state()
        obs.append(env._flat_obs(env._env.task.get_observation(env._env.physics)))
        qpos.append(s[: len(s) // 2]); ep.append(e)
        _, _, done = env.step(rng.uniform(env.act_low, env.act_high).astype(np.float32))
        if done:
            env.reset(); e += 1
    return np.array(obs, np.float32), np.array(qpos, np.float64), np.array(ep)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task", default="reacher-easy")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--collect-steps", type=int, default=20000)
    p.add_argument("--head-steps", type=int, default=20000)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--k-max", type=int, default=30)
    p.add_argument("--rho-gate", type=float, default=0.6)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    set_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    env = DMCEnv(a.task, seed=a.seed, action_repeat=2)
    wm, ld = load_encoder(a.ckpt, env.obs_dim, env.act_dim, dev)

    obs, qpos, ep = collect(env, a.collect_steps, rng)
    with torch.no_grad():
        Z = wm.encode(torch.as_tensor(obs, device=dev))
    N = len(obs)
    print(f"[r22] {N} states; qpos range {qpos.min(0)}..{qpos.max(0)} "
          f"span {(qpos.max(0) - qpos.min(0))}", flush=True)

    qm = QuasimetricHead(ld, k=a.k).to(dev)
    opt = torch.optim.Adam(qm.parameters(), lr=3e-4)
    ep_t = torch.as_tensor(ep, device=dev)
    for step in range(a.head_steps):
        i = torch.randint(0, N, (256,), device=dev)
        gap = torch.randint(1, a.k_max + 1, (256,), device=dev)
        j = torch.clamp(i + gap, max=N - 1)
        same = ep_t[i] == ep_t[j]
        d_local = qm(Z[i], Z[j])
        local = (torch.relu(d_local - (j - i).float())[same] ** 2).mean()
        r1 = torch.randint(0, N, (256,), device=dev); r2 = torch.randint(0, N, (256,), device=dev)
        spread = F.softplus(a.k_max - qm(Z[r1], Z[r2])).mean()
        loss = local + spread
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 4000 == 0:
            print(f"[r22] step {step} local {local.item():.3f} spread {spread.item():.3f}", flush=True)
    qm.eval()

    with torch.no_grad():
        ii = rng.integers(0, N, 2000); jj = rng.integers(0, N, 2000)
        ti = torch.as_tensor(ii, device=dev); tj = torch.as_tensor(jj, device=dev)
        dq = qm(Z[ti], Z[tj]).cpu().numpy()
        dl2 = ((Z[ti] - Z[tj]) ** 2).mean(1).cpu().numpy()
        tp = np.linalg.norm(qpos[ii] - qpos[jj], axis=1)
    rho_qm = float(spearmanr(dq, tp).correlation)
    rho_l2 = float(spearmanr(dl2, tp).correlation)
    print(f"[r22] RHO quasimetric={rho_qm:.3f}  latent-L2={rho_l2:.3f}  gate>={a.rho_gate}", flush=True)

    os.makedirs("runs/R22_quasimetric", exist_ok=True)
    tag = f"{a.task.replace('-', '_')}_s{a.seed}"
    head_path = f"runs/R22_quasimetric/{tag}_qmhead.pt"
    torch.save({"state_dict": qm.state_dict(), "latent_dim": ld, "k": a.k}, head_path)
    out = {"task": a.task, "seed": a.seed, "n_states": int(N), "k": a.k, "k_max": a.k_max,
           "rho_quasimetric": rho_qm, "rho_latent_l2": rho_l2, "rho_gate": a.rho_gate,
           "qpos_span": (qpos.max(0) - qpos.min(0)).tolist(),
           "verdict": "PRECHECK_PASS" if rho_qm >= a.rho_gate else "PRECHECK_FAIL",
           "head_path": head_path}
    print(json.dumps(out, indent=2))
    with open(f"runs/R22_quasimetric/{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
