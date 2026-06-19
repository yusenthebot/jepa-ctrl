"""R13 diagnostic: WHY does reward-free latent control degrade with DoF (3D quadruped)?

H1 (capacity) already refuted in R12 (lat512 quad reward-free = 281, within the lat256
variance band; latent stays collapsed regardless). This script discriminates the two
remaining hypotheses, EVAL-ONLY (no retraining), on already-trained models:

  H2  planner under-searches the 12-D action space
      -> re-evaluate saved models with progressively beefier MPPI (samples/iters/horizon).
         If return rises toward the reward-grounded ceiling, control is search-limited.

  H3  the consistency-only latent under-encodes the task subspace in high DoF
      -> linear-probe the frozen online latent for reward / next-reward / obs decodability.
         Compare cheetah-reward-free (works ~496) vs quad-reward-free (degrades ~184) vs
         quad-reward-grounded (~450). Low task-R^2 only for quad-RF => representation is the limit.

Usage: python scripts/diag_dof.py --part both --episodes 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jepa_ctrl.envs import DMCEnv
from jepa_ctrl.model import ModelConfig, WorldModel
from jepa_ctrl.model.jepa_controller import JepaController
from jepa_ctrl.model.mppi import MPPIConfig

# (label, dir, task, latent_norm, latent_dim) — latent_norm "none" for reward-free (sigreg/raw) arms
MODELS = [
    ("quad_RF_s0", "runs/R11_quad200k/rawcons_s0", "quadruped-walk", "none", 256),
    ("quad_RF_s1", "runs/R11_quad200k/rawcons_s1", "quadruped-walk", "none", 256),
    ("quad_RW_s0", "runs/R11_quad200k/reward_s0", "quadruped-walk", "simnorm", 256),
    ("quad_RW_s1", "runs/R11_quad200k/reward_s1", "quadruped-walk", "simnorm", 256),
    ("chee_RF_s0", "runs/R07_groundless_redteam/ctrl_consistency_only_s0", "cheetah-run", "none", 256),
    ("chee_RW_s0", "runs/R06_stability/cheetah-run_s0_r6", "cheetah-run", "simnorm", 256),
]

# MPPI configs for the H2 sweep (baseline eval -> wider -> more iters -> longer horizon)
PLANNERS = {
    "base":  MPPIConfig(horizon=3, iters=6, num_samples=512, num_elites=64),
    "wide":  MPPIConfig(horizon=3, iters=6, num_samples=2048, num_elites=128),
    "iters": MPPIConfig(horizon=3, iters=12, num_samples=1024, num_elites=128),
    "long":  MPPIConfig(horizon=5, iters=8, num_samples=1024, num_elites=128),
}


def load_model(d: str, task: str, latent_norm: str, latent_dim: int, device: str):
    env = DMCEnv(task, seed=0, action_repeat=2)
    mcfg = ModelConfig(obs_dim=env.obs_dim, act_dim=env.act_dim,
                       latent_dim=latent_dim, latent_norm=latent_norm)
    wm = WorldModel(mcfg)
    sd = torch.load(Path(d) / "model.pt", map_location=device)
    # strict=False: older checkpoints (pre inverse-dynamics head) lack a few keys we don't use
    wm.load_state_dict(sd, strict=False)
    wm.to(device).eval()
    return wm, env


def eval_return(wm, env, mppi_cfg, episodes, device, seed=0):
    ctrl = JepaController(wm, env.act_low, env.act_high, mppi_cfg, device)
    rets = []
    for _ in range(episodes):
        ctrl.reset()
        obs = env.reset()
        total, done, steps = 0.0, False, 0
        while not done and steps < 1000:
            obs, r, done = env.step(ctrl.act(obs))
            total += r
            steps += 1
        rets.append(total)
    return float(np.mean(rets)), [round(x, 1) for x in rets]


def collect_states(wm, env, device, n_steps=1500, seed=0):
    """Roll the trained controller on-distribution. Returns obs, reward, action, and the
    discounted Monte-Carlo return-to-go per state (within-episode, gamma=0.99)."""
    ctrl = JepaController(wm, env.act_low, env.act_high, PLANNERS["base"], device)
    ctrl.reset()
    obs = env.reset()
    O, R, A, ep_id = [], [], [], []
    cur_ep, done, steps = 0, False, 0
    while steps < n_steps:
        if done:
            ctrl.reset(); obs = env.reset(); cur_ep += 1
        a = ctrl.act(obs)
        O.append(np.asarray(obs, np.float32)); A.append(np.asarray(a, np.float32))
        obs, r, done = env.step(a)
        R.append(float(r)); ep_id.append(cur_ep)
        steps += 1
    O, R, A, ep_id = np.array(O), np.array(R), np.array(A), np.array(ep_id)
    # discounted return-to-go within each episode
    gamma, g = 0.99, np.zeros(len(R), np.float32)
    run = 0.0
    for i in range(len(R) - 1, -1, -1):
        if i < len(R) - 1 and ep_id[i] != ep_id[i + 1]:
            run = 0.0
        run = R[i] + gamma * run
        g[i] = run
    return O, R, A, g


def value_quality(wm, O, A, mc_return, device):
    """How well does the trained value head match the realized MC return-to-go?
    The planner's H=3 bootstrap leans entirely on this. Returns held-out R^2."""
    with torch.no_grad():
        z = wm.encode(torch.as_tensor(O, dtype=torch.float32, device=device))
        a = torch.as_tensor(A, dtype=torch.float32, device=device)
        v_logits = wm.value_head.logits(z, a)            # (num_q, N, bins)
        v = wm.value_head.to_scalar(v_logits).amin(0)    # pessimistic ensemble min, (N,)
        v = v.cpu().numpy()
    # value head is trained in symlog space; compare in raw return units via R^2 + correlation
    ss_res = ((mc_return - v) ** 2).sum()
    ss_tot = ((mc_return - mc_return.mean()) ** 2).sum() + 1e-9
    r2 = float(1 - ss_res / ss_tot)
    corr = float(np.corrcoef(mc_return, v)[0, 1])
    return r2, corr, float(v.mean()), float(mc_return.mean())


def ridge_r2(X, y, alpha=1.0):
    """Closed-form ridge with a train/test split; returns held-out R^2 (>=0 useful)."""
    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]
    Xtr, Xte = X[tr], X[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    ytr, yte = y[tr], y[te]
    ymu = ytr.mean(0)
    A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
    if y.ndim == 1:
        w = np.linalg.solve(A, Xtr.T @ (ytr - ymu))
        pred = Xte @ w + ymu
        ss_res = ((yte - pred) ** 2).sum()
        ss_tot = ((yte - yte.mean()) ** 2).sum() + 1e-9
        return float(1 - ss_res / ss_tot)
    W = np.linalg.solve(A, Xtr.T @ (ytr - ymu))
    pred = Xte @ W + ymu
    ss_res = ((yte - pred) ** 2).sum()
    ss_tot = ((yte - yte.mean(0)) ** 2).sum() + 1e-9
    return float(1 - ss_res / ss_tot)


def latent_of(wm, O, device):
    with torch.no_grad():
        z = wm.encode(torch.as_tensor(O, dtype=torch.float32, device=device)).cpu().numpy()
    return z


def controlled_sweep(episodes, device, out, only=None):
    """Red-team: vary ONE MPPI knob at a time (elites fixed at 64) to attribute the H2 effect.
    Confounds in the coarse sweep (samples/elites/iters co-varied) are removed here."""
    models = [
        ("quad_RF_s0", "runs/R11_quad200k/rawcons_s0", "quadruped-walk", "none", 256),
        ("quad_RW_s0", "runs/R11_quad200k/reward_s0", "quadruped-walk", "simnorm", 256),
    ]
    E = 64
    sweeps = {
        "samples": [MPPIConfig(horizon=3, iters=6, num_samples=n, num_elites=E)
                    for n in (256, 512, 1024, 2048, 4096)],
        "iters":   [MPPIConfig(horizon=3, iters=k, num_samples=1024, num_elites=E)
                    for k in (3, 6, 9, 12)],
        "horizon": [MPPIConfig(horizon=h, iters=6, num_samples=1024, num_elites=E)
                    for h in (2, 3, 4, 5)],
        "elites":  [MPPIConfig(horizon=3, iters=6, num_samples=2048, num_elites=e)
                    for e in (32, 64, 128, 256)],
    }
    res = {}
    for label, d, task, ln, ld in models:
        wm, env = load_model(d, task, ln, ld, device)
        res[label] = {}
        for sweep_name, cfgs in sweeps.items():
            if only and sweep_name not in only:
                continue
            res[label][sweep_name] = []
            for c in cfgs:
                m, raw = eval_return(wm, env, c, episodes, device)
                tag = {"samples": c.num_samples, "iters": c.iters,
                       "horizon": c.horizon, "elites": c.num_elites}[sweep_name]
                res[label][sweep_name].append({"knob": tag, "mean": round(m, 1), "eps": raw})
                print(f"[CTRL] {label} {sweep_name}={tag:<5} -> {m:6.1f}  {raw}")
        del wm, env
        torch.cuda.empty_cache()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", default="both", choices=["h2", "h3", "both", "ctrl"])
    p.add_argument("--sweep", default="", help="ctrl: comma list of sweeps to run (samples,iters,horizon,elites)")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="runs/R13_dof_diag/result.json")
    a = p.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"
    if a.part == "ctrl":
        controlled_sweep(a.episodes, dev, a.out, only=a.sweep.split(",") if a.sweep else None)
        return
    out = {"part": a.part, "h2": {}, "h3": {}}

    if a.part in ("h2", "both"):
        # H2: planner-scaling sweep on quad reward-free (does harder search recover return?)
        for label, d, task, ln, ld in MODELS:
            if "quad" not in label:
                continue
            wm, env = load_model(d, task, ln, ld, dev)
            row = {}
            for pname, pcfg in PLANNERS.items():
                m, raw = eval_return(wm, env, pcfg, a.episodes, dev)
                row[pname] = {"mean": round(m, 1), "eps": raw}
                print(f"[H2] {label} {pname:5s} -> {m:6.1f}  {raw}")
            out["h2"][label] = row
            del wm, env
            torch.cuda.empty_cache()

    if a.part in ("h3", "both"):
        # H3: latent task-decodability (ridge R^2 latent->reward, latent->obs)
        for label, d, task, ln, ld in MODELS:
            wm, env = load_model(d, task, ln, ld, dev)
            O, R, A, G = collect_states(wm, env, dev, n_steps=1500)
            Z = latent_of(wm, O, dev)
            r2_reward = ridge_r2(Z, R)
            r2_obs = ridge_r2(Z, O)
            # baseline: how well does raw OBS predict reward (task difficulty reference)
            r2_obs_reward = ridge_r2(O, R)
            v_r2, v_corr, v_mean, g_mean = value_quality(wm, O, A, G, dev)
            row = {"r2_latent_to_reward": round(r2_reward, 3),
                   "r2_latent_to_obs": round(r2_obs, 3),
                   "r2_obs_to_reward": round(r2_obs_reward, 3),
                   "value_vs_mc_r2": round(v_r2, 3),
                   "value_vs_mc_corr": round(v_corr, 3),
                   "value_pred_mean": round(v_mean, 1),
                   "mc_return_mean": round(g_mean, 1),
                   "obs_dim": int(O.shape[1]), "n": int(len(O)),
                   "reward_mean": round(float(R.mean()), 3)}
            out["h3"][label] = row
            print(f"[H3] {label:12s} z->r={r2_reward:+.3f} z->obs={r2_obs:+.3f} "
                  f"V~MC r2={v_r2:+.3f} corr={v_corr:+.3f} (Vpred={v_mean:.0f} MC={g_mean:.0f})")
            del wm, env
            torch.cuda.empty_cache()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
