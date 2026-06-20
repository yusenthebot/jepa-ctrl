"""General 3D collapse-rate eval for ANY trained checkpoint.

The 3D quadruped metric (R13 bimodal finding): an episode is "collapsed" if its return < THRESH
(=150, which sits in the clean bimodal valley). A single eval is bimodal NOISE — always use >=20
episodes and report collapse_rate AND good_basin_mean (a knob that lowers collapse but craters the
good-basin gait is not a win). Eval protocol = base MPPI (eval_mppi: h3 i6 s512 e64).

Usage: python scripts/collapse_eval.py --ckpt runs/.../model.pt --task quadruped-walk \
           --latent-norm none --episodes 20
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from jepa_ctrl.envs import DMCEnv
from jepa_ctrl.model import ModelConfig, WorldModel
from jepa_ctrl.model.jepa_controller import JepaController
from jepa_ctrl.model.mppi import eval_mppi


def collapse_eval(ckpt, task, latent_norm, latent_dim, episodes, thresh, device="cuda") -> dict:
    e = DMCEnv(task, seed=0, action_repeat=2)
    cfg = ModelConfig(obs_dim=e.obs_dim, act_dim=e.act_dim, latent_dim=latent_dim,
                      latent_norm=latent_norm)
    wm = WorldModel(cfg)
    wm.load_state_dict(torch.load(ckpt))
    wm.eval().to(device)
    e.close()
    rets = []
    for sd in range(episodes):
        env = DMCEnv(task, seed=1000 + sd, action_repeat=2)
        ctrl = JepaController(wm, env.act_low, env.act_high, eval_mppi(), device)
        ctrl.reset()
        obs = env.reset()
        total, done, steps = 0.0, False, 0
        while not done and steps < 2000:
            obs, r, done = env.step(ctrl.act(obs))
            total += r
            steps += 1
        rets.append(round(total, 1))
        env.close()
    arr = np.array(rets)
    good = arr[arr >= thresh]
    return {
        "collapse_rate": float((arr < thresh).mean()),
        "n_collapse": int((arr < thresh).sum()),
        "n_good": int((arr >= thresh).sum()),
        "good_basin_mean": float(good.mean()) if len(good) else 0.0,
        "mean": float(arr.mean()),
        "eps": arr.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task", default="quadruped-walk")
    p.add_argument("--latent-norm", default="none")
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--thresh", type=float, default=150.0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    r = collapse_eval(a.ckpt, a.task, a.latent_norm, a.latent_dim, a.episodes, a.thresh, a.device)
    print(f"{a.ckpt}: collapse_rate={r['collapse_rate']:.2f} ({r['n_collapse']}/{a.episodes}) "
          f"good_basin={r['good_basin_mean']:.0f} mean={r['mean']:.0f}")
    print("eps:", r["eps"])


if __name__ == "__main__":
    main()
