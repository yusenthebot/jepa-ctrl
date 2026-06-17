"""Round-2 training driver for jepa-ctrl. Trains the JEPA world-model with latent MPPI on a
dm_control task, evaluates the JepaController periodically (real sim, deterministic), records a
sample-efficiency curve, and renders the final eval episode for REAL-VERIFY.

Includes wall-clock instrumentation: prints the post-seed steps/sec and the extrapolated ETA for
100k env-steps, so a full run can be checked against the 2h budget before launching it.

Usage (probe):  python scripts/train.py --task cheetah-run --steps 1200 --seed-steps 200 \
                    --eval-episodes 1 --outdir runs/probe --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from jepa_ctrl.envs import DMCEnv
from jepa_ctrl.metrics import collapse_diagnostics, is_collapsed, plot_sample_efficiency
from jepa_ctrl.model import ModelConfig, WorldModel
from jepa_ctrl.model.jepa_controller import JepaController
from jepa_ctrl.model.mppi import eval_mppi
from jepa_ctrl.model.trainer import TrainConfig, Trainer
from jepa_ctrl.render import contact_sheet, save_mp4
from jepa_ctrl.seeding import set_seed


def run_eval(model, task, n_episodes, seed, device, action_repeat, render=False, out=None, step=0):
    env = DMCEnv(task, seed=seed, action_repeat=action_repeat)
    ctrl = JepaController(model, env.act_low, env.act_high, eval_mppi(), device)
    returns, frames, cum = [], [], []
    try:
        for ep in range(n_episodes):
            ctrl.reset()
            obs = env.reset()
            total, steps, done = 0.0, 0, False
            do_r = render and ep == 0
            if do_r:
                frames.append(env.render())
                cum.append(0.0)
            while not done and steps < 2000:
                obs, r, done = env.step(ctrl.act(obs))
                total += r
                steps += 1
                if do_r:
                    frames.append(env.render())
                    cum.append(total)
            returns.append(total)
        if render and frames and out is not None:
            out = Path(out)
            fps = max(10, round(1.0 / (env.control_timestep() * action_repeat)))
            save_mp4(frames, out / f"eval_step{step}.mp4", fps=fps)
            contact_sheet(
                frames, out / f"eval_step{step}.png",
                title=f"{task} | JEPA-MPPI | step {step} | return {returns[0]:.1f}",
                step_returns=cum,
            )
    finally:
        env.close()
    return float(np.mean(returns)), returns


def collapse_probe(model, task, seed, device, action_repeat, n=256) -> dict:
    """Encode a batch of real obs and report collapse health + whether the latent actually
    tracks the observation (obs_latent_corr ~0 means the encoder ignores its input)."""
    import numpy.linalg as la

    env = DMCEnv(task, seed=seed, action_repeat=action_repeat)
    rng = np.random.default_rng(seed)
    try:
        obs, o = [], env.reset()
        for _ in range(n):
            obs.append(o.copy())
            o, _, d = env.step(rng.uniform(env.act_low, env.act_high).astype(np.float32))
            if d:
                o = env.reset()
        ob = np.asarray(obs, np.float32)
        with torch.no_grad():
            Z = model.to(device).eval().encode(torch.tensor(ob, device=device)).cpu().numpy()
        diag = collapse_diagnostics(Z)
        i, j = rng.integers(0, len(ob), (2, 200))
        od = la.norm(ob[i] - ob[j], axis=1)
        zd = la.norm(Z[i] - Z[j], axis=1)
        corr = float(np.corrcoef(od, zd)[0, 1])
    finally:
        env.close()
    return {
        "rank_fraction": diag["rank_fraction"],
        "participation_ratio": diag["participation_ratio"],
        "mean_pairwise": diag["mean_pairwise_dist"],
        "obs_latent_corr": corr,
        "collapsed": is_collapsed(diag),
    }


def main() -> None:
    p = argparse.ArgumentParser("jepa-ctrl train")
    p.add_argument("--task", default="cheetah-run")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed-steps", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=2)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--action-repeat", type=int, default=2)
    p.add_argument("--latent-dim", type=int, default=0, help="0 = auto (256 if obs>8 else 128)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", default="runs/train")
    a = p.parse_args()

    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    set_seed(a.seed)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    env = DMCEnv(a.task, seed=a.seed, action_repeat=a.action_repeat)
    latent_dim = a.latent_dim or (256 if env.obs_dim > 8 else 128)
    mcfg = ModelConfig(obs_dim=env.obs_dim, act_dim=env.act_dim, latent_dim=latent_dim)
    wm = WorldModel(mcfg)
    tcfg = TrainConfig(seed_steps=a.seed_steps, eval_every=a.eval_every)
    trainer = Trainer(wm, tcfg, env.act_low, env.act_high, device=device)
    print(f"device={device} task={a.task} obs={env.obs_dim} act={env.act_dim} latent={latent_dim} "
          f"params={sum(q.numel() for q in wm.parameters() if q.requires_grad)/1e6:.2f}M")

    curve: list[tuple[int, float]] = []
    collapse_log: list[dict] = []

    def eval_hook(step: int) -> None:
        if step == 0:
            return
        ret, rs = run_eval(wm, a.task, a.eval_episodes, a.seed + 1000, device, a.action_repeat)
        cp = collapse_probe(wm, a.task, a.seed + 2000, device, a.action_repeat)
        curve.append((step, ret))
        collapse_log.append({"step": step, "return": ret, **cp})
        el = time.time() - t_start
        print(f"[eval] step {step:6d}  return {ret:7.1f}  rank_frac {cp['rank_fraction']:.2f} "
              f"PR {cp['participation_ratio']:.1f} obs_corr {cp['obs_latent_corr']:.3f} "
              f"collapsed {cp['collapsed']}  elapsed {el/60:.1f}min", flush=True)

    # explicit loop = precise wall-clock instrumentation
    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=device)
    trainer.planner.reset()
    t_start = time.time()
    t_postseed, step_postseed = None, a.seed_steps
    losses: list[float] = []
    while trainer.step < a.steps:
        if t_postseed is None and trainer.step >= a.seed_steps:
            t_postseed, step_postseed = time.time(), trainer.step
        next_obs, _, done = trainer.collect_step(env, obs)
        obs = next_obs
        if done:
            obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=device)
            trainer.planner.reset()
        if len(trainer.buffer) > mcfg.horizon + 1 and trainer.step >= a.seed_steps:
            for _ in range(a.updates_per_step):
                losses.append(trainer.update()["consistency"])
        if trainer.step % a.eval_every == 0:
            eval_hook(trainer.step)
        if trainer.step % 200 == 0 and losses:
            print(f"  step {trainer.step:6d}  consistency(mean200) {np.mean(losses[-200:]):.4f}",
                  flush=True)

    # post-seed wall-clock + 100k ETA
    if t_postseed is not None:
        dt = time.time() - t_postseed
        rate = (trainer.step - step_postseed) / max(dt, 1e-9)
        eta100k = 100_000 / max(rate, 1e-9) / 60
        print(f"[PROBE] post-seed rate {rate:.1f} env-steps/s -> 100k ETA {eta100k:.1f} min "
              f"({'OK <2h' if eta100k < 120 else 'OVER 2h — reduce scope'})", flush=True)

    # final eval + REAL-VERIFY render
    ret, rs = run_eval(wm, a.task, max(a.eval_episodes, 1), a.seed + 1000, device,
                       a.action_repeat, render=True, out=out, step=trainer.step)
    cp_final = collapse_probe(wm, a.task, a.seed + 2000, device, a.action_repeat)
    curve.append((trainer.step, ret))
    collapse_log.append({"step": trainer.step, "return": ret, **cp_final})
    print(f"[final] step {trainer.step} return {ret:.1f} collapsed {cp_final['collapsed']} "
          f"obs_corr {cp_final['obs_latent_corr']:.3f} eps {rs}", flush=True)

    if len(curve) >= 2:
        steps_arr = [c[0] for c in curve]
        means = [c[1] for c in curve]
        plot_sample_efficiency(
            {a.task: (steps_arr, means, [0.0] * len(means))},
            out / "sample_efficiency.png", title=f"{a.task} JEPA-MPPI (seed {a.seed})",
        )
    (out / "result.json").write_text(json.dumps(
        {"task": a.task, "seed": a.seed, "steps": trainer.step, "curve": curve,
         "collapse_log": collapse_log, "final_return": ret, "latent_dim": latent_dim,
         "wall_clock_min": (time.time() - t_start) / 60}, indent=2))
    env.close()
    torch.save(wm.state_dict(), out / "model.pt")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
