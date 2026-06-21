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
import fcntl
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from jepa_ctrl.envs import DMCEnv
from jepa_ctrl.metrics import collapse_diagnostics, is_collapsed, plot_sample_efficiency
from jepa_ctrl.model import ModelConfig, WorldModel
from jepa_ctrl.model.jepa_controller import JepaController
from jepa_ctrl.model.mppi import eval_mppi, train_mppi
from jepa_ctrl.model.trainer import TrainConfig, Trainer
from jepa_ctrl.render import contact_sheet, save_mp4
from jepa_ctrl.seeding import set_seed

_PIXELS = {"on": False, "distractor": False, "size": 84, "frame_stack": 3, "masked_target": False}


def make_env(task: str, seed: int, action_repeat: int):
    """DMCEnv (state) or PixelDMCEnv (pixels + optional background distractor) per module config."""
    if _PIXELS["on"]:
        from jepa_ctrl.pixel_env import PixelDMCEnv
        return PixelDMCEnv(task, seed=seed, action_repeat=action_repeat, size=_PIXELS["size"],
                           frame_stack=_PIXELS["frame_stack"], distractor=_PIXELS["distractor"],
                           distractor_seed=seed + 7, masked_target=_PIXELS["masked_target"])
    return DMCEnv(task, seed=seed, action_repeat=action_repeat)


def run_eval(model, task, n_episodes, seed, device, action_repeat, render=False, out=None, step=0):
    env = make_env(task, seed, action_repeat)
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

    env = make_env(task, seed, action_repeat)
    rng = np.random.default_rng(seed)
    try:
        obs, o = [], env.reset()
        for _ in range(n):
            obs.append(o.copy())
            o, _, d = env.step(rng.uniform(env.act_low, env.act_high).astype(np.float32))
            if d:
                o = env.reset()
        ob = np.asarray(obs, np.float32)
        model = model.to(device).eval()
        with torch.no_grad():
            z = model.encode(torch.tensor(ob, device=device))
            Z = z.cpu().numpy()
            a0 = torch.zeros(len(ob), model.cfg.act_dim, device=device)
            r_mag = float(model.reward_head.to_scalar(model.reward_head.logits(z, a0)).abs().mean())
            v_mag = float(model.value_head.to_scalar(model.value_head.logits(z, a0)).abs().mean())
        diag = collapse_diagnostics(Z)
        obf = ob.reshape(len(ob), -1)  # flatten (pixel obs is 4D); NB pixel-dist is distractor-
        i, j = rng.integers(0, len(ob), (2, 200))  # dominated, so obs_corr is unreliable here
        od = la.norm(obf[i] - obf[j], axis=1)
        zd = la.norm(Z[i] - Z[j], axis=1)
        corr = float(np.corrcoef(od, zd)[0, 1])
    finally:
        env.close()
    return {
        "rank_fraction": diag["rank_fraction"],
        "participation_ratio": diag["participation_ratio"],
        "mean_pairwise": diag["mean_pairwise_dist"],
        "obs_latent_corr": corr,
        "pred_reward_mag": r_mag,
        "pred_value_mag": v_mag,
        "collapsed": is_collapsed(diag),
    }


def build_parser() -> argparse.ArgumentParser:
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
    p.add_argument("--grounding", default="reward",
                   choices=["reward", "inverse_dynamics", "sigreg", "reconstruction"],
                   help="reward/recon = grounded; inverse_dynamics/sigreg are reward-free")
    p.add_argument("--latent-norm", default="auto", choices=["auto", "simnorm", "none"],
                   help="auto = none for sigreg arm else simnorm; override for matched ablations")
    p.add_argument("--sigreg-coef", type=float, default=1.0)
    p.add_argument("--id-coef", type=float, default=1.0)
    # R15: exploration-noise schedule on the behaviour policy during data collection. The
    # late-training floor (explore-std-end) is the one-knob lever for the 3D bimodal-collapse
    # round (initial-condition robustness via wider late-training state coverage).
    p.add_argument("--explore-std", type=float, default=0.3,
                   help="initial gaussian action-noise std on the MPPI behaviour action")
    p.add_argument("--explore-std-end", type=float, default=0.05,
                   help="annealed exploration floor (R15 lever: raise to widen state coverage)")
    p.add_argument("--explore-anneal-steps", type=int, default=100_000,
                   help="linear anneal explore-std -> explore-std-end over this many steps")
    p.add_argument("--freeze-repr", action="store_true",
                   help="red-team control: freeze random repr, train only reward/value heads")
    # R17 reset-curriculum (cover-to-recover): with prob --reset-p, start an episode from a
    # banked near-fall state (harvested from low-return episodes) to teach recovery. Default OFF
    # -> the non-curriculum collect path stays byte-identical to the pre-R17 driver.
    p.add_argument("--reset-curriculum", action="store_true",
                   help="R17: start a fraction of episodes from banked near-fall states")
    p.add_argument("--reset-p", type=float, default=0.3,
                   help="probability a reset draws a banked near-fall state (curriculum on)")
    # R18+ Plan2Explore disagreement exploration. --n-pred-heads>=2 builds the ensemble;
    # --explore-objective=disagreement collects data by maximising ensemble disagreement (task
    # reward IGNORED at collection — pure intrinsic). Eval always plans on reward (zero-shot task).
    p.add_argument("--n-pred-heads", type=int, default=1,
                   help="independent predictor heads for epistemic disagreement (1 = no ensemble)")
    p.add_argument("--explore-objective", default="reward",
                   choices=["reward", "disagreement", "hybrid"],
                   help="data-collection planning objective. disagreement => Plan2Explore; "
                        "hybrid => extrinsic + annealed intrinsic (explore-early, exploit-late)")
    p.add_argument("--intrinsic-value", action="store_true",
                   help="Plan2Explore proper: train an intrinsic value head on disagreement-reward "
                        "and bootstrap it at the plan horizon (long-horizon exploration). Needs "
                        "--n-pred-heads>=2 and --explore-objective disagreement.")
    p.add_argument("--pixels", action="store_true", help="pixel obs + CNN encoder instead of state")
    p.add_argument("--distractor", action="store_true",
                   help="composite a time-varying background distractor (pixels only)")
    p.add_argument("--masked-target", action="store_true",
                   help="R20: JEPA masked-target stream — EMA consistency target sees robot-only "
                        "(bg-zeroed) frames while online sees the full distractor obs (pixels only)")
    p.add_argument("--size", type=int, default=84)
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", default="runs/train")
    return p


def train_config_from_args(a: argparse.Namespace, pixels: bool) -> TrainConfig:
    """Build the (immutable) TrainConfig from parsed CLI args. Kept separate from main() so the
    CLI->config wiring (esp. the R15 exploration-floor knob) is unit-testable without a sim."""
    return TrainConfig(
        seed_steps=a.seed_steps, eval_every=a.eval_every, grounding=a.grounding,
        sigreg_coef=a.sigreg_coef, id_coef=a.id_coef, freeze_repr=a.freeze_repr,
        explore_std=a.explore_std, explore_std_end=a.explore_std_end,
        explore_anneal_steps=a.explore_anneal_steps,
        reset_curriculum=a.reset_curriculum, reset_p=a.reset_p,
        masked_target=a.masked_target,
        capacity=50_000 if pixels else 1_000_000)  # stacked uint8 buffer ~6.3GB


_LOCK_FILE = os.path.expanduser("~/.cache/jepa-ctrl/train.lock")


def acquire_singleton_lock():
    """OOM GUARD: serialize ALL jepa-ctrl trainings to one at a time. A second concurrent training
    (e.g. an interactive launch racing a supervisor-loop round) REFUSES and exits, instead of piling
    onto RAM. Root cause of the 2026-06 OOM: ~20 overlapping ~2.3GB dm_control trainings -> 51GB,
    OOM-killed. Each training holds an exclusive flock for its whole lifetime. Returns the file
    handle — the caller MUST keep it alive (kept as a local in main())."""
    os.makedirs(os.path.dirname(_LOCK_FILE), exist_ok=True)
    fh = open(_LOCK_FILE, "w")  # noqa: SIM115 — handle intentionally kept open to hold the lock
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("REFUSING to start: another jepa-ctrl training holds the lock (trainings are "
              "serialized — OOM guard). Exiting cleanly. Check: pgrep -af scripts/train.py")
        raise SystemExit(0) from None
    fh.write(str(os.getpid()))
    fh.flush()
    return fh


def main() -> None:
    a = build_parser().parse_args()
    _lock = acquire_singleton_lock()  # OOM guard — only one training at a time (held until exit)
    assert _lock is not None  # keep the handle alive for the process lifetime

    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    set_seed(a.seed)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    _PIXELS.update(on=a.pixels, distractor=a.distractor, size=a.size, frame_stack=a.frame_stack,
                   masked_target=a.masked_target)
    env = make_env(a.task, a.seed, a.action_repeat)
    latent_norm = a.latent_norm if a.latent_norm != "auto" else (
        "none" if a.grounding == "sigreg" else "simnorm")  # SIGReg needs RAW; else override-able
    if a.pixels:
        latent_dim = a.latent_dim or 256
        mcfg = ModelConfig(obs_dim=int(np.prod(env.obs_shape)), act_dim=env.act_dim,
                           latent_dim=latent_dim, latent_norm=latent_norm, n_pred_heads=a.n_pred_heads,
                           explore_value=a.intrinsic_value,
                           encoder_type="cnn", obs_shape=tuple(int(x) for x in env.obs_shape))
    else:
        latent_dim = a.latent_dim or (256 if env.obs_dim > 8 else 128)
        mcfg = ModelConfig(obs_dim=env.obs_dim, act_dim=env.act_dim, latent_dim=latent_dim,
                           latent_norm=latent_norm, n_pred_heads=a.n_pred_heads,
                           explore_value=a.intrinsic_value)
    wm = WorldModel(mcfg)
    tcfg = train_config_from_args(a, pixels=a.pixels)
    # data-collection planner objective: reward (default) or disagreement (Plan2Explore). Eval is
    # always reward-MPC (run_eval/JepaController use eval_mppi()), so the disagreement arm is a
    # genuine zero-shot task evaluation of a reward-free-explored world model.
    train_cfg_mppi = replace(train_mppi(), objective=a.explore_objective)
    trainer = Trainer(wm, tcfg, env.act_low, env.act_high, mppi_cfg=train_cfg_mppi, device=device)
    nparam = sum(q.numel() for q in wm.parameters() if q.requires_grad) / 1e6
    print(f"device={device} task={a.task} grounding={a.grounding} "
          f"latent={latent_dim}/{latent_norm} params={nparam:.2f}M "
          f"explore_std={tcfg.explore_std}->{tcfg.explore_std_end}@{tcfg.explore_anneal_steps}")

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
        print(f"[eval] step {step:6d}  return {ret:7.1f}  PR {cp['participation_ratio']:.1f} "
              f"obs_corr {cp['obs_latent_corr']:.3f} v_mag {cp['pred_value_mag']:.1f} "
              f"r_mag {cp['pred_reward_mag']:.2f}  elapsed {el/60:.1f}min", flush=True)

    # explicit loop = precise wall-clock instrumentation. reset_env() routes through the R17
    # reset-curriculum when enabled and is a plain env.reset() otherwise (byte-identical path).
    obs = trainer.reset_env(env)
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
            obs = trainer.reset_env(env)
            trainer.planner.reset()
        if len(trainer.buffer) > mcfg.horizon + 1 and trainer.step >= a.seed_steps:
            for _ in range(a.updates_per_step):
                mtr = trainer.update()
                # representation loss key varies by arm (consistency | reconstruction)
                losses.append(mtr.get("consistency", mtr.get("reconstruction", mtr["loss"])))
        if trainer.step % a.eval_every == 0:
            eval_hook(trainer.step)
        if trainer.step % 200 == 0 and losses:
            print(f"  step {trainer.step:6d}  repr_loss(mean200) {np.mean(losses[-200:]):.4f} "
                  f"reward_hits {trainer.reward_hits}", flush=True)

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
         "reward_hits": trainer.reward_hits, "explore_objective": a.explore_objective,
         "intrinsic_value": bool(a.intrinsic_value),
         "wall_clock_min": (time.time() - t_start) / 60}, indent=2))
    env.close()
    torch.save(wm.state_dict(), out / "model.pt")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
