"""R18 — JEPA ensemble disagreement-CALIBRATION diagnostic (gates the sparse-exploration campaign).

Question: on a decoder-free raw-latent JEPA with a SINGLE shared EMA target encoder, does an
ensemble of N independent latent predictors produce CALIBRATED epistemic uncertainty — i.e. is
cross-head disagreement Var(z_hat_{t+1}) positively rank-correlated with the true one-step
prediction error, and higher on NOVEL states than VISITED ones? Or does the smooth EMA target
collapse the disagreement signal (the skeptic's Prong-2 / noisy-TV-adjacent failure)?

PASS (greenlight the campaign): Spearman rho(D,E) >= 0.4 (p<0.01) AND median D_novel/D_visited >= 1.5.
KILL: rho < 0.2 OR ratio < 1.2 -> EMA mode-averaging confirmed; pivot (bootstrapped targets / climb ladder).

Protocol: load a frozen JEPA encoder (reward-free RAW cartpole ckpt from train.py), collect VISITED
transitions via the trained controller (on-policy, narrow) and NOVEL transitions via random actions
from perturbed initial states (env.set_state — the R17 moat), train N independent predictor heads on
the frozen encoder's latents, then on held-out visited+novel compute D=disagreement and
E=||mean_pred - f_xi(o_{t+1})||. Red-team baked in: random-pairing null, partial-out ||z_t||,
pairwise head divergence, novel/visited + magnitude bins.

SIM-TOUCHING: run serialized (one sim at a time); the parent driver nukes + checks free -g.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from jepa_ctrl.envs import DMCEnv  # noqa: E402
from jepa_ctrl.model import ModelConfig, WorldModel  # noqa: E402
from jepa_ctrl.model.jepa_controller import JepaController  # noqa: E402
from jepa_ctrl.model.mppi import eval_mppi  # noqa: E402
from jepa_ctrl.model.nets import PredictorEnsemble  # noqa: E402


def _load_wm(ckpt, task, latent_norm, latent_dim, device):
    e = DMCEnv(task, seed=0, action_repeat=2)
    sd = torch.load(ckpt, map_location="cpu")
    # infer latent_dim from the checkpoint (train.py auto-sizes it per task: 128 small, 256 big)
    inferred = int(sd["encoder.proj.weight"].shape[0]) if "encoder.proj.weight" in sd else latent_dim
    if inferred != latent_dim:
        print(f"[r18_calib] latent_dim from ckpt = {inferred} (overriding --latent-dim {latent_dim})")
    cfg = ModelConfig(obs_dim=e.obs_dim, act_dim=e.act_dim, latent_dim=inferred,
                      latent_norm=latent_norm)
    wm = WorldModel(cfg)
    wm.load_state_dict(sd)
    wm.eval().to(device)
    for p in wm.parameters():
        p.requires_grad_(False)
    e.close()
    return wm, cfg


@torch.no_grad()
def _collect(task, wm, device, mode, n_eps, seed0, rng):
    """Return (o_t, a_t, o_next) float32 arrays. mode='visited' -> controller (on-policy);
    'novel' -> random actions from a state perturbed via env.set_state (OOD coverage)."""
    obs_t, act, obs_n = [], [], []
    for ep in range(n_eps):
        env = DMCEnv(task, seed=seed0 + ep, action_repeat=2)
        ctrl = JepaController(wm, env.act_low, env.act_high, eval_mppi(), device) \
            if mode == "visited" else None
        if ctrl is not None:
            ctrl.reset()
        obs = env.reset()
        if mode == "novel":
            # perturb the start state (R17 moat) so random rollouts cover OOD regions
            st = env.get_state()
            obs = env.reset(from_state=st + rng.normal(0, 0.3, size=st.shape).astype(st.dtype))
        steps, done = 0, False
        while not done and steps < 200:
            if mode == "visited":
                a = ctrl.act(obs)
            else:
                a = rng.uniform(env.act_low, env.act_high).astype(np.float32)
            nxt, _, done = env.step(a)
            obs_t.append(np.asarray(obs, np.float32))
            act.append(np.asarray(a, np.float32))
            obs_n.append(np.asarray(nxt, np.float32))
            obs = nxt
            steps += 1
        env.close()
    return np.array(obs_t), np.array(act), np.array(obs_n)


def _train_ensemble(ens, wm, ot, at, on, device, steps, bs, lr, rng):
    """Train N independent heads on the FROZEN encoder's latents: predict z_hat_{t+1} vs the
    stop-grad EMA target f_xi(o_{t+1}). Vanilla deep ensemble (shared data, independent init)."""
    ot_t = torch.as_tensor(ot, device=device)
    at_t = torch.as_tensor(at, device=device)
    on_t = torch.as_tensor(on, device=device)
    with torch.no_grad():
        z_pre = wm.encode_pre(ot_t)             # (M, LD) frozen
        z_tgt = wm.encode_target(on_t)          # (M, LD) stop-grad target
    opt = torch.optim.Adam(ens.parameters(), lr=lr)
    m = z_pre.shape[0]
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(0, m, size=bs), device=device)
        preds = ens(z_pre[idx], at_t[idx])      # (N, bs, LD)
        tgt = z_tgt[idx].unsqueeze(0).expand_as(preds)
        loss = torch.nn.functional.smooth_l1_loss(preds, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.item())


@torch.no_grad()
def _eval_DE(ens, wm, ot, at, on, device):
    ot_t = torch.as_tensor(ot, device=device)
    at_t = torch.as_tensor(at, device=device)
    on_t = torch.as_tensor(on, device=device)
    z_pre = wm.encode_pre(ot_t)
    z_tgt = wm.encode_target(on_t)
    D = ens.disagreement(z_pre, at_t).cpu().numpy()
    err = (ens.mean_prediction(z_pre, at_t) - z_tgt).pow(2).mean(dim=-1).sqrt().cpu().numpy()
    z_mag = z_pre.pow(2).mean(dim=-1).sqrt().cpu().numpy()
    return D, err, z_mag


def _pairwise_head_cosine(ens, wm, ot, at, device):
    """Red-team: heads must be distinct functions (mean pairwise cosine < 0.99) or Var is fake."""
    with torch.no_grad():
        z_pre = wm.encode_pre(torch.as_tensor(ot[:256], device=device))
        a = torch.as_tensor(at[:256], device=device)
        preds = ens(z_pre, a)  # (N, b, LD)
    n = preds.shape[0]
    cos = torch.nn.functional.cosine_similarity
    vals = [cos(preds[i].flatten(), preds[j].flatten(), dim=0).item()
            for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals))


def _partial_spearman(D, E, Z):
    """Spearman(D,E) controlling for ||z_t|| via rank-residuals (does the corr survive?)."""
    from scipy.stats import rankdata
    rd, re, rz = rankdata(D), rankdata(E), rankdata(Z)
    def resid(y, x):
        x1 = np.c_[np.ones_like(x), x]
        beta, *_ = np.linalg.lstsq(x1, y, rcond=None)
        return y - x1 @ beta
    return float(spearmanr(resid(rd, rz), resid(re, rz)).statistic)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-ckpt", required=True)
    p.add_argument("--task", default="cartpole-balance")
    p.add_argument("--latent-norm", default="none")
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=5)
    p.add_argument("--train-eps", type=int, default=40)
    p.add_argument("--test-eps", type=int, default=15)
    p.add_argument("--ens-steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", default="runs/R18_calib")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    dev = a.device
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    wm, cfg = _load_wm(a.encoder_ckpt, a.task, a.latent_norm, a.latent_dim, dev)
    ens = PredictorEnsemble(a.n_heads, cfg.latent_dim, cfg.act_dim, cfg.pred_hidden,
                            cfg.action_head_dim, cfg.simnorm_groups, a.latent_norm).to(dev)

    # collect: train = visited+novel mixed; test held out separately for visited & novel
    vtr = _collect(a.task, wm, dev, "visited", a.train_eps, 1000, rng)
    ntr = _collect(a.task, wm, dev, "novel", a.train_eps, 2000, rng)
    ot = np.concatenate([vtr[0], ntr[0]]); at = np.concatenate([vtr[1], ntr[1]])
    on = np.concatenate([vtr[2], ntr[2]])
    final_loss = _train_ensemble(ens, wm, ot, at, on, dev, a.ens_steps, 256, 1e-3, rng)

    vte = _collect(a.task, wm, dev, "visited", a.test_eps, 5000, rng)
    nte = _collect(a.task, wm, dev, "novel", a.test_eps, 6000, rng)
    Dv, Ev, Zv = _eval_DE(ens, wm, *vte, dev)
    Dn, En, Zn = _eval_DE(ens, wm, *nte, dev)
    D = np.concatenate([Dv, Dn]); E = np.concatenate([Ev, En]); Z = np.concatenate([Zv, Zn])

    rho = spearmanr(D, E)
    # red-team: random-pairing null (shuffle E vs D), partial-out ||z||, head divergence
    perm = rng.permutation(len(E))
    rho_null = spearmanr(D, E[perm]).statistic
    rho_partial = _partial_spearman(D, E, Z)
    head_cos = _pairwise_head_cosine(ens, wm, ot, at, dev)
    ratio = float(np.median(Dn) / max(np.median(Dv), 1e-12))

    PASS = bool(rho.statistic >= 0.4 and rho.pvalue < 0.01 and ratio >= 1.5)
    KILL = bool(rho.statistic < 0.2 or ratio < 1.2)
    res = {
        "task": a.task, "n_heads": a.n_heads, "latent_norm": a.latent_norm, "seed": a.seed,
        "n_test": int(len(D)), "ens_final_loss": round(final_loss, 6),
        "spearman_rho": round(float(rho.statistic), 4), "spearman_p": float(rho.pvalue),
        "rho_null_shuffled": round(float(rho_null), 4),
        "rho_partial_ctrl_zmag": round(rho_partial, 4),
        "D_visited_median": round(float(np.median(Dv)), 6),
        "D_novel_median": round(float(np.median(Dn)), 6),
        "novel_visited_ratio": round(ratio, 3),
        "E_visited_median": round(float(np.median(Ev)), 4),
        "E_novel_median": round(float(np.median(En)), 4),
        "pairwise_head_cosine": round(head_cos, 4),
        "VERDICT": "PASS" if PASS else ("KILL" if KILL else "INCONCLUSIVE"),
    }
    (out / f"calib_seed{a.seed}.json").write_text(json.dumps(res, indent=2))

    # eyes-on plot: D vs E scatter (visited vs novel) + medians
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter(Dv, Ev, s=8, alpha=0.4, label="visited", c="tab:blue")
    ax[0].scatter(Dn, En, s=8, alpha=0.4, label="novel", c="tab:red")
    ax[0].set_xlabel("ensemble disagreement D"); ax[0].set_ylabel("true 1-step error E")
    ax[0].set_title(f"{a.task}  rho={rho.statistic:.2f} (p={rho.pvalue:.1e})  "
                    f"null={rho_null:.2f} partial={rho_partial:.2f}")
    ax[0].legend()
    ax[1].boxplot([Dv, Dn], tick_labels=["visited", "novel"])
    ax[1].set_ylabel("disagreement D"); ax[1].set_title(
        f"novel/visited={ratio:.2f}x  head_cos={head_cos:.3f}  VERDICT={res['VERDICT']}")
    fig.tight_layout(); fig.savefig(out / f"calib_seed{a.seed}.png", dpi=110); plt.close(fig)

    print(json.dumps(res, indent=2))
    print(f"\n>>> R18 CALIBRATION VERDICT: {res['VERDICT']} <<<  -> {out}/calib_seed{a.seed}.png")


if __name__ == "__main__":
    main()
