# jepa-ctrl

**Can a small "JEPA" world model, running on a laptop, learn to control a robot by *imagining* the future in its own latent space — with no video, no images, no pixel prediction?**

This repo answers **yes**. A ~1.3M-parameter model, trained for ~20 minutes on a single laptop
GPU, learns to run a simulated cheetah by planning inside a learned latent space.

<p align="center">
  <img src="docs/cheetah_run.gif" width="320"><br/>
  <em>Cheetah running. No animation, no scripting — every action is chosen at runtime by
  planning ~3 steps ahead inside the model's learned latent world. Trained from scratch in ~20 min
  on an RTX 5080 laptop.</em>
</p>

---

## What is this, in plain language?

Most "world models" for robots learn by **predicting the next image** (pixels). That's expensive
and wasteful — you don't need to repaint every blade of grass to decide how to move a leg.

A **JEPA** (Joint-Embedding Predictive Architecture, the idea behind Meta's V-JEPA 2) instead
predicts the *next compressed idea* of the world — an abstract latent — and throws the pixels
away. Meta showed a giant version of this (**V-JEPA 2-AC**) can control real robot arms.

**This project asks: does the same idea work at *laptop scale*?** Build the smallest faithful
version — a tiny latent world model + planning in latent space — and see if it can actually
control a simulated robot. It can.

## Does it work?

Yes — and the project became a small research study of *what makes latent-only control work*.
All results are cross-seed and verified by watching the real sim rollout, not by "the loss went down".

| Task (dm_control) | Random | reward-grounded | **reward-FREE** (our key result) | "solved" |
|---|---|---|---|---|
| **cheetah-run** (2D, 6-DoF) | ~7 | 522 ± 139 | **496 ± 31** ✅ | ~850 |
| reacher-easy (2D) | ~40 | ~752 cross-seed | — | ~960 |
| **quadruped-walk** (3D, 12-DoF) | low | 450 ± 234 | 184 ± 141 (degrades) | — |
| humanoid-stand (3D, 21-DoF) | — | ✗ failed @80k | — | — |

<p align="center"><img src="docs/cheetah_curve.png" width="460"><br/>
<em>cheetah-run: episode return vs. env steps. 7 → ~550 in ~20 min on the laptop.</em></p>

## How it works — system block diagram

Three pieces: an **encoder** that turns observations into a latent, an **action-conditioned
predictor** that rolls that latent forward under hypothetical actions (the "world model"), and a
**planner (MPPI)** that imagines many action sequences in latent space and picks the best one.
Crucially there is **no decoder** — the model never reconstructs the observation.

```mermaid
flowchart TB
    ENV(["dm_control / MuJoCo<br/>robot simulator"])

    subgraph WM["Learned latent world model — ~1.3M params, NO pixel/state decoder"]
        direction TB
        ENC["Encoder f_θ<br/>MLP → SimNorm latent"]
        PRED["Action-conditioned predictor g_φ<br/>residual MLP: z, a → next latent"]
        REW["Reward head<br/>(distributional)"]
        VAL["Value head + EMA target<br/>(5-net ensemble)"]
        TENC["EMA target encoder f_ξ<br/>(stop-grad)"]
    end

    subgraph PLAN["Latent MPPI planner — this is the 'controller'"]
        MPPI["① sample action sequences<br/>② roll g_φ forward IN LATENT (no sim!)<br/>③ score = Σ γ·reward + γ·value<br/>④ execute best first action, replan"]
    end

    ENV -- "observation" --> ENC
    ENC -- "latent z" --> MPPI
    PRED -. "imagined future latents" .-> MPPI
    REW -. "score" .-> MPPI
    VAL -. "score" .-> MPPI
    MPPI -- "action" --> ENV

    ENV == "transitions" ==> BUF[("Replay buffer")]
    BUF == "sub-trajectories" ==> TRAIN["Training step (every env step)"]
    TRAIN -- "JEPA consistency loss:<br/>predicted latent ≈ f_ξ(real future obs)" --> ENC
    TRAIN -- "reward + value grounding<br/>(full rollout, SARSA value target)" --> REW
    TENC -. "stop-grad target" .-> TRAIN
```

**The two loops:**
- **Control (solid arrows):** observe → encode to latent → MPPI imagines action sequences purely
  in latent space (fast, no simulator) → execute the best first action → repeat.
- **Learning (double/▱ arrows):** store real transitions → train the world model so its
  *predicted* latents match the EMA target encoder's latents of the *real* future (the JEPA bet),
  while reward/value heads keep the latent grounded in the task.

## Key findings (the research story, honestly)

1. **Reward-free latent control works (GROUNDLESS).** The headline result: a JEPA can control a sim
   robot with **zero task reward in its representation** — pure multi-step latent consistency on a
   *raw* (un-normalized) latent with an EMA target. On cheetah it **matches the reward-grounded
   model** (496 ± 31 reward-free vs 522 reward), at *lower* variance.
2. **An earlier "reward is required" conclusion was wrong — and we caught it.** Mid-project we
   believed pure consistency collapses (encoder ignores the observation) so reward grounding was
   required. A controlled 2×2 ablation showed that collapse was an artifact of the **SimNorm
   simplex** latent, *not* of going reward-free: on a **raw** latent, consistency-only does **not**
   collapse (496) while on SimNorm it does (→4). The load-bearing variable is the latent
   parameterization, not the reward.
3. **It has a complexity boundary.** Reward-free control is strong in low-DoF 2D (cheetah) but
   **degrades in high-DoF 3D** (quadruped 184 vs reward-grounded 450) — the self-supervised
   consistency signal alone stops being enough as the action space grows. (Why is the current
   open question.)
4. **Honest negatives.** *Distractor robustness* — the hoped-for "JEPA ignores an unpredictable
   background where a reconstruction model can't" — **did not hold** at laptop scale. *Humanoid*
   (21-DoF) was not solved within budget.

Every headline here survived an adversarial **red-team** pass; three plausible-but-wrong claims
(including two of the above before correction) were refuted by re-seeding / full cross-eval before
being recorded. Single-seed and single-eval numbers misled repeatedly — the discipline is the point.

## Run it

```bash
# 1. install (Blackwell GPUs need the cu128 PyTorch build)
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. watch the random baseline in the real simulator (renders an mp4 + contact sheet)
PYTHONPATH=. MUJOCO_GL=egl python -m jepa_ctrl.cli --task cheetah-run --controller random --seeds 0,1,2

# 3. train the JEPA-MPPI controller (~20 min on an RTX 5080) and render the result
PYTHONPATH=. MUJOCO_GL=egl python scripts/train.py --task cheetah-run --steps 100000 --seed 0 --outdir runs/cheetah

# reward-FREE (GROUNDLESS): consistency-only on a raw latent — no reward in the representation
PYTHONPATH=. MUJOCO_GL=egl python scripts/train.py --task cheetah-run --grounding sigreg --sigreg-coef 0 --steps 100000

# 3D high-DoF control
PYTHONPATH=. MUJOCO_GL=egl python scripts/train.py --task quadruped-walk --steps 200000

# 4. tests
PYTHONPATH=. MUJOCO_GL=egl python -m pytest -q
```

## Project layout

```
jepa_ctrl/
  envs.py            dm_control wrapper (flat obs, action-repeat, offscreen render)
  controllers.py     Controller interface + random/zero baselines
  metrics.py         cross-seed returns, collapse diagnostics, latent-fidelity, ACCEPTANCE gates
  render.py          mp4 + annotated keyframe "contact sheet" for eyes-on verification
  evaluate.py        cross-seed evaluation loop
  model/             SimNorm encoder + EMA target, predictor, reward/value heads, latent MPPI, trainer
scripts/train.py     training driver (real-sim eval, collapse trajectory, wall-clock, render)
progress.md          full results table, the frontier roadmap, and what did/didn't work
```

## Status & roadmap (all-sim — no sim2real)

- [x] Latent-MPPI control of dm_control robots (cheetah, reacher).
- [x] **Reward-free latent control (GROUNDLESS)** on cheetah, red-teamed + SimNorm-vs-raw attributed.
- [x] 3D quadruped control (reward-grounded); reward-free generalizes but **degrades with DoF**.
- [x] Pixels + distractor-robustness head-to-head — **honest negative** (JEPA not auto-robust here).
- [ ] **Open / next:** *why* does reward-free degrade with DoF (latent capacity vs planning vs
      task-relevant info)? Can a minimal task-aware signal recover 3D control while staying mostly
      reward-free?
- [ ] Intrinsic-motivation exploration (latent-ensemble disagreement) on sparse tasks.
- [ ] Temporal-abstraction JEPA for long-horizon planning.

Full round-by-round log + experiment tables: [`progress.md`](progress.md). The original autonomous
research directive that drove all of this: [`LOOP_PROMPT.md`](LOOP_PROMPT.md).

## How this was built

This is an autonomous research loop: each round forms a hypothesis, implements it test-first,
**verifies in the real simulator** (cross-seed, with the rendered rollout inspected by eye — never
"the loss went down"), and records honest results including negative ones. Acceptance is real
control, not a passing metric. The full round-by-round log lives in [`progress.md`](progress.md).

Hardware: laptop **RTX 5080** (16 GB, Blackwell), 64 GB RAM, PyTorch cu128. Every training run
fits in well under 2 hours.

## Background / credits

Builds on the ideas in **I-JEPA** & **V-JEPA 2 / V-JEPA 2-AC** (Meta AI, joint-embedding
prediction + action-conditioned latent planning), **TD-MPC2** (Hansen et al. — latent MPC with
learned reward/value, our primary baseline), and **DreamerV3** (Hafner et al. — latent world
models). The contribution here is a minimal, laptop-scale, from-scratch instantiation and an
honest study of what makes latent-only control actually work.
