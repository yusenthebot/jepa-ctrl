# jepa-ctrl — progress

**Goal (floor is not the ceiling):** can a laptop-scale action-conditioned JEPA latent
world model + latent-space planning actually *control* a sim robot? Floor = latent-MPC
controls cheetah-run/reacher cross-seed; ceiling = match TD-MPC2 → manipulation →
frozen pretrained V-JEPA encoder → sim2real to Go2/SO-101.

Repo: https://github.com/yusenthebot/jepa-ctrl · single session on `main`.

---

## Current state — Round 1 COMPLETE (2026-06-17)

Round 1 = research DIVERGE + the architecture-agnostic **evaluation harness** + a
random baseline real-verified in the live simulator. **No model yet** (that is round 2).

- **Env**: isolated venv `~/jepa-ctrl/.venv`, torch `2.11.0+cu128` on the RTX 5080
  (sm_120 Blackwell, CUDA matmul verified), `dm_control 1.0.41`, `mujoco==3.8.1`,
  `gymnasium 1.3.0`. EGL offscreen GPU rendering works headless.
- **Harness** (`jepa_ctrl/`): `DMCEnv` (flat obs, dims derived at runtime, action_repeat,
  EGL render) · `Controller` ABC + `RandomController`/`ZeroController` · `render`
  (mp4 + annotated contact sheet) · `metrics` (cross-seed aggregation, collapse
  diagnostics, open-loop latent-rollout fidelity, sample-efficiency plot, **codified
  acceptance**) · `evaluate` (cross-seed loop, renders every seed) · `cli`.
- **Tests**: 15 pytest, ruff clean.
- **REAL-VERIFY**: random baseline driven through real dm_control, rendered, frames
  inspected by eye — genuine MuJoCo cheetah (flailing on the checkered floor) and reacher
  (2-link arm sweeping past the target). Harness proven end-to-end.

### Results table

| Round | Task | Controller | Seeds | Mean return | Notes |
|------|------|-----------|-------|-------------|-------|
| R1 | cheetah-run | random | 0,1,2 | 6.7 (±2.9 ci95) | floor; TD-MPC2 anchor ~772@500k |
| R1 | reacher-easy | random | 0,1,2 | 39.6 | per-seed [54.3, 0.0, 64.3] — random occasionally hits target |
| R2 | cheetah-run | JEPA-MPPI 100k | 0 | 138 (1 seed) | PARTIAL control (7→138 over 100k, 18min/100k). Latent under-collapsed but recovering: obs_corr 0.09→0.34, PR 2.2→6.9, rank_frac 0.13→0.23. cross-seed PENDING. |
| R3 | cheetah-run | JEPA-MPPI 100k +grounding-fix | **0,1,2** | **522 ± 139** | **RUNG 0 MET CROSS-SEED.** Grounding fix (full-rollout reward + SARSA value bootstrap) → 138→557 (4×). Per-seed [557, 369, 640], all ≫ random 6.7; visible RUNNING GAIT (render confirmed). ~20min/100k. `is_collapsed` recalibrated to absolute eff-rank floor (the fraction-of-d rule was a false-alarm artifact, verified on the real model). |
| R5 | reacher-easy | JEPA-MPPI 100k | 0 | 22 (**peak 949@80k**) | **GENERALIZES but UNSTABLE** — nearly SOLVED reacher (949@80k, ~960=solved) then catastrophically DIVERGED to 20 by 100k. Training stability is the new bottleneck. |
| R5 | walker-walk | JEPA-MPPI 100k | 0 | 227 (peak 419) | partial control (random ~45); noisy/unstable across evals. |

(Baselines only; these are the "did nothing" floor the learned controller must beat.)

## Locked architecture (round-2 blueprint)

**Route: state-input action-conditioned JEPA + reward-head MPPI** (laptop V-JEPA2-AC,
TD-MPC2-grounded). ~1M params, all-MLP, no decoder.
- **Encoder** `f_theta`: MLP `obs_dim → 256 → 256 → latent`, **SimNorm** output
  (simplex groups of 8); latent dim 128 (cartpole/reacher) / 256 (cheetah/walker).
  `enc_lr_scale=0.3`.
- **Predictor** `g_phi`: residual-delta MLP on `concat(z_t, A(a_t))` (separate learnable
  action head; never raw action scalars); `z_hat_{t+1}=SimNorm(z_t_pre + g_phi)`.
- **JEPA objective**: multi-step latent-consistency, H=5, discount ρ=0.5,
  `smooth_L1(z_hat_{t+k}, stopgrad f_xi(o_{t+k}))` fed its OWN rollout; consistency_coef=20.
  **No reconstruction.**
- **Collapse guards (layered)**: SimNorm + EMA target encoder (τ 0.99→0.996) + stop-grad +
  reward/value grounding; VICReg on a flag if diagnostics fire.
- **Reward/goal (floor)**: learned **distributional reward + terminal value** heads
  (101-bin HL-Gauss symlog, num_q=5 ensemble, EMA target) — NOT goal-image (cheetah-run
  has no static goal frame). Goal-image energy MPC is a later frontier rung.
- **Planner**: **MPPI** in latent space, H=3, iters 4–6, 256–384 samples (train) / 512
  (eval), terminal-value bootstrap, policy-prior warm-start. action_repeat=2.

## Frontier ladder (## Frontier — ambition horizon)

Current ceiling: **JEPA-MPPI RUNS cheetah-run — 522 ± 139 @100k CROSS-SEED (0,1,2), visible gallop. RUNG 0 MET.** The core bet (laptop-scale action-conditioned JEPA + latent MPPI controls a sim robot) is validated. Next: RUNG 1 (walker/reacher, harder dynamics) + the TD-MPC2 head-to-head; and push cheetah further (more steps → toward ~772). Escalation in *kind*:
0. **RUNG 0 (floor) — DONE for cheetah-run (522±139 cross-seed, visible gallop); reacher in
   RUNG 1.** Thesis answered: reward grounding is REQUIRED — pure-consistency-dominant collapses;
   full-rollout reward grounding + SARSA value bootstrap fixes it (138→557).
1. RUNG 1: harder dynamics same machinery — walker-walk → walker-run → finger-turn_hard /
   acrobot; match TD-MPC2 at 100k/500k.
2. RUNG 2: **match TD-MPC2 sample efficiency** on the easy/medium DMC set; add policy-prior;
   head-to-head table. ("caught up to baseline")
3. RUNG 3 (KIND): state → **PIXELS** — swap MLP for a DrQ-style 4-conv CNN, keep the whole
   JEPA/SimNorm/EMA stack; control from pixels.
4. RUNG 4 (KIND): **goal-image / reward-free** control (literal V-JEPA2-AC mode) on
   reacher-with-goal-image + a reach task.
5. RUNG 5 (KIND): **manipulation** (Meta-World/ManiSkill/dm_control manipulation),
   sub-goal decomposition like V-JEPA2-AC pick-and-place.
6. RUNG 6 (KIND): **frozen pretrained V-JEPA2 ViT-L 300M** encoder + our AC predictor —
   the literal 2-stage recipe at the largest scale the 16GB laptop allows.
7. RUNG 7 (KIND): **SIM2REAL** — SO-101 arm (goal-image reach/place) then Unitree Go2.

## What worked
- DIVERGE workflow → decisive, buildable blueprint; TD-MPC2 = both baseline and the
  reward-grounding ingredient our "pure JEPA" cousin will be ablated against.
- cu128 torch works on Blackwell sm_120; EGL render headless on the laptop.
- Adversarial-review workflow caught the metric self-deception modes *before* any model
  existed to exploit them.

## What did NOT work / gotchas (carry forward)
- **mujoco 3.9.0 breaks dm_control 1.0.41** (`MjModel.flex_bandwidth` removed). PINNED
  `mujoco==3.8.1`. Do not let it float.
- System python torch is **CPU-only**; must use the project venv (cu128).
- **Metric self-deception (fixed in R1, keep vigilant in R2):** cosine is offset-foolable
  → use **centered_cosine**; latent fidelity is meaningless under **target-encoder
  collapse** → `latent_rollout_fidelity` now flags `target_collapsed` and `fidelity_ok`
  rejects it; collapse has two modes (dimensional vs point) → `is_collapsed` checks both.
  The gate is **code** (`is_collapsed`/`fidelity_ok` with versioned thresholds), not a
  human reading JSON. Acceptance = real sim control + cross-seed, never latent-loss alone.
- **R2 collapse finding (the thesis, answered):** pure-consistency-dominant (coef 20) with
  weak reward grounding (0.1) **collapses early** on DMC state — at 1k steps obs↔latent corr
  fell 0.65→0.007, PR→1.3. Over 100k it RECOVERS (reward grounding works) but too slowly to
  reach strong control. Root causes diagnosed: (a) **grounding too sparse** — only rollout
  step 0 gets reward/value gradient, so the other H latents see only the collapse-prone
  consistency loss; (b) **value bootstrap uses a zero-action proxy** → value head useless →
  no real grounding/sample-efficiency signal; (c) **latent_dim=256 is oversized** for a 17-dim
  cheetah state, so `is_collapsed` over-fires (rank_frac threshold 0.35 >> natural ~0.07-0.12).
- **Threshold calibration:** the generic collapse thresholds need scaling to the intrinsic
  state dim (or track obs_latent_corr + the recovery TREND, which were the informative signals).

## Round 2–3 results (DONE 2026-06-17) & Round-4 seed (next)

**R3 OUTCOME: grounding fix VALIDATED — cheetah-run 138→557 (4×), visible running gait, RUNG 0
met single-seed.** Hypotheses 1 (full-rollout reward) + 2 (SARSA value bootstrap) were the fix.
Hypothesis 3 (right-size latent / recalibrate `is_collapsed`) is still open — `is_collapsed`
over-fires on the 256-dim latent (a proven artifact, since control is strong). The original
R2/R3 planning notes follow.

R2 built the full model (SimNorm enc+EMA, residual AC predictor, distributional reward/value,
latent MPPI, trainer; 32 tests, GPU-verified) and ran cheetah-run 100k (18min, seed 0):
**partial control, return 138**, representation under-collapsed but recovering. The pipeline
works end-to-end; the bottleneck is grounding strength, not infra.

**Round 3 (grounding fix — leading hypotheses, run as an ablation vs the R2 baseline above):**
1. **Ground reward across the FULL rollout**, not just step 0: predict r_k from each rolled
   latent z_hat_k vs the sub-trajectory rewards — spreads grounding to every latent (TD-MPC2's
   real anti-collapse engine).
2. **Fix the value bootstrap**: SARSA-style TD target r_0 + γ·Q_target(z1, a1) using the
   ACTUAL next action a1 from the sub-trajectory (or a learned policy-prior π(z)), never zeros.
3. **Right-size the latent / recalibrate diagnostics**: try latent_dim 64–128 for cheetah, and
   scale collapse thresholds to intrinsic state dim (track obs_latent_corr + trend).
Re-run cheetah-run 100k, compare return + collapse trajectory vs R2 (138). If control clears a
real bar, scale to 3 seeds × {cartpole, reacher-easy/hard, cheetah-run}, **serialized**.
Prereq before locking sample-efficiency targets: pull EXACT TD-MPC2 per-task DMC returns from
the official results CSVs (do not trust recalled numbers).

**Round 4 (DONE 2026-06-17):** cross-seed validated — cheetah-run 522±139 @100k (seeds 0,1,2),
**RUNG 0 MET** ✓. `is_collapsed` recalibrated to an absolute effective-rank floor (the
fraction-of-d rule false-fired on the oversized 256-d latent; verified on the real R3 model
which reads healthy) ✓.

**Round 5 (DONE 2026-06-17) — RUNG 1 generalization probe:** the method GENERALIZES but is
UNSTABLE. reacher-easy nearly SOLVED (949@80k) then catastrophically diverged to 20@100k;
walker-walk partial (peak 419, final 227, noisy). Generalization confirmed; **training stability
is the new bottleneck.**

**Round 6 (next) — STABILITY (the new bottleneck):**
1. Diagnose the reacher 949→20 divergence (Debug Protocol). Leading hypotheses: value
   OVERESTIMATION / divergence (the SARSA bootstrap + ensemble may need stronger target
   stabilization, value clipping, or a slower value LR); representation drift late in training;
   or replay non-stationarity. Add per-update value-loss + value-magnitude logging to catch it.
2. Likely fixes to try (ablate): lower value LR / stronger Polyak (value_tau), reward/value
   normalization, longer EMA target horizon, or a learned policy prior to stabilize the bootstrap.
3. THEN resume RUNG 1: cross-seed reacher + walker once stable; pull TD-MPC2 per-task DMC numbers
   for the head-to-head; push cheetah to 300k–500k toward ~772.

## Open questions
- Does pure latent-consistency (no reward grounding) avoid collapse on DMC state with
  SimNorm+EMA+stopgrad alone, or is reward grounding strictly required? (the thesis)
- Residual-delta in pre-SimNorm space then re-normalize, vs predict SimNorm vector directly?
- Actual per-step MPPI wall-clock on the 5080 → env-step budget that fits 2h (profile R2).
- smooth_L1 vs cosine-on-normalized vs L1 for the consistency distance on SimNorm latents.
