# jepa-ctrl — progress

**Goal (floor is not the ceiling):** can a laptop-scale action-conditioned JEPA latent
world model + latent-space planning actually *control* a sim robot? Floor = latent-MPC
controls cheetah-run/reacher cross-seed; ceiling = match TD-MPC2 → manipulation →
frozen pretrained V-JEPA encoder → sim2real to Go2/SO-101.

Repo: https://github.com/yusenthebot/jepa-ctrl · single session on `main`.

**Runs naming convention (gitignored `runs/`):** `runs/R<NN>_<phase>/<arm>_s<seed>/` — round number
prefixes everything so an alphabetical listing is chronological. R01 harness_random · R02
first_control · R03 grounding_fix · R05 rung1_generalize · R06 stability · R07 groundless_{matrix,
redteam,2x2} · R08 distractor_pilot · R09 distractor_powered (p64_*) · R10+ = 3D control. Smokes are
deleted, not kept. Every new `--outdir` uses this scheme.

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
| R6 | reacher-easy | JEPA-MPPI 100k + explore-anneal | 0,1,2 | ~752 (744/904/608) | STABILITY FIX: anneal explore_std 0.3→0.05 → no more 949→20 crash, controls cross-seed. Value-divergence hypothesis **REFUTED** (v_mag stayed ≤4.1); real cause was variance/oscillation. |

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

Current ceiling: **RUNG 0 solid — cheetah 522±139 + reacher ~752 cross-seed, visible gait.** Core
bet validated. **STRATEGIC PIVOT (Yusen, 2026-06-18): stop reproducing TD-MPC2; pursue FRONTIER
breakthroughs only** — see the ⭐ FRONTIER PIVOT section below (active bet: GROUNDLESS). The old
physical ladder below remains the long-horizon path; the frontier ladder is now the spine.
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

**Round 6 (DONE 2026-06-18) — STABILITY:** Debug Protocol REFUTED the value-divergence hypothesis
— value magnitude stayed bounded (≤4.1) and the reacher rerun reached 987 and HELD through 120k;
the R5 "949→20" was a high-variance dip (constant explore_std=0.3 thrashing a near-solved policy +
GPU nondeterminism), not a systematic blow-up. Fix = anneal explore_std 0.3→0.05. Result: reacher
cross-seed ~752 (744/904/608), no crashes. Floor solid.

---

## ⭐ FRONTIER PIVOT (2026-06-18, Yusen directive: "frontier breakthroughs, not reproduction")

"Match TD-MPC2 on cheetah-run" is reproduction and is now BANNED as a goal. Every round forward
must earn JEPA's keep — do something a reconstruction/reward world model **structurally cannot**.

### Active bet — GROUNDLESS (Frontier round 1)
*What minimally REPLACES reward as the JEPA anti-collapse grounding?* A controlled 3-arm head-to-head
on identical infra: **A = consistency + inverse-dynamics** (action-from-latent), **B = consistency +
SIGReg** (LeJEPA isotropic-Gaussian latent regularizer, raw latent), **C = consistency + reward**
(R3 positive control). In A & B the reward/value heads are trained on DETACHED latents (post-hoc
probe) so the representation is **genuinely reward-free**; MPPI still plans with them.
- **Breakthrough claim:** a task-AGNOSTIC self-supervised signal replaces task reward as the
  collapse-preventer → one reward-free JEPA latent that re-plans arbitrary test-time rewards.
- **Why frontier not reproduction:** opposite of TD-MPC2/Dreamer (need reward); SIGReg/LeJEPA is
  image-SSL never tested in an action-conditioned world model; no published 3-way ablation in a
  decoder-free MPPI planner on dm_control locomotion from scratch. Tests our own R2/R3 finding with
  a 2025 tool that didn't exist when we found it.
- **Falsifiable success:** a reward-free arm reaches ≥365 cheetah-run (≥70% of 522) cross-seed AND
  keeps obs_latent_corr>0.4 + eff_rank above floor through 100k, while consistency-only collapses.
  A clean NEGATIVE (both reward-free arms collapse) is also publishable — it bounds task-agnostic grounding.
- **Builds on:** existing WorldModel/MPPI/metrics verbatim; adds an inverse-dyn head + SIGReg loss +
  arm flag + detach. ~20min/run; full 3-arm × 3-seed matrix is an overnight serialized loop.

### GROUNDLESS R1 matrix (2026-06-18) — superseded by the RED-TEAM VERDICT below
cheetah-run @100k, cross-seed (0,1,2):
- **SIGReg (reward-free): 387 ± 87** [457, 290, 415] — **CLEARS the ≥365 bar**, ~74% of reward-grounded.
  All 3 seeds gallop (render confirmed). obs_corr rising (s0 0.46; mean ~0.32 — the >0.4 sub-criterion
  is only partly met, reported honestly). ⇒ a TASK-AGNOSTIC distributional constraint replaces reward.
- **inverse-dynamics (reward-free): 31 ± 24** — COLLAPSED (obs_corr ~0). Action-from-latent does NOT
  prevent collapse here. Sharp contrast: not any self-supervised signal works — the distributional one does.
- Controls: reward 522 ± 139 (positive); consistency-only 138 (negative, R2).

### RED-TEAM VERDICT (2026-06-18) — SIGReg claim REFUTED; a BETTER, truer finding emerged
The discriminating controls overturned the headline (the red-team working as intended):
- **Control A — consistency-ONLY (raw latent, NO SIGReg, NO reward): 493** (524, 462), obs_corr
  **0.50–0.60.** Does NOT collapse; controls cheetah reward-free BETTER than SIGReg (387) and at
  **~94% of the reward-grounded arm (522)**. ⇒ SIGReg is **NOT causal** — unnecessary, slightly HURTS.
  "SIGReg replaces reward" = **REFUTED**.
- **Control B — frozen-random repr + planning: 16** (16, 17). Planner + post-hoc reward head over a
  RANDOM latent cannot control ⇒ the **learned latent is essential** (CONFIRMED, not a planner artifact;
  its obs_corr 0.67 is a random-projection artifact — return, not corr, is the gold signal).

**Real finding (stronger) — OVERTURNS our own R2 conclusion:** plain **multi-step latent consistency
on a RAW (un-normalized) latent** (EMA target + stop-grad — the JEPA objective itself) is **sufficient
for reward-free locomotion control**: 493 cross-seed, no SimNorm / SIGReg / reward / inverse-dynamics.
R2's "pure consistency collapses; reward required" was a **SimNorm artifact** — the simplex lets the
encoder collapse to a point under reward-free consistency; an un-normalized latent with EMA/stop-grad
does not. The load-bearing variable is the **latent parameterization**, not the grounding signal.
Caveat (honest): reward-arm used SimNorm vs consistency-only raw — matched reward-on-raw run is the
clean follow-up (running). **Cross-seed CONFIRMED: consistency-only RAW = 496 ± 31** (524/462/501),
gallop rendered, obs_corr 0.34–0.57 — and LOWER variance than reward-grounded (±31 vs ±139).
**Matched 2×2 attribution — CONFIRMED (latent parameterization is load-bearing, not grounding):**

| latent | reward-free (consistency-only) | reward-grounded |
|--------|--------------------------------|-----------------|
| **RAW**     | **496** (524/462/501, ±31) ✓ | 438 (543/333, noisier) |
| **SimNorm** | **4** (3/5) ✗ TOTAL COLLAPSE | 522 (R3, ±139) ✓ |

⇒ Reward-free consistency **collapses on SimNorm (4)** but **controls on a raw latent (496)**. The
R2 "consistency collapses; reward required" was a **SimNorm simplex artifact** — the simplex permits
point-collapse under reward-free consistency; an un-normalized latent + EMA/stop-grad does not.
Reward grounding *rescues* SimNorm (4→522) but is **unnecessary** on a raw latent (496 > the noisier
reward-grounded-raw 438). **Frontier finding, red-teamed + fully attributed:** a raw-latent JEPA
(multi-step consistency, EMA+stop-grad, NO decoder, NO reward) gives reward-free cheetah control at
496±31 — the cleanest reward-free result, and it overturns our own earlier conclusion. GROUNDLESS done.

### R8 — Distractor robustness pilot (2026-06-18): INCONCLUSIVE / underpowered (honest)
Pixel cheetah-run @45k, seed 0, 2×2 JEPA(reward,no-decoder) vs Reconstruction(decoder), clean vs distractor:

| arm | clean | distractor | drop |
|-----|-------|-----------|------|
| JEPA | 184 | 75 | 109 (59%) |
| Reconstruction | 51 | 34 | 17 (33%) |

**Hypothesis NOT supported at this scale.** The relative drop went the "wrong" way, but the cause is
UNDERPOWERING: at 45k pixel steps the reconstruction arm barely learned (51 clean ≈ noise floor; vs
state-based 522), so its small distractor drop is uninformative — you can't show "recon craters under
distractors" when recon never learned clean control. Eyes-on: even pixel-JEPA-clean (184) only weakly
controls (shuffles, then tips). The pixel rung is COMPUTE-BOUND on the laptop (recon decoder ~2× slower;
100k pushes >2h). Mild confounded signal: pixel-JEPA learns ~3.6× better than pixel-recon at low budget
(184 vs 51). 5 integration bugs were caught+fixed by smokes (env↔buffer, planner flatten, 4D probe, CLI,
loss key). NEXT: re-run at 64×64 + more steps so both arms reach a real clean baseline before judging
robustness; if still inconclusive, this rung needs compute beyond a clean laptop 2h run.

### R9 — Distractor robustness POWERED (64×64, 90k, seed0): hypothesis REFUTED (honest)
| arm | clean | distractor | drop |
|-----|-------|-----------|------|
| JEPA | **341** | 60 | 82% |
| Reconstruction | 77 | 44 | 44% |

64×64/90k gave JEPA a REAL clean baseline (341, up from 184@45k) — but the distractor **crushes** it
(341→60, 82% drop), a LARGER relative drop than reconstruction's (44%). **JEPA is NOT automatically
distractor-robust** in this setup: the latent-consistency objective did not, on its own, teach the CNN
encoder to discard the unpredictable background. (Recon still underlearned at 77, so its drop stays
confounded — but JEPA's own collapse under the distractor is the decisive negative.) **Killer-app claim
does NOT hold here.** Likely needs: consistency as the *dominant* loss + much more training for the
"ignore-unpredictable" effect to emerge, or an explicit distractor-invariance pressure — i.e. robustness
is not free from the JEPA objective alone. Recorded straight; moved on to 3D per Yusen. (NOTE: the
strongest, cleanest frontier result of the project remains GROUNDLESS — reward-free raw-latent control.)

### R10 — 3D CONTROL (quadruped-walk, state, 80k): GROUNDLESS generalizes to 3D (honest, cross-seed)
quadruped-walk (obs 78, act 12 — the Go2 sim2real bridge), 3D state-based (~20min/100k, cheap).
**TRAINING-cross-seed (separate seeds 0/1/2, 3 eval-eps):**
- **reward-FREE raw-latent consistency: 202 ± 179** [391, 180, 35]
- **reward-grounded: 311 ± 240** [175, 588, 171]
- ⇒ **GROUNDLESS GENERALIZES to 3D**: reward-free raw-latent consistency genuinely *learns to control*
  the quadruped (391/180 on 2/3 seeds — not collapsed; forward locomotion confirmed by return + render),
  so the reward-free finding is NOT 2D-specific. BUT it does **NOT beat reward-grounded cross-seed**
  (202 vs 311) and **both arms are high-variance at 80k** — 80k is too few for stable 3D quadruped control.
- **Correction (red-team save #3):** the single-seed "reward-free 391 > reward 174" was seed-0 LUCK;
  cross-seed reverses the means. Recorded straight — the honest claim is "reward-free control works in
  3D", NOT "beats reward in 3D".
- **humanoid-stand (act 21): 6.9 — FAILED** at 80k (21-DOF too hard for this budget; honest negative).
### R11 — Quadruped 200k cross-seed: the GROUNDLESS BOUNDARY (honest, well-powered)
quadruped-walk, 200k, separate seeds 0/1/2, 3 eval-eps:
- **reward-free raw: 184 ± 141** [169, 332, 52] — did NOT stabilize even at 200k (seed2 stuck 52).
- **reward-grounded: 450 ± 234** [424, 696, 230] — clearly better, still climbing at 200k.
⇒ **Refined GROUNDLESS conclusion (the real scientific result):** reward-free raw-latent consistency
controls *low-DOF 2D* (cheetah ~496, matches reward-grounded) but **degrades in high-DOF 3D**
(quadruped 184 ≪ reward-grounded 450). Reward-free latent control WORKS in 3D (learns, 169-332 on
2/3 seeds) but hits a **complexity/DOF ceiling** — as action-dim rises the self-supervised consistency
signal alone is insufficient and reward grounding becomes necessary for stable control. A genuine,
well-powered BOUNDARY (not "scales everywhere"). For the Go2 sim2real path, reward-grounded is the
practical config (450@200k, climbing). Open: WHY does reward-free degrade with DOF (predictor
capacity? exploration? planning horizon in high-dim action space?).

### R12 — 3D capacity probe (H1): REFUTED. lat512 reward-free quad-walk 200k seed0 = **281**
[curve 290/306/168/284] — within the lat256 reward-free variance band [169,332,52]; latent stays
`collapsed=true` (obs_latent_corr≈0) throughout. Doubling latent capacity does NOT lift reward-free
into reward-grounded territory ⇒ **H1 (representation capacity) is NOT the DoF lever.**

### R13 — WHY reward-free degrades with DoF: a RED-TEAM SAVE (2026-06-19, eval-only)
Diagnosed the R11 "DoF degradation" on already-trained models (no retrain). `scripts/diag_dof.py`.
- **H3 (latent under-encodes task) — REFUTED, clean.** Ridge probe of the frozen online latent →
  per-step reward: R²≈ **0.81–0.94 (reward-free quad)**, 0.96 (reward-grounded), 0.99 (cheetah).
  The reward-free latent encodes the task reward almost as well as the grounded one — representation
  is *not* the bottleneck. (Value head is miscalibrated everywhere — underestimates MC return,
  corr≈0 on cheetah — yet cheetah controls great ⇒ planner leans on summed per-step reward over a
  short horizon, not the terminal bootstrap.)
- **H2 (planner search) — the clean story COLLAPSED under red-team.** A coarse sweep *looked* like
  "more MPPI samples (512→2048) reliably fixes it" (RF 184→412, RW 450→865, tight variance). But a
  CONTROLLED one-knob-at-a-time sweep (elites fixed; 5–8 eps) shows **NO clean monotonic lever** for
  any of {samples 256–4096, iters 3–12, horizon 2–5, elites 32–256}. The coarse "tight wide" result
  was a 3-episode fluke + a `num_elites=128` confound.
- **The real, red-teamed finding: 3D quad control is BIMODAL, not a smooth "DoF degradation."** Every
  config has episodes that either **catch a gait** (RF ~250–450, RW ~830–875) or **collapse early**
  (~15–125). Which basin is hit is dominated by initial-condition/planning stochasticity. Verified
  eyes-on: the SAME controller+config gives {494,498,488,505} vs {72} across 5 episodes
  (`runs/R13_dof_diag/{good_gait,collapsed}.png`). ⇒ The prior R10/R11 point estimates (RF 184,
  RW 450, 3 eps each) were **underpowered means over a bimodal distribution** — the "degradation"
  conflated a real reward-vs-RF *good-basin* gap (~400 vs ~850) with collapse-rate variance.
- **Causes cleanly refuted: H1 capacity (R12), H3 representation (R13).** The actual lever for 3D is
  the **collapse rate / gait-acquisition reliability** — a stability/exploration/training problem,
  NOT planner-tuning and NOT representation. This red-team saved the campaign from a wrong
  "tune the planner" / "fix the latent" conclusion. (red-team save #4.)

### R14 — COLLAPSE RATE characterized (20 eps) + eval-time levers REFUTED (2026-06-19, eval-only)
Followed R13's seed: stop tuning planner search-breadth; properly measure the BIMODAL collapse rate
over many eps and test EVAL-TIME action-smoothness levers on the FIXED R11 checkpoints (no retrain).
`scripts/collapse_rate.py`. THRESH=150 sits in a clean wide gap in every arm (e.g. RW_s0 collapse
cluster 101-129 vs gait cluster 869-896). 20 eps/cell. runs/R14_collapse/.

- **Collapse rate is HIGH and ARM-INDEPENDENT (the genuinely new finding):** base collapse_rate
  RF_s0 **0.55**, RF_s1 **0.70**, RW_s0 **0.70**. Reward grounding does **NOT** reduce the collapse
  rate (RW 0.70 = RF 0.70); it only RAISES the good-basin return (RW good ~674-817 vs RF ~400-486).
  ⇒ The reward-vs-reward-free gap is **entirely a good-basin QUALITY gap, not a reliability gap.**
  Overturns the naive read that the reward-grounded arm is "more reliable" — it tips over just as often.
- **Eval-time action-smoothness levers REFUTED as a collapse fix.** One-knob-at-a-time sweep
  (corr {0.3,0.6,0.85} · std_min {0.15,0.3} · temperature {0.1,0.05} · momentum {0.3,0.5}, all
  verified wired into the planner): **no lever robustly lowers collapse across seeds+arms.** Best
  apparent (corr0.3 on s0: 0.55→0.40) is **NOT significant** (Fisher p=0.53) and **REVERSES** on
  s1/RW (p=1.0). The only robust, consistent effect is the WRONG direction — **greedier planning
  (temp 0.5→0.1/0.05) makes collapse strictly worse (0.75-0.95 across all 3 arms)** ⇒ eval-time
  exploration noise is load-bearing; collapse is NOT eval-time jerkiness/over-exploitation.
- **Verdict (red-teamed):** the bimodal collapse is **NOT fixable at eval/planning time**. Causes
  now refuted: H1 capacity (R12), H3 representation (R13), H2 planner-search (R13), **eval-time
  action-smoothness (R14)**. By elimination the lever is a **TRAINING / gait-acquisition-reliability**
  problem (initial-condition robustness) — exactly where R13 pointed. red-team passed: levers proven
  wired; significance tested (Fisher); bimodality crisp + eyes-on (R13 good_gait/collapsed.png, same
  R11 checkpoints).
- **NEXT (training-side, the only remaining lever):** retrain quad reward-free with ONE controlled
  training change and measure collapse_rate over ≥20 eps vs the R11 base (0.55-0.70). Candidates,
  pick one: (a) **updates-per-step ↑** (undertrained-policy hypothesis — cheapest, ~20min/100k);
  (b) **exploration/action-noise SCHEDULE during data collection** (widen initial-state coverage so
  the policy learns to recover from bad starts); (c) terminal-value recalibration (value head under-
  estimates MC return ~2×). Report 3D ALWAYS as good-basin return + collapse rate over ≥20 eps.

### R15 — TRAINING-side collapse lever (IN-FLIGHT 2026-06-19): exploration-floor knob
R13/R14 eliminated every NON-training cause of the 3D bimodal collapse (H1 capacity R12, H3 repr
R13, H2 planner R13, eval-time action-smoothness R14). The only lever left = training-side
gait-acquisition reliability (initial-condition robustness). R15 tests ONE controlled knob:
**raise the late-training exploration floor** `explore_std_end` 0.05->0.20 (anneal 0.3->floor over
100k, then hold floor 100k-200k) so the behaviour policy keeps wider late-training state coverage
-> learns to recover from bad starts -> fewer early collapses. Reward-free raw quad (sigreg coef 0,
latent none/256), 200k, ALL-FRESH retrain (controls retrain variance), base(0.05) vs treat(0.20)
x seeds{0,1} = 4 runs. Scaffold committed 47f2889 (CLI knob refactor build_parser/
train_config_from_args, sim-free unit-testable; r15_collapse.py 20-eps collapse_rate + Fisher-exact;
knob verified consumed by trainer `_explore_std`). 4 runs in background (~37min each, batch ETA
~15:20). **PRE-REGISTERED RED-TEAM (do NOT trust collapse_rate without all of these):**
1. THRESH=150 was verified in-gap for R11 BASE checkpoints only; the treat checkpoint's bimodal
   clusters may SHIFT — plot the per-ep `eps` histogram per cell, confirm 150 sits in the valley,
   eyes-on render 1 good + 1 collapsed ep per cell before believing the rate.
2. Report BOTH collapse_rate AND good_basin_mean. More late-training noise could LOWER collapse
   yet CRATER good-basin return — that is NOT a win. The headline must be the pair, never the rate alone.
3. Only 2 seeds. If s0 and s1 disagree (R14's eval levers reversed on s1) the result is INCONCLUSIVE
   — add seed 2 before any directional claim.
4. Fisher is 1-sided (treat collapses LESS, pre-registered H1). Also report the honest direction if
   treat collapses MORE — raising late noise hurting gait acquisition is a real, plausible outcome.
If the knob is null/negative: next levers = updates-per-step up (undertrained-policy hypothesis) or
terminal-value recalibration (value head underestimates MC return ~2x).

### Frontier ladder (escalation in KIND — the new spine)
1. **GROUNDLESS** (DONE): reward-free raw-latent control 496±31, red-teamed + attributed.
2. **Distractor robustness** (JEPA's killer app): JEPA-MPC stays in control under visual distractors
   that collapse a reconstruction baseline. Pilot @45k INCONCLUSIVE (underpowered, compute-bound);
   powered 64×64/90k re-run running. The clearest "JEPA does what Dreamer can't" demo if it holds.
2b. **★ 3D CONTROL — reframed by R13/R14.** Quad-walk control is BIMODAL (gait vs early collapse).
   R14 (20 eps): collapse_rate ~0.55-0.70 and **ARM-INDEPENDENT** — reward grounding raises the
   good-basin (~674-817 vs RF ~400-486) but does NOT lower collapse. Refuted as causes: H1 capacity
   (R12), H3 representation (R13), H2 planner-search (R13), **eval-time action-smoothness (R14)**.
   ⇒ The lever is **TRAINING / gait-acquisition reliability** (initial-condition robustness), the
   only thing left standing. **Next round = ONE controlled TRAINING change** (updates-per-step ↑, or
   action-noise schedule during data collection, or terminal-value recal), measure collapse_rate over
   ≥20 eps vs R11 base. Report 3D ALWAYS as good-basin return + collapse rate over ≥20 eps.
3. **Latent-disagreement intrinsic motivation**: ensemble disagreement in latent space → crack
   sparse / hard-exploration tasks reward-MPC fails on.
4. **Temporal-abstraction JEPA**: predict far-future latents directly → long-horizon planning where
   1-step rollouts drift.
5. (then the prior physical ladder: pixels → goal-image → manipulation → frozen V-JEPA enc → sim2real.)

## Open questions
- Does pure latent-consistency (no reward grounding) avoid collapse on DMC state with
  SimNorm+EMA+stopgrad alone, or is reward grounding strictly required? (the thesis)
- Residual-delta in pre-SimNorm space then re-normalize, vs predict SimNorm vector directly?
- Actual per-step MPPI wall-clock on the 5080 → env-step budget that fits 2h (profile R2).
- smooth_L1 vs cosine-on-normalized vs L1 for the consistency distance on SimNorm latents.
