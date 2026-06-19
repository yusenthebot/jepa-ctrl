# STATUS — main
updated: 2026-06-19T13:06 · loop 15 (IN-FLIGHT, supervisor-driven round)
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    green/execute — R15 scaffold COMMITTED (47f2889); 4 training runs in flight (background)
owns:     whole repo (single session; supervisor evolving-loop.sh drives cadence)
state:    R15 = TRAINING-side lever for 3D bimodal collapse (only cause left after R12/R13/R14
  refuted capacity/repr/planner/eval-smoothness). ONE knob: raise late-training explore FLOOR
  explore_std_end 0.05->0.2 (wider state coverage -> initial-condition robustness). Reward-free
  raw quad (sigreg coef 0, latent none/256), 200k, ALL-FRESH retrain to control retrain variance.
  Design: base(0.05) vs treat(0.2) x seeds{0,1} = 4 runs. THIS ROUND (non-GPU, training busy):
  committed CLI knob + eval + tests (resume invariant restored); verified knob wired end-to-end
  (_explore_std anneal trainer.py:134) + tests green CPU + ruff clean. Did NOT start a 2nd sim
  (serialize; 16GB/64GB).
in_flight: bash /tmp/r15_train.sh -> base_s0 RUNNING (~13min in @13:04); batch base_s0,base_s1,
  treat_s0,treat_s1 serial ~37min each, ALL-DONE ETA ~15:20. log runs/R15_explore_train.log.
next:     when 4 ckpts land: run scripts/r15_collapse.py --episodes 20. PRE-REGISTERED RED-TEAM
  (do NOT trust collapse_rate without these): (1) THRESH=150 verified in-gap for R11 BASE only;
  treat dist may shift -> plot histogram of `eps` per cell, confirm 150 sits in the valley, eyes-on
  render 1 good + 1 collapsed ep/cell; (2) report BOTH collapse_rate AND good_basin_mean — a knob
  that lowers collapse but craters good-basin return is NOT a win (more late noise can hurt gait);
  (3) only 2 seeds — if s0/s1 disagree (R14 levers reversed on s1) = INCONCLUSIVE, add seed 2;
  (4) Fisher is 1-sided treat<base; also report if treat collapses MORE (honest direction).
  If knob null -> next lever = updates-per-step up (undertrained) or terminal-value recal.
notes:    PIN mujoco==3.8.1. .venv torch cu128. MUJOCO_GL=egl. Report 3D ALWAYS as good-basin
  return + collapse rate >=20 eps. progress.md = full record; LOOP_PROMPT.md = directive.
