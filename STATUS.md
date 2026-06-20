# STATUS — main
updated: 2026-06-20 · loop 17
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    review — R17 COVER-TO-RECOVER probe RUNNING (reset-curriculum alone, the cheap discriminator)
owns:     whole repo (single session)
state:    R17 build DONE + verified: envs get/set_state + reset(from_state=) — LIVE roundtrip PASS on
  quadruped (state_restored/reproducible/shape_ok/differs_from_fresh). NearFallBank reservoir +
  inject p=0.3. Full suite GREEN 104p/1skip. ALSO fixed a 2nd OOM bomb this session: pixel replay
  buffer reused the STATE default capacity (1e6 -> ~127GB at pixel slot) — clamped to 8GiB byte
  budget (pixel_buffer_capacity) + allocation-free estimate_nbytes; rule -> performance.md
  (incl. ulimit -v is WRONG for torch/CUDA — CUDA reserves tens of GB VIRTUAL).
in_flight: reward-free RAW quad 200k reset-curriculum probe. seed0 RUNNING (~5k steps/min, ~40min/seed);
  detached driver scripts/r17_probe.sh (NOT harness-tracked) waits seed0 -> trains seed1
  (flock-serialized) -> collapse_eval both (20 eps THRESH=150) -> runs/R17_resetcurr/RESULT.txt.
blocked:  none
next:     when RESULT.txt ready: RED-TEAM (both-metrics collapse_rate+good_basin, seed agree, valley
  check) + eyes-on render. If collapse_rate <0.55 Fisher-sig vs base 0.65-0.85 -> coverage CONFIRMED
  -> build full ensemble (disagreement pessimism + harvest). If null -> ensemble-pessimism lead;
  whole attack null + noisy-TV -> PIVOT to reward-free disagreement-explore on SPARSE tasks.
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root (NOT empty — empty drops
  jepa_ctrl). flock serializes trainings (refuse-not-block). progress.md=record; LOOP_PROMPT.md=directive.
