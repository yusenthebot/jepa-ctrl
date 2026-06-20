# STATUS — main
updated: 2026-06-19 · loop 17
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    frontier — R17 COVER-TO-RECOVER: attack the 3D bimodal collapse via epistemic coverage
owns:     whole repo (single session)
state:    3D quadruped collapse is a CHARACTERIZED METHOD BOUNDARY — bimodal (catch-a-gait ~400-500
  OR collapse ~40), collapse_rate ~0.70 (R11/R15), arm-independent, robust to 6 refuted levers
  (capacity/repr/planner/eval-smooth/explore-floor/training-amount; R16 300k made it WORSE 0.80-0.85
  = bad-start coverage shrinks as buffer narrows to good-gait). DIVERGE+judge chose to ATTACK it.
  R17 = COVER-TO-RECOVER: predictor-ensemble disagreement -> MPPI pessimism + near-fall harvest +
  reset-curriculum injection. FIRST PROBE (cheap, no ensemble): reset-curriculum ALONE.
in_flight: building the reset-curriculum probe (envs.py get/set_state + reset(from_state=); trainer
  low-return-tail state bank + inject p=0.3 of resets; --reset-curriculum CLI). TDD sim-free.
next:     verify build (env state roundtrip + harvest/inject) -> run reward-free raw quad 200k seeds{0,1}
  --reset-curriculum -> collapse_eval 20 eps vs base 0.70/0.80, Fisher + good_basin + valley + eyes-on.
  If <0.55 sig -> build full ensemble (pessimism + disagreement-harvest). If null -> ensemble-pessimism
  lead; if whole attack null + noisy-TV confirmed -> PIVOT to reward-free disagreement-explore on SPARSE
  tasks (ball_in_cup/cartpole-sparse/acrobot). RED-TEAM all (both-metrics, seed agree).
notes:    PIN mujoco==3.8.1. .venv torch cu128. MUJOCO_GL=egl. 3D metric = collapse_rate(>=20 eps,
  scripts/collapse_eval.py) + good_basin (single-eval is bimodal noise). progress.md=record; LOOP_PROMPT.md=directive.
