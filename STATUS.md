# STATUS — main
updated: 2026-06-17 · loop 1
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI to control sim robots (dm_control); floor=cheetah/reacher, ceiling=match TD-MPC2 → manipulation → frozen V-JEPA enc → sim2real Go2/SO-101
phase:    review (R1 done)
owns:     whole repo (single session)
doing:    R1 COMPLETE — venv(cu128)+eval harness+random baseline real-verified; metrics hardened vs self-deception (centered cosine, target-collapse gate, codified is_collapsed/fidelity_ok)
blocked:  none
next:     R2 = RUNG 0 — SimNorm enc + EMA target + residual AC predictor + distributional reward/value + GPU MPPI; FIRST a cheetah-run 100k-step <2h wall-clock probe, THEN consistency-only vs +reward-grounding ablation, THEN 3 seeds × {cartpole,reacher-easy/hard,cheetah-run} serialized
notes:    PIN mujoco==3.8.1 (dm_control 1.0.41 breaks on 3.9). venv torch=cu128 (system torch is CPU). MUJOCO_GL=egl for render. run via PYTHONPATH=$HOME/jepa-ctrl. acceptance=real sim control cross-seed, never latent-loss alone.
