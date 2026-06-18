# STATUS — main
updated: 2026-06-18 · loop 7
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE (breakthroughs not reproduction)
phase:    frontier (GROUNDLESS red-teamed — headline overturned to a truer finding)
owns:     whole repo (single session)
doing:    GROUNDLESS red-team VERDICT: "SIGReg replaces reward" REFUTED. Real finding (stronger): RAW-latent multi-step latent-consistency (EMA+stop-grad, the JEPA objective itself) gives REWARD-FREE cheetah control 493 cross-seed (~94% of reward-grounded 522), NO SimNorm/SIGReg/reward/inverse-dyn. This OVERTURNS our R2 "consistency collapses" — that was a SimNorm artifact. Frozen-random-repr control fails (16) → learned latent essential. seed 2 (b1koihwz7) running to finish cross-seed.
blocked:  none
next:     (1) finish consistency-only cross-seed (seed 2). (2) MATCHED follow-up: reward-on-RAW + SimNorm-vs-raw ablation, to cleanly attribute (latent parameterization is the load-bearing variable, not grounding). (3) investigate WHY raw+EMA+stop-grad avoids collapse where SimNorm doesn't. (4) next frontier rung: distractor robustness (JEPA killer app, pixels). RED-TEAM every headline.
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. scripts/train.py --grounding {reward,inverse_dynamics,sigreg} [--sigreg-coef 0 = consistency-only] [--freeze-repr = red-team control]. 100k≈20min. acceptance=real sim control cross-seed + eyes-on + RED-TEAM. return is the gold signal; obs_corr is a noisy probe.
