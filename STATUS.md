# STATUS — main
updated: 2026-06-18 · loop 8
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE (breakthroughs not reproduction)
phase:    frontier (distractor-robustness head-to-head RUNNING)
owns:     whole repo (single session)
doing:    GROUNDLESS done (raw-latent reward-free control 496±31, red-teamed+2x2-attributed). Now RUNG distractor-robustness (JEPA killer app): pixel CNN + seg-mask distractor + matched reconstruction baseline all BUILT+verified (5 integration bugs caught by smokes, fixed; eyes-on distractor render confirmed). Head-to-head bvodghk80 RUNNING (~5h): JEPA vs reconstruction × {clean, distractor}, cheetah-run pixels, 45k steps each.
blocked:  none
next:     (1) read powered distractor 2x2 bhgw8mmmb (64x64/90k); RED-TEAM + eyes-on; if recon still underlearns, this pixel rung is laptop-compute-bound (say so). (2) ★ Yusen QUEUED: 3D CONTROL — dm_control quadruped-walk/run (Go2 sim2real bridge) + humanoid; STATE-BASED so cheap (<2h); tests if GROUNDLESS reward-free raw-latent scales to 3D high-DOF. Likely the better next rung than chasing the compute-bound pixel test.
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. scripts/train.py --pixels [--distractor] --grounding {reward=JEPA, reconstruction=baseline}. pixel ~7-14 steps/s (recon 2x slower, decoder); 45k fits <2h. pixel buffer stacked uint8 cap 5e4. obs_corr UNRELIABLE on pixels (distractor dominates pixel-dist) — return is the gold signal. RED-TEAM every headline.
