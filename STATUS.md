# STATUS — main
updated: 2026-06-19 · loop 9
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE (breakthroughs not reproduction)
phase:    frontier (3D control RUNNING)
owns:     whole repo (single session)
doing:    R9 distractor-robustness POWERED (64x64/90k) = hypothesis REFUTED (honest): JEPA clean 341 but distractor crushes it 341->60 (82% drop > recon 44%); JEPA NOT auto distractor-robust, recon underlearned. Killer-app claim does NOT hold at this scale. Now R10 3D CONTROL running (bhb8m5m3u): quadruped-walk reward + reward-free-raw (GROUNDLESS-scaling test) + humanoid-stand, state-based 80k seed0.
blocked:  none
next:     read bhb8m5m3u 3D 2x: (1) does latent-MPPI control 3D quadruped/humanoid at all? (2) KEY: does reward-FREE raw-latent consistency control 3D like reward (GROUNDLESS scales to 3D high-DOF)? RED-TEAM + eyes-on render. If quadruped controls, cross-seed + push toward Go2 sim2real. runs/ now R<NN>_<phase>/ chronological.
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. scripts/train.py [--pixels] --grounding {reward,inverse_dynamics,sigreg,reconstruction} [--sigreg-coef 0 = reward-free raw consistency] [--size]. 3D needs NO new code (DMCEnv dynamic dims: quad obs78/act12, humanoid obs67/act21). return=gold signal. RED-TEAM every headline. Best result so far = GROUNDLESS (reward-free raw-latent 496±31).
