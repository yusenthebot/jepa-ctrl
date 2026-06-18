# STATUS — main
updated: 2026-06-18 · loop 8
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE (breakthroughs not reproduction)
phase:    frontier (distractor-robustness head-to-head RUNNING)
owns:     whole repo (single session)
doing:    GROUNDLESS done (raw-latent reward-free control 496±31, red-teamed+2x2-attributed). Now RUNG distractor-robustness (JEPA killer app): pixel CNN + seg-mask distractor + matched reconstruction baseline all BUILT+verified (5 integration bugs caught by smokes, fixed; eyes-on distractor render confirmed). Head-to-head bvodghk80 RUNNING (~5h): JEPA vs reconstruction × {clean, distractor}, cheetah-run pixels, 45k steps each.
blocked:  none
next:     read bvodghk80 2x2; hypothesis = recon's distractor DROP >> JEPA's drop (recon wastes capacity on background). RED-TEAM + eyes-on render both arms (recon should flail under distractor, JEPA run). If signal clear, cross-seed; if 45k too few for pixel learning, go 64x64 / reduce MPPI samples for more steps under 2h.
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. scripts/train.py --pixels [--distractor] --grounding {reward=JEPA, reconstruction=baseline}. pixel ~7-14 steps/s (recon 2x slower, decoder); 45k fits <2h. pixel buffer stacked uint8 cap 5e4. obs_corr UNRELIABLE on pixels (distractor dominates pixel-dist) — return is the gold signal. RED-TEAM every headline.
