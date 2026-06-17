# jepa-ctrl

**Can a laptop-scale, action-conditioned JEPA latent world model + planning-in-latent-space actually control a simulated robot?**

This is a from-scratch research project testing one bet: a *small* action-conditioned
Joint-Embedding Predictive Architecture (JEPA) world model, with sampling-based planning
(CEM/MPPI) done entirely in latent space, is enough to control a `dm_control` / MuJoCo
robot on a single laptop GPU — i.e. a laptop-sized cousin of **V-JEPA 2-AC**.

The distinguishing feature vs. classic model-based RL (DreamerV3) is **no pixel/observation
decoder**: the world model predicts *latents*, not reconstructions. The distinguishing
feature vs. TD-MPC2 is that the latent is shaped by a **JEPA objective** (predict the
representation of the next observation), with TD-MPC2 used as the primary baseline to
catch up to.

## Hardware target

- Laptop **NVIDIA RTX 5080** (16 GB VRAM, Blackwell sm_120), 64 GB RAM, CUDA 12.8, PyTorch cu128
- Hard constraint: **any single training run finishes in ≤ ~2 hours**

## Direction (floor → ceiling)

- **Floor:** latent-space MPC actually controls `cheetah-run` and `reacher`, demonstrated
  visually and across seeds.
- **Ceiling / frontier:** match TD-MPC2 sample efficiency on the DeepMind Control suite →
  manipulation tasks → a *frozen pretrained* V-JEPA encoder → sim2real onto a Unitree Go2
  quadruped / SO-101 arm.

## Method (research-driven, runs autonomously)

Development proceeds in rounds — each round is a *leap in kind*, validated only by
**real training + real control in the simulator**, cross-seed, with replays inspected by
eye. No seed-picking, no latent-loss-only claims, no faked control. Negative results are
recorded honestly. See `progress.md` for the current state, the experiment-results table,
and the `## Frontier` ladder.

## Status

**Round 1 complete** — literature DIVERGE done, the cross-seed evaluation harness is built
and real-verified against live dm_control (random baseline, frames inspected by eye), and the
acceptance metrics are hardened against self-deception (centered cosine, target-collapse
gating, codified `is_collapsed`/`fidelity_ok`). The locked architecture blueprint, results
table, and the frontier ladder live in `progress.md`. Round 2 implements the model (RUNG 0).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
PYTHONPATH=. MUJOCO_GL=egl python -m jepa_ctrl.cli --task cheetah-run --controller random --seeds 0,1,2
```
