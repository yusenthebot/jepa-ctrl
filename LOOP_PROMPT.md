# LOOP_PROMPT — the autonomous research directive driving this repo

This project is run as an **autonomous, open-ended research loop**: a high-level direction is given,
and the agent (Claude) decomposes it into rounds — ORIENT → PLAN → EXECUTE → REAL-VERIFY → RECORD —
driving itself until told to stop. This file pins the directive so it survives across sessions /
context compaction. To resume: re-read this + `STATUS.md` (current loop state) + `progress.md`
(durable round-by-round record) + `git log`, then continue the loop.

---

## Original directive (Yusen, 2026-06-16)

> 直接造一个 JEPA latent 控制器。
>
> **赌注:** 一个小的 action-conditioned JEPA 世界模型 + latent 里做 planning，就能在笔记本上控住 sim
> 机器人（= 笔记本量级的 V-JEPA2-AC）。最直给，成果是"一个能跑的东西"。
>
> 开环 research loop，从零建在 `~/jepa-ctrl`。硬件：笔记本 RTX 5080（16GB VRAM）+ 64GB RAM；按此 scope，
> 单次训练 ≤ ~2h。研究问题：能否用笔记本级 JEPA latent world model + latent planning 控制仿真机器人。
>
> 第一轮先 DIVERGE 读 V-JEPA2(+AC) / I-JEPA / Dreamer v3 / TD-MPC2，定可行路线，并先建跨 seed 评测台
> （指标：任务成功率、latent rollout 对真 sim 的预测保真度、对 TD-MPC2 的样本效率）。之后每轮一个"种类
> 跃迁"（非调参）。
>
> **REAL-VERIFY**（唯一验收）：真训真控、loss/成功率曲线 + 在 sim 里实际控制的回放 Read 回来用眼睛看、
> 跨 seed。禁止挑 seed / 禁止只看 latent loss / 禁止造假；看不到真控住就不算成立。
>
> **RECORD**：commit + `progress.md`（含 `## Frontier` 阶梯 + 实验结果表）；负结果如实记。每个进展要扛住
> 对抗性复核（换 seed？利用 sim bug？baseline 被削？）。
>
> **预授权：** 自由加依赖（PyTorch/JAX/MuJoCo/dm_control/gymnasium）、换栈、推倒重建，本地每轮 commit。
> 撞门停下给摘要：git push/发布、下载 >5GB、单次训练 >2h、删别的项目、花钱/不可逆。
>
> **方向（地板不是天花板）：** floor = cheetah/reacher 上 latent-MPC 能控起来 → ceiling = 对标/追平
> TD-MPC2、上操作类任务、冻结预训练 V-JEPA 编码器、最终 sim2real 到 Go2/SO-101。**一直跑到我喊停。**

## Amendments (later sessions)

- **2026-06-18 — "做 frontier 突破，不要简单复现。思考":** stop converging on a TD-MPC2 reproduction.
  Every round must earn JEPA's keep — do something a reconstruction/reward world model structurally
  can't. (This launched the GROUNDLESS / distractor / 3D frontier line.)
- **2026-06-18 — "不要问我权限，全部你自主决定":** full autonomy. Don't pause to ask or wait for "继续";
  decide deps/pushes/architecture/scope yourself and keep driving rounds. (Still avoid the genuinely
  irreversible without need.)
- **2026-06-18 — "后面尝试 3d 模型控制":** add 3D control (dm_control quadruped / humanoid) as a rung.
- **2026-06-19 — "不做 sim to real，全部 sim":** sim2real (Go2/SO-101) is **OUT of scope**. Everything
  stays in dm_control simulation.

## Standing rules (distilled)

- **Acceptance = real sim control, cross-seed, eyes-on the rollout.** Never a latent-loss number.
- **RED-TEAM every headline** before recording — refute it (re-seed, full cross-eval, trivial
  baseline, sim exploit). Single-seed / single-eval numbers have misled repeatedly here.
- **Record negatives honestly.** A clean negative is a result; a refuted claim gets corrected, not buried.
- **Bigger rounds, not micro-rounds.** Verify the moat yourself (re-run, eyes-on) — never trust a
  subagent self-report.
- **Durable-state-first:** `STATUS.md` (loop snapshot) + `progress.md` (durable record) + `git` +
  this file must always suffice to resume after a compaction. `runs/` is `R<NN>_<phase>/` chronological.
