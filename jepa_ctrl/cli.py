from __future__ import annotations

import argparse
import json

from .config import EvalConfig
from .evaluate import evaluate_controller


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser("jepa_ctrl.cli", description="cross-seed eval harness")
    p.add_argument("--task", default="cheetah-run", help="dm_control 'domain-task'")
    p.add_argument("--controller", default="random", choices=["random", "zero"])
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--action-repeat", type=int, default=2)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--outdir", default="runs")
    a = p.parse_args(argv)

    cfg = EvalConfig(
        task=a.task,
        controller=a.controller,
        seeds=tuple(int(s) for s in a.seeds.split(",") if s != ""),
        episodes=a.episodes,
        action_repeat=a.action_repeat,
        render=not a.no_render,
        outdir=a.outdir,
    )
    res = evaluate_controller(cfg)
    payload = {"task": cfg.task, "controller": cfg.controller, **res["aggregate"]}
    print(json.dumps(payload, indent=2))
    if res["any_truncated"]:
        print("WARNING: at least one episode hit the safety cap (truncated, not terminal)")
    for r in res["renders"]:
        print(f"seed{r['seed']} contact_sheet:", r["contact_sheet"])
    return res


if __name__ == "__main__":
    main()
