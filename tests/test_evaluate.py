from __future__ import annotations

from jepa_ctrl.config import EvalConfig
from jepa_ctrl.evaluate import evaluate_controller


def test_evaluate_zero_controller_no_render(tmp_path):
    cfg = EvalConfig(
        task="cartpole-swingup",
        controller="zero",
        seeds=(0, 1),
        episodes=1,
        action_repeat=2,
        render=False,
        outdir=str(tmp_path),
    )
    res = evaluate_controller(cfg)
    agg = res["aggregate"]
    assert agg["n_seeds"] == 2
    assert agg["low_confidence"] is False
    # cartpole return is in [0, 1000]; zero action gives a low-ish but finite return
    assert 0.0 <= agg["mean"] <= 1000.0
    assert res["renders"] == []
    assert res["any_truncated"] is False  # 1000/2 = 500 env-steps << 2000 cap
