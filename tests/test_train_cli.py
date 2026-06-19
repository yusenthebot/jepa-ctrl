"""R15: the exploration-schedule knobs must be reachable from the CLI so a one-knob
controlled training change (raise the late-training exploration floor) is a flag, not an
edit. Guards that --explore-std / --explore-std-end / --explore-anneal-steps parse and flow
verbatim into TrainConfig (the schedule logic itself is covered by test_trainer_schedule)."""

from __future__ import annotations

from scripts.train import build_parser, train_config_from_args


def test_explore_flags_default_to_trainconfig_defaults():
    a = build_parser().parse_args(["--task", "quadruped-walk"])
    cfg = train_config_from_args(a, pixels=False)
    assert cfg.explore_std == 0.3
    assert cfg.explore_std_end == 0.05  # current annealed floor
    assert cfg.explore_anneal_steps == 100_000


def test_explore_std_end_override_flows_to_trainconfig():
    a = build_parser().parse_args(
        ["--task", "quadruped-walk", "--grounding", "sigreg", "--sigreg-coef", "0",
         "--explore-std-end", "0.2"]
    )
    cfg = train_config_from_args(a, pixels=False)
    assert cfg.explore_std_end == 0.2  # the R15 one-knob change
    assert cfg.explore_std == 0.3  # start unchanged -> isolates the floor
    assert cfg.grounding == "sigreg" and cfg.sigreg_coef == 0.0  # reward-free raw recipe
