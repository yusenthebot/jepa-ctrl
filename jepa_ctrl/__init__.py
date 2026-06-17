"""jepa-ctrl: laptop-scale action-conditioned JEPA latent world model + latent planning.

Round 1 ships the architecture-agnostic evaluation harness (envs, controllers,
metrics, render, eval loop) + a random baseline. The JEPA world model + MPPI
planner land in round 2 and plug into the `Controller` interface.
"""

__version__ = "0.1.0"
