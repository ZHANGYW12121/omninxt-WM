"""Pure latent-imagination Dreamer for the factorized UAV world model.

This package deliberately has no candidate generator, online planner,
behaviour-cloning teacher, or staged Event/Actor curriculum.  The existing
factorized perception and RSSM are reused, while training and deployment are
kept separate from the historical planning stack.
"""

from .trainer import PureDreamerConfig, PureDreamerTrainer

__all__ = ("PureDreamerConfig", "PureDreamerTrainer")
