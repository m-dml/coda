from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass
class L96Parametrized:
    """Hydra config for Lorenz 96 simulator parametrized by Neural Network."""

    _target_: str = "hpl.model.lorenz96.L96Parametrized"
    _recursive_: bool = True
    forcing: Any = 8
    parametrization: Any = MISSING
    method: str = "rk4"
    options: Any = None
