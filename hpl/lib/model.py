from dataclasses import dataclass
from typing import Any


@dataclass
class Lorenz96Base:
    """Hydra config for Lorenz 96 model."""

    _target_: str = "hpl.model.lorenz96.L96Simulator"
    _recursive_: bool = False
    f: Any = 8
    network: Any = None
    method: str = "rk4"
    options: Any = None
