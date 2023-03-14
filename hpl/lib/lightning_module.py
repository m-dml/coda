from dataclasses import dataclass
from typing import Any


@dataclass
class DataAssimilationModule:
    _target_: str = "hpl.model.lightning_model.DataAssimilationModule"
    _recursive_: bool = False
    model: Any = None
    assimilation_network: Any = None
    rollout_length: int = 5
    time_step: float = 0.01
    loss: Any = None


@dataclass
class ParameterTuningModule:
    _target_: str = "hpl.model.lightning_model.ParameterTuningModule"
    _recursive_: bool = False
    model: Any = None
    assimilation_network: Any = None
    rollout_length: int = 5
    time_step: float = 0.01
    loss: Any = None
