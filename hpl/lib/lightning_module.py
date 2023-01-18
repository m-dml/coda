from dataclasses import dataclass
from typing import List, Any
from hydra.conf import MISSING


@dataclass
class LitModule:
    _target_: str = "hpl.model.lightning_model.LightningModel"
    _recursive_: bool = False
    model: Any = None
    encoder: Any = None
    rollout_length: int = 5
    time_step: float = 0.01
    loss: Any = None
