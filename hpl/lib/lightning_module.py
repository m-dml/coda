from dataclasses import dataclass
from typing import Any


@dataclass
class LitModule:
    _target_: str = "hpl.model.lightning_model.LightningModel"
    _recursive_: bool = False
    model: Any = None
    assimilation_network: Any = None
    rollout_length: int = 5
    time_step: float = 0.01
    loss: Any = None
    save_onnx_model: bool = False
