from dataclasses import dataclass
from typing import List, Any
from hydra.conf import MISSING


@dataclass
class LitModule:
    _target_: str = "hpl.model.lightning_model.LightningModel"
    _recursive_: bool = False
    model: Any = None
    encoder: Any = None
    # chunk_size: int = 2

