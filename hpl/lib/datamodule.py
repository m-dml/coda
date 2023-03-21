from dataclasses import dataclass
from typing import Any

from hydra.conf import MISSING


@dataclass
class L96DataModule:
    _target_: str = "hpl.datamodule.DataLoader.L96DataModule"
    _recursive_: bool = False
    path: str = MISSING
    training_split: float = 1.0
    shuffle_train: bool = True
    shuffle_valid: bool = False
    batch_size: int = 1
    drop_last_batch: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    dataset: Any = None
