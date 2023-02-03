from dataclasses import dataclass

from hydra.conf import MISSING


@dataclass
class L96DataModule:
    _target_: str = "hpl.datamodule.DataLoader.L96DataModule"
    path: str = MISSING
    chunk_size: int = 2
    window: tuple[int, int] = (25, 25)
    training_split: float = 1.0
    shuffle_train: bool = True
    shuffle_valid: bool = False
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
