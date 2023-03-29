from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class L96DataModule:
    _target_: str = "hpl.datamodule.DataLoader.L96DataModule"
    _recursive_: bool = False
    path: Optional[str] = None
    training_split: float = 1.0
    shuffle_train: bool = True
    shuffle_valid: bool = False
    batch_size: int = 1
    drop_last_batch: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    dataset: Any = None
    generator: Any = None


@dataclass
class L96OneGenerator:
    _target_: str = "hpl.utils.l96observations.generate_l96_1level_observations"
    f: float = 8
    k: int = 40
    dt: float = 0.01
    n_steps: int = 500
    spin_up_steps: int = 200
    sigma: float = 0
    missing: float = 0
    random_seed: int = 100


@dataclass
class L96TwoGenerator:
    _target_: str = "hpl.utils.l96observations.generate_l96_2level_observations"
    f: float = 10
    b: float = 10
    c: float = 1
    h: float = 10
    k: int = 40
    j: int = 10
    dt: float = 0.01
    n_steps: int = 500
    spin_up_steps: int = 500
    sigma: float = 0
    missing: float = 0
    random_seed: int = 100
