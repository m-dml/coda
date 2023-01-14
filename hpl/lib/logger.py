from dataclasses import dataclass
from typing import List, Any
from hydra.conf import MISSING


@dataclass
class TensorBoardLogger:
    _target_: str = "pytorch_lightning.loggers.TensorBoardLogger"
    save_dir: str = "./outputs/tensorboard"
    default_hp_metric: bool = False
    log_graph: bool = True
    version: str = "all"
