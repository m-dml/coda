from dataclasses import dataclass


@dataclass
class CheckpointCallback:
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    monitor: str = "TotalLoss/Validation"
    save_top_k: int = 10
    save_last: bool = True
    mode: str = "min"
    verbose: bool = False
    dirpath: str = "./logs/checkpoints/"  # use  relative path, so it can be adjusted by hydra
    filename: str = "{epoch:02d}"


@dataclass
class EarlyStoppingCallback:
    _target_: str = "pytorch_lightning.callbacks.EarlyStopping"
    monitor: str = "TotalLoss/Validation"
    min_delta: float = 0.00
    patience: int = 5  # set default arbitrarily high, so it won't be triggered accidentally.
    verbose: bool = True
    mode: str = "min"
