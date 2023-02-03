from dataclasses import dataclass


@dataclass
class Adam:
    _target_: str = "torch.optim.Adam"
    lr: float = 0.001
    betas: list = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: int = 0
    amsgrad: bool = False
    maximize: bool = False
