from dataclasses import dataclass
from typing import Union


@dataclass
class Four4DVarLoss:
    _target_: str = "hpl.utils.Loss4DVar.Four4DVarLoss"
    alpha: Union[float] = 0
