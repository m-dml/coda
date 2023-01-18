from dataclasses import dataclass


@dataclass
class WeakConstraintLoss:
    _target_:  str = "hpl.utils.Loss4DVar.WeakConstraintLoss"
    alpha: int = 1


@dataclass
class StrongConstraintLoss:
    _target_: str = "hpl.utils.Loss4DVar.StrongConstraintLoss"
