from dataclasses import dataclass


@dataclass
class WeakConstraintLoss:
    _target_:  str = "hpl.utils.Loss4DVar.WeakConstraintLoss"
    alphas: int = 1


@dataclass
class StrongConstraintLoss:
    _target_: str = "hpl.utils.Loss4DVar.StrongConstraintLoss"
