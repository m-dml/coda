import torch
import torch.nn as nn


class BaseLossDA(nn.Module):
    """Abstract data assimilation loss
    Implements some basic functions for working with observations
    """
    @staticmethod
    def masked_mse(x, y, m=None):
        if m is None:
            return ((y - x)**2).mean()
        return ((y - x)**2 * m).sum() / m.sum()


class WeakConstraintLoss(BaseLossDA):
    """Weak Constraint 4DVar loss
    Implementation of 4DVar objective function to train models consistent
    with observations and itself
        Args:
            alpha (float): model loss scaling parameter
    """
    def __init__(self, alpha: float = 1):

        super(WeakConstraintLoss, self).__init__()
        self.alpha = alpha
        self.data_loss = None
        self.model_loss = None

    def __call__(
            self,
            target_and_mask: torch.Tensor,
            rollout: torch.Tensor,
            ic_next: torch.Tensor,
    ):
        pass
        # mask = target_and_mask[:, 1, ...]
        # target = target_and_mask[:, 0, ...]
        # self.data_loss = self.masked_mse(rollout, target, mask)
        #
        # ic_predicted = rollout[0, ..., -1].unsqueeze(0)
        # self.model_loss = self.masked_mse(ic_next, ic_predicted)
        # loss = self.data_loss + self.alpha * self.model_loss
        # return loss
