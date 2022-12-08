import torch
import torch.nn as nn


class WeakConstraintLoss(nn.Module):

    def __init__(self, alpha: float = 1):
        super(WeakConstraintLoss, self).__init__()
        self.alpha = alpha
        self.data_loss = None
        self.model_loss = None

    @staticmethod
    def masked_mse(x, y, m):
        return ((y - x)**2 * m).sum() / m.sum()

    def __call__(
            self,
            target_and_mask: torch.Tensor,
            rollout: torch.Tensor,
            ic_next: torch.Tensor,
    ):
        mask = target_and_mask[:, 1]
        target = target_and_mask[:, 0]
        self.data_loss = self.masked_mse(rollout, target, mask)

        ic_target = target_and_mask[0, 0, ..., -1]
        ic_mask = target_and_mask[0, 1, ..., -1]
        self.model_loss = self.masked_mse(ic_next, ic_target, ic_mask)
        loss = self.data_loss + self.alpha * self.model_loss
        return loss
