from typing import Union

import torch


class Four4DVarLoss:
    """4DVar loss function.

    Args:
        use_model_term (bool): calculate missmatch between sub-windows
        alpha (float): model error scaler
    """

    def __init__(
        self,
        use_model_term: bool = False,
        alpha: float = None,
    ):
        self.use_model_term = use_model_term
        self.alpha = alpha
        self.device = None

    def __call__(
        self,
        data: list[torch.Tensor],
        target: list[torch.Tensor],
        mask: torch.Tensor = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor]]:
        self.device = data[0].device
        loss = torch.zeros(1, device=self.device)
        loss += self._calculate_data_loss(data[0], target[0], mask)
        data_loss = loss.detach().clone()
        if self.use_model_term:
            model_loss = self._calculate_model_loss(data[1], target[1])
            loss += model_loss
            return data_loss, model_loss, loss
        return loss

    @staticmethod
    def _calculate_data_loss(
        data: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if mask is not None:
            data = torch.masked_select(data, mask)
            target = torch.masked_select(target, mask)
        return torch.nn.functional.mse_loss(data, target)

    def _calculate_model_loss(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
    ):
        alpha = self.alpha
        if alpha is None:
            alpha = 1 / (data - target).var()
        return torch.nn.functional.mse_loss(data, target) * alpha
