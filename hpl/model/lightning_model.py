from typing import Any, Union

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from mdml_tools.utils.logging import get_logger
from omegaconf import DictConfig

from hpl.model.lorenz96 import BaseSimulator


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model: DictConfig,
        assimilation_network: DictConfig,
        optimizer: DictConfig,
        rollout_length: int,
        time_step: float,
        loss: DictConfig,
        save_onnx_model: bool = True,
    ):
        super().__init__()
        self.model: Union[nn.Module, BaseSimulator] = hydra.utils.instantiate(model)
        self.assimilation_network: nn.Module = hydra.utils.instantiate(assimilation_network)
        self.cfg_optimizer: DictConfig = optimizer
        self.learning_rate: float = optimizer.lr
        self.rollout_length = rollout_length
        self.time_step = time_step
        self.loss_func = hydra.utils.instantiate(loss)
        self.save_onnx_model = save_onnx_model
        self.console_logger = get_logger(__name__)

    def configure_optimizers(self) -> Any:
        params = [*self.assimilation_network.parameters()]
        if hasattr(self.model, "network"):
            if self.model.network is not None:  # if simulator is parametrized
                params += [*self.model.parameters()]
        optimizer = hydra.utils.instantiate(self.cfg_optimizer, lr=self.learning_rate, params=params)
        return optimizer

    def rollout(self, ic: torch.Tensor) -> torch.Tensor:
        if isinstance(self.model, BaseSimulator):
            t = torch.arange(0, self.rollout_length * self.time_step, self.time_step)
            if ic.size(0) > 1:
                rollouts = []
                for the_ic in ic:
                    the_rollout = self.model.integrate(t, the_ic).squeeze()
                    rollouts.append(the_rollout)
                return torch.stack(rollouts, dim=0)  # stack rollouts over batch dim
            return self.model.integrate(t, ic).squeeze().unsqueeze(0)
        else:
            raise NotImplementedError("The model should be child of BaseSimulator")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.assimilation_network.forward(x)

    def do_step(self, batch: dict[str : torch.Tensor], stage: str = "Training") -> torch.Tensor:
        left_ff = batch["feedforward_left"]
        right_ff = batch["feedforward_right"]
        observations_data = batch["observations_data"]
        observations_mask = batch["observations_mask"]

        left_ics = self.forward(left_ff)
        right_ics = self.forward(right_ff)
        rollout = self.rollout(left_ics)

        if self.loss_func.use_model_term:
            loss_dict: dict = self.loss_func(
                prediction=[rollout, rollout[:, -1, :].unsqueeze(1)],
                target=[observations_data, right_ics],
                mask=observations_mask,
            )
        else:
            loss_dict: dict = self.loss_func(rollout, observations_data, observations_mask)

        for key, value in loss_dict.items():
            if value is not None:
                self.log(f"{key}/{stage}", value)
        return loss_dict["TotalLoss"]

    def training_step(self, batch, **kwargs):
        return self.do_step(batch, "Training")

    def validation_step(self, batch, *args, **kwargs):
        return self.do_step(batch, "Validation")
