from typing import Any, Union

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

from hpl.model.lorenz96 import BaseSimulator
from hpl.utils.Loss4DVar import WeakConstraintLoss


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model: DictConfig,
        encoder: DictConfig,
        optimizer: DictConfig,
        rollout_length: int,
        time_step: float,
        loss: DictConfig,
    ):
        super().__init__()
        self.model: Union[nn.Module, BaseSimulator] = hydra.utils.instantiate(model)
        self.encoder: nn.Module = hydra.utils.instantiate(encoder)
        self.cfg_optimizer: DictConfig = optimizer

        self.rollout_length = rollout_length
        self.time_step = time_step
        self.loss_func = hydra.utils.instantiate(loss)
        if isinstance(self.loss_func, WeakConstraintLoss):
            self.use_model_loss = True
        self.use_model_loss = False

    def configure_optimizers(self) -> Any:
        params = [*self.encoder.parameters()]
        if hasattr(self.model, "network"):
            if self.model.network is not None:  # if simulator is parametrized
                params += [*self.simulator.parameters()]
        optimizer = hydra.utils.instantiate(self.cfg_optimizer, params=params)
        return optimizer

    def rollout(self, ics: tuple[torch.Tensor]) -> torch.Tensor:
        if isinstance(self.model, BaseSimulator):
            t = torch.arange(0, self.rollout_length * self.time_step, self.time_step)
            return self.model.integrate(t, ics).squeeze()

    def forward(self, feed_forward_input: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward(feed_forward_input)

    def _do_step(self, batch):
        left_ff, right_ff, observations = batch
        left_ics = self.forward(left_ff)
        right_ics = self.forward(right_ff)
        rollout = self.rollout(left_ics)
        return left_ics, right_ics, rollout

    def training_step(self, batch, **kwargs):
        _, _, observations = batch
        left_ics, right_ics, rollout = self._do_step(batch)
        if self.use_model_loss:
            loss = self.loss_func(observations, rollout, right_ics)
            self.log("TotalLoss/Training", loss)
            self.log("DataLoss/Training", self.loss_func.data_loss)
            self.log("ModelLoss/Training", self.loss_func.model_loss)
        else:
            loss = self.loss_func(observations, rollout)
            self.log("TotalLoss/Training", loss)
        return loss

    def validation_step(self, batch, **kwargs):
        _, _, observations = batch
        left_ics, right_ics, rollout = self._do_step(batch)
        if self.use_model_loss:
            loss = self.loss_func(observations, rollout, right_ics)
            self.log("TotalLoss/Validation", loss)
            self.log("DataLoss/Validation", self.loss_func.data_loss)
            self.log("ModelLoss/Validation", self.loss_func.model_loss)
        else:
            loss = self.loss_func(observations, rollout)
            self.log("TotalLoss/Validation", loss)
        return loss
