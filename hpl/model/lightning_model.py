import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra
from hpl.model.lorenz96 import BaseSimulator
from typing import Any, Tuple


class LightningModel(pl.LightningModule):

    def __init__(
            self,
            model: DictConfig,
            encoder: DictConfig,
            optimizer: DictConfig,
            # rollout_length: int,
            # time_step: float,
    ):
        super().__init__()
        self.model: nn.Module | BaseSimulator = hydra.utils.instantiate(model)
        self.encoder: nn.Module = hydra.utils.instantiate(encoder)
        self.cfg_optimizer: DictConfig = optimizer

        # self.cfg_optimizer = optimizer
        # self.rollout_length = rollout_length
        # self.time_step = time_step

    def configure_optimizers(self) -> Any:
        params = [*self.encoder.parameters()]
        if hasattr(self.model, 'network'):
            if self.model.network is not None:  # if simulator is parametrized
                params += [*self.simulator.parameters()]
        optimizer = hydra.utils.instantiate(self.cfg_optimizer, params=params)
        return optimizer

    def rollout(self, ics: Tuple[torch.Tensor]) -> torch.Tensor:
        if isinstance(self.model, BaseSimulator):
            t = torch.arange(0, self.rollout_length*self.time_step, self.time_step)
            return self.model.integrate(t, ics)
        #TODO Do self iterations if model isn't simulator class

    def forward(self, feed_forward_input: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward(feed_forward_input)

    def training_step(self, batch, batch_index):
        feed_forward_input, observations = batch
        ics = self.forward(feed_forward_input)
        print(ics)
        pass
    #
    # def validation_step(self, batch, batch_index):
    #     #TODO: test parametrization model with observations and model error
    #     pass
    #
