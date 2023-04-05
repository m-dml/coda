from typing import Any

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from mdml_tools.simulators.base import BaseSimulator
from mdml_tools.utils.logging import get_logger
from omegaconf import DictConfig


class BaseLightningModel(pl.LightningModule):
    """Base Lightning Module. This module shares common functionality for three tasks:

    - Data Assimilation
    - Patameter Tining
    - Parametrization Learning
    """

    def __init__(
        self,
        simulator: DictConfig,
        assimilation_network: DictConfig,
        loss: DictConfig,
        rollout_length: int,
        time_step: int,
    ):
        super().__init__()
        self.console_logger = get_logger(__name__)
        self.console_logger.info(f"Instantiating simulator <{simulator._target_}>")
        self.simulator: BaseSimulator = hydra.utils.instantiate(simulator)
        self.console_logger.info(f"Instantiating assimilation network <{assimilation_network._target_}>")
        self.assimilation_network: nn.Module = hydra.utils.instantiate(assimilation_network)
        self.console_logger.info(f"Instantiating loss function <{loss._target_}>")
        self.loss_function = hydra.utils.instantiate(loss)

        self.rollout_length = rollout_length
        self.time_step = time_step

    def rollout(self, ic: torch.Tensor) -> torch.Tensor:
        if isinstance(self.simulator, BaseSimulator):
            t = torch.arange(0, self.rollout_length * self.time_step, self.time_step)
            return self.simulator.integrate(t, ic)
        else:
            raise NotImplementedError("The simulator should be child of BaseSimulator class")

    def do_step(self, batch: dict[str : torch.Tensor], stage: str = "Training") -> torch.Tensor:
        left_ff = batch["feedforward_left"]
        right_ff = batch["feedforward_right"]
        observations_data = batch["observations_data"]
        observations_mask = batch["observations_mask"]

        left_ics = self.assimilation_network.forward(left_ff)
        right_ics = self.assimilation_network.forward(right_ff)
        rollout = self.rollout(left_ics.squeeze(1))

        if self.loss_function.use_model_term:
            loss_dict: dict = self.loss_function(
                prediction=[rollout, rollout[:, -1, :].unsqueeze(1)],
                target=[observations_data, right_ics],
                mask=observations_mask,
            )
        else:
            loss_dict: dict = self.loss_function(rollout, observations_data, observations_mask)

        for key, value in loss_dict.items():
            if value is not None:
                self.log(f"{key}/{stage}", value)
        return loss_dict["TotalLoss"]


class DataAssimilationModule(BaseLightningModel):
    def __init__(
        self,
        simulator: DictConfig,
        assimilation_network: DictConfig,
        optimizer: DictConfig,
        loss: DictConfig,
        rollout_length: int,
        time_step: int,
    ):
        super().__init__(simulator, assimilation_network, loss, rollout_length, time_step)
        self.cfg_optimizer: DictConfig = optimizer.data_assimilation

    def configure_optimizers(self) -> Any:
        params = [*self.assimilation_network.parameters()]
        optimizer = hydra.utils.instantiate(self.cfg_optimizer, params=params)
        return optimizer

    def training_step(self, batch, **kwargs):
        return self.do_step(batch, "Training")

    def validation_step(self, batch, *args, **kwargs):
        return self.do_step(batch, "Validation")


# class ParameterTuningModule(BaseLightningModel):
#     def __init__(
#         self,
#         simulator: DictConfig,
#         assimilation_network: DictConfig,
#         optimizer: DictConfig,
#         loss: DictConfig,
#         rollout_length: int,
#         time_step: int,
#     ):
#         super().__init__(simulator, assimilation_network, loss, rollout_length, time_step)
#         self.cfg_optimizer_da: DictConfig = optimizer.data_assimilation
#         self.cfg_optimizer_param: DictConfig = optimizer.parametrization
#         self.automatic_optimization = False
#
#     def configure_optimizers(self) -> Any:
#         params_da = [*self.assimilation_network.parameters()]
#         optimizer_da = hydra.utils.instantiate(self.cfg_optimizer_da, params=params_da)
#         params_param = [*self.simulator.parameters()]
#         optimizer_param = hydra.utils.instantiate(self.cfg_optimizer_param, params=params_param)
#         return optimizer_da, optimizer_param
#
#     def training_step(self, batch, **kwargs):
#         optimizers = self.optimizers()
#         loss = self.do_step(batch, "Training")
#         for optimizer in optimizers:
#             optimizer.zero_grad()
#         self.manual_backward(loss)
#         for optimizer in optimizers:
#             optimizer.step()
#
#         free_parameter = self.simulator.f
#         if isinstance(free_parameter, nn.Parameter):
#             self.log("Parameter/Training", free_parameter)
#
#     def validation_step(self, batch, *args, **kwargs):
#         self.do_step(batch, "Validation")
#         free_parameter = self.simulator.f
#         if isinstance(free_parameter, nn.Parameter):
#             self.log("Parameter/Validation", free_parameter)
