import os
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

    def rollout(self, ics: tuple[torch.Tensor]) -> torch.Tensor:
        if isinstance(self.model, BaseSimulator):
            t = torch.arange(0, self.rollout_length * self.time_step, self.time_step)
            return self.model.integrate(t, ics).squeeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.assimilation_network.forward(x)

    def do_step(self, batch, stage: str = "Training"):
        left_ff = batch["feedforward_left"]
        right_ff = batch["feedforward_right"]
        observations_data = batch["observations_data"]
        observations_mask = batch["observations_mask"]

        if self.global_step < 1:
            self.example_input_array = left_ff

        left_ics = self.forward(left_ff)
        right_ics = self.forward(right_ff)
        rollout = self.rollout(left_ics)

        if self.loss_func.use_model_term:
            data_loss, model_loss, loss = self.loss_func(
                [rollout, right_ics.squeeze()],
                [observations_data, rollout[..., -1, :].squeeze()],
                observations_mask,
            )
            self.log(f"DataLoss/{stage}", data_loss)
            self.log(f"ModelLoss/{stage}", model_loss)
        else:
            loss = self.loss_func([rollout], observations_data, observations_mask)
        self.log(f"TotalLoss/{stage}", loss)
        return loss

    def training_step(self, batch, **kwargs):
        return self.do_step(batch, "Training")

    def validation_step(self, batch, *args, **kwargs):
        return self.do_step(batch, "Validation")

    def on_save_checkpoint(self, *args, **kwargs) -> None:
        # save model to onnx:
        if self.global_step > 0 and self.save_onnx_model:
            folder = self.trainer.checkpoint_callback.dirpath
            if not os.path.exists(folder):
                os.makedirs(folder)
            onnx_file = os.path.join(folder, f"danet_{self.global_step}.onnx")
            torch.onnx.export(
                model=self.assimilation_network.float(),
                args=self.example_input_array,
                f=onnx_file,
                opset_version=17,
                verbose=False,
                export_params=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )
            self.console_logger.info(f"Model saved to {onnx_file}")
