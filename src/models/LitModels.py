import pytorch_lightning as pl
import hydra
import torch
import torch.nn as nn
import logging
from src.utils.Loss4DVar import WeakConstraintLoss


class DataAssimilationModule(pl.LightningModule):

    def __init__(self, model, assimilation_model, optimizer, chunk_size=10, alpha=1):
        super().__init__()
        self.model = model
        self.assimilation_model = assimilation_model
        self.cfg_optimizer = optimizer
        self.chunk_size = chunk_size
        self.objective = WeakConstraintLoss(alpha)

    def configure_optimizers(self):
        logging.info(f"Instantiating <{self.cfg_optimizer._target_}>")
        optimizer = hydra.utils.instantiate(self.cfg_optimizer, params=self.assimilation_model.parameters())
        return optimizer

    def get_rollout(self, ic, **kwargs):
        rollout = [ic]
        for i in range(self.chunk_size - 1):
            x = self.model.forward(rollout[-1], **kwargs)
            rollout.append(x)
        return torch.stack(rollout, -1).squeeze()

    def forward(self, batch):
        ff_input, target, params = batch
        ff_input = ff_input.squeeze()
        target = target.squeeze()
        initial_conditions = self.assimilation_model.forward(ff_input)
        rollout_combined = []
        for ic in initial_conditions:
            rollout = self.get_rollout(ic.unsqueeze(0), **params)
            rollout_combined.append(rollout)
        rollout_combined = torch.stack(rollout_combined, 0)
        return initial_conditions[-1], rollout_combined

    def training_step(self, batch, batch_index):
        _, target, _ = batch
        ic_next, rollout_combined = self.forward(batch)
        loss = self.objective(target, rollout_combined, ic_next)

        self.log('total_loss/train', loss)
        self.log('data_loss/train', self.objective.data_loss)
        self.log('model_loss/train', self.objective.model_loss)
        return loss

    def validation_step(self, batch, batch_index):
        _, target, _ = batch
        ic_next, rollout_combined = self.forward(batch)
        loss = self.objective(target, rollout_combined, ic_next)

        self.log('total_loss/valid', loss)
        self.log('data_loss/valid', self.objective.data_loss)
        self.log('model_loss/valid', self.objective.model_loss)
        return loss
