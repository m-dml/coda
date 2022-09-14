import pytorch_lightning as pl
import hydra
import torch
import torch.nn as nn
import logging


class MaskedMSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
        if mask is None:
            return ((x - y) ** 2).mean()
        return (((x - y) ** 2) * mask).sum() / mask.sum()


class DataAssimilationModule(pl.LightningModule):

    def __init__(self, model, assimilation_model, optimizer, chunk_size=10, alpha=1):
        super().__init__()
        self.model = model
        self.assimilation_model = assimilation_model
        self.cfg_optimizer = optimizer
        self.chunk_size = chunk_size
        self.alpha = alpha
        self.objective = MaskedMSE()

    def configure_optimizers(self):
        logging.info(f"Instantiating <{self.cfg_optimizer._target_}>")
        optimizer = hydra.utils.instantiate(self.cfg_optimizer, params=self.assimilation_model.parameters())
        return optimizer

    def forward(self, ic, **kwargs):
        rollout = [ic]
        for i in range(self.chunk_size):
            x = self.model.forward(rollout[-1], **kwargs)
            rollout.append(x)
        return torch.cat(rollout, 0)

    def training_step(self, batch, batch_index):
        neighbors, target, target_mask, params = batch
        neighbors = neighbors.squeeze()
        target = target.squeeze()
        target_mask = target_mask.squeeze()
        print(neighbors[0].shape)
        ic_left = self.assimilation_model.forward(neighbors[0])
        ic_right = self.assimilation_model.forward(neighbors[1])

        rollout = self.forward(ic_left.unsqueeze(0), **params)
        rollout = torch.permute(rollout, (1, -1, 0))

        model_loss = self.objective(ic_right, rollout[..., -1])
        data_loss = self.objective(target[0], rollout[..., :-1].squeeze(), target_mask[0])

        rollout = self.forward(ic_right.unsqueeze(0), **params)
        rollout = torch.permute(rollout, (1, -1, 0))
        data_loss += self.objective(target[1], rollout[..., :-1].squeeze(), target_mask[1])

        loss = data_loss + self.alpha * model_loss
        self.log('total_loss/train', loss)
        self.log('data_loss/train', data_loss)
        self.log('model_loss/train', model_loss)
        return loss

    def validation_step(self, batch, batch_index):
        neighbors, target, target_mask, params = batch
        neighbors = neighbors.squeeze()
        target = target.squeeze()
        target_mask = target_mask.squeeze()

        ic_left = self.assimilation_model.forward(neighbors[0])
        ic_right = self.assimilation_model.forward(neighbors[1])

        rollout = self.forward(ic_left.unsqueeze(0), **params)
        rollout = torch.permute(rollout, (1, -1, 0))

        model_loss = self.objective(ic_right, rollout[..., -1])
        data_loss = self.objective(target[0], rollout[..., :-1].squeeze(), target_mask[0])

        rollout = self.forward(ic_right.unsqueeze(0), **params)
        rollout = torch.permute(rollout, (1, -1, 0))
        data_loss += self.objective(target[1], rollout[..., :-1].squeeze(), target_mask[1])

        loss = data_loss + self.alpha * model_loss
        self.log('total_loss/valid', loss)
        self.log('data_loss/valid', data_loss)
        self.log('model_loss/valid', model_loss)
        return loss
