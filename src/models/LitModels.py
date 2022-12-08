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
        for i in range(self.chunk_size-1):
            x = self.model.forward(rollout[-1], **kwargs)
            rollout.append(x)
        return torch.cat(rollout, 0)

    def training_step(self, batch, batch_index):
        ff_input, target, params = batch
        ff_input = ff_input.squeeze()
        target = target.squeeze()
        ic = self.assimilation_model.forward(ff_input[0].unsqueeze(0))
        ic_next = self.assimilation_model.forward(ff_input[1].unsqueeze(0))

        rollout = self.forward(ic, **params)
        rollout = torch.permute(rollout, (1, -1, 0))
        model_loss = self.objective(ic_next, rollout[..., -1])
        data_loss = self.objective(target[0, 0], rollout.squeeze(), target[0, 1])

        rollout = self.forward(ic_next, **params)
        rollout = torch.permute(rollout, (1, -1, 0))
        data_loss += self.objective(target[1, 0], rollout.squeeze(), target[1, 1])

        loss = data_loss + self.alpha * model_loss
        self.log('total_loss/train', loss)
        self.log('data_loss/train', data_loss)
        self.log('model_loss/train', model_loss)
        return loss

    def on_save_checkpoint(self, checkpoint):
        state_dict = self.assimilation_model.state_dict()
        torch.save(state_dict, "assimilator.ckpt")

    def validation_step(self, batch, batch_index):
        ff_input, target, params = batch
        ff_input = ff_input.squeeze()
        target = target.squeeze()
        ic = self.assimilation_model.forward(ff_input[0].unsqueeze(0))
        ic_next = self.assimilation_model.forward(ff_input[1].unsqueeze(0))

        rollout = self.forward(ic, **params)
        rollout = torch.permute(rollout, (1, -1, 0))
        model_loss = self.objective(ic_next, rollout[..., -1])
        data_loss = self.objective(target[0, 0], rollout.squeeze(), target[0, 1])

        rollout = self.forward(ic_next, **params)
        rollout = torch.permute(rollout, (1, -1, 0))
        data_loss += self.objective(target[1, 0], rollout.squeeze(), target[1, 1])

        loss = data_loss + self.alpha * model_loss
        self.log('total_loss/valid', loss)
        self.log('data_loss/valid', data_loss)
        self.log('model_loss/valid', model_loss)
        return loss
