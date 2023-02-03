import logging

import pytorch_lightning as pl
from omegaconf import DictConfig


def get_logger(name=__name__, level="INFO") -> logging.Logger:
    log = logging.getLogger(name)
    level_obj = logging.getLevelName(level)
    log.setLevel(level_obj)
    return log


def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
) -> dict:
    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["lightning_module"]
    hparams["datamodule"] = config["datamodule"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params_not_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    for key, value in config.items():
        if key not in hparams:
            hparams[key] = value

    return hparams
