import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch.random

from hpl.lib.config import register_configs
from hpl.utils import get_logger, log_hyperparameters
from hpl.utils.tb_logging_handler import set_tb_logger

register_configs()


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.2.0")
def main(cfg: omegaconf.DictConfig):
    log = get_logger()
    log.info(f"Hydra version: {hydra.__version__}")

    if cfg.random_seed:
        log.info(f"Applying random seed: {cfg.random_seed}")
        torch.random.manual_seed(cfg.random_seed)
        pl.seed_everything(cfg.random_seed)
        np.random.seed(cfg.random_seed)

    cfg_logger: list = []
    if "logger" in cfg:
        for _, lg_conf in cfg["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                cfg_logger.append(hydra.utils.instantiate(lg_conf))

    for this_logger in cfg_logger:
        if "tensorboard" in str(this_logger):
            set_tb_logger(this_logger)

    lit_model: pl.LightningModule = hydra.utils.instantiate(
        cfg.lightning_module,
        model=cfg.model,
        encoder=cfg.da_encoder,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    )

    callbacks: list = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    log.info("Logging hparams to tensorboard")
    hydra_params = log_hyperparameters(config=cfg, model=lit_model)
    for this_logger in cfg_logger:
        if "tensorboard" in str(this_logger):
            log.info("Add hparams to tensorboard")
            this_logger.log_hyperparams(hydra_params, {"hp/loss": 0, "hp/accuracy": 0, "hp/epoch": 0})
        else:
            this_logger.log_hyperparams(hydra_params)

    log.info(f"Instantiating <{cfg.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    # datamodule.setup()

    log.info(f"Instantiating <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)
    log.info("Training...")
    trainer.fit(model=lit_model, datamodule=datamodule)
    log.info("Training finished")


if __name__ == "__main__":
    main()
