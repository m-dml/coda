import logging
import hydra
import omegaconf
import torch.nn as nn
import pytorch_lightning as pl

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path='conf', config_name='config.yaml')
def main(cfg: omegaconf.DictConfig):

    print(cfg.datamodule.path)
    if cfg.random_seed:
        logging.info(f"Appling random seed: {cfg.random_seed}")
        pl.seed_everything(cfg.random_seed)

    logging.info(f"Instantiating <{cfg.model._target_}>")
    model: nn.Module = hydra.utils.instantiate(cfg.model)

    logging.info(f"Instantiating <{cfg.assimilator._target_}>")
    assimilator: nn.Module = hydra.utils.instantiate(cfg.assimilator)
    print(assimilator)

    logger: list = []
    logging.info(f"Instantiating logger <{cfg.logger._target_}>")
    logger.append(hydra.utils.instantiate(cfg.logger))

    callbacks: list = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg["callbacks"].items():
            if "_target_" in cb_conf:
                logging.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    logging.info(f"Instantiating <{cfg.lightning_module._target_}>")
    lit_model: pl.LightningModule = hydra.utils.instantiate(
        cfg.lightning_module,
        model=model,
        assimilation_model=assimilator,
        optimizer=cfg.optimizer,
    )

    logging.info(f"Instantiating <{cfg.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()

    logging.info(f"Instantiating <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)
    trainer.fit(model=lit_model, datamodule=datamodule)


if __name__ == '__main__':
    main()
