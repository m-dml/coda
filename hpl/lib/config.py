from dataclasses import dataclass
from typing import Any
from hydra.conf import ConfigStore, MISSING

from hpl.lib.model import Lorenz96Base
from hpl.lib.da_encoder import Unet
from hpl.lib.lightning_module import LitModule

from hpl.lib.logger import TensorBoardLogger
from hpl.lib.callbacks import CheckpointCallback, EarlyStoppingCallback
from hpl.lib.optimizer import Adam
from hpl.lib.datamodule import L96DataModule
from hpl.lib.trainer import Trainer
from hpl.lib.loss import WeakConstraintLoss, StrongConstraintLoss


def register_configs() -> None:
    cs = ConfigStore.instance()

    # model:
    model_group = "model"
    cs.store(name="l96_base", node=Lorenz96Base, group=model_group)

    # encoder:
    cs.store(name="unet", node=Unet, group="da_encoder")

    # lightning module:
    cs.store(name="base_lightning_module", node=LitModule, group="lightning_module")

    # loss
    cs.store(name="weak_4dvar", node=WeakConstraintLoss, group="loss")
    cs.store(name="strong_4dvar", node=StrongConstraintLoss, group="loss")

    # optimizer
    optimizer_group = "optimizer"
    cs.store(name="adam", node=Adam, group=optimizer_group)

    # datamodule
    cs.store(name="l96_datamodule", node=L96DataModule, group="datamodule")

    # trainer
    cs.store(name="base_trainer", node=Trainer, group="trainer")

    # logger:
    cs.store(name="tensorboard", node=TensorBoardLogger, group="logger/tensorboard")

    # callbacks:
    cs.store(name="model_checkpoint", node=CheckpointCallback, group="callbacks/checkpoint")
    cs.store(name="early_stopping", node=EarlyStoppingCallback, group="callbacks/early_stopping")

    # register the base config class (this name has to be called in config.yaml):
    cs.store(name="base_config", node=Config)


@dataclass
class Config:
    output_dir_base_path: str = MISSING
    random_seed: int = 101

    model: Any = MISSING
    da_encoder: Any = MISSING
    loss: Any = MISSING
    lightning_module: Any = MISSING
    optimizer: Any = MISSING

    datamodule: Any = MISSING
    trainer: Trainer = MISSING
    logger: Any = MISSING
    callbacks: Any = MISSING
