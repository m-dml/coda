from dataclasses import dataclass
from typing import Any

from hydra.conf import ConfigStore, MISSING
from mdml_tools.utils import add_hydra_models_to_config_store

from hpl.lib.da_encoder import Unet
from hpl.lib.datamodule import L96DataModule
from hpl.lib.lightning_module import DataAssimilationModule, ParameterTuningModule
from hpl.lib.loss import Four4DVarLoss
from hpl.lib.model import Lorenz96Base


def register_configs() -> None:
    cs = ConfigStore.instance()
    # add hydra models from mdml_tools
    rename_groups = {
        "optimizer": ["optimizer/data_assimilation", "optimizer/parametrization"],
        "models": "model/network",
        # model/network/fully_connected_model/fully_connected_model_base
        # models/fully_connected_model
    }
    add_hydra_models_to_config_store(cs, rename_groups)

    # loss:
    cs.store(name="4dvar", node=Four4DVarLoss, group="loss")

    # model:
    model_group = "model"
    cs.store(name="l96_base", node=Lorenz96Base, group=model_group)

    # encoder:
    cs.store(name="unet", node=Unet, group="assimilation_network")

    # lightning module:
    cs.store(name="data_assimilation_module", node=DataAssimilationModule, group="lightning_module")
    cs.store(name="parameter_tuning_module", node=ParameterTuningModule, group="lightning_module")

    # datamodule
    cs.store(name="l96_datamodule", node=L96DataModule, group="datamodule")

    # register the base config class (this name has to be called in config.yaml):
    cs.store(name="base_config", node=Config)


@dataclass
class Config:
    output_dir_base_path: str = MISSING
    print_config: bool = True
    random_seed: int = 101

    model: Any = MISSING
    assimilation_network: Any = MISSING
    loss: Any = MISSING
    lightning_module: Any = MISSING
    optimizer: Any = MISSING

    datamodule: Any = MISSING
    lightning_trainer: Any = MISSING
    lightning_logger: Any = MISSING
    lightning_callback: Any = None
