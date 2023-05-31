from dataclasses import dataclass
from typing import Any

from hydra.conf import ConfigStore, MISSING
from mdml_tools.utils import add_hydra_models_to_config_store

from hpl.lib.da_encoder import Unet
from hpl.lib.datamodule import L96DataModule, L96Dataset, L96InferenceDataset
from hpl.lib.lightning_module import DataAssimilationModule, ParameterTuningModule, ParametrizationLearningModule
from hpl.lib.loss import Four4DVarLoss
from hpl.lib.model import L96Parametrized


def register_configs() -> None:
    cs = ConfigStore.instance()
    # add hydra models from mdml_tools
    rename_groups = {
        "optimizer": ["optimizer/data_assimilation", "optimizer/parametrization"],
        "simulator": ["simulator", "datamodule/simulator"],
        "model": "simulator/parametrization",
        "activation": "simulator/parametrization/activation",
    }
    add_hydra_models_to_config_store(cs, rename_groups)

    # loss:
    cs.store(name="4dvar", node=Four4DVarLoss, group="loss")

    # simulator:
    model_group = "simulator"
    cs.store(name="l96_parametrized_base", node=L96Parametrized, group=model_group)

    # encoder:
    cs.store(name="unet", node=Unet, group="assimilation_network")

    # lightning module:
    cs.store(name="data_assimilation_module", node=DataAssimilationModule, group="lightning_module")
    cs.store(name="parameter_tuning_module", node=ParameterTuningModule, group="lightning_module")
    cs.store(name="parametrization_learning_module", node=ParametrizationLearningModule, group="lightning_module")

    # datamodule:
    cs.store(name="l96_datamodule_base", node=L96DataModule, group="datamodule")
    cs.store(name="l96_dataset_base", node=L96Dataset, group="datamodule/dataset")
    cs.store(name="l96_dataset_base", node=L96Dataset, group="dataset")
    cs.store(name="l96_inference_dataset_base", node=L96InferenceDataset, group="datamodule/dataset")
    cs.store(name="l96_inference_dataset_base", node=L96InferenceDataset, group="dataset")

    # register the base config class (this name has to be called in config.yaml):
    cs.store(name="base_config", node=Config)
    cs.store(name="evaluate_config", node=ConfigEvaluateDA)


@dataclass
class Config:
    output_dir_base_path: str = MISSING
    print_config: bool = True
    random_seed: int = 101
    debug: bool = False
    rollout_length: int = MISSING
    input_window_length: int = MISSING
    time_step: float = MISSING

    simulator: Any = MISSING
    assimilation_network: Any = MISSING
    loss: Any = MISSING
    lightning_module: Any = MISSING
    assimilation_network_checkpoint: Any = None
    optimizer: Any = MISSING

    datamodule: Any = MISSING
    lightning_trainer: Any = MISSING
    lightning_logger: Any = MISSING
    lightning_callback: Any = None


@dataclass
class ConfigEvaluateDA:
    output_dir_base_path: str = MISSING
    exp_base_dir: str = MISSING
    random_seed: int = 101
    n_trials: int = 1000
    observations_length: int = 100
    search_in_directories: Any = MISSING
    simulator: Any = MISSING
    dataset: Any = MISSING
