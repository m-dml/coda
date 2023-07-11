from dataclasses import dataclass
from typing import Any

from hydra.conf import ConfigStore, MISSING
from mdml_tools.utils import add_hydra_models_to_config_store

from hpl.lib.datamodule import L96DataModule, L96Dataset, L96InferenceDataset
from hpl.lib.lightning_module import DataAssimilationModule, ParameterTuningModule, ParametrizationLearningModule
from hpl.lib.loss import Four4DVarLoss
from hpl.lib.model import FullyConvolutionalNetwork, L96Parametrized
from hpl.lib.unet import (
    ConvolutionalDecoder,
    ConvolutionalDecodingBlock,
    ConvolutionalEncoder,
    ConvolutionalEncodingBlock,
    GlobalAvgPool,
    GlobalMaxPool,
    PeriodicConv1d,
    PeriodicConv2d,
    Unet,
)


def register_configs() -> None:
    cs = ConfigStore.instance()
    # add hydra models from mdml_tools
    rename_groups = {
        "optimizer": ["optimizer/data_assimilation", "optimizer/parametrization"],
        "simulator": ["simulator", "datamodule/simulator"],
        "model": "simulator/parametrization",
        "activation": [
            "simulator/parametrization/activation",
            "assimilation_network/encoder/block/activation",
            "assimilation_network/decoder/block/activation",
        ],
        "convolution": [
            "simulator/parametrization/convolution",
            "assimilation_network/encoder/block/convolution",
            "assimilation_network/decoder/block/convolution",
            "assimilation_network/decoder/block/upscale",
            "assimilation_network/output_convolution",
        ],
        "batch_norm": [
            "simulator/parametrization/batch_norm",
            "assimilation_network/encoder/block/batch_norm",
            "assimilation_network/decoder/block/batch_norm",
        ],
        "dropout": [
            "simulator/parametrization/dropout",
            "assimilation_network/encoder/block/dropout",
            "assimilation_network/decoder/block/dropout",
        ],
        "pooling": [
            "assimilation_network/encoder/block/pooling",
        ],
    }
    add_hydra_models_to_config_store(cs, rename_groups)

    # loss:
    cs.store(name="4dvar", node=Four4DVarLoss, group="loss")

    # simulator:
    model_group = "simulator"
    cs.store(name="l96_parametrized_base", node=L96Parametrized, group=model_group)

    # encoder:
    cs.store("unet_base", node=Unet, group="assimilation_network")
    cs.store("conv_encoder_base", node=ConvolutionalEncoder, group="assimilation_network/encoder")
    cs.store("conv_block_encoding_base", node=ConvolutionalEncodingBlock, group="assimilation_network/encoder/block")
    cs.store("conv_decoder_base", node=ConvolutionalDecoder, group="assimilation_network/decoder")
    cs.store("conv_block_decoding_base", node=ConvolutionalDecodingBlock, group="assimilation_network/decoder/block")
    cs.store("global_max_pool_base", node=GlobalMaxPool, group="assimilation_network/global_pool")
    cs.store("global_avg_pool_base", node=GlobalAvgPool, group="assimilation_network/global_pool")

    for group_name in rename_groups["convolution"]:
        cs.store("periodic_conv1d_base", node=PeriodicConv1d, group=group_name)
        cs.store("periodic_conv2d_base", node=PeriodicConv2d, group=group_name)

    # parametrization:
    cs.store(name="fully_convolutional_network_base", node=FullyConvolutionalNetwork, group="simulator/parametrization")

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
